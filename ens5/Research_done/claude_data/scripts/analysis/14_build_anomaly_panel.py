from _rp import _RP  # portable-path shim (added when vendoring)
"""
14_build_anomaly_panel.py — Build daily anomaly characteristic panel for BIST50

Computes ~12-15 anomaly characteristics for every (ticker, day) over 2021-2026.
This is the input to:
  - Stage A monthly tests (script 15)
  - Stage B intraday risk regression (script 17)

Anomalies built (Brogaard-style + BIST-specific):
  Tier 1 — price-based (no fundamentals lag needed):
    R1  size      = log(market_cap)
    R3  beta_36m  = 36-month rolling β to MKT_BIST_daily (or 1y / 6mo as alternatives)
    R4  mom_12_1  = cumret_{t-252, t-22}
    R5  rev_1m    = cumret_{t-21, t-1}
    R15 ivol_60d  = std of FF3 residuals over 60 days
    R16 amh_60d   = Amihud illiquidity = mean(|r| / dollar_volume) over 60 days
    R17 max_21d   = max daily return over last 21 days  (lottery preference)
    R18 ceq_12m   = composite equity issuance = ΔlogME_12m − return_12m (Round 12)

  Tier 1 — fundamental-based (60-day filing lag, TAS-29 deflated):
    R2  bm        = book equity / market cap
    R6  ag        = annual asset growth = (TA_t - TA_{t-4q}) / TA_{t-4q}
    R6b inv_to_ta = (Inventory_t − Inventory_{t-252d}) / lagged TA (Round 12)
    R7  gp        = gross profitability = GROSS_PROFIT_TTM / TA  [FIXED Round 12 — uses code 3D]
    R7c margin    = gross margin = GROSS_PROFIT_TTM / Revenue_TTM (Round 12)
    R8  roa       = quarterly NI / lagged TA
    R9  nsi       = net stock issues = log(shares_t / shares_{t-252d})
    R11 cash      = Cash_PIT / TA_PIT (Round 12)
    R12 acc       = accruals = (NI_TTM - OCF_TTM) / TA_avg
    R13 cf_yield  = OCF_TTM / MarketCap (Round 12)
    R14 cf_ta     = OCF_TTM / lagged TA (Round 12)
    R19 net_debt_fin = ΔTotalDebt / lagged TA (Round 12)
    R20 capex_ta  = |CapEx_TTM| / lagged TA (Round 12)
    R24 int_cov   = EBIT_TTM / |FinancialExpense_TTM| (Round 12)
    R25 dwc_ta    = ΔWorkingCapital_TTM / lagged TA (Round 12)
    R26 lev_book  = (STDebt+LTDebt) / Equity (Round 12)
    R26b lev_mkt  = (STDebt+LTDebt) / (STDebt+LTDebt+ME) (Round 12)
    R28 sales_g   = (Revenue_TTM_t − Revenue_TTM_{t-252d}) / Revenue_TTM_{t-252d} (Round 12)
    R29 op_margin = EBIT_TTM / Revenue_TTM (Round 12)
    R36 fcf_yield = FreeCashFlow_TTM / MarketCap (Round 12)

  Tier 3 — BIST-specific (FX/hedge/sector):
    R23 try_beta_60d   = 60-day rolling β to ΔUSDTRY
    R23b try_beta_252d = 1-year rolling β to ΔUSDTRY (Round 12)
    R31 net_fx_ta      = Net FX Position_PIT / TA (pre-hedge, Round 12)
    R32 net_fx_hedged_ta = Net FX Position incl. hedge / TA (Round 12)
    R33 hedge_intensity  = |raw_FX − hedged_FX| / |raw_FX| (Round 12)
    R34 export_share     = ExportSales_TTM / Revenue_TTM (Round 12)
    R35 try_dn_beta_60d  = Downside FX beta — regression on TRY-weakening days only (Round 12)
    R27 sector           = one of {Banks, Industrials, Holdings, Consumer, Energy, Telecom}

  Round 13 (2026-05-12) — 32 new factors across 7 tiers:
    Tier 1 sector-neutralized canonical 5:
      R12_acc_sn, MKT_beta_6m_sn, R1_size_sn, R4_mom_12_1_sn, R8_roa_sn
    Tier 2A orthogonalized: R26_lev_book_orth
    Tier 2B z-score composites:
      Quality_z, Value_z, Accruals_z, Investment_z, LowVol_z
    Tier 3 price-based:
      R4b_mom_6_1, R5b_rev_60m, R5c_rev_5d, R15b_rvol_60d, R21_turnover_60d,
      R22_skew_60d, R22b_iskew_60d, R22c_coskew_60d, R7_mom_consistency
    Tier 4 fundamentals:
      R10_noa, R29b_oat, R29c_em, R38_inv_intensity, R40_oa_growth,
      R41_earnings_stability, R42_sales_stability
    Tier 5: R43_F_score (8-point Piotroski)
    Tier 6 interactions: R46_mom_x_vol, R47_mom_x_size, R48_acc_x_size
    Tier 7: R50_sector_xs_beta

Output:
  results/cross_factor/anomaly_panel_daily.parquet
  results/cross_factor/anomaly_panel_monthly.parquet  (resampled to month-end)

Usage:
  python analysis/14_build_anomaly_panel.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(_RP(r""))

sys.path.insert(0, str(ROOT / 'claude_data' / 'scripts'))
from shared.data import (
    load_prices, load_fundamentals, load_sectors,
    BANKS, NI_CODE_BANK, NI_CODE_NB, EQ_CODE_BANK, EQ_CODE_NB
)
from shared import configs as CFG

DATA_DIR    = CFG.DATA_DIR
RESULTS_DIR = CFG.RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────
# Constants — pulled from shared.configs
# ─────────────────────────────────────────────────────────────────────
LAG_DAYS   = CFG.LAG_FUNDAMENTALS_DAYS
START_DATE = CFG.PERIODS['FULL'][0]
END_DATE   = CFG.PERIODS['FULL'][1]

# ─────────────────────────────────────────────────────────────────────
# Step 0: Load everything
# ─────────────────────────────────────────────────────────────────────
print('=' * 70)
print('ANOMALY PANEL BUILDER')
print('=' * 70)

print('\n[1/8] Loading prices, fundamentals, sectors, FF3...')
px = load_prices()
px = px[(px['date'] >= START_DATE) & (px['date'] <= END_DATE)].copy()

ret_wide = px.pivot(index='date', columns='ticker', values='ret')
mcap_wide = px.pivot(index='date', columns='ticker', values='mktcap_try')
close_wide = px.pivot(index='date', columns='ticker', values='close_adj')
vol_wide = px.pivot(index='date', columns='ticker', values='volume') if 'volume' in px.columns else None

fund = load_fundamentals()
fine_sec, macro_sec = load_sectors()

# Load BIST FF3 (must run script 13 first)
ff3_path = RESULTS_DIR / 'bist_ff3_daily.parquet'
if not ff3_path.exists():
    print('  ERROR: Run analysis/13_build_bist_ff3.py first.')
    sys.exit(1)
ff3 = pd.read_parquet(ff3_path)
print(f'      FF3 loaded: {ff3.shape}')

# Load USDTRY for TRY beta
usdtry_path = DATA_DIR / 'usdtry_daily.parquet'
if usdtry_path.exists():
    usdtry = pd.read_parquet(usdtry_path)
    usdtry['date'] = pd.to_datetime(usdtry['date'])
    # File schema is ['Tarih', 'TP_DK_USD_A_YTL', 'date'] (TCMB feed). The
    # FX rate column is 'TP_DK_USD_A_YTL' (the Tarih column is a string date).
    # Pick the first numeric column to be robust to schema variants.
    fx_col = next((c for c in ['close', 'TP_DK_USD_A_YTL', 'usdtry', 'rate']
                   if c in usdtry.columns and pd.api.types.is_numeric_dtype(usdtry[c])),
                  None)
    if fx_col is None:
        numeric_cols = [c for c in usdtry.columns
                         if pd.api.types.is_numeric_dtype(usdtry[c])]
        fx_col = numeric_cols[0] if numeric_cols else None
    if fx_col is None:
        print(f'      WARNING: no numeric column in {usdtry_path.name}; '
              f'columns are {list(usdtry.columns)}. Skipping TRY beta.')
        usdtry_ret = None
    else:
        usdtry_series = usdtry.set_index('date')[fx_col]
        usdtry_ret = usdtry_series.pct_change()
        print(f'      USDTRY loaded ({fx_col}): {usdtry_series.iloc[-1]:.4f} '
              f'as of {usdtry_series.index[-1].date()}')
else:
    print('      WARNING: no usdtry data, skipping TRY beta')
    usdtry_ret = None

# ─────────────────────────────────────────────────────────────────────
# Step 1: Tier 1 price-based anomalies
# ─────────────────────────────────────────────────────────────────────
print('\n[2/8] Building price-based anomalies (R1, R3, R4, R5, R15, R16, R17)...')

# R1 SIZE = log(mktcap)
size = np.log(mcap_wide)

# R3 BETA_36M (36-month = ~756 trading days)
# Use rolling regression: r_i = a + β·MKT + ε
mkt = ff3['MKT'].reindex(ret_wide.index)
def rolling_beta(stock_ret, mkt_ret, window):
    """Rolling β = Cov(stock, mkt) / Var(mkt)."""
    cov = stock_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var

# Use MIN of (36m, available data) — start with whatever we have
WINDOW_36M = min(756, len(ret_wide) - 1)  # cap at available history
beta_36m = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)
for ticker in ret_wide.columns:
    beta_36m[ticker] = rolling_beta(ret_wide[ticker], mkt, WINDOW_36M)

# Also build a 1-year (252d) and 6-month (126d) version for robustness
beta_1y = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)
beta_6m = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)
for ticker in ret_wide.columns:
    beta_1y[ticker] = rolling_beta(ret_wide[ticker], mkt, 252)
    beta_6m[ticker] = rolling_beta(ret_wide[ticker], mkt, 126)

# R4 MOM_12_1 = cumret_{t-252, t-22}
log_close = np.log(close_wide)
mom_12_1 = log_close.shift(22) - log_close.shift(252)

# R5 REV_1M = cumret_{t-21, t-1}
rev_1m = log_close.shift(1) - log_close.shift(21)

# R17 MAX_21D = max daily return over last 21 days  (single-day max, Bali-Cakici-Whitelaw 2011)
max_21d = ret_wide.rolling(21).max()

# R17b MAX5_21D = mean of top 5 daily returns over last 21 days  (smoothed BCW)
#   Less noisy than single max; reduces sensitivity to a single outlier day.
print('      Computing MAX5 (top-5-day mean over 21d)...')
def _top5_mean(window):
    if np.isnan(window).all():
        return np.nan
    valid = window[~np.isnan(window)]
    if len(valid) < 5:
        return np.nan
    return np.sort(valid)[-5:].mean()
max5_21d = ret_wide.rolling(21, min_periods=10).apply(_top5_mean, raw=True)

# R15 IVOL_60D = std of FF3 residuals
# Run rolling regression: r_i = a + b·MKT + c·SMB + d·HML + ε
# Idio vol = std(ε) over 60 days
print('      Computing 60-day idiosyncratic volatility...')
ff3_aligned = ff3[['MKT', 'SMB', 'HML']].reindex(ret_wide.index)

def rolling_ivol(stock_ret, factors, window=60):
    """Rolling residual std from r = a + β·F + ε."""
    n = len(stock_ret)
    ivol = pd.Series(index=stock_ret.index, dtype=float)
    valid = stock_ret.notna() & factors.notna().all(axis=1)
    
    for i in range(window, n):
        s_win = stock_ret.iloc[i - window:i]
        f_win = factors.iloc[i - window:i]
        valid_win = s_win.notna() & f_win.notna().all(axis=1)
        if valid_win.sum() < 30:
            continue
        s = s_win[valid_win].values
        F = f_win[valid_win].values
        # OLS: β = (X'X)^-1 X'y
        X = np.column_stack([np.ones(len(F)), F])
        try:
            beta = np.linalg.lstsq(X, s, rcond=None)[0]
            resid = s - X @ beta
            ivol.iloc[i] = resid.std()
        except np.linalg.LinAlgError:
            continue
    return ivol

ivol_60d = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
for ticker in ret_wide.columns:
    ivol_60d[ticker] = rolling_ivol(ret_wide[ticker], ff3_aligned, 60)

# R16 AMIHUD = mean(|r| / dollar_volume) over 60 days
print('      Computing Amihud illiquidity...')
if vol_wide is not None and not vol_wide.empty:
    dollar_vol = vol_wide * close_wide
    illiq_daily = ret_wide.abs() / dollar_vol.replace(0, np.nan)
    amh_60d = illiq_daily.rolling(60, min_periods=20).mean() * 1e6  # scale for readability
else:
    print('      WARNING: no volume data, Amihud = NaN')
    amh_60d = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)

# ─────────────────────────────────────────────────────────────────────
# Step 2: Fundamental-based anomalies (60-day lag)
# ─────────────────────────────────────────────────────────────────────
print('\n[3/8] Building fundamental anomalies (R2 BM, R8 ROA, R6 AG)...')

def build_pit_panel(fund_long, code_bank, code_nb, value_name):
    """Build PIT-lagged ticker × date panel for a fundamental item."""
    bank_data = fund_long[fund_long['FINANCIAL_ITEM_CODE'] == code_bank][['SYMBOL', 'period_end', 'value']]
    nb_data = fund_long[fund_long['FINANCIAL_ITEM_CODE'] == code_nb][['SYMBOL', 'period_end', 'value']]
    df = pd.concat([bank_data, nb_data], ignore_index=True)
    df.columns = ['ticker', 'period_end', value_name]
    df = df.dropna(subset=[value_name])
    df['available_from'] = df['period_end'] + pd.Timedelta(days=LAG_DAYS)
    df = df.sort_values(['ticker', 'available_from'])
    
    # For each ticker, merge_asof to get latest available value per date
    records = []
    for ticker in fund_long['SYMBOL'].unique():
        df_t = df[df['ticker'] == ticker]
        if len(df_t) == 0:
            continue
        dates = pd.DataFrame({'date': ret_wide.index})
        merged = pd.merge_asof(
            dates.sort_values('date'),
            df_t[['available_from', value_name]].rename(columns={'available_from': 'date'}),
            on='date',
            direction='backward'
        )
        merged['ticker'] = ticker
        records.append(merged[['ticker', 'date', value_name]])
    
    if not records:
        return pd.DataFrame()
    panel_long = pd.concat(records, ignore_index=True)
    panel_wide = panel_long.pivot(index='date', columns='ticker', values=value_name)
    return panel_wide.reindex(ret_wide.index)


def build_ttm_panel_nb(fund_long, code, lag_days=LAG_DAYS, abs_value=False):
    """Build TTM panel for a single non-bank cash-flow / income-statement code.

    Uses shared.data.compute_ttm to de-cumulate Turkish cumulative-YTD reports
    and roll to TTM. Banks excluded (their chart of accounts is different).
    Returns wide DataFrame (date × ticker) with banks as NaN columns.

    Parameters
    ----------
    code : str
        KAP FINANCIAL_ITEM_CODE for the non-bank item.
    abs_value : bool
        If True, return |TTM value|. Useful for items stored as negative
        (CapEx 4CAI, Financial Expenses 4BB) where we want a magnitude.
    """
    records = []
    for ticker in ret_wide.columns:
        if ticker in BANKS:
            continue
        ttm_df = compute_ttm(fund_long, ticker, code, lag_days=lag_days)
        if ttm_df.empty:
            continue
        dates = pd.DataFrame({'date': ret_wide.index}).sort_values('date')
        src = ttm_df[['signal_date', 'ttm']].rename(columns={'signal_date': 'date'}).sort_values('date')
        merged = pd.merge_asof(dates, src, on='date', direction='backward')
        merged['ticker'] = ticker
        records.append(merged[['ticker', 'date', 'ttm']])

    if not records:
        return pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)

    long_df = pd.concat(records, ignore_index=True)
    wide = long_df.pivot(index='date', columns='ticker', values='ttm').reindex(ret_wide.index)
    # Reindex to ALL tickers (banks get NaN)
    for t in ret_wide.columns:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[ret_wide.columns]
    if abs_value:
        wide = wide.abs()
    return wide


# ──────────────────────────────────────────────────────────────────
# Round 13 (2026-05-12) — helper functions for sector-neutralization,
# orthogonalization, z-score composites, and higher-moment rolling stats.
# ──────────────────────────────────────────────────────────────────
def cross_sectional_zscore(panel):
    """Per-row (per-date) z-score standardization of a wide panel.

    Returns the panel with each row's mean=0 and std=1. Rows with constant
    or all-NaN values return NaN to avoid divide-by-zero.
    """
    row_std = panel.std(axis=1)
    row_mean = panel.mean(axis=1)
    return panel.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)


def composite_z(signed_panels):
    """Combine multiple panels into a single z-score composite.

    Parameters
    ----------
    signed_panels : list of (panel, sign) tuples
        sign ∈ {+1, -1}. The output equals the mean of (sign × z(panel))
        across all panels, NaN-aware (uses np.nanmean).
    """
    z_panels = [cross_sectional_zscore(p) * s for (p, s) in signed_panels]
    # nanmean across the stack
    stacked = np.stack([z.values for z in z_panels], axis=0)
    avg = np.nanmean(stacked, axis=0)
    out = pd.DataFrame(avg, index=z_panels[0].index, columns=z_panels[0].columns)
    # If all components were NaN at a cell, np.nanmean returns NaN — keep that.
    return out


def sector_neutralize_panel(signal_panel, sector_panel):
    """Subtract sector mean from a wide signal panel, per date.

    sector_panel has the same shape as signal_panel and contains integer
    sector codes. Both NaN-safe; sectors with <2 stocks at a date leave
    those cells unchanged (mean would equal the value).
    """
    result = pd.DataFrame(np.nan, index=signal_panel.index, columns=signal_panel.columns)
    for date in signal_panel.index:
        sig_row = signal_panel.loc[date]
        sec_row = sector_panel.loc[date]
        valid = sig_row.notna() & sec_row.notna()
        if valid.sum() < 5:
            continue
        df = pd.DataFrame({'signal': sig_row[valid], 'sector': sec_row[valid]})
        sector_means = df.groupby('sector')['signal'].transform('mean')
        residual = df['signal'] - sector_means
        result.loc[date, residual.index] = residual.values
    return result


def cross_sectional_residualize(target_panel, control_panels):
    """Per-date OLS residualization: target ~ intercept + controls.

    For each date, regress target_panel.loc[date] on the controls cross-sectionally
    and return the residuals. Used to test whether a factor adds incremental signal
    beyond known controls (e.g. R26_lev_book residualized on R1_size + R8_roa).
    """
    result = pd.DataFrame(np.nan, index=target_panel.index, columns=target_panel.columns)
    for date in target_panel.index:
        y = target_panel.loc[date].values.astype(float)
        Xs = [c.loc[date].values.astype(float) for c in control_panels]
        X_mat = np.column_stack(Xs) if Xs else None
        if X_mat is None:
            continue
        valid = ~np.isnan(y) & ~np.isnan(X_mat).any(axis=1)
        if valid.sum() < 10:
            continue
        y_v = y[valid]
        X_v = X_mat[valid]
        X_v_aug = np.column_stack([np.ones(X_v.shape[0]), X_v])
        try:
            beta, *_ = np.linalg.lstsq(X_v_aug, y_v, rcond=None)
            resid = y_v - X_v_aug @ beta
            valid_idx = np.where(valid)[0]
            row_idx = result.index.get_loc(date)
            for j, col_idx in enumerate(valid_idx):
                result.iat[row_idx, col_idx] = resid[j]
        except np.linalg.LinAlgError:
            continue
    return result


def rolling_coskew(stock_ret, mkt_ret, window=60, min_obs=30):
    """Rolling coskewness with the market:
       coskew = E[(r_s - mu_s)(r_m - mu_m)^2] / (std(r_s) * var(r_m)).
    """
    result = pd.Series(index=stock_ret.index, dtype=float)
    sr = stock_ret.values
    mr = mkt_ret.values
    for i in range(window, len(stock_ret)):
        s_win = sr[i - window:i]
        m_win = mr[i - window:i]
        mask = ~np.isnan(s_win) & ~np.isnan(m_win)
        if mask.sum() < min_obs:
            continue
        s = s_win[mask]
        m = m_win[mask]
        s_c = s - s.mean()
        m_c = m - m.mean()
        var_m = (m_c ** 2).mean()
        std_s = s.std()
        if var_m == 0 or std_s == 0:
            continue
        coskew = (s_c * (m_c ** 2)).mean() / (std_s * var_m)
        result.iloc[i] = coskew
    return result


def rolling_idio_skew(stock_ret, factors, window=60, min_obs=30):
    """Skewness of FF3 residuals over a rolling window."""
    result = pd.Series(index=stock_ret.index, dtype=float)
    for i in range(window, len(stock_ret)):
        s_win = stock_ret.iloc[i - window:i]
        f_win = factors.iloc[i - window:i]
        valid = s_win.notna() & f_win.notna().all(axis=1)
        if valid.sum() < min_obs:
            continue
        s = s_win[valid].values
        F = f_win[valid].values
        X = np.column_stack([np.ones(len(F)), F])
        try:
            beta, *_ = np.linalg.lstsq(X, s, rcond=None)
            resid = s - X @ beta
            result.iloc[i] = pd.Series(resid).skew()
        except np.linalg.LinAlgError:
            continue
    return result


# Equity → BM
eq_wide = build_pit_panel(fund, EQ_CODE_BANK, EQ_CODE_NB, 'equity')
bm = eq_wide / mcap_wide
bm = bm.where(bm > 0)  # drop negative BM (distressed)

# Net Income → ROA needs Total Assets too
# For non-banks: NI = 3L (cumulative YTD), TA = 1BL (stock variable)
# For banks:     NI = 3Z (cumulative YTD), TA = 1Z
#
# CORRECTNESS FIX (2026-05-08): Turkish reports are CUMULATIVE within fiscal year.
# Q3 NI is Jan-Sep, NOT Jul-Sep. Direct latest-NI / latest-TA gives quarter-of-year-
# dependent values (Q4 4x larger than Q1). Brogaard's ROA = TTM NI / TA, so we use
# shared.data.compute_ttm() which de-cumulates and rolls 4 quarters into TTM.
print('      Computing TTM NI per ticker (de-cumulating quarterly reports)...')
from shared.data import compute_ttm

ni_ttm_records = []
for ticker in ret_wide.columns:
    code = NI_CODE_BANK if ticker in BANKS else NI_CODE_NB
    ttm_df = compute_ttm(fund, ticker, code, lag_days=LAG_DAYS)
    if ttm_df.empty:
        continue
    # Build PIT panel: for each daily date, latest available TTM
    dates = pd.DataFrame({'date': ret_wide.index}).sort_values('date')
    src = ttm_df[['signal_date', 'ttm']].rename(columns={'signal_date': 'date'}).sort_values('date')
    merged = pd.merge_asof(dates, src, on='date', direction='backward')
    merged['ticker'] = ticker
    ni_ttm_records.append(merged[['ticker', 'date', 'ttm']].rename(columns={'ttm': 'ni_ttm'}))

if ni_ttm_records:
    ni_ttm_long = pd.concat(ni_ttm_records, ignore_index=True)
    ni_ttm_wide = ni_ttm_long.pivot(index='date', columns='ticker', values='ni_ttm').reindex(ret_wide.index)
else:
    ni_ttm_wide = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)

# Keep ni_wide (stock-of-cumulative) for AG/accruals legacy; but use ni_ttm_wide for ROA
ni_wide = build_pit_panel(fund, NI_CODE_BANK, NI_CODE_NB, 'ni')
ta_wide = build_pit_panel(fund, '1Z', '1BL', 'ta')
ta_wide = ta_wide.where(ta_wide > 0)

# ROA = NI_TTM / TA_lagged_one_quarter (~63 trading days)
ta_lag_q = ta_wide.shift(63)  # ~1 quarter lag, per Brogaard
roa = ni_ttm_wide / ta_lag_q
print(f'      ROA panel: coverage {roa.notna().mean().mean()*100:.1f}%, '
      f'mean {roa.values[~np.isnan(roa.values)].mean():.4f}')

# Asset Growth: ΔTA / TA(prev year)
# Approximate via 252-day shift on the panel
ta_lag = ta_wide.shift(252)
ag = (ta_wide - ta_lag) / ta_lag

# Gross Profitability: GROSS_PROFIT_TTM / TA  (FIXED Round 12 — 2026-05-12)
# Previous formula used (3C − 3DA) / TA where 3DA = Marketing/Selling/Distribution
# Expenses, NOT COGS. Real COGS is 3CA; gross profit (3C − |3CA|) is reported
# directly as code 3D. Additionally, the old code used build_pit_panel for
# revenue/COGS — these are flow items reported cumulative-YTD, so build_pit_panel
# gave quarter-of-year-dependent values. Both issues fixed by using
# build_ttm_panel_nb on code 3D directly.
print('      Computing gross profitability via TTM (non-banks only)...')
gp_ttm_wide = build_ttm_panel_nb(fund, '3D')
gp = gp_ttm_wide / ta_wide

# Accruals = (NI_TTM - OCF_TTM) / TA   (using TTM versions per Sloan 1996)
print('      Computing accruals via TTM NI and TTM OCF (non-banks only)...')
ocf_ttm_records = []
for ticker in ret_wide.columns:
    if ticker in BANKS:
        continue
    ttm_df = compute_ttm(fund, ticker, '4C', lag_days=LAG_DAYS)
    if ttm_df.empty:
        continue
    dates = pd.DataFrame({'date': ret_wide.index}).sort_values('date')
    src = ttm_df[['signal_date', 'ttm']].rename(columns={'signal_date': 'date'}).sort_values('date')
    merged = pd.merge_asof(dates, src, on='date', direction='backward')
    merged['ticker'] = ticker
    ocf_ttm_records.append(merged[['ticker', 'date', 'ttm']].rename(columns={'ttm': 'ocf_ttm'}))

if ocf_ttm_records:
    ocf_ttm_long = pd.concat(ocf_ttm_records, ignore_index=True)
    ocf_ttm_wide = ocf_ttm_long.pivot(index='date', columns='ticker', values='ocf_ttm').reindex(ret_wide.index)
    # Reindex to all tickers (banks get NaN)
    for t in ret_wide.columns:
        if t not in ocf_ttm_wide.columns:
            ocf_ttm_wide[t] = np.nan
    ocf_ttm_wide = ocf_ttm_wide[ret_wide.columns]
else:
    ocf_ttm_wide = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)

acc = (ni_ttm_wide - ocf_ttm_wide) / ta_wide
for ticker in BANKS:
    if ticker in acc.columns:
        acc[ticker] = np.nan

# Net stock issues — log change in shares outstanding
# We don't have shares panel directly; approximate via mcap/price:
# shares = mcap / close → log change over 252 days
shares_implied = mcap_wide / close_wide
log_shares = np.log(shares_implied.replace(0, np.nan))
nsi = log_shares - log_shares.shift(252)

# ===================================================================
# Added 2026-05-09 — explicit F02 (earnings yield) and F04 (ROE)
# These mirror the legacy F02/F03/F04 portfolio characteristics so we
# can run Test A + Test B on them under their canonical names.
# ===================================================================
print('\n[3b/8] Building F02 (E/P) and F04 (ROE)...')

# F02 EARNINGS YIELD = NI_TTM / market_cap  (Basu 1977 / Fama-French)
# Inverse of P/E. We use TTM NI so the signal is consistent year-round.
f02_ep = ni_ttm_wide / mcap_wide
print(f'      F02_EP   coverage: {f02_ep.notna().mean().mean()*100:.1f}%')

# F04 ROE = NI_TTM / EQUITY  (equity is the same PIT-lagged panel as in BM)
# Note: ROA is NI/TA (asset-based); ROE is NI/Equity (equity-based).
f04_roe = ni_ttm_wide / eq_wide.where(eq_wide > 0)
print(f'      F04_ROE  coverage: {f04_roe.notna().mean().mean()*100:.1f}%')

# ===================================================================
# Round 12 (2026-05-12) — new fundamental + FX factors
#   Tier B/C/D/E/F items per Factor_Batch_Round12_Plan.md
#   16 new KAP codes; 19 new factor columns + R7_gp fix
# ===================================================================
print('\n[3c/8] Round 12: building new fundamental + FX factor inputs...')

# --- PIT stock items (non-bank only; build_pit_panel will leave banks empty,
#     we explicitly NaN them below for safety) ---
cash_wide          = build_pit_panel(fund, 'NA_BANK_NO_X', '1AA',  'cash')
inv_wide           = build_pit_panel(fund, 'NA_BANK_NO_X', '1AF',  'inv')
st_debt_wide       = build_pit_panel(fund, 'NA_BANK_NO_X', '2AA',  'st_debt')
lt_debt_wide       = build_pit_panel(fund, 'NA_BANK_NO_X', '2BA',  'lt_debt')
fx_pos_wide        = build_pit_panel(fund, 'NA_BANK_NO_X', '4BE',  'fx_pos')
fx_pos_mon_wide    = build_pit_panel(fund, 'NA_BANK_NO_X', '4BEA', 'fx_pos_mon')
fx_pos_hedged_wide = build_pit_panel(fund, 'NA_BANK_NO_X', '4BEB', 'fx_pos_hedged')

for w in [cash_wide, inv_wide, st_debt_wide, lt_debt_wide,
          fx_pos_wide, fx_pos_mon_wide, fx_pos_hedged_wide]:
    for ticker in BANKS:
        if ticker in w.columns:
            w[ticker] = np.nan

# --- TTM flow items (non-bank only by construction in build_ttm_panel_nb) ---
print('      Computing TTM panels for 8 income-statement / cash-flow items...')
ebit_ttm_wide     = build_ttm_panel_nb(fund, '3DF')
finexp_ttm_wide   = build_ttm_panel_nb(fund, '4BB',  abs_value=True)
domsales_ttm_wide = build_ttm_panel_nb(fund, '4BC')
expsales_ttm_wide = build_ttm_panel_nb(fund, '4BD')
capex_ttm_wide    = build_ttm_panel_nb(fund, '4CAI', abs_value=True)
dwc_ttm_wide      = build_ttm_panel_nb(fund, '4CAF')   # keep sign — informative for accruals
fcf_ttm_wide      = build_ttm_panel_nb(fund, '4CB')    # keep sign — distressed firms negative
sales_ttm_wide    = build_ttm_panel_nb(fund, '3C')

# Total debt = ST + LT. Use .add(fill_value=0) so a NaN in ONE leg still
# contributes (treated as 0), but NaN in BOTH legs (banks) stays NaN.
# This is more robust than fillna(0)+fillna(0)+masking which can fail when
# the two frames have slightly different column sets (object-dtype invert).
total_debt_wide = st_debt_wide.add(lt_debt_wide, fill_value=0)

# Helper: cross-sectional winsorization for outlier-prone ratios
def winsorize_xs(panel, lo=0.01, hi=0.99):
    """Per-row (per-date) cross-sectional winsorization at [lo, hi] quantiles."""
    lo_q = panel.quantile(lo, axis=1)
    hi_q = panel.quantile(hi, axis=1)
    return panel.clip(lower=lo_q, upper=hi_q, axis=0)

# Round 12 factor definitions (19 new + R7_gp fixed above)
print('      Computing 19 new Round 12 factors...')

# R18 Composite Equity Issuance (price-based; all tickers including banks)
# CEI = ΔlogME(252) − (logP(t-22) − logP(t-252)). Sign: long low (heavy
# issuers underperform). Daniel-Titman 2006.
log_mcap_wide = np.log(mcap_wide.replace(0, np.nan))
ceq_12m = ((log_mcap_wide - log_mcap_wide.shift(252))
           - (log_close.shift(22) - log_close.shift(252)))

# R7c Gross Margin (scale-free; non-bank only)
gp_margin = gp_ttm_wide / sales_ttm_wide.replace(0, np.nan)

# R13 Cash Flow Yield (non-bank only; ocf_ttm_wide already NaN for banks)
cf_yield = ocf_ttm_wide / mcap_wide

# R14 OCF-ROA = OCF_TTM / lagged TA (non-bank only)
cf_ta = ocf_ttm_wide / ta_lag_q

# R28 Sales Growth YoY (non-bank only) — log-difference for robustness
# to (a) hyperinflation magnitude blow-up and (b) near-zero prior-year sales
# producing 100x outliers. log(S_t / S_{t-252}) is bounded for positive
# inputs and is the standard factor-research form for growth rates.
sales_lag = sales_ttm_wide.shift(252)
sales_g = (np.log(sales_ttm_wide.where(sales_ttm_wide > 0))
           - np.log(sales_lag.where(sales_lag > 0)))

# R11 Cash / TA (non-bank only; Palazzo 2012)
cash_ta = cash_wide / ta_wide

# R6b Inventory growth scaled by lagged TA (non-bank only)
inv_to_ta = (inv_wide - inv_wide.shift(252)) / ta_wide.shift(252)

# R20 CapEx / lagged TA (non-bank only; capex_ttm_wide already abs())
capex_ta = capex_ttm_wide / ta_lag_q

# R24 Interest Coverage — winsorized cross-sectional 1/99%
int_cov_raw = ebit_ttm_wide / finexp_ttm_wide.replace(0, np.nan)
int_cov = winsorize_xs(int_cov_raw, 0.01, 0.99)

# R29 Operating Margin (non-bank only)
op_margin = ebit_ttm_wide / sales_ttm_wide.replace(0, np.nan)

# R26 Book Leverage = TotalDebt / Equity (non-bank only)
lev_book = total_debt_wide / eq_wide.where(eq_wide > 0)

# R26b Market Leverage = TotalDebt / (TotalDebt + ME)
lev_mkt = total_debt_wide / (total_debt_wide + mcap_wide)

# R19 Net Debt Issuance = ΔTotalDebt / lagged TA
net_debt_fin = (total_debt_wide - total_debt_wide.shift(252)) / ta_wide.shift(252)

# R25 ΔWorkingCapital / lagged TA — signed (cash-flow-statement direct measure)
dwc_ta = dwc_ttm_wide / ta_lag_q

# R36 FCF Yield — signed (distressed firms negative)
fcf_yield = fcf_ttm_wide / mcap_wide

# R31 Net FX Position / TA (pre-hedge; signed). Positive = net FX-asset firm
# (TRY-weakness beneficiary); negative = net FX-liability firm.
net_fx_ta = fx_pos_wide / ta_wide

# R32 Net FX (incl. hedge) / TA — post-hedge exposure
net_fx_hedged_ta = fx_pos_hedged_wide / ta_wide

# R33 Hedge Intensity = |raw − hedged| / |raw|, clipped to [0, 2].
# Novel BIST-specific signal: most data feeds don't expose pre/post-hedge
# positions separately. Tests whether actively-hedged firms have a return
# premium (distress-aversion or quality interpretation).
hedge_diff = (fx_pos_wide - fx_pos_hedged_wide).abs()
hedge_denom = fx_pos_wide.abs().replace(0, np.nan)
hedge_intensity = (hedge_diff / hedge_denom).clip(lower=0, upper=2)

# R34 Export Share = ExportSales_TTM / TotalSales_TTM (non-bank only).
# Many firms genuinely have zero exports — keep raw zeros, just guard /0.
export_share = expsales_ttm_wide / sales_ttm_wide.replace(0, np.nan)

print(f'      Round 12 fundamentals coverage spot-check:')
for nm, p in [('R7_gp (fixed)', gp), ('R11_cash', cash_ta),
              ('R24_int_cov', int_cov), ('R29_op_margin', op_margin),
              ('R26_lev_book', lev_book), ('R31_net_fx_ta', net_fx_ta),
              ('R33_hedge_intensity', hedge_intensity),
              ('R34_export_share', export_share),
              ('R36_fcf_yield', fcf_yield)]:
    cov = p.notna().mean().mean() * 100
    print(f'         {nm:25s}: coverage {cov:5.1f}%')

# ===================================================================
# Round 13 (2026-05-12) — 32 new factors across 7 tiers
# ===================================================================
print('\n[3d/8] Round 13: Tier 3 price-based factors (9 new)...')

# R4b 6-1 momentum: cumret over [t-126, t-22]
R4b_mom_6_1 = log_close.shift(22) - log_close.shift(126)

# R5b 60-month long-term reversal: cumret over [t-60m, t-13m] = [t-1260d, t-252d]
# WARNING: 5-year horizon on a 5-year sample — coverage will be very low.
# Document this limitation; the verdict will be PRE-blind.
R5b_rev_60m = log_close.shift(252) - log_close.shift(1260)

# R5c 5-day reversal: very short-term reversal
R5c_rev_5d = log_close.shift(1) - log_close.shift(5)

# R15b realized vol (raw, no FF3 residualization)
R15b_rvol_60d = ret_wide.rolling(window=60, min_periods=30).std()

# R21 turnover: shares_out = mcap / close. Daily volume / shares -> turnover.
shares_out_wide = mcap_wide / close_wide
turnover_daily = (vol_wide / shares_out_wide) if vol_wide is not None else None
if turnover_daily is not None:
    R21_turnover_60d = turnover_daily.rolling(window=60, min_periods=20).mean()
else:
    R21_turnover_60d = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)

# R22 raw skewness of daily returns over 60d
R22_skew_60d = ret_wide.rolling(window=60, min_periods=30).skew()

# R22b idiosyncratic skewness: skew of FF3 residuals over 60d
print('      Computing 60d idio skew per ticker (~30s)...')
R22b_iskew_60d = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
for ticker in ret_wide.columns:
    R22b_iskew_60d[ticker] = rolling_idio_skew(ret_wide[ticker], ff3_aligned, window=60)

# R22c coskewness with the market over 60d
print('      Computing 60d coskew per ticker (~30s)...')
R22c_coskew_60d = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
mkt_aligned = ff3['MKT'].reindex(ret_wide.index)
for ticker in ret_wide.columns:
    R22c_coskew_60d[ticker] = rolling_coskew(ret_wide[ticker], mkt_aligned, window=60)

# R7_mom_consistency: std of daily returns in the 12-1 window
# Lower = more consistent winner/loser. Grinblatt-Moskowitz style.
R7_mom_consistency = ret_wide.shift(22).rolling(window=230, min_periods=120).std()

print(f'      Tier 3 coverage spot-check:')
for nm, p in [('R4b_mom_6_1', R4b_mom_6_1), ('R5b_rev_60m', R5b_rev_60m),
              ('R5c_rev_5d', R5c_rev_5d), ('R15b_rvol_60d', R15b_rvol_60d),
              ('R21_turnover_60d', R21_turnover_60d), ('R22_skew_60d', R22_skew_60d),
              ('R22b_iskew_60d', R22b_iskew_60d), ('R22c_coskew_60d', R22c_coskew_60d),
              ('R7_mom_consistency', R7_mom_consistency)]:
    cov = p.notna().mean().mean() * 100
    print(f'         {nm:25s}: coverage {cov:5.1f}%')

print('\n[3e/8] Round 13: Tier 4 fundamentals + Tier 5 F-score + Tier 2 + Tier 6 (17 new)...')

# ---------- Tier 4 fundamental constructs (7 factors) ----------
# R10 Net Operating Assets / lagged TA — Hirshleifer-Hou-Teoh-Zhang 2004.
# NOA = OperAssets − OperLiabilities ≈ (TA − Cash) − (TA − Equity − TotalDebt)
#      = Equity + TotalDebt − Cash. Long low NOA (low investment intensity).
R10_noa = (eq_wide + total_debt_wide - cash_wide) / ta_wide.shift(252)

# R29b Asset Turnover = Sales_TTM / TA. Long high.
R29b_oat = sales_ttm_wide / ta_wide

# R29c Operating ROA = EBIT_TTM / TA (DuPont decomp: margin × turnover).
R29c_em = ebit_ttm_wide / ta_wide

# R38 Inventory intensity = Inventory / Sales_TTM. Long low.
R38_inv_intensity = inv_wide / sales_ttm_wide.replace(0, np.nan)

# R40 Operating Asset Growth = Δ(TA − Cash) / TA_{t-1}. Long low (low investment).
oa_now = ta_wide - cash_wide
oa_lag = oa_now.shift(252)
R40_oa_growth = (oa_now - oa_lag) / ta_wide.shift(252)

# R41 Earnings stability = std of (NI_TTM / TA) over ~5 quarters (315 daily obs).
# Long low (= stable). Note this is daily-rolling so the 5-quarter window
# captures the same value across consecutive days; std is essentially over
# the 5 most-recent quarter-end values that have been ffilled.
ni_to_ta = ni_ttm_wide / ta_wide
R41_earnings_stability = ni_to_ta.rolling(window=315, min_periods=126).std()

# R42 Sales growth stability = std(log_sales_growth_quarterly) over 5 quarters.
# Approximate: std of log(Sales_TTM / Sales_TTM.shift(63)) over rolling 315d.
sales_g_q = np.log(sales_ttm_wide.where(sales_ttm_wide > 0)) - np.log(sales_ttm_wide.shift(63).where(sales_ttm_wide.shift(63) > 0))
R42_sales_stability = sales_g_q.rolling(window=315, min_periods=126).std()

# ---------- Tier 5 Piotroski F-score (1 factor) ----------
# 8-point version (full Piotroski is 9; we skip ΔLiquidity which needs CL):
f1 = (ni_ttm_wide > 0).astype(float)
f2 = (ocf_ttm_wide > 0).astype(float)
f3 = (roa > roa.shift(252)).astype(float)
f4 = (ocf_ttm_wide > ni_ttm_wide).astype(float)
f5 = (lev_book < lev_book.shift(252)).astype(float)
f6 = (nsi <= 0).astype(float)
f7 = (gp_margin > gp_margin.shift(252)).astype(float)
f8 = (R29b_oat > R29b_oat.shift(252)).astype(float)
# Mask NaN inputs so the sum is NaN if any component missing
mask_any_nan = (ni_ttm_wide.isna() | ocf_ttm_wide.isna() | roa.isna() |
                lev_book.isna() | nsi.isna() | gp_margin.isna() | R29b_oat.isna())
R43_F_score = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8).where(~mask_any_nan)

# ---------- Tier 2A orthogonalized R26_lev_book (1 factor) ----------
print('      Computing R26_lev_book_orth (per-date cross-sec residualization, ~30s)...')
R26_lev_book_orth = cross_sectional_residualize(lev_book, [size, roa])

# ---------- Tier 2B z-score composites (5 factors) ----------
print('      Computing 5 z-score composite factors...')
Quality_z = composite_z([(roa, +1), (op_margin, +1), (lev_mkt, -1)])
Value_z   = composite_z([(f02_ep, +1), (bm, +1), (fcf_yield, +1)])
Accruals_z = composite_z([(acc, -1), (dwc_ta, -1)])
Investment_z = composite_z([(ag, -1), (capex_ta, -1), (net_debt_fin, -1)])
LowVol_z = composite_z([(beta_6m, -1), (ivol_60d, -1)])

# ---------- Tier 6 interactions (3 factors) ----------
# Multiply z-scores so the interaction is mean-zero and scale-free.
print('      Computing interaction factors...')
z_mom = cross_sectional_zscore(mom_12_1)
z_size = cross_sectional_zscore(size)
z_ivol = cross_sectional_zscore(ivol_60d)
z_acc = cross_sectional_zscore(acc)

R46_mom_x_vol = z_mom * z_ivol
R47_mom_x_size = z_mom * (-z_size)   # long high momentum AND small size
R48_acc_x_size = z_acc * (-z_size)   # high accruals (bad signal) for small firms

print(f'      Round 13 [3d+3e] coverage spot-check:')
for nm, p in [('R10_noa', R10_noa), ('R29b_oat', R29b_oat),
              ('R29c_em', R29c_em), ('R41_earnings_stability', R41_earnings_stability),
              ('R43_F_score', R43_F_score), ('R26_lev_book_orth', R26_lev_book_orth),
              ('Quality_z', Quality_z), ('Value_z', Value_z),
              ('R46_mom_x_vol', R46_mom_x_vol)]:
    cov = p.notna().mean().mean() * 100
    print(f'         {nm:25s}: coverage {cov:5.1f}%')

# ─────────────────────────────────────────────────────────────────────
# Step 3: BIST-specific factors
# ─────────────────────────────────────────────────────────────────────
print('\n[4/8] Building BIST-specific factors (TRY beta, sector dummies)...')

if usdtry_ret is not None:
    usdtry_aligned = usdtry_ret.reindex(ret_wide.index)
    try_beta_60d = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
    for ticker in ret_wide.columns:
        try_beta_60d[ticker] = rolling_beta(ret_wide[ticker], usdtry_aligned, 60)

    # Round 12: 252d long-window FX beta (structural exposure)
    print('      Round 12 — building 252d FX beta...')
    try_beta_252d = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
    for ticker in ret_wide.columns:
        try_beta_252d[ticker] = rolling_beta(ret_wide[ticker], usdtry_aligned, 252)

    # Round 12: Downside FX beta — β conditional on TRY weakening (ΔUSDTRY > 0)
    # Tests asymmetric exposure: exporters / FX-asset firms GAIN on TRY weakness,
    # importers / FX-debt firms get hurt. Symmetric β averages these out.
    def rolling_downside_beta(stock_ret, fx_ret, window=60, min_obs=15):
        result = pd.Series(index=stock_ret.index, dtype=float)
        sr = stock_ret.values
        fr = fx_ret.values
        for i in range(window, len(stock_ret)):
            s_win = sr[i - window:i]
            f_win = fr[i - window:i]
            mask = (f_win > 0) & ~np.isnan(s_win) & ~np.isnan(f_win)
            if mask.sum() < min_obs:
                continue
            s = s_win[mask]
            f = f_win[mask]
            f_c = f - f.mean()
            var_f = (f_c ** 2).sum()
            if var_f == 0:
                continue
            s_c = s - s.mean()
            result.iloc[i] = (f_c * s_c).sum() / var_f
        return result

    print('      Round 12 — building 60d downside FX beta (slower; ~2 min)...')
    try_dn_beta_60d = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
    for ticker in ret_wide.columns:
        try_dn_beta_60d[ticker] = rolling_downside_beta(ret_wide[ticker], usdtry_aligned, 60)
else:
    try_beta_60d    = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)
    try_beta_252d   = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)
    try_dn_beta_60d = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)

# Sector dummies — static per ticker
# We'll store sector as integer code; one-hot expansion happens in the regression
sector_codes = {'Financials': 1, 'Consumer': 2, 'Industrials': 3, 'Materials': 4, 'Energy': 5, 'Telecom': 6, 'Other': 0}
sector_int = pd.Series({t: sector_codes.get(macro_sec.get(t, 'Other'), 0) for t in ret_wide.columns})
sector_panel = pd.DataFrame(np.broadcast_to(sector_int.values, ret_wide.shape),
                            index=ret_wide.index, columns=ret_wide.columns)

# ===================================================================
# Round 13 [4a/8] — Tier 1 sector-neutralized canonical 5 + Tier 7 sector-xs-beta
# (must come AFTER sector_panel is built)
# ===================================================================
print('\n[4a/8] Round 13: Tier 1 sector-neutralized + Tier 7 sector-xs-beta (6 new)...')

# Tier 1 — sector-neutralized variants of canonical 5
print('      Computing 5 sector-neutralized canonical factors...')
R12_acc_sn       = sector_neutralize_panel(acc, sector_panel)
MKT_beta_6m_sn   = sector_neutralize_panel(beta_6m, sector_panel)
R1_size_sn       = sector_neutralize_panel(size, sector_panel)
R4_mom_12_1_sn   = sector_neutralize_panel(mom_12_1, sector_panel)
R8_roa_sn        = sector_neutralize_panel(roa, sector_panel)

# Tier 7 — Sector cross-sectional beta
# For each ticker, compute rolling 60d beta of its return on the
# equal-weighted within-sector peer return (excluding itself). Tests
# within-sector defensive vs aggressive positioning.
print('      Computing sector peer-mean returns (vectorized)...')
sector_groups = {sec_code: sector_int[sector_int == sec_code].index.tolist()
                 for sec_code in sector_int.unique()}
sector_returns = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)
for sec_code, tickers_in_sec in sector_groups.items():
    if len(tickers_in_sec) < 2:
        continue
    sub = ret_wide[tickers_in_sec]
    sec_ret = sub.mean(axis=1)
    n = sub.notna().sum(axis=1)
    for tk in tickers_in_sec:
        # peer return = (n*mean - own_ret) / (n-1)
        peer = (sec_ret * n - ret_wide[tk]) / (n - 1).replace(0, np.nan)
        sector_returns[tk] = peer

print('      Computing 60d beta vs sector peer-mean (per ticker, ~30s)...')
R50_sector_xs_beta = pd.DataFrame(np.nan, index=ret_wide.index, columns=ret_wide.columns)
for tk in ret_wide.columns:
    if tk in sector_returns.columns and not sector_returns[tk].isna().all():
        R50_sector_xs_beta[tk] = rolling_beta(ret_wide[tk], sector_returns[tk], 60)

print(f'      Round 13 [4a] coverage spot-check:')
for nm, p in [('R12_acc_sn', R12_acc_sn), ('MKT_beta_6m_sn', MKT_beta_6m_sn),
              ('R1_size_sn', R1_size_sn), ('R4_mom_12_1_sn', R4_mom_12_1_sn),
              ('R8_roa_sn', R8_roa_sn), ('R50_sector_xs_beta', R50_sector_xs_beta)]:
    cov = p.notna().mean().mean() * 100
    print(f'         {nm:25s}: coverage {cov:5.1f}%')

# ─────────────────────────────────────────────────────────────────────
# Step 4: Stack into long format
# ─────────────────────────────────────────────────────────────────────
print('\n[5/8] Stacking into long format...')

panels = {
    'R1_size': size,
    'R3_beta_36m': beta_36m,
    'R3b_beta_1y': beta_1y,
    'R3c_beta_6m': beta_6m,
    'R4_mom_12_1': mom_12_1,
    'R5_rev_1m': rev_1m,
    'R15_ivol_60d': ivol_60d,
    'R16_amh_60d': amh_60d,
    'R17_max_21d': max_21d,
    'R17b_max5_21d': max5_21d,
    'R2_bm': bm,
    'R8_roa': roa,
    'R6_ag': ag,
    'R7_gp': gp,            # FIXED Round 12 — now uses code 3D directly
    'R12_acc': acc,
    'R9_nsi': nsi,
    'R23_try_beta_60d': try_beta_60d,
    # ============================================================
    # Legacy F01–F04 canonical lineup. These are characteristic-form
    # versions of the legacy factor strategies, suitable for Test A+B.
    # ============================================================
    'F01_BM':         bm,            # alias of R2_bm — the value/B/M signal
    'F02_EP':         f02_ep,        # earnings yield NI_TTM / market_cap
    'F03_MOM_12_1':   mom_12_1,      # 12-1 momentum (alias of R4)
    'F04_ROE':        f04_roe,       # return on equity NI_TTM / equity
    # ============================================================
    # "Market" exposure characteristic. With only ~5 years of data,
    # 36-month rolling β only covers ~Jan 2024 onward (41% coverage).
    # 1-year covers ~Jan 2022 (79%); 6-month covers ~Jul 2021 (89%).
    # MKT_beta is the canonical name; we default to 1-year as the most
    # data-efficient choice that still has economic meaning.
    # ============================================================
    'MKT_beta':       beta_1y,       # default "market beta" — 1-year window
    'MKT_beta_6m':    beta_6m,       # alt: shorter window, more responsive
    'MKT_beta_1y':    beta_1y,       # explicit 1-year alias
    'MKT_beta_36m':   beta_36m,      # only usable from 2024 onward in this sample
    # ============================================================
    # Round 12 (2026-05-12) — 19 new factors. See methodology guides/
    # Factor_Batch_Round12_Plan.md for formulas, signs, bank-treatment.
    # ============================================================
    'R18_ceq_12m':            ceq_12m,           # composite equity issuance (price-only)
    'R7c_margin':             gp_margin,         # gross margin (non-bank)
    'R13_cf_yield':           cf_yield,          # OCF / mcap (non-bank)
    'R14_cf_ta':              cf_ta,             # OCF / lagged TA (non-bank)
    'R28_sales_g':            sales_g,           # sales growth YoY (non-bank)
    'R11_cash':               cash_ta,           # Cash / TA (non-bank)
    'R6b_inv_to_ta':          inv_to_ta,         # ΔInventory / lagged TA (non-bank)
    'R20_capex_ta':           capex_ta,          # |CapEx_TTM| / lagged TA (non-bank)
    'R24_int_cov':            int_cov,           # EBIT / |FinExp| (non-bank, winsorized)
    'R29_op_margin':          op_margin,         # EBIT / Sales (non-bank)
    'R26_lev_book':           lev_book,          # TotalDebt / Equity (non-bank)
    'R26b_lev_mkt':           lev_mkt,           # TotalDebt / (TD+ME) (non-bank)
    'R19_net_debt_fin':       net_debt_fin,      # ΔTotalDebt / lagged TA (non-bank)
    'R25_dwc_ta':             dwc_ta,            # ΔWC_TTM / lagged TA (non-bank; signed)
    'R36_fcf_yield':          fcf_yield,         # FCF_TTM / mcap (non-bank; signed)
    'R31_net_fx_ta':          net_fx_ta,         # Net FX_PIT / TA (non-bank; signed)
    'R32_net_fx_hedged_ta':   net_fx_hedged_ta,  # Net FX incl. hedge / TA (non-bank; signed)
    'R33_hedge_intensity':    hedge_intensity,   # |raw−hedged| / |raw| (non-bank; clipped)
    'R34_export_share':       export_share,      # ExportSales_TTM / Sales_TTM (non-bank)
    'R23b_try_beta_252d':     try_beta_252d,     # 252d USDTRY beta (price-only)
    'R35_try_dn_beta_60d':    try_dn_beta_60d,   # 60d downside-FX beta (price-only)
    # ============================================================
    # Round 13 (2026-05-12) — 32 new factors. See
    # methodology/guides/Factor_Batch_Round13_Plan.md for full specs.
    # ============================================================
    # Tier 1: sector-neutralized canonical 5
    'R12_acc_sn':            R12_acc_sn,
    'MKT_beta_6m_sn':        MKT_beta_6m_sn,
    'R1_size_sn':            R1_size_sn,
    'R4_mom_12_1_sn':        R4_mom_12_1_sn,
    'R8_roa_sn':             R8_roa_sn,
    # Tier 2A: orthogonalized R26_lev_book
    'R26_lev_book_orth':     R26_lev_book_orth,
    # Tier 2B: z-score composites
    'Quality_z':             Quality_z,
    'Value_z':               Value_z,
    'Accruals_z':            Accruals_z,
    'Investment_z':          Investment_z,
    'LowVol_z':              LowVol_z,
    # Tier 3: price-based
    'R4b_mom_6_1':           R4b_mom_6_1,
    'R5b_rev_60m':           R5b_rev_60m,
    'R5c_rev_5d':            R5c_rev_5d,
    'R15b_rvol_60d':         R15b_rvol_60d,
    'R21_turnover_60d':      R21_turnover_60d,
    'R22_skew_60d':          R22_skew_60d,
    'R22b_iskew_60d':        R22b_iskew_60d,
    'R22c_coskew_60d':       R22c_coskew_60d,
    'R7_mom_consistency':    R7_mom_consistency,
    # Tier 4: fundamentals (non-bank)
    'R10_noa':               R10_noa,
    'R29b_oat':              R29b_oat,
    'R29c_em':               R29c_em,
    'R38_inv_intensity':     R38_inv_intensity,
    'R40_oa_growth':         R40_oa_growth,
    'R41_earnings_stability': R41_earnings_stability,
    'R42_sales_stability':   R42_sales_stability,
    # Tier 5: F-score
    'R43_F_score':           R43_F_score,
    # Tier 6: interactions
    'R46_mom_x_vol':         R46_mom_x_vol,
    'R47_mom_x_size':        R47_mom_x_size,
    'R48_acc_x_size':        R48_acc_x_size,
    # Tier 7: sector-xs-beta
    'R50_sector_xs_beta':    R50_sector_xs_beta,
    'R27_sector': sector_panel,
}

long_records = []
for fname, panel in panels.items():
    df_l = panel.stack().reset_index()
    df_l.columns = ['date', 'ticker', fname]
    long_records.append(df_l.set_index(['date', 'ticker']))

panel_long = pd.concat(long_records, axis=1).reset_index()
print(f'      Panel shape: {panel_long.shape}')
print(f'      Tickers: {panel_long["ticker"].nunique()}, '
      f'dates: {panel_long["date"].nunique()}')

# ─────────────────────────────────────────────────────────────────────
# Step 5: Coverage report
# ─────────────────────────────────────────────────────────────────────
print('\n[6/8] Coverage report:')
factor_cols = [c for c in panel_long.columns if c not in ('date', 'ticker')]
print(f'\n{"Anomaly":<22} {"Coverage":>10} {"Mean":>12} {"Std":>12}')
print('─' * 56)
for col in factor_cols:
    s = panel_long[col]
    cov = s.notna().mean()
    mu = s.mean() if pd.api.types.is_numeric_dtype(s) else np.nan
    sd = s.std() if pd.api.types.is_numeric_dtype(s) else np.nan
    cov_flag = '✓' if cov > 0.7 else '✗'
    print(f'{col:<22} {cov*100:>9.1f}%{cov_flag} {mu:>12.4f} {sd:>12.4f}')

# ─────────────────────────────────────────────────────────────────────
# Step 6: Save
# ─────────────────────────────────────────────────────────────────────
print('\n[7/8] Saving panel...')
panel_long.to_parquet(RESULTS_DIR / 'anomaly_panel_daily.parquet')
print(f'      Saved anomaly_panel_daily.parquet')

# Monthly version: take last value per month for each (ticker, anomaly)
panel_long['month'] = panel_long['date'].dt.to_period('M')
monthly = panel_long.groupby(['month', 'ticker']).last().drop(columns=['date'])
monthly = monthly.reset_index()
monthly['date'] = monthly['month'].dt.to_timestamp(how='end').dt.normalize()
monthly = monthly.drop(columns=['month'])
monthly.to_parquet(RESULTS_DIR / 'anomaly_panel_monthly.parquet')
print(f'      Saved anomaly_panel_monthly.parquet: {monthly.shape}')

# ─────────────────────────────────────────────────────────────────────
# Step 7: Done
# ─────────────────────────────────────────────────────────────────────
print('\n[8/8] Done.')
print('=' * 70)
print('Next: run analysis/15_factor_validator.py to test each anomaly.')
print('=' * 70)
