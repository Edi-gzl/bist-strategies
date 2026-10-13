from _rp import _RP  # portable-path shim (added when vendoring)
"""
45_build_attention_paper_factors.py - Build NEW factors from Epstein-Wang-Choi-
Pelger "Attention Factors for Statistical Arbitrage" (ICAIF 2025) paper.

R16 ADDITIONS (14 new factors):
  Past Returns (paper Table 1):
    R55_ret_d1        - previous day return
    R56_ret_w1        - previous 5-day compounded return
    R57_std_w1        - past 5-day daily return std
    R58_r36_13        - long-term momentum (cumret t-273..t-756)
    R61_r2_1          - short-term momentum (cumret t-21..t-42, skip-1m)        *** NEW
    R62_r12_7         - intermediate momentum (cumret t-147..t-252)             *** NEW
  Trading Frictions:
    R52_rel2high_52w  - close / max_252d(close) - 1
    R53_suv_60d       - Standard Unexplained Volume
    R54_ltm_turnover  - log(60d avg turnover)
    R60_log_vol       - log(60d avg trading volume TRY)
  Value:
    R59_tobins_q      - MktCap_equity / Total Assets (Q approximation)
    R63_a2me          - Total Assets / Market Equity                            *** NEW
    R64_cf2b          - Free Cash Flow / Book Equity (FCF yield variant)        *** NEW
  Profitability:
    R65_cto           - Sales / Total Assets (capital turnover)                 *** NEW

NOT BUILT (data limitations):
    Spread (no BBO data), FC2Y (no fixed-cost breakdown), PCM (no cost data),
    DPI2A (no clean PP&E code), OL (operating leverage requires cost detail).

WRITES TO: results/cross_factor/anomaly_panel_daily_r16.parquet
(preserves original anomaly_panel_daily.parquet)

CITATION: Epstein, Wang, Choi, Pelger (2025), Table 1.

USAGE:
    python claude_data\\scripts\\analysis\\45_build_attention_paper_factors.py
    py -3 claude_data\\scripts\\analysis\\45_build_attention_paper_factors.py --smoke
"""
import sys, io, argparse, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(_RP(r""))
sys.path.insert(0, str(ROOT / 'claude_data' / 'scripts'))
from shared import configs as CFG
from shared.data import load_prices, load_fundamentals, BANKS

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true',
                help='Smoke test: 5 tickers, output to *_smoke.parquet')
args = ap.parse_args()

print('=' * 78)
print('R16 SCRIPT 45 - Build 14 new factors (R52-R65) from Attention Factors paper')
print('=' * 78)

# ─── Load price data with KNOWN BIST column names ───
px = load_prices()
print(f'Loaded prices: {px.shape}')
print(f'Columns: {list(px.columns)}')

# BIST-specific column names (verified from shared/data.py and smoke test output)
CLOSE_COL  = 'close_adj'        # split-adjusted close
VOL_COL    = 'volume'           # daily volume (shares)
MCAP_COL   = 'mktcap_try'       # market cap in TRY
SHARES_COL = 'shares_out'       # shares outstanding

required_cols = [CLOSE_COL, VOL_COL, MCAP_COL]
missing_cols = [c for c in required_cols if c not in px.columns]
if missing_cols:
    print(f'FATAL: missing required columns: {missing_cols}')
    print(f'Available: {list(px.columns)}')
    sys.exit(1)
print(f'Using close={CLOSE_COL}, volume={VOL_COL}, mkt_cap={MCAP_COL}')

if args.smoke:
    tk_smoke = sorted(px['ticker'].unique())[:5]
    px = px[px['ticker'].isin(tk_smoke)]
    print(f'[smoke] subset to {len(tk_smoke)} tickers: {tk_smoke}')

# Wide pivots
ret_w   = px.pivot(index='date', columns='ticker', values='ret')
close_w = px.pivot(index='date', columns='ticker', values=CLOSE_COL)
vol_w   = px.pivot(index='date', columns='ticker', values=VOL_COL)
mcap_w  = px.pivot(index='date', columns='ticker', values=MCAP_COL)

for df in [ret_w, close_w, vol_w, mcap_w]:
    df.index = pd.to_datetime(df.index)

# Volume in TRY (price × volume_shares):
# vol_w is in SHARES (per shared/data.py). Volume in TRY = vol_w * close.
vol_try_w = vol_w * close_w
print(f'Price/vol/mcap pivots: ret={ret_w.shape}, close={close_w.shape}, vol={vol_w.shape}')

# ─── R52: rel2high_52w ───
print('\n[R52] rel2high_52w (distance from 52-week high)...')
max_252 = close_w.rolling(window=252, min_periods=60).max()
R52 = close_w.div(max_252).sub(1.0)  # ∈ [-1, 0]

# ─── R55: previous day return ───
print('[R55] ret_d1 (1-day lag return)...')
R55 = ret_w.shift(1)

# ─── R56: previous 5-day compounded return ───
print('[R56] ret_w1 (5-day compounded return, lagged 1d)...')
R56 = ((1 + ret_w).rolling(window=5, min_periods=4).apply(np.prod, raw=True) - 1.0).shift(1)

# ─── R57: previous 5-day return std ───
print('[R57] std_w1 (5-day rolling std, lagged 1d)...')
R57 = ret_w.rolling(window=5, min_periods=4).std().shift(1)

# ─── R53: SUV (Standard Unexplained Volume) ───
print('[R53] SUV_60d (residual std of log_volume ~ |return|, 60-day rolling)...')
vol_log = np.log(vol_w.replace(0, np.nan))
abs_ret = ret_w.abs()
R53 = pd.DataFrame(index=vol_log.index, columns=vol_log.columns, dtype=float)
WIN_SUV = 60
for tk in vol_log.columns:
    y = vol_log[tk]
    x = abs_ret[tk]
    mask = (~y.isna()) & (~x.isna())
    if mask.sum() < 100:
        continue
    x_m = x.rolling(WIN_SUV, min_periods=20).mean()
    y_m = y.rolling(WIN_SUV, min_periods=20).mean()
    xy_m = (x * y).rolling(WIN_SUV, min_periods=20).mean()
    xx_m = (x * x).rolling(WIN_SUV, min_periods=20).mean()
    cov_xy = xy_m - x_m * y_m
    var_x  = xx_m - x_m * x_m
    # Ridge-protect against zero variance
    var_x_safe = var_x.where(var_x.abs() > 1e-10, np.nan)
    slope = cov_xy / var_x_safe
    intercept = y_m - slope * x_m
    yhat = intercept + slope * x
    resid = y - yhat
    R53[tk] = resid.rolling(WIN_SUV, min_periods=20).std()
print(f'  R53 computed for {R53.notna().any().sum()}/{R53.shape[1]} tickers')

# ─── R54: log turnover (60-day avg) ───
print('[R54] ltm_turnover (log of 60-day avg turnover)...')
turnover = vol_try_w / mcap_w.replace(0, np.nan)
turnover_60d = turnover.rolling(60, min_periods=20).mean()
R54 = np.log1p(turnover_60d.clip(lower=0))

# ─── R58: r36_13 long-term momentum (R16 NEW) ───
# Per Attention paper: cumulative return from t-756 to t-273 (3-yr to 1-yr ago)
# We approximate with 1+ret rolling product over (252*3 - 252) ~ 504-day window
# starting 273 days ago.
print('[R58] r36_13 long-term momentum (cumulative ret t-756..t-273)...')
log_ret = np.log1p(ret_w)
cum_756 = log_ret.rolling(window=756, min_periods=400).sum()
cum_273 = log_ret.rolling(window=273, min_periods=180).sum()
# r36_13 = (sum from t-756 to t) - (sum from t-273 to t) = sum from t-756 to t-273
R58 = (cum_756 - cum_273).shift(1)  # shift 1 to avoid lookahead at t
# Convert back from log
R58 = np.exp(R58) - 1.0

# ─── R61: r2_1 short-term momentum (cumret t-21..t-42, skip-1-month) (R16 NEW) ───
# Per Attention paper: cumulative return from t-42 to t-21 (~1 month of return
# observed two months ago — captures the very-short-horizon momentum signal,
# different from R5_rev_1m which is 1m reversal)
print('[R61] r2_1 short-term momentum (cumret t-42..t-21)...')
cum_42 = log_ret.rolling(window=42, min_periods=30).sum()
cum_21 = log_ret.rolling(window=21, min_periods=15).sum()
R61 = (cum_42 - cum_21).shift(1)  # shift 1 to avoid same-day usage
R61 = np.exp(R61) - 1.0

# ─── R62: r12_7 intermediate momentum (cumret t-147..t-252) (R16 NEW) ───
# Per Attention paper: cumulative return from t-252 to t-147 (5-month window
# from ~12 to ~7 months ago)
print('[R62] r12_7 intermediate momentum (cumret t-252..t-147)...')
cum_252 = log_ret.rolling(window=252, min_periods=180).sum()
cum_147 = log_ret.rolling(window=147, min_periods=100).sum()
R62 = (cum_252 - cum_147).shift(1)
R62 = np.exp(R62) - 1.0

# ─── R60: log volume (R16 NEW) ───
print('[R60] log_vol (log of 60-day avg trading volume in TRY)...')
vol_60d_avg = vol_try_w.rolling(60, min_periods=20).mean()
R60 = np.log1p(vol_60d_avg.clip(lower=0))

# ─── Fundamentals-based factors: R59, R63, R64, R65 (R16) ───
# All four need quarterly fundamentals (TA, Sales, Book Equity, FCF).
# We try multiple candidate codes per metric and gracefully skip if missing.
print('\n[R59/R63/R64/R65] Fundamentals-based factors (Tobin Q, A2ME, CF, CTO)...')

def _build_pit_metric(fund_df, codes_to_try, metric_label):
    """Try multiple candidate codes; return PIT (signal-date indexed) series
    keyed by ticker, or None if nothing found."""
    for code in codes_to_try:
        sub = fund_df[fund_df.get('FINANCIAL_ITEM_CODE', '') == code]
        if len(sub) > 1000:
            print(f'  {metric_label}: using code {code} (n={len(sub)})')
            sub = sub.copy()
            sub['period_end'] = pd.to_datetime(sub['period_end'])
            sub['signal_date'] = sub['period_end'] + pd.Timedelta(days=CFG.LAG_FUNDAMENTALS_DAYS)
            sub['val'] = pd.to_numeric(sub['value'], errors='coerce')
            return sub[['SYMBOL', 'signal_date', 'val']].dropna()
    print(f'  {metric_label}: NO matching code found (tried {codes_to_try})')
    return None

def _ffill_to_daily(pit_df, close_w_index, all_tickers, value_col='val'):
    """Forward-fill PIT values to a daily date index for all tickers."""
    out = pd.DataFrame(index=close_w_index, columns=all_tickers, dtype=float)
    if pit_df is None or pit_df.empty:
        return out
    for tk in all_tickers:
        sub = pit_df[pit_df['SYMBOL'] == tk][['signal_date', value_col]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values('signal_date').drop_duplicates('signal_date', keep='last')
        sub = sub.set_index('signal_date')
        out[tk] = sub[value_col].reindex(close_w_index, method='ffill')
    return out

R59 = close_w * np.nan  # default
R63 = close_w * np.nan
R64 = close_w * np.nan
R65 = close_w * np.nan

try:
    fund = load_fundamentals()
    print(f'  Fundamentals: {fund.shape}, columns={list(fund.columns)[:10]}')

    if 'FINANCIAL_ITEM_CODE' in fund.columns:
        top_codes = fund['FINANCIAL_ITEM_CODE'].value_counts().head(30)
        print(f'  Top 30 codes by count:')
        for c, n in top_codes.items():
            print(f'    {c}: {n}')

        # Total Assets (1A is the standard BIST code; try fallbacks)
        ta_pit = _build_pit_metric(fund, ['1A', '1AT', 'TA', '1F', 'TOTAL_ASSETS'], 'Total Assets')
        ta_daily = _ffill_to_daily(ta_pit, close_w.index, list(close_w.columns))

        # Sales / Revenue (3A for non-banks; 3A1 / 3I variants)
        sales_pit = _build_pit_metric(fund, ['3A', '3A1', '3I', 'SALES', 'REVENUE'], 'Sales/Revenue')
        sales_daily = _ffill_to_daily(sales_pit, close_w.index, list(close_w.columns))

        # Book Equity (2N non-banks, 2O banks). We'll union via a per-ticker switch
        eq_pit_nb = _build_pit_metric(fund, ['2N'], 'Equity (non-bank)')
        eq_pit_b  = _build_pit_metric(fund, ['2O'], 'Equity (bank)')
        # Merge: bank tickers use 2O, others use 2N
        eq_combined = None
        if eq_pit_nb is not None and eq_pit_b is not None:
            eq_combined = pd.concat([
                eq_pit_nb[~eq_pit_nb['SYMBOL'].isin(BANKS)],
                eq_pit_b[eq_pit_b['SYMBOL'].isin(BANKS)],
            ], ignore_index=True)
        elif eq_pit_nb is not None:
            eq_combined = eq_pit_nb
        elif eq_pit_b is not None:
            eq_combined = eq_pit_b
        eq_daily = _ffill_to_daily(eq_combined, close_w.index, list(close_w.columns))

        # FCF — typically not a single code; we approximate as Operating CF minus CapEx
        # if those exist, else use Net Income as a fallback proxy.
        # Codes: 5A (op CF), 5B (CapEx) are typical, but vary.
        ocf_pit = _build_pit_metric(fund, ['5A', '5A1', 'OCF', 'CASH_FROM_OPS'], 'Operating CF')
        capex_pit = _build_pit_metric(fund, ['5B', '5B1', 'CAPEX'], 'CapEx')
        ocf_daily = _ffill_to_daily(ocf_pit, close_w.index, list(close_w.columns))
        capex_daily = _ffill_to_daily(capex_pit, close_w.index, list(close_w.columns))
        fcf_daily = ocf_daily - capex_daily.fillna(0)

        # ─── R59: Tobin's Q = MktCap / TA (no book_debt addition; biased but informative) ───
        if not ta_daily.isna().all().all():
            R59 = mcap_w / ta_daily.replace(0, np.nan)
            print(f'  R59 tobins_q: built for {R59.notna().any().sum()}/{R59.shape[1]} tickers')

        # ─── R63: A2ME = TA / MktCap (paper Table 1) ───
        if not ta_daily.isna().all().all():
            R63 = ta_daily / mcap_w.replace(0, np.nan)
            print(f'  R63 a2me: built for {R63.notna().any().sum()}/{R63.shape[1]} tickers')

        # ─── R64: CF / Book Equity (paper Table 1 "CF") ───
        if not fcf_daily.isna().all().all() and not eq_daily.isna().all().all():
            R64 = fcf_daily / eq_daily.replace(0, np.nan)
            print(f'  R64 cf2b: built for {R64.notna().any().sum()}/{R64.shape[1]} tickers')
        else:
            print(f'  R64 cf2b: SKIPPED (FCF or Equity unavailable)')

        # ─── R65: CTO = Sales / TA (paper Table 1 "CTO" capital turnover) ───
        if not sales_daily.isna().all().all() and not ta_daily.isna().all().all():
            R65 = sales_daily / ta_daily.replace(0, np.nan)
            print(f'  R65 cto: built for {R65.notna().any().sum()}/{R65.shape[1]} tickers')
        else:
            print(f'  R65 cto: SKIPPED (Sales or TA unavailable)')
except Exception as e:
    print(f'  WARN: fundamentals factor construction failed: {e}; R59/R63/R64/R65 will be NaN')

# ─── Stack & merge ───
print('\nStacking new factors and merging into panel...')

new_factors = {
    'R52_rel2high_52w':  R52,
    'R53_suv_60d':       R53,
    'R54_ltm_turnover':  R54,
    'R55_ret_d1':        R55,
    'R56_ret_w1':        R56,
    'R57_std_w1':        R57,
    'R58_r36_13':        R58,
    'R59_tobins_q':      R59,
    'R60_log_vol':       R60,
    'R61_r2_1':          R61,
    'R62_r12_7':         R62,
    'R63_a2me':          R63,
    'R64_cf2b':          R64,
    'R65_cto':           R65,
}

panel_path = CFG.RESULTS_DIR / 'anomaly_panel_daily.parquet'
panel = pd.read_parquet(panel_path)
panel['date'] = pd.to_datetime(panel['date'])
print(f'Existing panel: {panel.shape}')

# In smoke mode, subset panel to same tickers
if args.smoke:
    panel = panel[panel['ticker'].isin(tk_smoke)]
    print(f'[smoke] panel subset to {panel.shape}')

added_count = 0
for name, df in new_factors.items():
    if df.isna().all().all():
        print(f'  {name:<22} ALL NaN — skip')
        continue
    long = df.stack(future_stack=True).reset_index()
    long.columns = ['date', 'ticker', name]
    long['date'] = pd.to_datetime(long['date'])
    panel = panel.merge(long, on=['date', 'ticker'], how='left')
    n_obs = panel[name].notna().sum()
    coverage = n_obs / len(panel) * 100 if len(panel) > 0 else 0
    print(f'  {name:<22} {n_obs:>7,} obs ({coverage:.1f}%)')
    added_count += 1

# Sanity: print value ranges in POST sample
print('\nNew factor value ranges (POST sample):')
start, end = CFG.PERIODS['POST']
post = panel[(panel['date'] >= start) & (panel['date'] <= end)]
for name in new_factors.keys():
    if name not in post.columns:
        continue
    s = post[name].dropna()
    if len(s) == 0:
        print(f'  {name:<22} all NaN in POST')
        continue
    # Cross-sectional stats give a better feel than time-series
    cs_std_by_date = post.groupby('date')[name].std()
    print(f'  {name:<22} n={len(s):>6,}  mean={s.mean():+.4f}  std={s.std():.4f}  '
          f'min={s.min():+.4f}  max={s.max():+.4f}  avg_cs_std={cs_std_by_date.mean():.4f}')

# Save
out_path = CFG.RESULTS_DIR / ('anomaly_panel_daily_r16_smoke.parquet' if args.smoke
                              else 'anomaly_panel_daily_r16.parquet')
panel.to_parquet(out_path, compression='snappy')
print(f'\n✅ Saved: {out_path}')
print(f'   New panel: {panel.shape} ({added_count}/{len(new_factors)} new factors added)')
sys.exit(0)
