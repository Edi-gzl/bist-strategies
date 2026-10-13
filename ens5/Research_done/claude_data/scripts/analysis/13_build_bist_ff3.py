from _rp import _RP  # portable-path shim (added when vendoring)
"""
13_build_bist_ff3.py — Build BIST Fama-French 3-factor returns

Constructs:
  MKT_BIST  = equal-weighted universe return  (or VW with mkt cap)
  SMB_BIST  = small minus big                  (size sort, monthly rebal)
  HML_BIST  = high B/M minus low B/M            (value sort, monthly rebal)

Both DAILY and MONTHLY frequencies are saved.

Usage:
    python analysis/13_build_bist_ff3.py

Output:
    results/cross_factor/bist_ff3_daily.parquet
    results/cross_factor/bist_ff3_monthly.parquet

Notes on construction (BIST adaptation, not strict FF1993):
- Sample is BIST50 only (~50 names), so we use 2x3 sort instead of 5x5
- Size breakpoint: median of universe (50/50 split, ~25 stocks each)
- B/M breakpoints: 30/40/30 within each size group (low/mid/high)
- SMB = avg(small_low, small_mid, small_high) - avg(big_low, big_mid, big_high)
- HML = avg(small_high, big_high) - avg(small_low, big_low)
- Rebalance monthly (start of month using prior month-end signals)
- MKT = equal-weighted universe minus risk-free (rf_daily.parquet)
- BM constructed from latest filed equity (60-day lag) / current mktcap

Why not 5x5: BIST50 is too small. Even 2x3 gives only ~8 stocks per portfolio.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(_RP(r""))

# Add shared modules to path
sys.path.insert(0, str(ROOT / 'claude_data' / 'scripts'))
from shared.data import load_prices, load_fundamentals, BANKS, EQ_CODE_BANK, EQ_CODE_NB
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
# Step 1: Load daily prices, compute returns, market cap
# ─────────────────────────────────────────────────────────────────────
print('=' * 70)
print('BIST FF3 BUILDER')
print('=' * 70)

print('\n[1/5] Loading daily prices...')
px = load_prices()
px = px[px['date'] >= START_DATE].copy()
print(f'      Loaded {len(px):,} rows, {px["ticker"].nunique()} tickers, '
      f'{px["date"].min().date()} → {px["date"].max().date()}')

# Wide panels for vectorized ops
ret_wide = px.pivot(index='date', columns='ticker', values='ret')
mcap_wide = px.pivot(index='date', columns='ticker', values='mktcap_try')
print(f'      Return panel shape: {ret_wide.shape}')

# ─────────────────────────────────────────────────────────────────────
# Step 2: Build Book-to-Market signal (PIT-safe)
# ─────────────────────────────────────────────────────────────────────
print('\n[2/5] Building B/M signal with 60-day filing lag...')
fund = load_fundamentals()

# Equity for banks (2O) and non-banks (2N)
eq_bank = fund[fund['FINANCIAL_ITEM_CODE'] == EQ_CODE_BANK][['SYMBOL', 'period_end', 'value']]
eq_nb = fund[fund['FINANCIAL_ITEM_CODE'] == EQ_CODE_NB][['SYMBOL', 'period_end', 'value']]
eq = pd.concat([eq_bank, eq_nb], ignore_index=True)
eq.columns = ['ticker', 'period_end', 'equity']
eq = eq.dropna(subset=['equity'])
eq = eq[eq['equity'] > 0]  # drop negative equity (e.g., distressed)

# Filing date = period_end + LAG_DAYS
eq['available_from'] = eq['period_end'] + pd.Timedelta(days=LAG_DAYS)
eq = eq.sort_values(['ticker', 'available_from'])

# For each (ticker, date), get the latest equity available
def get_pit_equity(ticker_dates, eq_ticker):
    """For each date, find the most recent equity value available."""
    eq_t = eq_ticker.sort_values('available_from')
    merged = pd.merge_asof(
        ticker_dates.sort_values('date'),
        eq_t[['available_from', 'equity']].rename(columns={'available_from': 'date'}),
        on='date',
        direction='backward'
    )
    return merged

# Build equity panel: ticker × date
eq_records = []
for ticker in px['ticker'].unique():
    eq_t = eq[eq['ticker'] == ticker]
    if len(eq_t) == 0:
        continue
    dates = px[px['ticker'] == ticker][['date']].copy()
    pit = get_pit_equity(dates, eq_t)
    pit['ticker'] = ticker
    eq_records.append(pit[['ticker', 'date', 'equity']])

eq_panel = pd.concat(eq_records, ignore_index=True)
eq_wide = eq_panel.pivot(index='date', columns='ticker', values='equity')
print(f'      Equity panel shape: {eq_wide.shape}, '
      f'{eq_wide.notna().sum().sum():,} non-null cells')

# B/M = Equity / MarketCap (both at the same date, equity is PIT-lagged)
bm_wide = eq_wide / mcap_wide
print(f'      B/M panel shape: {bm_wide.shape}, mean: {bm_wide.values[~np.isnan(bm_wide.values)].mean():.3f}')

# ─────────────────────────────────────────────────────────────────────
# Step 3: Build MKT (equal-weighted universe minus rf)
# ─────────────────────────────────────────────────────────────────────
print('\n[3/5] Building MKT factor (EW universe excess return)...')

# Load risk-free rate
rf_path = DATA_DIR / 'rf_daily.parquet'
if rf_path.exists():
    rf = pd.read_parquet(rf_path)
    rf['date'] = pd.to_datetime(rf['date'])
    # rf_daily.parquet schema is ['date', 'rf_annual_pct', 'rf_daily']
    # — the daily-decimal rate is in 'rf_daily'.
    rf_col = next((c for c in ['rf_daily', 'rf', 'risk_free', 'rate']
                   if c in rf.columns), None)
    if rf_col is None:
        print(f'      WARNING: no rf column found in {rf_path.name}; '
              f'columns are {list(rf.columns)}. Falling back to rf=0.')
        rf = pd.Series(0.0, index=ret_wide.index, name='rf')
    else:
        rf = rf.set_index('date')[rf_col]
        # If the annual column was picked by accident, downscale defensively.
        if rf_col == 'rf_annual_pct':
            rf = rf / 100.0 / 252.0
        print(f'      Risk-free loaded ({rf_col}): min {rf.min():.5f}  '
              f'max {rf.max():.5f} (daily decimal)')
else:
    print('      WARNING: rf_daily.parquet not found, using rf=0')
    rf = pd.Series(0.0, index=ret_wide.index, name='rf')

mkt_ew = ret_wide.mean(axis=1)  # equal-weighted return
mkt_ex = mkt_ew - rf.reindex(mkt_ew.index).fillna(0)
print(f'      MKT EW excess return: mean {mkt_ex.mean()*252*100:.2f}%/yr, '
      f'vol {mkt_ex.std()*np.sqrt(252)*100:.2f}%/yr')

# ─────────────────────────────────────────────────────────────────────
# Step 4: Build SMB and HML via 2×3 sort, monthly rebalance
# ─────────────────────────────────────────────────────────────────────
print('\n[4/5] Building SMB and HML via 2×3 sort (monthly rebal)...')

# Rebalance dates = first trading day of each month
rebal_dates = pd.DatetimeIndex(
    sorted(set(d for d in ret_wide.index if d == ret_wide.index[ret_wide.index.to_period('M') == d.to_period('M')].min()))
)
print(f'      {len(rebal_dates)} monthly rebalances')

# Build position dictionaries
positions = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype='object')

for r in rebal_dates:
    if r not in mcap_wide.index or r not in bm_wide.index:
        continue
    
    mc = mcap_wide.loc[r].dropna()
    bm = bm_wide.loc[r].dropna()
    
    # Universe at rebalance: stocks with valid mcap AND valid B/M
    universe = mc.index.intersection(bm.index)
    if len(universe) < 20:
        continue
    
    mc_u = mc.loc[universe]
    bm_u = bm.loc[universe]
    
    # Size split: median
    size_median = mc_u.median()
    small = mc_u[mc_u <= size_median].index
    big = mc_u[mc_u > size_median].index
    
    # B/M split: 30/40/30 within each size group
    def bm_split(grp_tickers):
        bm_g = bm_u.loc[grp_tickers].sort_values()
        n = len(bm_g)
        n30 = int(n * 0.30)
        low = bm_g.iloc[:n30].index
        high = bm_g.iloc[n - n30:].index
        return low, high
    
    sl, sh = bm_split(small)
    bl, bh = bm_split(big)
    
    # Assign portfolio labels
    pos_dict = {}
    for t in sl: pos_dict[t] = 'SL'
    for t in sh: pos_dict[t] = 'SH'
    for t in bl: pos_dict[t] = 'BL'
    for t in bh: pos_dict[t] = 'BH'
    # mid B/M stocks get 'M' label, won't enter SMB/HML
    for t in small.difference(sl).difference(sh): pos_dict[t] = 'SM'
    for t in big.difference(bl).difference(bh): pos_dict[t] = 'BM'
    
    positions.loc[r] = pd.Series(pos_dict).reindex(positions.columns)

# Forward-fill positions between rebalances
positions = positions.ffill()

# Compute portfolio returns each day
def portfolio_return(label):
    mask = (positions == label)
    return ret_wide.where(mask).mean(axis=1)

r_SL = portfolio_return('SL')
r_SH = portfolio_return('SH')
r_BL = portfolio_return('BL')
r_BH = portfolio_return('BH')
r_SM = portfolio_return('SM')
r_BM = portfolio_return('BM')

# SMB = (SL+SM+SH)/3 - (BL+BM+BH)/3
smb = (r_SL + r_SM + r_SH) / 3 - (r_BL + r_BM + r_BH) / 3

# HML = (SH+BH)/2 - (SL+BL)/2
hml = (r_SH + r_BH) / 2 - (r_SL + r_BL) / 2

print(f'      SMB: mean {smb.mean()*252*100:.2f}%/yr, vol {smb.std()*np.sqrt(252)*100:.2f}%/yr')
print(f'      HML: mean {hml.mean()*252*100:.2f}%/yr, vol {hml.std()*np.sqrt(252)*100:.2f}%/yr')

# ─────────────────────────────────────────────────────────────────────
# Step 5: Save daily and monthly versions
# ─────────────────────────────────────────────────────────────────────
print('\n[5/5] Saving outputs...')

ff3_daily = pd.DataFrame({
    'MKT': mkt_ex,
    'SMB': smb,
    'HML': hml,
    'rf': rf.reindex(mkt_ex.index).fillna(0),
}).dropna(how='all')

ff3_daily.to_parquet(RESULTS_DIR / 'bist_ff3_daily.parquet')
print(f'      Saved bist_ff3_daily.parquet: {ff3_daily.shape}')

# Monthly aggregation: compound within month
def compound(r):
    return (1 + r).prod() - 1

ff3_monthly = ff3_daily[['MKT', 'SMB', 'HML']].resample('ME').apply(compound)
ff3_monthly['rf'] = ff3_daily['rf'].resample('ME').sum()
ff3_monthly.to_parquet(RESULTS_DIR / 'bist_ff3_monthly.parquet')
print(f'      Saved bist_ff3_monthly.parquet: {ff3_monthly.shape}')

# Print sanity checks
print('\n' + '=' * 70)
print('SUMMARY (annualized)')
print('=' * 70)
print(f'{"Factor":<8} {"Daily mean":>12} {"Daily vol":>12} {"Sharpe":>8}')
for col in ['MKT', 'SMB', 'HML']:
    s = ff3_daily[col].dropna()
    ann_ret = s.mean() * 252 * 100
    ann_vol = s.std() * np.sqrt(252) * 100
    sr = ann_ret / ann_vol if ann_vol > 0 else 0
    print(f'{col:<8} {ann_ret:>11.2f}% {ann_vol:>11.2f}% {sr:>8.2f}')

print('\nMonthly correlations:')
print(ff3_monthly[['MKT', 'SMB', 'HML']].corr().round(3))

print('\nDone.')
