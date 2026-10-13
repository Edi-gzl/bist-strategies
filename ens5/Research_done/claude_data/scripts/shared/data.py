from _rp import _RP  # portable-path shim (added when vendoring)
"""
shared.data — Single source of truth for loading BIST50 raw data.

Every script that needs price/fundamental/sector data uses this module.
Replaces ad-hoc loading code that was duplicated across F01/F02/F03/F04.

Usage:
    from shared.data import load_all
    
    data = load_all()
    px = data['prices']           # daily prices DataFrame
    fund = data['fundamentals']    # quarterly fundamentals long format
    sect = data['sectors']         # ticker → fine sector mapping
    macro = data['macro_sectors']  # ticker → macro sector mapping
    ret_wide = data['ret_wide']    # date × ticker daily returns matrix
    mcap_wide = data['mcap_wide']  # date × ticker market cap matrix
    mkt_ret = data['mkt_ret']      # equal-weighted market return series
    tickers = data['tickers']      # sorted list of all 50 tickers
    banks = data['banks']          # set of 7 bank tickers
"""
from pathlib import Path
import pandas as pd
import numpy as np

# === Paths ===
ROOT = Path(_RP(r"\Research_done"))
DATA_DIR = ROOT / 'claude_data' / 'raw'
SAMPLE_CSV = ROOT / 'claude_data' / 'bist50_factor_sample_2024Q4_v2.csv'

# === Constants verified across F01-F04 audits ===
BANKS = {'AKBNK', 'GARAN', 'HALKB', 'ISCTR', 'TSKB', 'VAKBN', 'YKBNK'}

# Bank vs non-bank financial-item codes
NI_CODE_BANK = '3Z'      # Net income — banks
NI_CODE_NB   = '3L'      # Net income — non-banks
EQ_CODE_BANK = '2O'      # Equity — banks
EQ_CODE_NB   = '2N'      # Equity — non-banks

# Sector aggregation — fine → macro
SECTOR_AGG = {
    'Banking':         'Financials',
    'Dev Banking':     'Financials',
    'Holding':         'Financials',
    'Holding-Koc':     'Financials',
    'Holding-Sabanci': 'Financials',
    'Real Estate':     'Financials',
    'REIT':            'Financials',
    'Food Retail':     'Consumer',
    'Beverages':       'Consumer',
    'Food':            'Consumer',
    'Apparel Retail':  'Consumer',
    'Auto Distrib':    'Consumer',
    'Automotive':      'Industrials',
    'Airlines':        'Industrials',
    'Airports':        'Industrials',
    'Construction':    'Industrials',
    'Cement':          'Industrials',
    'Defense':         'Industrials',
    'Steel':           'Industrials',
    'Steel Processing':'Industrials',
    'Energy Equip':    'Industrials',
    'Electric Util':   'Industrials',
    'Refining':        'Energy',
    'Chemicals':       'Materials',
    'Synthetics':      'Materials',
    'Petrochemicals':  'Materials',
    'Glass & Matls':   'Materials',
    'Telecom':         'Telecom',
    'Technology':      'Telecom',
    'Consumer Elec':   'Industrials',
    'Electronics':     'Industrials',
    'Small Cap':       'Industrials',
}

# Standard analysis periods
# Authoritative source = shared.configs.PERIODS (uppercase keys)
# Lowercase aliases below are kept for legacy script backward compatibility.
try:
    from . import configs as _cfg
    PERIODS = {
        # Legacy lowercase keys (used by scripts 01-12)
        'full':       _cfg.PERIODS['FULL'],
        'pre_tas29':  _cfg.PERIODS['PRE'],
        'post_tas29': _cfg.PERIODS['POST'],
        'oos_2026':   _cfg.PERIODS['OOS'],
        # Uppercase aliases (forward-compatible, match configs.PERIODS)
        'FULL': _cfg.PERIODS['FULL'],
        'PRE' : _cfg.PERIODS['PRE'],
        'POST': _cfg.PERIODS['POST'],
        'OOS' : _cfg.PERIODS['OOS'],
    }
except Exception:
    # configs not yet available (e.g. during initial bootstrap)
    PERIODS = {
        'full':       ('2021-01-01', '2026-05-05'),
        'pre_tas29':  ('2021-01-01', '2022-12-31'),
        'post_tas29': ('2023-01-01', '2026-05-05'),
        'oos_2026':   ('2026-01-01', '2026-05-05'),
        'FULL':       ('2021-01-01', '2026-05-05'),
        'PRE' :       ('2021-01-01', '2022-12-31'),
        'POST':       ('2023-01-01', '2026-05-05'),
        'OOS' :       ('2026-01-01', '2026-05-05'),
    }

PERIOD_LABELS = {
    'full':       'Full (2021-2025)',
    'pre_tas29':  'Pre-TAS29 (2021-2022)',
    'post_tas29': 'Post-TAS29 (2023-2025)',
    'oos_2026':   'OOS Holdout (Jan-May 2026)',
}

TDAYS = 252  # trading days per year


def load_prices():
    """Load daily prices with returns and market cap pre-computed."""
    px = pd.read_parquet(DATA_DIR / 'bist50_prices_daily.parquet')
    px['date'] = pd.to_datetime(px['date'])
    px = px.sort_values(['ticker', 'date']).reset_index(drop=True)
    px['ret'] = px.groupby('ticker')['close_adj'].pct_change()
    return px


def load_fundamentals():
    """Load quarterly fundamentals in long format."""
    fund = pd.read_parquet(DATA_DIR / 'bist50_fundamentals_long.parquet')
    fund['period_end'] = pd.to_datetime(fund['period_end'])
    return fund


def load_sectors():
    """Load fine and macro sector mappings."""
    sect = pd.read_csv(SAMPLE_CSV)
    fine = dict(zip(sect['ticker'], sect['sector']))
    macro = {t: SECTOR_AGG.get(s, 'Other') for t, s in fine.items()}
    return fine, macro


def build_wide_panels(px):
    """Pivot to date × ticker matrices for vectorized strategy code."""
    ret_wide = px.pivot(index='date', columns='ticker', values='ret')
    mcap_wide = px.pivot(index='date', columns='ticker', values='mktcap_try')
    close_wide = px.pivot(index='date', columns='ticker', values='close_adj')
    return ret_wide, mcap_wide, close_wide


def load_all():
    """Load everything needed for any factor script."""
    px = load_prices()
    fund = load_fundamentals()
    fine_sect, macro_sect = load_sectors()
    ret_wide, mcap_wide, close_wide = build_wide_panels(px)
    mkt_ret = ret_wide.mean(axis=1)  # equal-weighted market return
    tickers = sorted(px['ticker'].unique())
    
    return {
        'prices':       px,
        'fundamentals': fund,
        'sectors':      fine_sect,
        'macro_sectors': macro_sect,
        'ret_wide':     ret_wide,
        'mcap_wide':    mcap_wide,
        'close_wide':   close_wide,
        'mkt_ret':      mkt_ret,
        'tickers':      tickers,
        'banks':        BANKS,
    }


def get_ni_code(ticker):
    """Return correct net-income code for a ticker."""
    return NI_CODE_BANK if ticker in BANKS else NI_CODE_NB


def get_eq_code(ticker):
    """Return correct equity code for a ticker."""
    return EQ_CODE_BANK if ticker in BANKS else EQ_CODE_NB


def compute_ttm(fund, ticker, code, lag_days=60):
    """De-cumulate Turkish-GAAP cumulative-within-year reports → TTM.
    
    Turkish reports are CUMULATIVE within fiscal year (Q3 = Jan-Sep, NOT Jul-Sep).
    This function:
    1. Subtracts prior cumulative to get standalone Q1, Q2, Q3, Q4
    2. Rolls 4 quarters into TTM
    3. Adds lag_days from period_end → signal_date (when the data is public)
    """
    sub = fund[(fund['SYMBOL']==ticker)&(fund['FINANCIAL_ITEM_CODE']==code)].copy()
    if sub.empty: return pd.DataFrame()
    sub = sub.sort_values('period_end').reset_index(drop=True)
    sub['month'] = sub['period_end'].dt.month
    sub['year'] = sub['period_end'].dt.year
    sub['value'] = pd.to_numeric(sub['value'], errors='coerce')
    rows = []
    for _, r in sub.iterrows():
        m, yr, cum = r['month'], r['year'], r['value']
        if m == 3: standalone = cum
        elif m == 6:
            q1 = sub[(sub['year']==yr)&(sub['month']==3)]
            standalone = cum - float(q1['value'].iloc[0]) if not q1.empty else np.nan
        elif m == 9:
            q2 = sub[(sub['year']==yr)&(sub['month']==6)]
            standalone = cum - float(q2['value'].iloc[0]) if not q2.empty else np.nan
        elif m == 12:
            q3 = sub[(sub['year']==yr)&(sub['month']==9)]
            standalone = cum - float(q3['value'].iloc[0]) if not q3.empty else cum
        else: standalone = np.nan
        rows.append({'period_end': r['period_end'], 'standalone': standalone})
    sa = pd.DataFrame(rows).sort_values('period_end').reset_index(drop=True)
    sa['standalone'] = pd.to_numeric(sa['standalone'], errors='coerce')
    sa['ttm'] = sa['standalone'].rolling(4, min_periods=4).sum()
    sa['signal_date'] = sa['period_end'] + pd.Timedelta(days=lag_days)
    return sa[['period_end', 'ttm', 'signal_date']].dropna(subset=['ttm'])


def get_pit_value(fund, ticker, code, lag_days=60):
    """Stock variable (e.g., equity, total assets) — latest reported balance, NOT TTM."""
    sub = fund[(fund['SYMBOL']==ticker)&(fund['FINANCIAL_ITEM_CODE']==code)].copy()
    if sub.empty: return pd.DataFrame()
    sub = sub.sort_values('period_end').reset_index(drop=True)
    sub['value_pit'] = pd.to_numeric(sub['value'], errors='coerce')
    sub['signal_date'] = sub['period_end'] + pd.Timedelta(days=lag_days)
    return sub[['period_end', 'value_pit', 'signal_date']].dropna()


if __name__ == '__main__':
    print("Sanity checking shared.data ...")
    data = load_all()
    print(f"  prices       : {data['prices'].shape}")
    print(f"  fundamentals : {data['fundamentals'].shape}")
    print(f"  tickers      : {len(data['tickers'])}")
    print(f"  date range   : {data['ret_wide'].index[0].date()} → {data['ret_wide'].index[-1].date()}")
    print(f"  banks        : {sorted(data['banks'])}")
    print(f"  macro sectors: {pd.Series(data['macro_sectors']).value_counts().to_dict()}")
    print("OK")
