from _rp import _RP  # portable-path shim (added when vendoring)
"""
BIST50 Daily Price + Market Cap Fetcher (v2)
=============================================
Fetches OHLCV + market cap (PD), float-adj mcap (HAO_PD), shares (SERMAYE).
Universe: Q2 2026 BIST50 (50 tickers verified 2026-05-06).

Output: C:\Research\Research_done\claude_data\raw\bist50_prices_daily.parquet
"""

import sys, io, logging, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import isyatirimhisse as iy
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

OUT_DIR = _RP(r"\Research_done\claude_data\raw")

# Q2 2026 BIST50 universe (verified 2026-05-06)
BIST50 = [
    # Banks (7)
    'AKBNK', 'GARAN', 'HALKB', 'ISCTR', 'TSKB', 'VAKBN', 'YKBNK',
    # Holdings (4)
    'ALARK', 'DOHOL', 'KCHOL', 'SAHOL',
    # Transportation (4)
    'PASEU', 'PGSUS', 'TAVHL', 'THYAO',
    # F&B Retail (5)
    'BIMAS', 'CCOLA', 'MGROS', 'SOKM', 'ULKER',
    # Chemicals (6)
    'GUBRF', 'HEKTS', 'PETKM', 'SASA', 'SISE', 'TUPRS',
    # Auto (3)
    'DOAS', 'FROTO', 'TOASO',
    # Telecom (2)
    'TCELL', 'TTKOM',
    # Cement (3)
    'BTCIM', 'CIMSA', 'OYAKC',
    # Steel/Metal (3)
    'BRSAN', 'EREGL', 'KRDMD',
    # GYO (2)
    'EKGYO', 'KUYAS',
    # Mining (2)
    'KOZAA', 'KOZAL',
    # Energy Tech (2)
    'ASTOR', 'KONTR',
    # Other (single-name sectors)
    'ASELS',  # Defense
    'ENKAI',  # Construction
    'ARCLK',  # Durable Goods
    'MAVI',   # Apparel
    'MIATK',  # Software
    'AEFES',  # Beverage
    'DSTKF',  # Factoring
]

assert len(BIST50) == 50, f'Expected 50 tickers, got {len(BIST50)}'

START_DATE = '01-01-2020'
END_DATE   = '06-05-2026'  # today 2026-05-06 (DD-MM-YYYY format for isyatirimhisse)


def main():
    logging.info(f'Fetching daily prices for {len(BIST50)} tickers...')
    prices = iy.fetch_stock_data(
        symbols=BIST50,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    logging.info(f'Raw data: {prices.shape}')

    # Rename key columns to readable names
    rename = {
        'HGDG_HS_KODU': 'ticker',
        'HGDG_TARIH':   'date',
        'HGDG_KAPANIS': 'close_adj',
        'HGDG_AOF':     'vwap_adj',
        'HGDG_MIN':     'low_adj',
        'HGDG_MAX':     'high_adj',
        'HGDG_HACIM':   'volume',
        'HG_KAPANIS':   'close_raw',
        'HG_MIN':       'low_raw',
        'HG_MAX':       'high_raw',
        'SERMAYE':      'shares_out',
        'PD':           'mktcap_try',
        'PD_USD':       'mktcap_usd',
        'HAO_PD':       'mktcap_float_try',
        'HAO_PD_USD':   'mktcap_float_usd',
        'DOLAR_BAZLI_FIYAT': 'close_usd',
    }
    prices = prices.rename(columns={k: v for k, v in rename.items() if k in prices.columns})

    # Keep useful columns
    keep = ['ticker', 'date', 'close_adj', 'vwap_adj', 'low_adj', 'high_adj', 'volume',
            'close_raw', 'shares_out', 'mktcap_try', 'mktcap_usd',
            'mktcap_float_try', 'mktcap_float_usd', 'close_usd']
    prices = prices[[c for c in keep if c in prices.columns]]

    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, 'bist50_prices_daily.parquet')
    prices.to_parquet(out_path, index=False)
    logging.info(f'Saved → {out_path}  shape={prices.shape}')

    # Coverage summary
    print('\n=== PRICE COVERAGE ===')
    cov = prices.groupby('ticker').agg(
        rows=('date', 'count'),
        first=('date', 'min'),
        last=('date', 'max'),
    )
    print(cov.to_string())
    
    missing = set(BIST50) - set(prices['ticker'].unique())
    if missing:
        print(f'\n⚠️  MISSING TICKERS: {missing}')


if __name__ == '__main__':
    main()
