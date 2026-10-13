from _rp import _RP  # portable-path shim (added when vendoring)
"""
BIST50 Fundamental Data Fetcher (v3 — current Q2 2026 universe)
================================================================
Fetches quarterly + annual financials for BIST50 (2020 → present).

FIXES vs v1:
- Updated BIST50 universe to current Q2 2026 period (verified 2026-05-06)
- Adds dtype conversion (period columns → numeric) before save
- Saves both wide and long format
- Persists ticker-level errors to JSON for re-runs
- Adds --strict flag (abort on any error) and --resume flag
- Conservative rate limiting (1.5s between calls)
- Tries financial_group=2 then falls back to 3 (UFRS_K) for banks

CHANGES vs v2:
  ADDED:   AEFES, BRSAN, BTCIM, CIMSA, DOHOL, DSTKF, KONTR, KUYAS, MAVI, MIATK, PASEU
  REMOVED: AKSEN, ENJSA, ODAS, SKBNK, TKFEN, VESTL  (left BIST50 in 2026 reviews)

Output:
    raw/bist50_fundamentals_wide.parquet   (one row per item, period cols wide)
    raw/bist50_fundamentals_long.parquet   (one row per item-period, melted)
    raw/bist50_fundamentals_errors.json    (ticker → error message)

Usage:
    python 01_fetch_fundamentals_v3.py             # full BIST50
    python 01_fetch_fundamentals_v3.py --test      # 4-ticker sample
    python 01_fetch_fundamentals_v3.py --strict    # abort on first error
    python 01_fetch_fundamentals_v3.py --resume    # skip tickers already in output
"""

import sys, io, os, json, time, logging, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import isyatirimhisse as iy
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

OUT_DIR = _RP(r"\Research_done\claude_data\raw")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# UNIVERSE — Q2 2026 period (April 1 - June 30, 2026)
# Source: KAP / TradingView / yatirimadeger.com cross-validated 2026-05-06
# ────────────────────────────────────────────────────────────────────
BIST50_NON_BANK = [
    # Holdings (4)
    'ALARK', 'DOHOL', 'KCHOL', 'SAHOL',
    # Transportation / Aviation (4)
    'PASEU', 'PGSUS', 'TAVHL', 'THYAO',
    # Food & Beverage Retail (5)
    'BIMAS', 'CCOLA', 'MGROS', 'SOKM', 'ULKER',
    # Chemicals & Plastics (6)
    'GUBRF', 'HEKTS', 'PETKM', 'SASA', 'SISE', 'TUPRS',
    # Automotive (3)
    'DOAS', 'FROTO', 'TOASO',
    # Telecom (2)
    'TCELL', 'TTKOM',
    # Cement (3)
    'BTCIM', 'CIMSA', 'OYAKC',
    # Steel/Metal (3)
    'BRSAN', 'EREGL', 'KRDMD',
    # Real Estate / GYO (2)
    'EKGYO', 'KUYAS',
    # Mining (2)
    'KOZAA', 'KOZAL',
    # Energy Tech (2)
    'ASTOR', 'KONTR',
    # Defense (1)
    'ASELS',
    # Construction (1)
    'ENKAI',
    # Durable Goods (1)
    'ARCLK',
    # Apparel (1)
    'MAVI',
    # Software (1)
    'MIATK',
    # Beverage (1)
    'AEFES',
    # Factoring (1) — special: technically a financial institution but uses non-bank schema
    'DSTKF',
]
BIST50_BANK = [
    # Banks (7) — use financial_group='2' (UFRS), fallback to '3' (UFRS_K)
    'AKBNK', 'GARAN', 'HALKB', 'ISCTR', 'TSKB', 'VAKBN', 'YKBNK',
]

assert len(BIST50_NON_BANK) + len(BIST50_BANK) == 50, \
    f'Expected 50 tickers, got {len(BIST50_NON_BANK) + len(BIST50_BANK)}'

TEST_NON_BANK = ['THYAO', 'ASELS', 'MGROS']
TEST_BANK     = ['AKBNK']

START_YEAR = '2020'
END_YEAR   = '2025'
SLEEP_SEC  = 1.5


def fetch_one(ticker: str, group: str, retries: int = 2):
    """Fetch one ticker with retries on transient failures."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = iy.fetch_financials(
                symbols=[ticker],
                start_year=START_YEAR,
                end_year=END_YEAR,
                exchange='TRY',
                financial_group=group,
            )
            if df is None or df.empty:
                raise ValueError(f'empty result for group={group}')
            return df, None
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                logging.warning(f'  {ticker} attempt {attempt+1} failed: {e}, retrying...')
                time.sleep(SLEEP_SEC * 2)
    return None, last_err


def fetch_bank_with_fallback(ticker: str):
    """Banks: try group 2 (UFRS), fall back to group 3 (UFRS_K consolidated) if empty."""
    df, err = fetch_one(ticker, group='2')
    if df is not None:
        return df, '2', None

    logging.warning(f'  {ticker} group=2 failed ({err}), trying group=3...')
    df, err = fetch_one(ticker, group='3')
    if df is not None:
        return df, '3', None

    return None, None, err


def fetch_universe(non_bank: list, bank: list, strict: bool = False, resume: bool = False):
    """Run full universe; return list of frames + errors dict."""
    frames, errors, completed = [], {}, set()

    if resume:
        existing_path = os.path.join(OUT_DIR, 'bist50_fundamentals_long.parquet')
        if os.path.exists(existing_path):
            old = pd.read_parquet(existing_path)
            completed = set(old['SYMBOL'].unique())
            logging.info(f'Resume mode: {len(completed)} tickers already in output')

    # Non-banks
    for i, ticker in enumerate(non_bank):
        if ticker in completed:
            logging.info(f'[non-bank] [{i+1}/{len(non_bank)}] {ticker}: SKIPPED (resume)')
            continue
        df, err = fetch_one(ticker, group='1')
        if df is not None:
            df['_group'] = '1'
            frames.append(df)
            logging.info(f'[non-bank] [{i+1}/{len(non_bank)}] {ticker}: {df.shape[0]} rows')
        else:
            errors[ticker] = err
            logging.error(f'[non-bank] [{i+1}/{len(non_bank)}] {ticker} FAILED: {err}')
            if strict:
                raise RuntimeError(f'Strict mode: aborting on {ticker}')
        time.sleep(SLEEP_SEC)

    # Banks (with fallback)
    for i, ticker in enumerate(bank):
        if ticker in completed:
            logging.info(f'[bank] [{i+1}/{len(bank)}] {ticker}: SKIPPED (resume)')
            continue
        df, group, err = fetch_bank_with_fallback(ticker)
        if df is not None:
            df['_group'] = group
            frames.append(df)
            logging.info(f'[bank] [{i+1}/{len(bank)}] {ticker}: {df.shape[0]} rows (group={group})')
        else:
            errors[ticker] = err
            logging.error(f'[bank] [{i+1}/{len(bank)}] {ticker} FAILED: {err}')
            if strict:
                raise RuntimeError(f'Strict mode: aborting on {ticker}')
        time.sleep(SLEEP_SEC)

    return frames, errors


def save_panel(frames: list):
    """Save both wide and long format with proper dtypes."""
    if not frames:
        logging.error('No frames to save')
        return

    panel = pd.concat(frames, ignore_index=True)

    # Convert period columns to numeric
    period_cols = [c for c in panel.columns if '/' in str(c)]
    for c in period_cols:
        panel[c] = pd.to_numeric(panel[c], errors='coerce')

    # Save wide
    wide_path = os.path.join(OUT_DIR, 'bist50_fundamentals_wide.parquet')
    panel.to_parquet(wide_path, index=False)
    logging.info(f'Saved wide → {wide_path}  shape={panel.shape}')

    # Save long
    id_cols = [c for c in panel.columns if '/' not in str(c)]
    long = panel.melt(
        id_vars=id_cols,
        value_vars=period_cols,
        var_name='period',
        value_name='value',
    )
    long['value'] = pd.to_numeric(long['value'], errors='coerce')

    def period_to_date(p: str) -> str:
        try:
            year, q = p.split('/')
            q_map = {'3': f'{year}-03-31', '6': f'{year}-06-30',
                     '9': f'{year}-09-30', '12': f'{year}-12-31'}
            return q_map.get(q, '')
        except Exception:
            return ''

    long['period_end'] = pd.to_datetime(long['period'].apply(period_to_date),
                                        errors='coerce')
    long['is_annual']  = long['period'].str.endswith('/12')

    long_path = os.path.join(OUT_DIR, 'bist50_fundamentals_long.parquet')
    long.to_parquet(long_path, index=False)
    logging.info(f'Saved long → {long_path}  shape={long.shape}')

    return panel, long


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='4-ticker sample')
    parser.add_argument('--strict', action='store_true', help='Abort on any error')
    parser.add_argument('--resume', action='store_true', help='Skip already-fetched tickers')
    args = parser.parse_args()

    if args.test:
        non_bank, bank = TEST_NON_BANK, TEST_BANK
    else:
        non_bank, bank = BIST50_NON_BANK, BIST50_BANK

    logging.info(f'BIST50 universe (Q2 2026 period):')
    logging.info(f'  {len(non_bank)} non-banks')
    logging.info(f'  {len(bank)} banks')
    logging.info(f'  Total: {len(non_bank) + len(bank)}')
    logging.info(f'  Estimated time: ~{(len(non_bank) + len(bank)) * (SLEEP_SEC + 1.5) / 60:.1f} minutes')

    frames, errors = fetch_universe(non_bank, bank,
                                     strict=args.strict, resume=args.resume)

    if errors:
        err_path = os.path.join(OUT_DIR, 'bist50_fundamentals_errors.json')
        with open(err_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        logging.warning(f'Errors saved → {err_path} ({len(errors)} tickers)')

    if frames:
        panel, long = save_panel(frames)

        print('\n' + '=' * 60)
        print('COVERAGE SUMMARY')
        print('=' * 60)
        summary = panel.groupby('SYMBOL').agg(
            n_codes=('FINANCIAL_ITEM_CODE', 'nunique'),
            group=('_group', 'first'),
        )
        print(summary.to_string())

        print(f'\nTotal frames: {len(frames)}')
        print(f'Total rows wide: {len(panel)}')
        print(f'Total rows long: {len(long)}')
        print(f'Errors: {len(errors)}')


if __name__ == '__main__':
    main()
