"""Load anomaly panel + PCs for intraday residual models (mirrors script 62)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared import configs as CFG


def load_factor_panel(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily factor panel with PC/R17 merges and per-ticker ffill."""
    panel = pd.read_parquet(CFG.RESULTS_DIR / 'anomaly_panel_daily_r16.parquet')
    panel['date'] = pd.to_datetime(panel['date'])

    pc_fp = CFG.RESULTS_DIR / 'pc_loadings_panel.parquet'
    if pc_fp.exists():
        pc_panel = pd.read_parquet(pc_fp)
        pc_panel['date'] = pd.to_datetime(pc_panel['date'])
        pc_cols = [c for c in pc_panel.columns if c not in ('date', 'ticker')]
        if pc_cols:
            all_dates = sorted(panel['date'].unique())
            pieces = []
            for c in pc_cols:
                wide = pc_panel.pivot(index='date', columns='ticker', values=c)
                wide = wide.reindex(pd.to_datetime(all_dates)).ffill()
                long = wide.stack().rename(c).reset_index()
                long.columns = ['date', 'ticker', c]
                pieces.append(long.set_index(['date', 'ticker']))
            pc_ffill = pd.concat(pieces, axis=1).reset_index()
            panel = panel.merge(pc_ffill, on=['date', 'ticker'], how='left')

    r17_fp = CFG.RESULTS_DIR / 'anomaly_panel_daily_r17.parquet'
    if r17_fp.exists():
        r17 = pd.read_parquet(r17_fp)
        r17['date'] = pd.to_datetime(r17['date'])
        new_cols = [c for c in r17.columns if c not in panel.columns]
        if new_cols:
            panel = panel.merge(
                r17[['date', 'ticker'] + new_cols], on=['date', 'ticker'], how='left',
            )

    t0 = pd.Timestamp(start or CFG.PERIODS['FULL'][0])
    t1 = pd.Timestamp(end or CFG.PERIODS['FULL'][1])
    panel = panel[(panel['date'] >= t0) & (panel['date'] <= t1)].copy()

    id_cols = {'date', 'ticker'}
    factor_cols = [c for c in panel.columns if c not in id_cols]
    panel = panel.sort_values(['ticker', 'date'])
    panel[factor_cols] = panel.groupby('ticker', sort=False)[factor_cols].ffill()
    return panel


def lag_factors_one_day(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    out = panel.sort_values(['ticker', 'date']).copy()
    for f in factor_cols:
        out[f] = out.groupby('ticker')[f].shift(1)
    return out


def copy_factor_models_from_daily() -> Path:
    """Sync model JSONs + index from D_residual to F_intraday/02_factor_models."""
    src = CFG.RESULTS_D_RESIDUAL / '01_factor_models'
    dst = CFG.F_INTRADAY_MODELS
    dst.mkdir(parents=True, exist_ok=True)
    for fp in src.glob('*.json'):
        (dst / fp.name).write_bytes(fp.read_bytes())
    for fp in src.glob('*.csv'):
        (dst / fp.name).write_bytes(fp.read_bytes())
    return dst


def load_model_meta(model_id: str) -> dict:
    import json
    idx = pd.read_csv(CFG.F_INTRADAY_MODELS / '_MODELS_INDEX.csv')
    row = idx[idx['id'] == model_id].iloc[0]
    with open(CFG.F_INTRADAY_MODELS / row['json_file'], encoding='utf-8') as f:
        return json.load(f)
