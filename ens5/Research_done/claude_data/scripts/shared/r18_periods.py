"""Round 18 portfolio / plot date windows (POST vs model-effective start)."""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from shared import configs as CFG

# POST = post-TAS29 inflation-restated fundamentals (project priority).
POST_START = pd.Timestamp(CFG.PERIODS['POST'][0])
POST_END = pd.Timestamp(CFG.PERIODS['POST'][1])

PRE_START = pd.Timestamp(CFG.PERIODS['PRE'][0])
PRE_END = pd.Timestamp(CFG.PERIODS['PRE'][1])
FULL_START = pd.Timestamp(CFG.PERIODS['FULL'][0])
FULL_END = pd.Timestamp(CFG.PERIODS['FULL'][1])


def pc_loadings_ready_date() -> pd.Timestamp:
    """First date in pc_loadings_panel.parquet (FULL build ≈ 2022; POST-only ≈ 2024)."""
    fp = CFG.RESULTS_DIR / 'pc_loadings_panel.parquet'
    if fp.exists():
        df = pd.read_parquet(fp, columns=['date'])
        return pd.Timestamp(df['date'].min())
    # ~252 trading days after FULL start if panel not built yet
    return FULL_START + pd.Timedelta(days=380)


SLICE_LABELS = {
    'pre': 'Pre-TAS29 hyperinflation era (2021-01-01 → 2022-12-31)',
    'post': 'Post-TAS29 clean accounting (2023-01-01 → end)',
    'full': 'Full sample (2021-01-01 → end)',
    'effective': 'First date with enough residuals (model-specific)',
    'pc_ready': 'First date PC loadings exist (from panel)',
}


def first_residual_date(R: pd.DataFrame, min_names: int = 10) -> Optional[pd.Timestamp]:
    """First index date with enough non-NaN residuals for portfolio sorts."""
    if R is None or R.empty:
        return None
    counts = R.notna().sum(axis=1)
    ok = counts[counts >= min_names]
    if ok.empty:
        return None
    return pd.Timestamp(ok.index.min())


def effective_start(R: pd.DataFrame, uses_pc: bool = False) -> pd.Timestamp:
    """Start after warmup: max(POST, first residual date [, PC ready if hybrid])."""
    starts = [POST_START]
    fr = first_residual_date(R)
    if fr is not None:
        starts.append(fr)
    if uses_pc:
        starts.append(pc_loadings_ready_date())
    return max(starts)


def model_uses_pc(model_id: str) -> bool:
    """Heuristic: hybrid / PC-only R18 models."""
    return model_id in {'M07', 'M08', 'M09', 'M13', 'M15', 'M25'}


def slice_window(
    slice_name: str,
    R: pd.DataFrame,
    model_id: str = '',
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return (start, end) inclusive for filtering residual / portfolio dates.

    slice_name: pre | post | full | effective | pc_ready
    """
    end = FULL_END
    name = slice_name.lower().strip()
    if name == 'pre':
        return PRE_START, PRE_END
    if name == 'full':
        return FULL_START, end
    if name == 'post':
        return POST_START, end
    if name == 'pc_ready':
        return pc_loadings_ready_date(), end
    if name == 'effective':
        return effective_start(R, uses_pc=model_uses_pc(model_id)), end
    raise ValueError(f'Unknown slice {slice_name!r}; use pre|post|full|effective|pc_ready')


def filter_residual_panel(R: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    R = R.copy()
    R.index = pd.to_datetime(R.index)
    return R.loc[(R.index >= start) & (R.index <= end)]
