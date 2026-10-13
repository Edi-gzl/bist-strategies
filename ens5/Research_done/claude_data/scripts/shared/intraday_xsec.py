"""
Cross-sectional utilities for Brogaard-style intraday residual decomposition.

Supports KNS normalization (paper default in script 17) and rank_l1 (Round 18 M05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_kns(df: pd.DataFrame, factor_cols: list[str], date_col: str = 'date') -> pd.DataFrame:
    """Rank → de-mean → divide by sum(|dev|) per (date, factor). Skips sec_* dummies."""
    out = df.copy()
    for f in factor_cols:
        if f.startswith('sec_'):
            continue
        ranked = df.groupby(date_col)[f].rank(method='average')
        n_per = df.groupby(date_col)[f].transform('count')
        rc = ranked / (n_per + 1)
        rc_mean = rc.groupby(df[date_col]).transform('mean')
        centered = rc - rc_mean
        denom = centered.abs().groupby(df[date_col]).transform('sum')
        out[f] = centered / denom.replace(0, np.nan)
    return out


def normalize_zscore_panel(
    df: pd.DataFrame,
    factor_cols: list[str],
    date_col: str = 'date',
) -> pd.DataFrame:
    """Cross-sectional z-score per calendar date (daily script 62 analogue; shared across periods)."""
    out = df.copy()
    for f in factor_cols:
        if f.startswith('sec_'):
            continue
        mu = df.groupby(date_col)[f].transform('mean')
        sd = df.groupby(date_col)[f].transform('std')
        sd_safe = sd.where(sd > 1e-12, 1.0)
        out[f] = (df[f] - mu) / sd_safe
    return out


def normalize_rank_l1(X: np.ndarray) -> np.ndarray:
    """Per cross-section matrix X (n × k): rank-L1 column normalization (M05 daily)."""
    out = np.zeros_like(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return out
    for j in range(X.shape[1]):
        col = X[:, j]
        order = np.argsort(np.argsort(col))
        cr = (order + 1) / (n + 1)
        centered = cr - cr.mean()
        l1 = np.sum(np.abs(centered))
        out[:, j] = centered / l1 if l1 > 1e-12 else centered
    return out


def normalize_factors_matrix(X: np.ndarray, mode: str) -> np.ndarray:
    """Matrix-level norm for one cross-section."""
    if mode in ('rank_l1', 'kns'):
        return normalize_rank_l1(X)
    if mode == 'zscore':
        out = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = X[:, j]
            cs = col.std()
            out[:, j] = (col - col.mean()) / (cs if cs > 1e-12 else 1.0)
        return out
    if mode == 'none':
        return X.copy()
    raise ValueError(f'Unknown normalization: {mode}')


def brogaard_xs_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    min_stocks: int = 20,
) -> dict | None:
    """
    One (date, period) cross-section. RISK = X @ beta (no intercept); RESIDUAL = y - RISK.
    Intercept estimated but absorbed into residual per Brogaard Eq. 3.
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    if len(y) < min_stocks:
        return None
    X_aug = np.column_stack([np.ones(len(y)), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    intercept = beta[0]
    slopes = beta[1:]
    risk = X @ slopes
    residual = y - risk
    fitted_full = intercept + risk
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - fitted_full) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n, k = len(y), X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
    return {
        'risk': risk,
        'residual': residual,
        'ret': y,
        'intercept': intercept,
        'r2': r2,
        'adj_r2': adj_r2,
        'n_stocks': n,
    }
