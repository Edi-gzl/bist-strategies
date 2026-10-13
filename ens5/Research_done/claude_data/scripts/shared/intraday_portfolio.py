"""30-min L/S portfolio construction (mirrors analysis/65 for daily)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared import configs as CFG

PERIODS_PER_YEAR = CFG.PERIODS_PER_YEAR_INTRADAY


def weights_from_baskets(longs: set[str], shorts: set[str]) -> dict[str, float]:
    """Dollar-neutral EW: +1/N_L on long names, -1/N_S on short names (gross 2.0; matches r_ls=mean_long-mean_short return basis so turnover and return share one scale)."""
    w: dict[str, float] = {}
    nL, nS = len(longs), len(shorts)
    if nL > 0:
        for t in longs:
            w[t] = 1.0 / nL
    if nS > 0:
        for t in shorts:
            w[t] = -1.0 / nS
    return w


def turnover_two_sided(prev_w: dict[str, float], w_new: dict[str, float]) -> float:
    """Σᵢ |w_new,i − w_old,i| over all tickers (two-sided traded weight)."""
    if not prev_w:
        return float(sum(abs(v) for v in w_new.values()))
    tickers = set(prev_w) | set(w_new)
    return float(sum(abs(w_new.get(t, 0.0) - prev_w.get(t, 0.0)) for t in tickers))


def turnover_overlap(prev_longs: set[str], prev_shorts: set[str],
                     longs: set[str], shorts: set[str]) -> float:
    """Legacy: 1 − |∩|/|∪| on ticker sets (long ∪ short). Bounded in [0, 1]."""
    if not prev_longs and not prev_shorts:
        return 1.0
    u_prev, u_new = prev_longs | prev_shorts, longs | shorts
    return 1.0 - len(u_prev & u_new) / len(u_prev | u_new) if u_prev | u_new else 1.0


def add_turnover_weighted(port_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute turnover_weighted from longs/shorts columns (sorted time order)."""
    sort_cols = ['signal_date', 'period'] if 'period' in port_df.columns else ['signal_date']
    out = port_df.sort_values(sort_cols).copy()
    prev_w: dict[str, float] = {}
    tw = []
    for _, row in out.iterrows():
        longs = set(str(row['longs']).split('|')) if pd.notna(row.get('longs')) else set()
        shorts = set(str(row['shorts']).split('|')) if pd.notna(row.get('shorts')) else set()
        longs.discard('')
        shorts.discard('')
        w_new = weights_from_baskets(longs, shorts)
        tw.append(turnover_two_sided(prev_w, w_new))
        prev_w = w_new
    out['turnover_weighted'] = tw
    return out


def resid_long_to_sorted(resid_fp) -> pd.DataFrame:
    df = pd.read_parquet(resid_fp)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['ticker', 'date', 'period']).reset_index(drop=True)


def add_next_period_return(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret_next'] = out.groupby('ticker')['ret'].shift(-1)
    return out


def build_ls_portfolio_30m(
    df: pd.DataFrame,
    threshold: float,
    direction: str = 'reversal',
    min_names: int = 10,
    signal_col: str = 'residual',
) -> pd.DataFrame:
    """Quintile/decile L/S at each (date, period); earn ret_next.

    signal_col: column to sort on.
      'residual' = Brogaard residual reversal/momentum (daily M05 spec)
      'risk'     = systematic component X@beta (LYZ-style systematic reversal/momentum)
    """
    if signal_col not in {'residual', 'risk'}:
        raise ValueError(f'signal_col must be residual|risk, got {signal_col!r}')
    rows = []
    prev_longs, prev_shorts = set(), set()
    prev_w: dict[str, float] = {}

    for (date, period), grp in df.groupby(['date', 'period']):
        g = grp.dropna(subset=[signal_col, 'ret_next'])
        if len(g) < min_names:
            continue
        eps = g.set_index('ticker')[signal_col]
        rank_pct = eps.rank(pct=True)
        if direction == 'reversal':
            long_mask = rank_pct <= threshold
            short_mask = rank_pct >= (1 - threshold)
        elif direction == 'momentum':
            long_mask = rank_pct >= (1 - threshold)
            short_mask = rank_pct <= threshold
        else:
            raise ValueError(direction)

        longs = set(eps[long_mask].index)
        shorts = set(eps[short_mask].index)
        min_leg = 2 if min_names >= 10 else 1
        if len(longs) < min_leg or len(shorts) < min_leg:
            continue

        r_next = g.set_index('ticker')['ret_next']
        r_long = r_next.reindex(list(longs)).mean()
        r_short = r_next.reindex(list(shorts)).mean()
        if pd.isna(r_long) or pd.isna(r_short):
            continue
        r_ls = r_long - r_short

        turnover = turnover_overlap(prev_longs, prev_shorts, longs, shorts)
        w_new = weights_from_baskets(longs, shorts)
        turnover_w = turnover_two_sided(prev_w, w_new)

        rows.append({
            'signal_date': date,
            'period': period,
            'r_long': r_long,
            'r_short': r_short,
            'r_ls': r_ls,
            'n_longs': len(longs),
            'n_shorts': len(shorts),
            'turnover': turnover,
            'turnover_weighted': turnover_w,
            'longs': '|'.join(sorted(longs)),
            'shorts': '|'.join(sorted(shorts)),
        })
        prev_longs, prev_shorts = longs, shorts
        prev_w = w_new

    return pd.DataFrame(rows)


def compute_metrics_30m(port_df: pd.DataFrame) -> dict | None:
    if port_df.empty:
        return None
    r = port_df['r_ls'].values
    n = len(r)
    if n == 0 or np.std(r) == 0:
        return None
    daily_mean = float(np.mean(r))
    daily_vol = float(np.std(r))
    ann_factor = np.sqrt(PERIODS_PER_YEAR)
    sharpe = daily_mean / daily_vol * ann_factor
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    dd = (cum / running_max) - 1
    return {
        'n_periods': n,
        'mean_per_period': daily_mean,
        'vol_per_period': daily_vol,
        'ann_return': daily_mean * PERIODS_PER_YEAR,
        'ann_vol': daily_vol * ann_factor,
        'sharpe_ann': sharpe,
        'total_return': float(cum[-1] - 1),
        'max_drawdown': float(dd.min()),
        'hit_rate': float((r > 0).mean()),
        'avg_turnover': float(port_df['turnover'].mean()),
        'avg_turnover_weighted': float(port_df['turnover_weighted'].mean())
        if 'turnover_weighted' in port_df.columns else np.nan,
        'avg_n_longs': float(port_df['n_longs'].mean()),
        'avg_n_shorts': float(port_df['n_shorts'].mean()),
    }
