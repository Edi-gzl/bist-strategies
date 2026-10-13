"""
Brogaard-style intraday tables and plots: decile sorts, period breakdown, K/M, costs.

Normalization note (NOT z-score / NOT inverse-normal):
  - rank_l1 / kns = rank percentile → demean → divide by sum(|centered|) per column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared import configs as CFG
from shared.stats import newey_west_var

PERIODS_PER_YEAR = CFG.PERIODS_PER_YEAR_INTRADAY

PERIOD_LABELS: dict[int, str] = {
    0: 'overnight 18:00→10:30',
    1: '10:30-11:00',
    2: '11:00-11:30',
    3: '11:30-12:00',
    4: '12:00-12:30',
    5: '12:30-13:00',
    6: '13:00-13:30',
    7: '13:30-14:00',
    8: '14:00-14:30',
    9: '14:30-15:00',
    10: '15:00-15:30',
    11: '15:30-16:00',
    12: '16:00-16:30',
    13: '16:30-17:00',
    14: '17:00-17:30',
    15: '17:30-18:00',
}


def nw_t(series: pd.Series) -> tuple[float, float, int]:
    """Mean with Newey-West t (lag=1 default in stats module)."""
    x = pd.Series(series).dropna().astype(float)
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    mu = float(x.mean())
    nv = newey_west_var(x.values - mu)
    t = mu / np.sqrt(nv / len(x)) if nv > 0 else np.nan
    return mu, t, len(x)


def ann_period_return(mean_per_period: float) -> float:
    return float(mean_per_period * PERIODS_PER_YEAR)


def prepare_residual_frame(
    resid_fp,
    panel_lagged: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load residuals, add ret_next and optional value weights."""
    from shared.intraday_portfolio import resid_long_to_sorted, add_next_period_return

    try:
        import pyarrow.parquet as pq
        filters = []
        if start:
            filters.append(('date', '>=', pd.Timestamp(start)))
        if end:
            filters.append(('date', '<=', pd.Timestamp(end)))
        if filters:
            df = pq.read_table(resid_fp, filters=filters).to_pandas()
        else:
            df = pd.read_parquet(resid_fp)
    except Exception:
        df = pd.read_parquet(resid_fp)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date', 'period']).reset_index(drop=True)
    df = add_next_period_return(df)
    if panel_lagged is not None and 'R1_size' in panel_lagged.columns:
        w = panel_lagged[['date', 'ticker', 'R1_size']].copy()
        w['mcap_w'] = np.exp(w['R1_size'].clip(-30, 30))
        df = df.merge(w[['date', 'ticker', 'mcap_w']], on=['date', 'ticker'], how='left')
    return df


def assign_buckets(eps: pd.Series, n_buckets: int) -> pd.Series:
    """1 = lowest residual (reversal long leg), n = highest."""
    ranks = eps.rank(method='first')
    return pd.qcut(ranks, n_buckets, labels=list(range(1, n_buckets + 1)))


def bucket_returns_one_cs(
    grp: pd.DataFrame,
    n_buckets: int,
    weight_col: str | None,
) -> dict[int, float]:
    g = grp.dropna(subset=['residual', 'ret_next'])
    if len(g) < n_buckets * 2:
        return {}
    eps = g.set_index('ticker')['residual']
    ret = g.set_index('ticker')['ret_next']
    buckets = assign_buckets(eps, n_buckets)
    out: dict[int, float] = {}
    for b in range(1, n_buckets + 1):
        tickers = buckets[buckets == b].index
        if len(tickers) < 1:
            continue
        if weight_col and weight_col in g.columns:
            w = g.set_index('ticker').loc[tickers, weight_col]
            w = w / w.sum()
            out[b] = float((ret.reindex(tickers) * w).sum())
        else:
            out[b] = float(ret.reindex(tickers).mean())
    return out


def monotonicity_curve(
    df: pd.DataFrame,
    n_buckets: int,
    weight_col: str | None = None,
    min_names: int = 10,
) -> pd.DataFrame:
    """Average next-period return by residual bucket (pooled over all cross-sections)."""
    g = df.dropna(subset=['residual', 'ret_next']).copy()
    if g.empty:
        return pd.DataFrame()

    def _assign(s: pd.Series) -> pd.Series:
        if len(s) < min_names:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(
            s.rank(method='first'),
            n_buckets,
            labels=list(range(1, n_buckets + 1)),
        )

    g['bucket'] = g.groupby(['date', 'period'], observed=True)['residual'].transform(_assign)
    g = g.dropna(subset=['bucket'])
    g['bucket'] = g['bucket'].astype(int)

    if weight_col and weight_col in g.columns:
        g['_w'] = g[weight_col].fillna(0)
        g['_wr'] = g['ret_next'] * g['_w']
        cs = (
            g.groupby(['date', 'period', 'bucket'], observed=True)
            .apply(lambda x: x['_wr'].sum() / x['_w'].sum() if x['_w'].sum() > 0 else np.nan)
            .rename('ret_next')
            .reset_index()
        )
    else:
        cs = (
            g.groupby(['date', 'period', 'bucket'], observed=True)['ret_next']
            .mean()
            .reset_index()
        )

    if cs.empty:
        return pd.DataFrame()
    agg = cs.groupby('bucket')['ret_next'].agg(['mean', 'std', 'count']).reset_index()
    agg['t_nw'] = np.nan
    for b in agg['bucket'].unique():
        _, t, _ = nw_t(cs.loc[cs['bucket'] == b, 'ret_next'])
        agg.loc[agg['bucket'] == b, 't_nw'] = t
    agg['ann_return'] = agg['mean'].apply(ann_period_return)
    return agg.sort_values('bucket')


def reversal_by_period_table(
    df: pd.DataFrame,
    n_buckets: int = 10,
    weight_col: str | None = None,
    min_names: int = 10,
) -> pd.DataFrame:
    """
    Rows = clock period; columns = D1..D10 mean ret + L-H (low ε minus high ε) + t(L-H).
    """
    g = df.dropna(subset=['residual', 'ret_next']).copy()

    def _assign(s: pd.Series) -> pd.Series:
        if len(s) < min_names:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(
            s.rank(method='first'),
            n_buckets,
            labels=list(range(1, n_buckets + 1)),
        )

    g['bucket'] = g.groupby(['date', 'period'], observed=True)['residual'].transform(_assign)
    g = g.dropna(subset=['bucket'])
    g['bucket'] = g['bucket'].astype(int)

    if weight_col and weight_col in g.columns:
        g['_w'] = g[weight_col].fillna(0)
        g['_wr'] = g['ret_next'] * g['_w']
        cs = (
            g.groupby(['date', 'period', 'bucket'], observed=True)
            .apply(lambda x: x['_wr'].sum() / x['_w'].sum() if x['_w'].sum() > 0 else np.nan)
            .rename('ret_next')
            .reset_index()
        )
    else:
        cs = (
            g.groupby(['date', 'period', 'bucket'], observed=True)['ret_next']
            .mean()
            .reset_index()
        )
    if cs.empty:
        return pd.DataFrame()

    wide = cs.pivot_table(
        index=['date', 'period'],
        columns='bucket',
        values='ret_next',
        aggfunc='first',
    )
    if 1 not in wide.columns or n_buckets not in wide.columns:
        return pd.DataFrame()

    wide['L_minus_H'] = wide[1] - wide[n_buckets]
    period_rows = []
    for period in sorted(wide.index.get_level_values('period').unique()):
        sub = wide.xs(period, level='period')
        lh_series = sub['L_minus_H'].dropna()
        if lh_series.empty:
            continue
        row = {'period': period, 'period_label': PERIOD_LABELS.get(int(period), str(period))}
        for b in range(1, n_buckets + 1):
            if b in sub.columns:
                row[f'D{b}'] = float(sub[b].mean())
                row[f'D{b}_ann'] = ann_period_return(row[f'D{b}'])
            else:
                row[f'D{b}'] = np.nan
                row[f'D{b}_ann'] = np.nan
        row['L_minus_H'] = float(lh_series.mean())
        row['L_minus_H_ann'] = ann_period_return(row['L_minus_H'])
        _, row['t_L_minus_H'], _ = nw_t(lh_series)
        row['n_cs'] = len(lh_series)
        period_rows.append(row)
    return pd.DataFrame(period_rows)


def reversal_ls_timeseries(
    df: pd.DataFrame,
    threshold: float = 0.10,
    weight_col: str | None = None,
    min_names: int = 10,
) -> pd.DataFrame:
    """Reversal L/S: long bottom threshold, short top; EW or VW.

    Returns columns: signal_date, period, r_long, r_short, r_ls, n_long, n_short, turnover.
    `turnover` measures basket overlap vs previous (date, period) step; used by cost-stress.
    """
    g = df.dropna(subset=['residual', 'ret_next']).copy()
    if g.empty:
        return pd.DataFrame()

    g['rank_pct'] = g.groupby(['date', 'period'], observed=True)['residual'].rank(pct=True)
    g['is_long'] = g['rank_pct'] <= threshold
    g['is_short'] = g['rank_pct'] >= (1 - threshold)
    min_leg = 2 if min_names >= 10 else 1

    cnt = g.groupby(['date', 'period'], observed=True).agg(
        n_long=('is_long', 'sum'),
        n_short=('is_short', 'sum'),
        n_all=('residual', 'count'),
    )
    valid = cnt[(cnt['n_all'] >= min_names) & (cnt['n_long'] >= min_leg) & (cnt['n_short'] >= min_leg)]
    if valid.empty:
        return pd.DataFrame()

    if weight_col and weight_col in g.columns:
        g['_w'] = g[weight_col].fillna(0)
        g['_wr'] = g['ret_next'] * g['_w']
        long_agg = (
            g[g['is_long']]
            .groupby(['date', 'period'], observed=True)
            .apply(lambda x: x['_wr'].sum() / x['_w'].sum() if x['_w'].sum() > 0 else np.nan)
        )
        short_agg = (
            g[g['is_short']]
            .groupby(['date', 'period'], observed=True)
            .apply(lambda x: x['_wr'].sum() / x['_w'].sum() if x['_w'].sum() > 0 else np.nan)
        )
    else:
        long_agg = g[g['is_long']].groupby(['date', 'period'], observed=True)['ret_next'].mean()
        short_agg = g[g['is_short']].groupby(['date', 'period'], observed=True)['ret_next'].mean()

    # Build a per-step basket membership table so we can compute turnover.
    long_members = (
        g[g['is_long']]
        .groupby(['date', 'period'], observed=True)['ticker']
        .apply(lambda s: frozenset(s))
    )
    short_members = (
        g[g['is_short']]
        .groupby(['date', 'period'], observed=True)['ticker']
        .apply(lambda s: frozenset(s))
    )

    port = pd.DataFrame({'r_long': long_agg, 'r_short': short_agg})
    port = port.join(valid, how='inner')
    port['r_ls'] = port['r_long'] - port['r_short']
    port = port.dropna(subset=['r_ls'])

    # Order by (date, period) and compute turnover against the previous step.
    port = port.sort_index()
    long_seq = [long_members.get(idx, frozenset()) for idx in port.index]
    short_seq = [short_members.get(idx, frozenset()) for idx in port.index]
    turnovers = []
    prev_all: frozenset[str] = frozenset()
    for L, S in zip(long_seq, short_seq):
        new_all = L | S
        if not prev_all:
            turnovers.append(1.0)
        else:
            union = prev_all | new_all
            inter = prev_all & new_all
            turnovers.append(1.0 - len(inter) / len(union) if union else 1.0)
        prev_all = new_all
    port['turnover'] = turnovers

    port = port.reset_index().rename(columns={'date': 'signal_date'})
    return port[['signal_date', 'period', 'r_long', 'r_short', 'r_ls',
                 'n_long', 'n_short', 'turnover']]


def persistency_table(
    df: pd.DataFrame,
    lags: list[int],
    threshold: float = 0.10,
    min_names: int = 10,
) -> pd.DataFrame:
    """Reversal L/S return at lag L periods (30-min bars). Single pass over cross-sections."""
    work = df.sort_values(['ticker', 'date', 'period']).copy()
    max_lag = max(lags)
    for L in range(1, max_lag + 1):
        work[f'_t{L}'] = work.groupby('ticker')['ret'].shift(-L)

    slopes_by_lag: dict[int, list[float]] = {L: [] for L in lags}
    for (_, _), grp in work.groupby(['date', 'period'], sort=False):
        g = grp.dropna(subset=['residual'])
        if len(g) < min_names:
            continue
        eps = g.set_index('ticker')['residual']
        rank_pct = eps.rank(pct=True)
        longs = list(eps[rank_pct <= threshold].index)
        shorts = list(eps[rank_pct >= (1 - threshold)].index)
        if len(longs) < 2 or len(shorts) < 2:
            continue
        for lag in lags:
            tgt = g.set_index('ticker')[f'_t{lag}']
            r_ls = tgt.reindex(longs).mean() - tgt.reindex(shorts).mean()
            if pd.notna(r_ls):
                slopes_by_lag[lag].append(float(r_ls))

    rows = []
    for lag in lags:
        s = slopes_by_lag.get(lag, [])
        if not s:
            continue
        mu, t, n = nw_t(pd.Series(s))
        rows.append({
            'lag_periods': lag,
            'lag_minutes': lag * 30,
            'mean_per_period': mu,
            'ann_return': ann_period_return(mu),
            't_NW': t,
            'n_obs': n,
        })
    return pd.DataFrame(rows)


# Sparse K/M grid (25 cells) vs full 16×16 (256) — use for --fast-km
KM_SPARSE_GRID = [1, 2, 4, 8, 16]

_KM_CTX: dict = {}


def _prepare_km_context(
    df: pd.DataFrame,
    threshold: float,
    min_names: int,
) -> dict:
    work = df.sort_values(['ticker', 'date', 'period']).copy()
    work['ret_fwd'] = work.groupby('ticker')['ret'].shift(-1)
    timeline = (
        work[['date', 'period']].drop_duplicates().sort_values(['date', 'period']).reset_index(drop=True)
    )
    t_keys = list(zip(timeline['date'], timeline['period']))
    n_t = len(t_keys)
    grouped = {
        k: g.dropna(subset=['residual'])
        for k, g in work.groupby(['date', 'period'], sort=False)
    }
    # fwd_by_step[i] = per-ticker return from bar i -> i+1 (same as ret_next at bar i).
    fwd_by_step: list[dict[str, float]] = []
    for i in range(n_t):
        d, p = t_keys[i]
        g = grouped.get((d, p))
        if g is not None:
            fwd_by_step.append(g.set_index('ticker')['ret_fwd'].to_dict())
        else:
            fwd_by_step.append({})
    formations: list[tuple[tuple[str, ...], tuple[str, ...]] | None] = []
    for i in range(n_t):
        d, p = t_keys[i]
        g = grouped.get((d, p))
        if g is None or len(g) < min_names:
            formations.append(None)
            continue
        eps = g.set_index('ticker')['residual']
        rank_pct = eps.rank(pct=True)
        longs = tuple(eps[rank_pct <= threshold].index)
        shorts = tuple(eps[rank_pct >= (1 - threshold)].index)
        if len(longs) < 2 or len(shorts) < 2:
            formations.append(None)
        else:
            formations.append((longs, shorts))
    return {'n_t': n_t, 'formations': formations, 'fwd_by_step': fwd_by_step}


def _km_one_pair(K: int, M: int, ctx: dict) -> dict | None:
    formations = ctx['formations']
    fwd_by_step = ctx['fwd_by_step']
    n_t = ctx['n_t']
    rets = []
    for i in range(0, n_t - M, K):
        fm = formations[i]
        if fm is None:
            continue
        longs, shorts = fm
        period_ls = []
        for j in range(1, M + 1):
            if i + j - 1 >= len(fwd_by_step):
                break
            r = fwd_by_step[i + j - 1]
            rl_vals = [r.get(t) for t in longs if t in r and pd.notna(r.get(t))]
            rs_vals = [r.get(t) for t in shorts if t in r and pd.notna(r.get(t))]
            if not rl_vals or not rs_vals:
                period_ls = []
                break
            rl = float(np.mean(rl_vals))
            rs = float(np.mean(rs_vals))
            if np.isnan(rl) or np.isnan(rs):
                period_ls = []
                break
            period_ls.append(rl - rs)
        if len(period_ls) != M:
            continue
        rets.append(float(np.prod(1 + np.array(period_ls)) - 1))
    if not rets:
        return None
    mu, t, n = nw_t(pd.Series(rets))
    return {
        'K_formation': K,
        'M_holding': M,
        'mean_holding_return': mu,
        'ann_return_per_period': ann_period_return(mu / M) if M > 0 else np.nan,
        't_NW': t,
        'n_formations': n,
    }


def _km_worker_init(ctx: dict) -> None:
    global _KM_CTX
    _KM_CTX = ctx


def _km_worker(pair: tuple[int, int]) -> dict | None:
    K, M = pair
    return _km_one_pair(K, M, _KM_CTX)


def km_strategy_table(
    df: pd.DataFrame,
    k_list: list[int],
    m_list: list[int],
    threshold: float = 0.20,
    min_names: int = 10,
    *,
    workers: int = 0,
    progress_label: str = '',
) -> pd.DataFrame:
    """
    Formation every K (date, period) steps; hold fixed reversal L/S for M bars (compound).

    workers>1: parallel over (K,M) pairs (ProcessPool). progress_label: printed as K/M tag.
    """
    import os
    import sys

    ctx = _prepare_km_context(df, threshold, min_names)
    if ctx['n_t'] == 0:
        return pd.DataFrame()

    pairs = [(K, M) for K in k_list for M in m_list]
    total = len(pairs)
    tag = f'{progress_label} ' if progress_label else ''
    rows: list[dict] = []

    w = workers if workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    use_pool = workers > 1 and total >= 4

    if use_pool:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f'  K/M {tag}0/{total} (prep done, {w} workers) ...', flush=True)
        with ProcessPoolExecutor(max_workers=min(w, total), initializer=_km_worker_init, initargs=(ctx,)) as ex:
            futs = {ex.submit(_km_worker, p): p for p in pairs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                row = fut.result()
                if row:
                    rows.append(row)
                if done == 1 or done % 5 == 0 or done == total:
                    print(f'  K/M {tag}{done}/{total}', flush=True)
    else:
        for idx, (K, M) in enumerate(pairs, 1):
            row = _km_one_pair(K, M, ctx)
            if row:
                rows.append(row)
            if idx == 1 or idx % 5 == 0 or idx == total:
                print(f'  K/M {tag}{idx}/{total}', flush=True)

    return pd.DataFrame(rows)


def cost_sensitivity_table(
    port_ls: pd.DataFrame,
    bps_list: list[int] | None = None,
    *, 
    turnover_col: str = 'turnover_weighted',  # default: CORRECT weighted (Σ|Δw|); legacy: 'turnover' (Jaccard)
    bps_divisor: float = 1.0,
) -> pd.DataFrame:
    """Round-trip cost per rebalance: net = r_ls − turnover_col × (bps/bps_divisor) / 10_000.

    turnover_col: 'turnover_weighted' (Σ|Δw|, CORRECT for costs) or 'turnover' (Jaccard, LEGACY — understates ~4x).
    bps_divisor: e.g. 16 to spread a daily 5 bps budget across 16 intraday periods.
    """
    bps_list = bps_list or CFG.COSTS_BPS
    r = port_ls['r_ls'].astype(float)
    if turnover_col in port_ls.columns:
        to = port_ls[turnover_col].astype(float).fillna(1.0)
        # NOTE: Jaccard ('turnover') clips to [0,1] and understates cost ~4x vs weighted.
        # Prefer 'turnover_weighted' for honest cost numbers.
        if turnover_col == 'turnover':
            to = to.clip(lower=0.0, upper=1.0)
        elif turnover_col == 'turnover_weighted':
            to = to.clip(lower=0.0, upper=4.0)  # max theoretical: 2× gross-2.0 churn
        cost_basis = f'{turnover_col} (bps/{bps_divisor:g})' if bps_divisor != 1 else turnover_col
    else:
        to = pd.Series(1.0, index=port_ls.index)
        cost_basis = 'full rebalance (no turnover column)'

    rows = []
    for bps in bps_list:
        eff_bps = bps / bps_divisor
        cost = to * (eff_bps / 10000.0)
        net = r - cost
        vol = net.std()
        sharpe = net.mean() / vol * np.sqrt(PERIODS_PER_YEAR) if vol > 0 else np.nan
        rows.append({
            'cost_bps_rt': bps,
            'bps_effective': eff_bps,
            'mean_net': float(net.mean()),
            'ann_return_net': ann_period_return(float(net.mean())),
            'sharpe_ann': float(sharpe),
            'hit_rate': float((net > 0).mean()),
            'avg_turnover': float(to.mean()),
            'turnover_col': turnover_col,
            'cost_basis': cost_basis,
        })
    return pd.DataFrame(rows)
