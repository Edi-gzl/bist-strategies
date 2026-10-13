"""
shared.engine — Vectorized strategy engine + cost model + rebal-date generators.

Single canonical implementation. All factor scripts use this.
Replaces the duplicated/inconsistent engines that existed in F02/F03/F04.

DESIGN DECISION (2026-05-07): winsorization is OFF by default. Pass winsor>0
explicitly only if you need it for a sensitivity test. Reason: with only 50
stocks the 1% winsor doesn't actually clip much (≈ ±0.5 stocks per tail) and
adds an arbitrary parameter to the methodology.
"""
import numpy as np
import pandas as pd
from .stats import perf

TDAYS = 252


# ====================================================================
# Rebalance date generators
# ====================================================================
def daily_rebal(period_idx):
    return pd.DatetimeIndex(period_idx)


def weekly_rebal(period_idx):
    df = pd.DataFrame({'date': period_idx})
    iso = df['date'].dt.isocalendar()
    df['key'] = iso['year'].astype(str) + '_' + iso['week'].astype(str)
    return pd.DatetimeIndex(df.groupby('key')['date'].min().values)


def monthly_rebal(period_idx):
    df = pd.DataFrame({'date': period_idx})
    df['ym'] = df['date'].dt.to_period('M').astype(str)
    return pd.DatetimeIndex(df.groupby('ym')['date'].min().values)


def quarterly_rebal(period_idx):
    df = pd.DataFrame({'date': period_idx})
    df['yq'] = df['date'].dt.to_period('Q').astype(str)
    return pd.DatetimeIndex(df.groupby('yq')['date'].min().values)


def yearly_rebal(period_idx):
    df = pd.DataFrame({'date': period_idx})
    return pd.DatetimeIndex(df.groupby(df['date'].dt.year)['date'].min().values)


def semi_apr_oct(period_idx):
    df = pd.DataFrame({'date': period_idx})
    apr = df[df['date'].dt.month == 4].groupby(df[df['date'].dt.month==4]['date'].dt.year)['date'].min()
    oct_ = df[df['date'].dt.month == 10].groupby(df[df['date'].dt.month==10]['date'].dt.year)['date'].min()
    return pd.DatetimeIndex(sorted(list(apr.values) + list(oct_.values)))


REBAL_FUNCS = {
    'D': daily_rebal,
    'W': weekly_rebal,
    'M': monthly_rebal,
    'Q': quarterly_rebal,
    'Y': yearly_rebal,
    'AprOct': semi_apr_oct,
}

REBAL_LABELS = {
    'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly',
    'Q': 'Quarterly', 'Y': 'Yearly', 'AprOct': 'Apr+Oct (Sol1a)',
}


def get_rebal_dates(freq, period_dates, period_start=None, period_end=None):
    """Generate rebalance dates within a period."""
    rd = REBAL_FUNCS[freq](period_dates)
    if period_start is not None:
        rd = rd[rd >= pd.Timestamp(period_start)]
    if period_end is not None:
        rd = rd[rd <= pd.Timestamp(period_end)]
    return rd


# ====================================================================
# Signal panel construction
# ====================================================================
def panel_to_wide(panel_data, ret_wide, signal_col='signal'):
    """Convert sparse signal points (date × ticker) to forward-filled daily wide matrix.
    
    Args:
        panel_data: dict {ticker: DataFrame with 'date' + signal_col + 'mcap' columns}
        ret_wide: returns wide matrix (provides date index)
        signal_col: name of signal column in each ticker's DataFrame
    
    Returns:
        sig_wide: date × ticker forward-filled signal
        mcap_wide_signal: date × ticker forward-filled market cap (for filtering)
    """
    tickers = list(panel_data.keys())
    all_dates = ret_wide.index
    sig_wide = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    mcap_wide_sig = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    for t in tickers:
        df = panel_data[t]
        if df.empty:
            continue
        s = df.set_index('date')[signal_col].sort_index()
        sig_wide[t] = s.reindex(all_dates, method='ffill')
        if 'mcap' in df.columns:
            m = df.set_index('date')['mcap'].sort_index()
            mcap_wide_sig[t] = m.reindex(all_dates, method='ffill')
    return sig_wide, mcap_wide_sig


# ====================================================================
# Strategy engine
# ====================================================================
def run_strategy(sig_wide, mcap_wide, rebal_dates, period_dates,
                 ret_wide, ticker_sector,
                 N=10, NL=2, NS=2, winsor=0.0, method='naive',
                 sign='+'):
    """Form long/short portfolios and compute returns + trade log + turnover.
    
    Args:
        sig_wide: date × ticker signal matrix (forward-filled)
        mcap_wide: date × ticker market cap (for liquidity filter)
        rebal_dates: dates to rebalance
        period_dates: full daily date range to compute returns over
        ret_wide: date × ticker daily returns
        ticker_sector: dict {ticker: macro_sector}
        N: top/bottom N for naive
        NL, NS: long/short per sector for SN
        winsor: winsorization fraction. DEFAULT 0.0 (no winsorization).
                Pass winsor>0 only for explicit sensitivity testing.
        method: 'naive' or 'sn'
        sign: '+' = long high, '-' = long low (e.g., for accruals/asset growth)
    
    Returns dict with:
        long_ret, short_ret, factor_ret  : pd.Series of daily returns
        positions     : DataFrame (date × ticker) of 'long'/'short'/'neutral'
        pos_dict      : dict {rebal_date: pd.Series of positions}
        trade_log     : DataFrame of position changes
        avg_turnover  : average one-sided turnover per rebalance
        n_rebalances  : count of rebalance events
        n_long_per_rebal  : list of long-leg sizes per rebalance
        n_short_per_rebal : list of short-leg sizes per rebalance
    """
    tickers = list(sig_wide.columns)
    pos_dict = {}
    n_long_per_rebal = []
    n_short_per_rebal = []
    
    for rd in rebal_dates:
        if rd not in sig_wide.index:
            continue
        sig = sig_wide.loc[rd]
        mcap = mcap_wide.loc[rd] if rd in mcap_wide.index else pd.Series(dtype=float)
        valid = sig.notna() & (mcap.notna() if not mcap.empty else True)
        if not mcap.empty:
            valid = valid & (mcap > 0)
        sig = sig[valid] * (1 if sign == '+' else -1)
        
        threshold = 2 * N if method == 'naive' else 4
        if len(sig) < threshold:
            pos_dict[rd] = pd.Series('neutral', index=tickers, dtype=object)
            n_long_per_rebal.append(0)
            n_short_per_rebal.append(0)
            continue
        
        if winsor > 0:
            q_lo = sig.quantile(winsor)
            q_hi = sig.quantile(1 - winsor)
            sig = sig.clip(q_lo, q_hi)
        
        pos = pd.Series('neutral', index=tickers, dtype=object)
        if method == 'naive':
            ranks = sig.rank(ascending=False, method='first')
            longs = ranks[ranks <= N].index.tolist()
            shorts = ranks[ranks > (len(sig) - N)].index.tolist()
            for t in longs: pos[t] = 'long'
            for t in shorts: pos[t] = 'short'
        else:  # sector-neutral
            sect_map = {t: ticker_sector.get(t, 'Other') for t in sig.index}
            df_t = pd.DataFrame({
                'ticker': sig.index, 'sig': sig.values,
                'sect': [sect_map[t] for t in sig.index]
            })
            for sec, grp in df_t.groupby('sect'):
                if len(grp) < NL + NS:
                    continue
                ranks = grp['sig'].rank(ascending=False, method='first')
                longs = grp.loc[ranks[ranks <= NL].index, 'ticker'].tolist()
                shorts = grp.loc[ranks[ranks > len(grp) - NS].index, 'ticker'].tolist()
                for t in longs: pos[t] = 'long'
                for t in shorts: pos[t] = 'short'
        pos_dict[rd] = pos
        n_long_per_rebal.append(int((pos == 'long').sum()))
        n_short_per_rebal.append(int((pos == 'short').sum()))
    
    if not pos_dict:
        return None
    
    pos_df = pd.DataFrame(pos_dict).T.reindex(period_dates).ffill().fillna('neutral')
    long_m = (pos_df == 'long')
    short_m = (pos_df == 'short')
    rp = ret_wide.reindex(period_dates)
    long_r = rp.where(long_m).mean(axis=1)
    short_r = rp.where(short_m).mean(axis=1)
    factor_r = long_r - short_r
    
    # Trade log
    trade_log = []
    rebal_keys = sorted(pos_dict.keys())
    for i, rd in enumerate(rebal_keys):
        curr = pos_dict[rd]
        prev = pos_dict[rebal_keys[i-1]] if i > 0 else pd.Series('neutral', index=tickers, dtype=object)
        for t in tickers:
            p_prev = prev.get(t, 'neutral')
            p_curr = curr.get(t, 'neutral')
            if p_prev != p_curr:
                trade_log.append({
                    'date': rd, 'ticker': t,
                    'sector': ticker_sector.get(t, '?'),
                    'prev_pos': p_prev, 'new_pos': p_curr,
                    'action': f'{p_prev}→{p_curr}',
                })
    trade_df = pd.DataFrame(trade_log)
    
    # Turnover — TWO metrics stored:
    #   1. Jaccard (name-overlap, legacy diagnostic, NOT for costs)
    #   2. weighted (Sigma|dw|, gross-2.0 basis, CORRECT for costs)
    turnover_jaccard = []
    turnover_weighted = []
    prev_w = {}
    for i in range(len(rebal_keys)):
        curr = pos_dict[rebal_keys[i]]
        curr_long = set(curr[curr == 'long'].index)
        curr_short = set(curr[curr == 'short'].index)
        # Jaccard (legacy, bounded [0,1])
        if i > 0:
            prev_long = set(pos_dict[rebal_keys[i-1]][pos_dict[rebal_keys[i-1]] == 'long'].index)
            prev_short = set(pos_dict[rebal_keys[i-1]][pos_dict[rebal_keys[i-1]] == 'short'].index)
            turn_l = len(prev_long.symmetric_difference(curr_long)) / (2 * max(1, len(prev_long)))
            turn_s = len(prev_short.symmetric_difference(curr_short)) / (2 * max(1, len(prev_short)))
            turnover_jaccard.append((turn_l + turn_s) / 2)
        # Weighted (Sigma|dw|, CORRECT for costs)
        nL, nS = len(curr_long), len(curr_short)
        w_new = {t: 1.0 / max(1, nL) for t in curr_long}
        for t in curr_short:
            w_new[t] = w_new.get(t, 0.0) - 1.0 / max(1, nS)
        if i == 0:
            weighted_to = sum(abs(v) for v in w_new.values())
        else:
            all_t = set(prev_w) | set(w_new)
            weighted_to = sum(abs(w_new.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_t)
        turnover_weighted.append(weighted_to)
        prev_w = w_new
    avg_turnover = np.mean(turnover_weighted) if turnover_weighted else 0.0
    avg_turnover_jaccard = np.mean(turnover_jaccard) if turnover_jaccard else 0.0
    
    return {
        'long_ret':     long_r,
        'short_ret':    short_r,
        'factor_ret':   factor_r,
        'positions':    pos_df,
        'pos_dict':     pos_dict,
        'trade_log':    trade_df,
        'avg_turnover': avg_turnover,           # weighted (Sigma|dw|) — CORRECT for costs
        'avg_turnover_jaccard': avg_turnover_jaccard,  # legacy diagnostic only
        'turnover_per_rebal': turnover_weighted,
        'n_rebalances': len(pos_dict),
        'rebal_dates':  rebal_keys,
        'n_long_per_rebal':  n_long_per_rebal,
        'n_short_per_rebal': n_short_per_rebal,
    }


# ====================================================================
# Cost model — per rebalance event, NOT daily drag
# ====================================================================
def apply_costs(factor_ret, rebal_dates, turnover,
                commission_bps=50, spread_bps=15, impact_bps=10):
    """Apply transaction costs at each rebalance event.
    
    Total = (commission + spread + impact) * turnover / 10000
    
    NOTE: 'turnover' should be WEIGHTED (Sigma|dw|), NOT Jaccard.
    For daily M05, typical weighted turnover ~0.71 (vs Jaccard ~0.17, which
    understates costs by ~4x).  The legacy default cost components below
    (50+15+10=75bps) were calibrated to Jaccard — use 5-10bps with weighted.
    
    Components calibrated for BIST50:
    - 50bp roundtrip commission (BIST broker rates)
    - 15bp bid-ask spread (large-caps average)
    - 10bp market impact (square-root model, ~$1M positions)
    Total = 75bp per stock turned over per rebalance.
    """
    cost_per_rebal = turnover * (commission_bps + spread_bps + impact_bps) / 10000
    net = factor_ret.copy()
    for rd in rebal_dates:
        if rd in net.index and not pd.isna(net.loc[rd]):
            net.loc[rd] = net.loc[rd] - cost_per_rebal
    return net, cost_per_rebal


def annual_cost_estimate(turnover, n_rebal_per_year, total_bps=75):
    """Quick annualized cost estimate for reporting."""
    return n_rebal_per_year * turnover * total_bps / 10000


# ====================================================================
# Convenience wrapper
# ====================================================================
def run_full(sig_wide, mcap_wide, ret_wide, ticker_sector,
             period_start, period_end, freq='M',
             N=10, NL=2, NS=2, winsor=0.0, method='sn', sign='+',
             apply_tc=True, **tc_kwargs):
    """End-to-end: pick period, generate rebal dates, run strategy, apply costs.
    
    Returns full result dict with both gross and net factor returns.
    
    NOTE: winsor=0 by default (no winsorization). Pass winsor>0 explicitly
    only for sensitivity testing.
    """
    period_dates = ret_wide.index[
        (ret_wide.index >= pd.Timestamp(period_start)) &
        (ret_wide.index <= pd.Timestamp(period_end))
    ]
    rebal_dates = get_rebal_dates(freq, period_dates, period_start, period_end)
    res = run_strategy(sig_wide, mcap_wide, rebal_dates, period_dates,
                       ret_wide, ticker_sector,
                       N=N, NL=NL, NS=NS, winsor=winsor, method=method, sign=sign)
    if res is None:
        return None
    res['period_dates'] = period_dates
    res['gross_perf'] = perf(res['factor_ret'])
    if apply_tc:
        net_ret, cost_per_rebal = apply_costs(res['factor_ret'], res['rebal_dates'],
                                                res['avg_turnover'], **tc_kwargs)
        res['net_ret'] = net_ret
        res['cost_per_rebal'] = cost_per_rebal
        res['net_perf'] = perf(net_ret)
        n_per_yr = res['n_rebalances'] / max(1, len(period_dates) / TDAYS)
        res['annual_cost'] = annual_cost_estimate(res['avg_turnover'], n_per_yr,
                                                    sum(tc_kwargs.values()) if tc_kwargs else 75)
    return res
