from _rp import _RP  # portable-path shim (added when vendoring)
﻿# -*- coding: utf-8 -*-
"""
intraday_sim.py -- GOLD-STANDARD intraday portfolio evaluator.

This is the *truth* evaluator. EXP10's per-trade Sharpe is a fast SCREEN that
treats overlapping trades as independent samples and overstates gross Sharpe;
this module simulates the actual held book day-by-day and is what deployable
numbers must come from.

Correct by construction:
  - dollar-neutral, gross-2.0 books (long +1/nL, short -1/nS)
  - overlapping sleeves blended at FIXED capital (average of the H live sleeves)
  - daily portfolio return = sum_i w_i * r_i  (SIMPLE returns -- portfolios
    aggregate linearly; log returns would be wrong here)
  - cost = bps * aggregate-book Sigma|dW| (same gross basis as the return)
  - selection routed through the FIXED EXP10.select_periods (ranks on realised
    buy-and-hold trailing return, no look-ahead)

verify() asserts the two structural invariants (net exposure 0, gross 2.0).

    from shared.intraday_sim import evaluate, verify
    r = evaluate(60, 'M10', basket='top5', top_n=2, hold=3); print(r)
"""
import sys, math
import numpy as np, pandas as pd

_EXPDIR = _RP(r"\Research_done\claude_data\scripts\experiments")
if _EXPDIR not in sys.path: sys.path.insert(0, _EXPDIR)
import EXP10_period_distribution as E
from EXP_lib import IS_START, IS_END, OOS_START, OOS_END, PERIODS_PER_YEAR

IS0, IS1 = pd.Timestamp(IS_START), pd.Timestamp(IS_END)
OO0, OO1 = pd.Timestamp(OOS_START), pd.Timestamp(OOS_END)


def _sharpe(a, mask, ann):
    r = np.asarray(a, float)[mask]; r = r[~np.isnan(r)]
    return r.mean()/r.std()*math.sqrt(ann) if len(r) > 5 and r.std() > 0 else float("nan")

def _breakeven(g, t, mask):
    g = np.asarray(g, float)[mask]; t = np.asarray(t, float)[mask]
    return g.mean()/t.mean()*1e4 if t.mean() > 0 else float("nan")


def _simulate(cells, selected, retmap, hold, overlap):
    """Day-by-day held-book simulation. Returns (daily_df, max_net, avg_gross, n_invested)."""
    cd = cells["date"].tolist(); cp = cells["period"].astype(int).tolist()
    L  = [s.split("|") if isinstance(s, str) and s else [] for s in cells["longs"]]
    S  = [s.split("|") if isinstance(s, str) and s else [] for s in cells["shorts"]]
    N  = len(cells)
    act = [cp[i] in selected.get(cd[i], set()) for i in range(N)]

    def rc(t, c):
        v = retmap.get((t, cd[c], cp[c]))
        return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v

    def bookw(k):  # dollar-neutral gross-2.0
        w = {}
        if L[k] and S[k]:
            for t in L[k]: w[t] = w.get(t, 0.0) + 1.0/len(L[k])
            for t in S[k]: w[t] = w.get(t, 0.0) - 1.0/len(S[k])
        return w

    rows = []; Wp = {}; max_net = 0.0; grosses = []; n_inv = 0
    for c in range(N):
        live = [k for k in range(max(0, c-hold), c) if act[k]]  # entered k in [c-H, c-1]
        W = {}
        for k in live:
            for t, wt in bookw(k).items(): W[t] = W.get(t, 0.0) + wt
        m = len(live)
        if overlap == 'blend' and m > 0:
            W = {t: v/m for t, v in W.items()}      # FIXED capital: average sleeves
        max_net = max(max_net, abs(sum(W.values())))
        if m > 0:
            grosses.append(sum(abs(v) for v in W.values())); n_inv += 1
        ret = sum(W.get(t, 0.0) * rc(t, c) for t in W)
        nm = set(W) | set(Wp)
        to = sum(abs(W.get(t, 0.0) - Wp.get(t, 0.0)) for t in nm)
        rows.append((cd[c], ret, to)); Wp = W
    df = pd.DataFrame(rows, columns=["date", "g", "t"]).groupby("date", as_index=False).sum()
    return df, max_net, (np.mean(grosses) if grosses else 0.0), n_inv


def evaluate(freq, model, basket='top5', top_n=2, hold=3,
             lookback='expanding', selection='S1', rank_horizon='B',
             overlap='blend', cache=None, _return_daily=False):
    """Full held-book evaluation of one intraday config. Returns a metrics dict."""
    port = E.build_portfolio_cached(freq, model, basket, cache if cache is not None else {})
    selected = E.select_periods(port, lookback, top_n, selection, rank_horizon, hold)

    resid = pd.read_parquet(E._resid_path(freq, model))
    resid["date"] = pd.to_datetime(resid["date"])
    retmap = resid.set_index(["ticker", "date", "period"])["ret"].to_dict()

    cells = port.copy(); cells["date"] = pd.to_datetime(cells["date"])
    cells = cells.sort_values(["date", "period"]).reset_index(drop=True)

    df, max_net, avg_gross, n_inv = _simulate(cells, selected, retmap, hold, overlap)
    ann = 252  # series is aggregated to DAILY (groupby date); annualize by trading days, not periods/yr
    d = pd.to_datetime(df["date"]); is_m = ((d >= IS0) & (d < IS1)).values
    oo = ((d >= OO0) & (d <= OO1)).values

    out = dict(freq=freq, model=model, basket=basket, top_n=top_n, hold=hold,
               overlap=overlap, n_days=len(df), n_invested_days=n_inv,
               max_net_exposure=max_net, avg_gross_exposure=avg_gross,
               avg_turnover=float(df["t"].mean()),
               gross_IS=_sharpe(df["g"], is_m, ann), gross_OOS=_sharpe(df["g"], oo, ann),
               breakeven_bp_OOS=_breakeven(df["g"], df["t"], oo))
    for b in (5, 10):
        n = df["g"] - df["t"] * (b/1e4)
        out[f"net{b}_IS"]  = _sharpe(n, is_m, ann)
        out[f"net{b}_OOS"] = _sharpe(n, oo, ann)
    return (out, df) if _return_daily else out


def verify(freq=60, model='M10', **kw):
    """Assert the structural invariants. Raises AssertionError on violation."""
    r = evaluate(freq, model, **kw)
    assert r["max_net_exposure"] < 1e-9, f"NOT dollar-neutral: max|net|={r['max_net_exposure']:.2e}"
    # gross <= 2.0 always (no leverage); blending offsetting sleeves can lower it below 2.0,
    # which is correct -- cost and return use the same blended book, so they stay consistent.
    assert r["avg_gross_exposure"] <= 2.0 + 1e-6, f"LEVERED (gross>2.0): {r['avg_gross_exposure']:.3f}"
    assert 1.0 < r["avg_gross_exposure"] <= 2.0 + 1e-6, f"gross out of range: {r['avg_gross_exposure']:.3f}"
    return r


if __name__ == "__main__":
    print("intraday_sim gold-standard evaluator -- self check\n")
    r = verify(60, 'M10', basket='top5', top_n=2, hold=3)
    print(f"  [verify] max|net exposure| = {r['max_net_exposure']:.2e}  (dollar-neutral OK)")
    print(f"  [verify] avg gross         = {r['avg_gross_exposure']:.3f} (2.00 => $1 long + $1 short)")
    print(f"  shipped config 60min M10 top5 top_n=2 hold=3:")
    print(f"    gross  IS {r['gross_IS']:+.2f}  OOS {r['gross_OOS']:+.2f}")
    print(f"    net@5  IS {r['net5_IS']:+.2f}  OOS {r['net5_OOS']:+.2f}")
    print(f"    net@10 IS {r['net10_IS']:+.2f}  OOS {r['net10_OOS']:+.2f}")
    print(f"    break-even OOS = {r['breakeven_bp_OOS']:.2f} bp")
