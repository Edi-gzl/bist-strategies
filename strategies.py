"""BIST strategies, reimplemented from the original research code on Yahoo data.

Each function takes (D, M, W):
    D  daily bars    dict[field] -> DataFrame(date x ticker), ~3 years
    M  1-minute bars dict[field] -> DataFrame(bar start x ticker), last ~7 days
    W  trading dates that have 1-minute data (the dates we can evaluate)
and returns a Result:
    returns  realised net daily return, indexed by REALISATION date
    trades   one dict per name-trade (pending exits have exit_price None)

Source of each rule is cited as <original file>: <lines>.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data import first_open_between, last_close_between


@dataclass
class Result:
    id: str
    name: str
    returns: pd.Series
    trades: list = field(default_factory=list)
    spec: str = ""
    deviations: list = field(default_factory=list)


def _prev(idx: pd.DatetimeIndex, T):
    p = idx.searchsorted(T, side="left")
    return idx[p - 1] if p > 0 else None


def _next(idx: pd.DatetimeIndex, T):
    p = idx.searchsorted(T, side="right")
    return idx[p] if p < len(idx) else None


def _f(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# =====================================================================
# Focus Long — focus_long_strategy_engine.py
#   "Daily pullback-in-uptrend top5": 60d OLS log-price trend, lag 1,
#   eligible score>0 and -1.5<residual<0, top 5 equal weight, daily,
#   weights set for close(t) earn close(t)->close(t+1), 5 bp x turnover.
# =====================================================================
def _trend(prices: pd.DataFrame, lookback: int = 60, lag: int = 1):
    v = np.log(prices.where(prices > 0)).shift(lag).to_numpy(float)
    n, m = v.shape
    slope = np.full((n, m), np.nan); r2 = slope.copy(); res = slope.copy()
    x = np.arange(lookback, dtype=float); xc = x - x.mean(); xss = float(xc @ xc)
    for j in range(m):
        col = v[:, j]
        for e in range(lookback - 1, n):
            y = col[e - lookback + 1:e + 1]
            if not np.isfinite(y).all():
                continue
            ym = y.mean(); yc = y - ym
            b = float(xc @ yc) / xss
            err = y - (ym + b * xc)
            sst = float(yc @ yc); sse = float(err @ err)
            slope[e, j] = b
            r2[e, j] = 1.0 - sse / sst if sst > 0 else 0.0
            mae = float(np.abs(err).mean())
            if mae > 0:
                res[e, j] = err[-1] / mae
    mk = lambda a: pd.DataFrame(a, index=prices.index, columns=prices.columns)
    return mk(slope) * mk(r2).clip(lower=0), mk(res)


def focus_long(D, M, W):
    close = D["Adj Close"]
    cov = close.notna().sum(axis=1)
    close = close.loc[cov >= int(np.ceil(0.80 * close.shape[1]))]      # engine L75-76
    score, resid = _trend(close, 60, 1)
    elig = close.notna() & score.gt(0) & resid.gt(-1.5) & resid.lt(0)   # engine L358-363
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for d in close.index:                                               # _rank_top_weights
        row = score.loc[d].where(elig.loc[d]).dropna().sort_values(ascending=False)
        sel = list(row.head(5).index)
        if sel:
            weights.loc[d, sel] = 1.0 / len(sel)
    fwd = close.shift(-1) / close - 1.0                                 # backtest_weights
    gross = (weights * fwd.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    net = gross - turnover * 5.0 / 10_000.0
    # net[t] is earned from close(t) to close(t+1): book it on t+1
    realised = pd.Series(net.values[:-1], index=close.index[1:])

    trades = []
    raw = D["Close"]
    for T in W:
        if T not in weights.index:
            continue
        T0 = _prev(weights.index, T)
        prev_w = weights.loc[T0] if T0 is not None else weights.loc[T] * 0
        for tk, dw in (weights.loc[T] - prev_w).items():
            if abs(dw) > 1e-12:
                trades.append({"date": str(T.date()), "ticker": tk,
                               "side": "BUY" if dw > 0 else "SELL",
                               "weight": _f(weights.at[T, tk]),
                               "entry_time": "closing auction",
                               "entry_price": _f(raw.at[T, tk], 2),
                               "exit_time": None, "exit_price": None, "ret": None})
    return Result("focus_long", "Focus Long — pullback in uptrend, top 5",
                  realised, trades,
                  "48-name universe; 60-day log-price trend (signal uses closes to t-1); "
                  "buy top 5 by slope x R2 among names with -1.5 < residual < 0; "
                  "equal weight, rebalance daily at the closing auction; 5 bp x turnover.",
                  ["Prices: Yahoo split/dividend-adjusted closes instead of the original "
                   "adjusted official closes."])


# =====================================================================
# Daily BETA phase-matched momentum — run_daily_beta_pm.py (Top3)
# =====================================================================
LOCK_PM = 0.094


def _rank_l1(values: np.ndarray) -> np.ndarray:                         # L48-54
    order = np.argsort(np.argsort(values))
    ranks = (order + 1) / (len(values) + 1)
    c = ranks - ranks.mean()
    s = np.abs(c).sum()
    return c / s if s > 1e-12 else c


def _beta_252(adj: pd.DataFrame) -> pd.DataFrame:
    """MKT_beta = 252d rolling Cov(r, mkt)/Var(mkt), mkt = EW universe return;
    ffill per ticker, missing -> cross-sectional median, lagged one session."""
    r = adj.pct_change(fill_method=None)
    mkt = r.mean(axis=1)
    beta = pd.DataFrame({c: r[c].rolling(252).cov(mkt) / mkt.rolling(252).var() for c in r})
    beta = beta.ffill()
    beta = beta.apply(lambda row: row.fillna(row.median()), axis=1)
    return beta.shift(1)


def daily_beta_pm(D, M, W, top_n=3):
    C, O, A = D["Close"], D["Open"], D["Adj Close"]
    idx = C.index
    beta = _beta_252(A)
    rets, trades = {}, []
    for T in W:
        prev, T1 = _prev(idx, T), _next(idx, T)
        if prev is None:
            continue
        p1800 = last_close_between(M, T, "17:00", "17:59")              # last continuous price
        adj_ch = A.loc[T] / A.loc[prev]
        raw_ch = C.loc[T] / C.loc[prev]
        bf = (adj_ch / raw_ch).where(lambda x: np.isfinite(x) & (x > 0.05) & (x < 20.0), 1.0)
        y = (p1800 / C.loc[prev]) * bf - 1.0                            # L96
        x = beta.loc[T] if T in beta.index else pd.Series(np.nan, index=C.columns)
        ok = np.isfinite(y.values) & np.isfinite(x.values)
        if ok.sum() < 15:                                               # L64
            continue
        yv = y.values[ok]
        design = np.column_stack([np.ones(ok.sum()), _rank_l1(x.values[ok])])
        coef = np.linalg.lstsq(design, yv, rcond=None)[0]
        score = pd.Series(yv - design @ coef, index=y.index[ok])
        ordered = list(score.sort_values().index)

        if T1 is not None:
            f_on = (A.loc[T1] / A.loc[T]) / (C.loc[T1] / C.loc[T])     # corporate-action factor
            f_on = f_on.where(lambda s: np.isfinite(s) & (s > 0.05) & (s < 20.0), 1.0)
            r_on = O.loc[T1] * f_on / C.loc[T] - 1.0
            r_id = C.loc[T1] / O.loc[T1] - 1.0
        longs = []
        for t in reversed(ordered):                                     # L182-190
            if T1 is not None and not np.isfinite(r_on[t]):
                continue
            if np.isfinite(y[t]) and y[t] >= LOCK_PM:
                continue
            longs.append(t)
            if len(longs) == top_n:
                break
        if T1 is None:                                                  # live: exits tomorrow
            for t in longs:
                trades.append({"date": str(T.date()), "ticker": t, "side": "LONG",
                               "entry_time": "closing auction", "entry_price": _f(C.at[T, t], 2),
                               "exit_time": "next opening auction", "exit_price": None,
                               "ret": None, "residual": _f(score[t], 5)})
            continue
        shorts = []
        for t in ordered:                                               # L192-207
            if not np.isfinite(r_id[t]) or not (C.at[T, t] > 0):
                continue
            if O.at[T1, t] * f_on[t] / C.at[T, t] - 1.0 <= -LOCK_PM:
                continue
            shorts.append(t)
            if len(shorts) == top_n:
                break
        if len(longs) < top_n or len(shorts) < top_n:                   # L209
            continue
        net = 0.5 * r_on[longs].mean() - 0.5 * r_id[shorts].mean() - 0.001
        rets[T1] = float(net)
        for t in longs:
            trades.append({"date": str(T.date()), "ticker": t, "side": "LONG",
                           "entry_time": "closing auction", "entry_price": _f(C.at[T, t], 2),
                           "exit_time": f"opening auction {T1.date()}",
                           "exit_price": _f(O.at[T1, t], 2), "ret": _f(r_on[t], 5),
                           "residual": _f(score[t], 5)})
        for t in shorts:
            trades.append({"date": str(T1.date()), "ticker": t, "side": "SHORT",
                           "entry_time": "opening auction", "entry_price": _f(O.at[T1, t], 2),
                           "exit_time": "closing auction", "exit_price": _f(C.at[T1, t], 2),
                           "ret": _f(-r_id[t], 5), "residual": _f(score[t], 5)})
    return Result("daily_beta_pm", "Daily BETA phase-matched momentum (long/short)",
                  pd.Series(rets, dtype=float), trades,
                  "At 18:00 regress each stock's previous-close -> 18:00 return on its ranked, "
                  "lagged 252-day market beta. Long the 3 highest residuals in today's closing "
                  "auction (exit next opening auction); short the 3 lowest at the next opening "
                  "auction (exit that day's closing auction). 50/50 legs; skip longs up >= 9.4% "
                  "and shorts opening <= -9.4%; 10 bp round trip.",
                  ["Market factor for beta = equal-weight of the 48 names (original: BIST50 FF3 MKT).",
                   "18:00 price = close of the last 1-minute bar up to 17:59."])


# =====================================================================
# EODR end-of-day loser short — run_eodr_causal_delay.py
# =====================================================================
def eodr_short(D, M, W):
    C, O = D["Close"], D["Open"]
    idx = C.index
    rets, trades = {}, []
    for T in W:
        prev = _prev(idx, T)
        if prev is None:
            continue
        sig = last_close_between(M, T, "15:55", "16:00")                # L63-68
        ent = first_open_between(M, T, "16:00", "16:05", inclusive_start=False)  # L66-72
        pa, op, pauc = C.loc[prev], O.loc[T], C.loc[T]
        w = pd.DataFrame({
            "overnight": op / pa - 1.0, "fac": sig / op - 1.0,
            "signal_move": sig / pa - 1.0, "entry_move": ent / pa - 1.0,
            "hold": pauc / ent - 1.0, "sig": sig, "ent": ent, "pauc": pauc})
        valid = (w.overnight.abs().le(0.15) & w.fac.abs().le(0.12) & w.hold.abs().le(0.12)
                 & w.sig.gt(0) & w.ent.gt(0) & w.pauc.gt(0))           # L115-122
        w = w.loc[valid].copy()
        w["z"] = (w.fac - w.fac.mean()) / w.fac.std(ddof=1)            # L124
        w = w.dropna(subset=["fac", "hold", "z"]).sort_values("fac")
        if len(w) < 12:                                                 # L129
            continue
        for rank, (t, row) in enumerate(w.head(2).iterrows(), start=1):  # r1 / r2 fallback
            if (row.z <= -2.0 and row.fac > -0.09
                    and row.signal_move > -0.095 and row.entry_move > -0.095):
                net = 1.0 - row.pauc / row.ent - 0.001
                rets[T] = float(net)
                trades.append({"date": str(T.date()), "ticker": t, "side": "SHORT",
                               "entry_time": "first 1-min open after 16:00",
                               "entry_price": _f(row.ent, 2), "exit_time": "closing auction",
                               "exit_price": _f(row.pauc, 2), "ret": _f(net, 5),
                               "z": _f(row.z, 2), "fallback_rank": rank})
                break
    s = pd.Series(rets, dtype=float).reindex(pd.DatetimeIndex(W)).fillna(0.0)
    return Result("eodr_short", "EODR — end-of-day loser short", s, trades,
                  "At 16:00 rank stocks by open -> 16:00 return. If the worst (or 2nd worst) is "
                  "z <= -2 vs the others, down < 9%, and not near the -10% floor, short it at the "
                  "first minute after 16:00 and cover in the closing auction. 10 bp round trip.",
                  [])


# =====================================================================
# Morning gap-up fade, "10:01 Core" — build_morning_short_repaired_report.py
# =====================================================================
def morning_gap_short(D, M, W):
    C, O, H, L, V, A = (D[k] for k in ("Close", "Open", "High", "Low", "Volume", "Adj Close"))
    idx = C.index
    value = ((H + L + C) / 3.0) * V                                    # daily traded value proxy
    liq = value.rolling(20, min_periods=10).median().shift(1).rank(axis=1, pct=True)
    fac = (A / C)
    action = (fac / fac.shift(1) - 1.0).abs().gt(1e-6)                 # corporate-action day
    rets, trades = {}, []
    for T in W:
        prev = _prev(idx, T)
        if prev is None:
            continue
        auct = O.loc[T]                                                 # opening auction print
        cont = first_open_between(M, T, "10:00", "10:00")               # p_cont_open
        entry = first_open_between(M, T, "10:01", "10:01")              # p_entry_after_first_1m
        pc = C.loc[prev]
        sig_ok = pc.gt(0) & ~action.loc[T].fillna(False) & liq.loc[T].gt(0.25) \
            & (auct.gt(0) | cont.gt(0))                                 # prep() L75-82
        w = pd.DataFrame({"signal": auct.where(auct.gt(0), cont), "entry": entry,
                          "exit": C.loc[T], "pc": pc}).loc[sig_ok]
        if w.empty:
            continue
        w["gap"] = w.signal / w.pc - 1.0
        w["entry_gap"] = w.entry / w.pc - 1.0
        w["rank_high"] = w.gap.rank(method="first", ascending=False)
        n = len(w)
        breadth = (w.gap > 0.01).mean()
        w["valid"] = (n >= 10) & w.entry.gt(0) & w.exit.gt(0) & w.entry_gap.abs().lt(0.09)
        pick = w[w.valid & w.rank_high.le(2) & w.gap.ge(0.01) & (breadth < 0.25)]
        if pick.empty:
            continue
        r = 1.0 - pick.exit / pick.entry - 0.001
        rets[T] = float(r.mean())
        for t, row in pick.iterrows():
            trades.append({"date": str(T.date()), "ticker": t, "side": "SHORT",
                           "entry_time": "10:01 open", "entry_price": _f(row.entry, 2),
                           "exit_time": "closing auction", "exit_price": _f(row.exit, 2),
                           "ret": _f(r[t], 5), "gap": _f(row.gap, 4)})
    s = pd.Series(rets, dtype=float).reindex(pd.DatetimeIndex(W)).fillna(0.0)
    return Result("morning_gap_short", "Morning gap-up fade (10:01 Core)", s, trades,
                  "Gap = opening auction vs previous close. If fewer than 25% of liquid names "
                  "gapped up > 1%, short the top-2 gap-ups (gap >= 1%, entry gap < 9%) at the "
                  "10:01 open and cover in the closing auction. 10 bp round trip.",
                  ["Liquidity filter uses daily volume x typical price (original: sum of "
                   "1-minute traded value) for the 20-day median ranking."])


# =====================================================================
# Fund Liquidity 80, config FL26131 — run_fund_liquidity_80.py
#   RAM25_TCLB75 blend, all-eligible universe (582 names), N=75,
#   liquidity-weighted, 85% equity floor, biweekly, 5 bp.
# =====================================================================
CASH_RATE = {2020: .10, 2021: .17, 2022: .12, 2023: .20, 2024: .45, 2025: .40, 2026: .40}


def _pct(df):
    return df.rank(axis=1, pct=True, method="average")


def _robust126(lr: pd.DataFrame) -> pd.DataFrame:
    """sqrt(126)*median/(1.4826*MAD) over lr[t-126..t-1]; >=100 values; std fallback."""
    from numpy.lib.stride_tricks import sliding_window_view
    v = lr.to_numpy(float)
    out = np.full(v.shape, np.nan)
    if len(v) < 126:
        return pd.DataFrame(out, index=lr.index, columns=lr.columns)
    for c0 in range(0, v.shape[1], 40):                    # column batches keep memory small
        win = sliding_window_view(v[:, c0:c0 + 40], 126, axis=0)   # (n-125, m, 126)
        cnt = np.isfinite(win).sum(axis=2)
        with np.errstate(all="ignore"):
            med = np.nanmedian(win, axis=2)
            mad = np.nanmedian(np.abs(win - med[..., None]), axis=2)
            scale = 1.4826 * mad
            sd = np.nanstd(win, axis=2, ddof=1)
            scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, sd)
            val = np.sqrt(126) * med / scale
        val[(cnt < 100) | ~np.isfinite(val)] = np.nan
        out[125:, c0:c0 + 40] = val
    return pd.DataFrame(out, index=lr.index, columns=lr.columns)


def _is_block_end(T: pd.Timestamp, idx: pd.DatetimeIndex) -> bool:
    """True if T is the last BIST session of its 2-week (W-FRI ordinal // 2) block."""
    import holidays
    code = lambda d: pd.Period(d, "W-FRI").ordinal // 2
    nxt = _next(idx, T)
    if nxt is None:                                        # live edge: next session by calendar
        tr = holidays.Turkey(years=[T.year, T.year + 1])
        nxt = T + pd.Timedelta(days=1)
        while nxt.weekday() >= 5 or nxt in tr:
            nxt += pd.Timedelta(days=1)
    return code(T) != code(nxt)


def fund_liquidity_80(D, M, W, live_start=None, top_n=75):
    A, C, V = D["Adj Close"], D["Close"], D["Volume"].fillna(0.0)
    obs = A.where(A > 0)
    close = obs.ffill()
    idx = close.index
    r = close.pct_change(fill_method=None)
    rl, Pl = r.shift(1), close.shift(1)
    hist = obs.notna().astype(float).rolling(252, min_periods=252).sum().shift(1)
    adv60 = (C * V).rolling(60, min_periods=40).median().shift(1)        # TRY traded value
    elig = hist.ge(252) & obs.notna() & adv60.gt(0)

    std = rl.rolling(126, min_periods=80).std(ddof=1).replace(0, np.nan)
    ram = ((Pl / Pl.shift(126) - 1) / std).where(elig)
    high52 = Pl / Pl.rolling(252, min_periods=200).max()
    robust = _robust126(np.log(close).diff().shift(1))
    mkt = rl.where(elig).mean(axis=1)
    mvar = mkt.rolling(126, min_periods=80).var().replace(0, np.nan)
    beta = pd.DataFrame({c: rl[c].rolling(126, min_periods=80).cov(mkt) for c in rl}) \
        .div(mvar, axis=0)
    cm = elig & _pct(beta.where(elig)).le(0.50)
    tclb = 0.4 * _pct(high52.where(cm)) + 0.6 * _pct(robust.where(cm))
    r_ram, r_tclb = _pct(ram.where(elig)), _pct(tclb.where(elig))
    score = 0.25 * r_ram.fillna(0) + 0.75 * r_tclb.fillna(0)
    valid = elig & (r_ram.notna() | r_tclb.notna())
    order = score.where(valid).rank(axis=1, ascending=False, method="first")
    selected = order.le(top_n).astype(float)
    liq = _pct(adv60.where(elig)).fillna(0.0)
    raw = selected * liq / top_n
    S = raw.sum(axis=1)
    scale = np.where(S < 0.85, 0.85, np.minimum(S, 1.0)) / S.replace(0, np.nan)
    target = raw.mul(scale, axis=0).fillna(0.0)

    start = pd.Timestamp(live_start) if live_start else W[0]
    days = [d for d in idx if d >= start]
    rets, trades = {}, []
    R = close.pct_change(fill_method=None).fillna(0.0)
    w_prev, cash_prev = None, 0.0
    for i, T in enumerate(days):
        if w_prev is None:
            pre = target.loc[T] * 0.0
            scheduled = True                                # initial trade (L227)
        else:
            g = 1 + float((w_prev * R.loc[T]).sum()) + (1 - w_prev.sum()) * cash_prev
            rets[T] = g - 1 - turnover_prev * 5 / 10_000      # realised return close(t-1)->t
            pre = w_prev * (1 + R.loc[T]) / g
            scheduled = _is_block_end(T, idx)
        tgt = target.loc[T]
        if scheduled or (pre.sum() < 0.80 and tgt.sum() >= 0.80):
            w = tgt
            for tk, dw in (w - pre).items():
                if abs(dw) > 1e-9:
                    trades.append({"date": str(T.date()), "ticker": tk,
                                   "side": "BUY" if dw > 0 else "SELL",
                                   "weight": _f(w[tk]), "change": _f(dw),
                                   "entry_time": "closing auction",
                                   "entry_price": _f(C.at[T, tk], 2),
                                   "exit_time": None, "exit_price": None, "ret": None})
        else:
            w = pre
        turnover_prev = float((w - pre).abs().sum())
        nxt = _next(idx, T)
        gap_days = (nxt - T).days if nxt is not None else 1
        cash_prev = CASH_RATE.get(T.year, 0.40) * gap_days / 365
        w_prev = w
    res = Result("fund_liquidity_80", "Fund Liquidity 80 — FL26131 (long-only)",
                 pd.Series(rets, dtype=float), trades,
                 "All 582 eligible BIST names. Score = 25% risk-adjusted 126d momentum + 75% "
                 "trend (52-week-high proximity + robust 126d trend) within the low-beta half. "
                 "Buy the top 75, weight by liquidity percentile, 85% invested (rest in TRY cash "
                 "at 40%/yr), rebalance on the last session of each 2-week block; 5 bp x turnover.",
                 ["Traded value = Yahoo close x volume (original: exchange TRY traded value).",
                  "Prices: Yahoo adjusted closes (original vendor adjustments may differ for "
                  "bonus/rights issues).",
                  "Market-cap > 0 gate treated as always true (no Yahoo history)."])
    res.holdings = {k: round(float(v), 4) for k, v in w_prev.items() if v > 1e-6} if w_prev is not None else {}
    return res


# =====================================================================
# ENS5v2 (basic and S10c) — computed by ens5/run_ens5.py with the friend's
# own pipeline; here we only turn its outputs into daily returns + trades.
# =====================================================================
ENS5_OUT = Path(__file__).resolve().parent / "ens5" / "out"
ENS5_DEVIATIONS = [
    "Inputs from Yahoo + Is Yatirim only: USD/TRY from Yahoo (original: central-bank EVDS), "
    "risk-free rate 0 (original: EVDS policy rate).",
    "Sessions before 2026-10-13 were built from Yahoo 5-minute bars; later sessions from 1-minute bars.",
]


def _ens5(D, variant: str):
    daily = pd.read_csv(ENS5_OUT / "ens5_daily.csv", index_col=0, parse_dates=True)
    picks = pd.read_csv(ENS5_OUT / "ens5_picks.csv", parse_dates=["date"])
    pend_fp = ENS5_OUT / "ens5_pending.csv"
    pend = pd.read_csv(pend_fp, parse_dates=["date"]) if pend_fp.exists() else pd.DataFrame()
    C, O = D["Close"], D["Open"]
    idx = C.index
    col = "TOP3" if variant == "basic" else "S10c"
    px = lambda F, d, t: _f(F.at[d, t], 2) if (t in F.columns and d in F.index) else None
    rets, trades = {}, []
    for T, row in daily.iterrows():
        T1 = _next(idx, T)
        if T1 is None or not np.isfinite(row[col]):
            continue
        rets[T1] = float(row[col])
        g = picks[picks.date == T]
        longs = g[g.leg == "LONG"].sort_values("signal", ascending=False)    # build_pick_panel order
        shorts = g[g.leg == "SHORT"].sort_values("signal", ascending=False)
        a3 = variant == "s10c" and bool(row["a3_long_top1_only"])
        b4 = variant == "s10c" and bool(row["b4_extend_long"])
        if a3:
            longs = longs.head(1)
        for r in longs.itertuples():
            ret = (1 + r.r_on) * (1 + r.r_id) - 1 if b4 else r.r_on
            note = "A3: whole long leg in top pick" if a3 else ""
            note += ("; " if note and b4 else "") + ("B4: held through next day" if b4 else "")
            trades.append({"date": str(T.date()), "ticker": r.ticker, "side": "LONG",
                           "entry_time": "closing auction", "entry_price": px(C, T, r.ticker),
                           "exit_time": f"{'closing' if b4 else 'opening'} auction {T1.date()}",
                           "exit_price": px(C if b4 else O, T1, r.ticker), "ret": _f(ret, 5), "note": note})
        if not bool(row["gap_skip_short"]):
            half = variant == "s10c" and bool(row["short_halved"])
            for r in shorts.itertuples():
                trades.append({"date": str(T1.date()), "ticker": r.ticker, "side": "SHORT",
                               "entry_time": "opening auction", "entry_price": px(O, T1, r.ticker),
                               "exit_time": "closing auction", "exit_price": px(C, T1, r.ticker),
                               "ret": _f(-r.r_id, 5), "note": "short leg halved (momentum streak)" if half else ""})
    if len(pend):
        T = pend.date.max()
        for r in pend[(pend.date == T) & (pend.leg == "LONG")].itertuples():
            trades.append({"date": str(T.date()), "ticker": r.ticker, "side": "LONG",
                           "entry_time": "closing auction", "entry_price": _f(r.p_auct, 2),
                           "exit_time": "next opening auction", "exit_price": None, "ret": None,
                           "note": "pending" + (" (S10c sizing set at execution)" if variant == "s10c" else "")})
    name = "ENS5v2 — 17:45-18:00 reversal, " + ("basic (top3 + gap skip)" if variant == "basic" else "S10c rules")
    spec = ("Each day at 18:00, five factor models explain every stock's 17:45-18:00 return; the "
            "unexplained part (residual) is ranked and averaged. Long the 3 biggest unusual drops in the "
            "closing auction (sell at the next opening auction); short the 3 biggest unusual rises at the next "
            "opening auction (cover at its closing auction). Short skipped when the market's overnight gap "
            "is unusually large. 5 bp per side.")
    if variant == "s10c":
        spec += (" S10c adds: A3 - whole long leg in the top pick when the 3 long signals are unusually close; "
                 "B4 - hold longs through the next day when their signals are unusually strong; "
                 "halve the short after a 5-day long momentum streak.")
    return Result(f"ens5_{variant}", name, pd.Series(rets, dtype=float), trades, spec, ENS5_DEVIATIONS)


def ens5_basic(D, M, W):
    return _ens5(D, "basic")


def ens5_s10c(D, M, W):
    return _ens5(D, "s10c")


ALL = [fund_liquidity_80, focus_long, daily_beta_pm, eodr_short, morning_gap_short, ens5_basic, ens5_s10c]
