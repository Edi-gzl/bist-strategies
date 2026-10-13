"""Detailed L/S report engine — gross exposure = 1.0 (0.5L / 0.5S).

Book dict keys:
  name, gross, turn, hl (optional)
  leg_long, leg_short — daily leg P&L on the 1.0 book (sum = gross)
  mkt — optional market return series aligned to gross index (for beta)
"""
from __future__ import annotations

import io, base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IS1 = pd.Timestamp("2025-01-01")
BG, FG, ACC, BLU, MUT, RED, ORG = "#0e1116", "#c9d1d9", "#3fd99a", "#6cb6ff", "#7d8794", "#e0795a", "#e8c468"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUT,
    "ytick.color": MUT, "axes.edgecolor": "#2b333c", "grid.color": "#222932", "axes.titleweight": "bold",
})
COST_LEVELS = [("gross", 0.0), ("5 bps/side", 5.0), ("10 bps/side", 10.0)]
DEFAULT_NET_BPS = 5.0


def book_turnover(weights: dict, prev: dict | None) -> float:
    if not prev:
        return sum(abs(v) for v in weights.values()) / 2.0
    keys = set(weights) | set(prev)
    return sum(abs(weights.get(x, 0.0) - prev.get(x, 0.0)) for x in keys) / 2.0


def ls_gross1(rl: float, rs: float) -> float:
    return 0.5 * rl - 0.5 * rs


def sh(x) -> float:
    x = np.asarray(pd.Series(x).dropna(), float)
    return x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 20 and x.std(ddof=1) > 0 else np.nan


BPS_LEVELS = [0, 5, 10]


def cnet(gross: pd.Series, turn: pd.Series, side: float) -> pd.Series:
    return gross - turn.reindex(gross.index).fillna(0) * 2 * side * 1e-4


def sh_at_costs(gross: pd.Series, turn: pd.Series) -> dict[str, str]:
    """Sharpe at 0 / 5 / 10 bps per side — always all three."""
    t = turn.reindex(gross.index).fillna(0)
    out = {}
    for bps in BPS_LEVELS:
        net = gross if bps == 0 else cnet(gross, t, bps)
        out[f"sh_{bps}bps"] = f"{sh(net):+.2f}"
    return out


def _mask(idx, period: str):
    if period == "IS":
        return idx <= IS1
    if period == "OOS":
        return idx > IS1
    return pd.Series(True, index=idx)


def stats_block(r: pd.Series, period: str = "FULL") -> dict:
    s = r.loc[_mask(r.index, period)].dropna()
    if len(s) < 20:
        return dict(n=len(s), ann=np.nan, vol=np.nan, mdd=np.nan, sharpe=np.nan, hit=np.nan, calmar=np.nan)
    eq = (1 + s).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    ann = float(s.mean() * 252)
    return dict(
        n=len(s),
        ann=ann,
        vol=float(s.std(ddof=1) * np.sqrt(252)),
        mdd=mdd,
        sharpe=sh(s),
        hit=float((s > 0).mean()),
        calmar=ann / abs(mdd) if mdd < 0 else np.nan,
    )


def nw_t(x, lags: int = 5) -> float:
    """Newey-West t-stat for the mean of a return series."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 30:
        return np.nan
    m = x.mean()
    e = x - m
    s = (e * e).mean()
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] * e[:-lag]).mean()
    return m / np.sqrt(s / n) if s > 0 else np.nan


def autocorr_lag1(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.autocorr(lag=1)) if len(x) > 30 else np.nan


def rolling_beta(r: pd.Series, mkt: pd.Series, win: int = 63) -> pd.Series:
    df = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    cov = df["r"].rolling(win).cov(df["m"])
    var = df["m"].rolling(win).var()
    return (cov / var).rename("beta")


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5)


def _cell_pair(full: float, oos: float, pct: bool = False) -> str:
    if pct:
        return (f"<div class='cell'><span class='full'>{full:+.0%}</span>"
                f"<span class='oos'>OOS {oos:+.0%}</span></div>")
    return (f"<div class='cell'><span class='full'>{full:+.2f}</span>"
            f"<span class='oos'>OOS {oos:+.2f}</span></div>")


def comparison_row(label: str, gross: pd.Series, turn: pd.Series, mkt: pd.Series | None,
                   hl: bool = False, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(gross, turn, net_bps).dropna()
    g_is, g_oos = stats_block(gross, "IS"), stats_block(gross, "OOS")
    n_is, n_oos = stats_block(net, "IS"), stats_block(net, "OOS")
    b_is = b_oos = np.nan
    if mkt is not None:
        b = rolling_beta(gross, mkt).dropna()
        if len(b):
            b_is = float(b.loc[_mask(b.index, "IS")].mean())
            b_oos = float(b.loc[_mask(b.index, "OOS")].mean())
    cls = " class=hl" if hl else ""
    turn_avg = turn.reindex(net.index).mean()
    return (
        f"<tr{cls}><td>{label}</td>"
        f"<td class='per'>{_cell_pair(g_is['sharpe'], g_oos['sharpe'])}</td>"
        f"<td class='per'>{_cell_pair(n_is['sharpe'], n_oos['sharpe'])}</td>"
        f"<td class='per'>{_cell_pair(n_is['ann'], n_oos['ann'], pct=True)}</td>"
        f"<td class='per'>{_cell_pair(n_is['vol'], n_oos['vol'], pct=True)}</td>"
        f"<td class='per'>{_cell_pair(n_is['mdd'], n_oos['mdd'], pct=True)}</td>"
        f"<td class='per'>{_cell_pair(n_is['hit'], n_oos['hit'], pct=True)}</td>"
        f"<td>{b_is:+.2f}</td><td>{b_oos:+.2f}</td>"
        f"<td>{turn_avg * 100:.0f}%</td></tr>"
    )


def leg_detail_table(book: dict, net_bps: float = DEFAULT_NET_BPS) -> str:
    """Long / short / combined leg stats for one book (net 5bps on combined only)."""
    rows = []
    net = cnet(book["gross"], book["turn"], net_bps).dropna()
    legs = [
        ("Long leg (0.5 book)", book.get("leg_long")),
        ("Short leg (0.5 book)", book.get("leg_short")),
        ("L/S combined (gross)", book["gross"]),
        ("L/S combined (net 5bps)", net),
    ]
    for nm, s in legs:
        if s is None:
            continue
        s = s.reindex(book["gross"].index).dropna()
        si, so = stats_block(s, "IS"), stats_block(s, "OOS")
        rows.append(
            f"<tr><td>{nm}</td>"
            f"<td class='per'>{si['sharpe']:+.2f}</td><td>{so['sharpe']:+.2f}</td>"
            f"<td class='per'>{si['ann']:+.0%}</td><td>{so['ann']:+.0%}</td>"
            f"<td class='per'>{si['vol']:.0%}</td><td>{so['vol']:.0%}</td>"
            f"<td>{si['mdd']:.0%}</td><td>{so['mdd']:.0%}</td>"
            f"<td>{si['hit']:.0%}</td><td>{so['hit']:.0%}</td></tr>"
        )
    return "".join(rows)


def cost_row(g, turn, nm, hl=False):
    cells = ""
    for _, side in COST_LEVELS:
        x = cnet(g, turn, side).dropna()
        cells += f"<td class='per'>{_cell_pair(sh(x), sh(x[x.index > IS1]))}</td>"
    cls = " class=hl" if hl else ""
    return f"<tr{cls}><td>{nm}</td>{cells}<td>{(1 + g).cumprod().iloc[-1]:.1f}×</td></tr>"


def cost_grid_table(books: list, bps_levels: tuple = (0, 5, 10)) -> str:
    """Sharpe grid: rows = books, cols = bps/side (cells FULL · OOS)."""
    hdr = "<tr><th>book · gross exp = 1.0</th>" + "".join(
        f"<th class='per'>{b} bps/side</th>" for b in bps_levels) + "</tr>"
    rows = []
    for bk in books:
        if bk.get("skip_cost_grid"):
            continue
        cells = ""
        for bps in bps_levels:
            net = cnet(bk["gross"], bk["turn"], bps).dropna()
            cells += f"<td class='per'>{_cell_pair(sh(net), sh(net[net.index > IS1]))}</td>"
        cls = " class=hl" if bk.get("hl") else ""
        rows.append(f"<tr{cls}><td>{bk['name']}</td>{cells}</tr>")
    return hdr + "".join(rows)


def cost_grid_ann_table(books: list, bps_levels: tuple = (0, 5, 10)) -> str:
    hdr = "<tr><th>book</th>" + "".join(
        f"<th class='per'>{b} bps/side</th>" for b in bps_levels) + "</tr>"
    rows = []
    for bk in books:
        if bk.get("skip_cost_grid"):
            continue
        cells = ""
        for bps in bps_levels:
            net = cnet(bk["gross"], bk["turn"], bps).dropna()
            si, so = stats_block(net, "IS"), stats_block(net, "OOS")
            cells += (f"<td class='per'><div class='cell'><span class='full'>{si['ann']:+.0%}</span>"
                      f"<span class='oos'>OOS {so['ann']:+.0%}</span></div></td>")
        cls = " class=hl" if bk.get("hl") else ""
        rows.append(f"<tr{cls}><td>{bk['name']}</td>{cells}</tr>")
    return hdr + "".join(rows)


def reference_row(label: str, r: pd.Series, hl: bool = False) -> str:
    """Pre-costed or external series — show net metrics only (no gross/turn model)."""
    r = r.dropna()
    si, so, sf = stats_block(r, "IS"), stats_block(r, "OOS"), stats_block(r, "FULL")
    cls = " class=hl" if hl else ""
    return (
        f"<tr{cls}><td>{label}</td>"
        f"<td class='per'>—</td>"
        f"<td class='per'>{_cell_pair(sf['sharpe'], so['sharpe'])}</td>"
        f"<td class='per'>{_cell_pair(si['ann'], so['ann'], pct=True)}</td>"
        f"<td class='per'>{_cell_pair(si['vol'], so['vol'], pct=True)}</td>"
        f"<td class='per'>{_cell_pair(si['mdd'], so['mdd'], pct=True)}</td>"
        f"<td class='per'>{_cell_pair(si['hit'], so['hit'], pct=True)}</td>"
        f"<td>—</td><td>—</td><td>—</td></tr>"
    )


def kpi_cards_html(book: dict, bench: pd.Series, bench_nm: str, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(book["gross"], book["turn"], net_bps).dropna()
    b = bench.reindex(net.index).dropna()
    idx = net.index.intersection(b.index)
    net, b = net.loc[idx], b.loc[idx]
    sf, bf = stats_block(net), stats_block(b)
    beta = float(net.cov(b) / b.var()) if len(b) > 20 and b.var() > 0 else np.nan
    rho = float(net.corr(b)) if len(net) > 20 else np.nan
    cards = [
        ("Sharpe (net)", f"{sf['sharpe']:+.2f}", f"{bf['sharpe']:+.2f}"),
        ("annual return", f"{sf['ann']:+.0%}", f"{bf['ann']:+.0%}"),
        ("volatility", f"{sf['vol']:.0%}", f"{bf['vol']:.0%}"),
        ("max drawdown", f"{sf['mdd']:.0%}", f"{bf['mdd']:.0%}"),
        ("hit rate", f"{sf['hit']:.0%}", f"{bf['hit']:.0%}"),
        ("Calmar", f"{sf['calmar']:.1f}", f"{bf['calmar']:.1f}"),
        ("β vs market", f"{beta:+.2f}", "+1.00"),
        ("corr ρ", f"{rho:+.2f}", "+1.00"),
    ]
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div>"
        f"<div class='b'>{bench_nm} {b}</div></div>"
        for k, v, b in cards
    ) + "</div>"


def robustness_table(book: dict, bench: pd.Series, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(book["gross"], book["turn"], net_bps).dropna()
    rows = []
    for nm, s in ((book["name"], net), (f"EW benchmark", bench.reindex(net.index).dropna())):
        rows.append(
            f"<tr><td>{nm}</td>"
            f"<td>{nw_t(s[s.index <= IS1]):+.2f}</td>"
            f"<td>{nw_t(s[s.index > IS1]):+.2f}</td>"
            f"<td>{nw_t(s):+.2f}</td>"
            f"<td>{autocorr_lag1(s):+.3f}</td></tr>"
        )
    return "".join(rows)


def plot_eq_is_oos_split(book: dict, bench: pd.Series, bench_nm: str, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(book["gross"], book["turn"], net_bps).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.2))
    for ax, msk, ttl in (
        (axes[0], net.index <= IS1, f"IS only (through {IS1.date()})"),
        (axes[1], net.index > IS1, "OOS only"),
    ):
        s, b = net.loc[msk], bench.reindex(net.index).loc[msk]
        ax.plot((1 + s).cumprod(), color=ACC, lw=1.5, label=book["name"])
        ax.plot((1 + b).cumprod(), color=MUT, lw=1.2, label=bench_nm)
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=8)
        clean(ax)
    return b64(fig)


def plot_yearly_vs_bench(book: dict, bench: pd.Series, bench_nm: str, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(book["gross"], book["turn"], net_bps).dropna()
    b = bench.reindex(net.index).dropna()
    years = sorted(set(net.index.year))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.0))
    x = np.arange(len(years))
    w = 0.38
    for ax, fn, ttl, fmt in (
        (axes[0], sh, "Sharpe by year (net)", "{:+.1f}"),
        (axes[1], lambda s: s.mean() * 252 * 100, "annual return by year (net, %)", "{:+.0f}"),
    ):
        a = [fn(net[net.index.year == y]) for y in years]
        bv = [fn(b[b.index.year == y]) for y in years]
        ax.bar(x - w / 2, a, w, color=ACC, label=book["name"])
        ax.bar(x + w / 2, bv, w, color=MUT, label=bench_nm)
        for xi, v in zip(x - w / 2, a):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        for xi, v in zip(x + w / 2, bv):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=MUT)
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_title(ttl)
        ax.axhline(0, color="#3a424c", lw=0.6)
        ax.legend(frameon=False, fontsize=7.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_yticks([])
    return b64(fig)


def plot_ret_dist_by_year(book: dict, bench: pd.Series, bench_nm: str, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(book["gross"], book["turn"], net_bps).dropna()
    b = bench.reindex(net.index).dropna()
    years = sorted(set(net.index.year))
    fig, axes = plt.subplots(1, len(years), figsize=(10.8, 2.6), sharey=True, sharex=True)
    if len(years) == 1:
        axes = [axes]
    bins = np.linspace(-0.04, 0.04, 41)
    for ax, y in zip(axes, years):
        ax.hist(b[b.index.year == y], bins=bins, color=MUT, alpha=0.55, density=True, label=bench_nm)
        ax.hist(net[net.index.year == y], bins=bins, color=ACC, alpha=0.65, density=True, label=book["name"])
        ax.set_title(str(y))
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("daily return distribution by year (net)", fontsize=10, fontweight="bold", y=1.04)
    return b64(fig)


def plot_gross_equity(hero: dict, bench: pd.Series | None, bench_nm: str) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    g = hero["gross"].dropna()
    ax.plot((1 + g).cumprod(), color=ACC, lw=2.0, label=f"{hero['name']} (gross)")
    if bench is not None:
        b = bench.reindex(g.index).dropna()
        ax.plot((1 + b).cumprod(), color=MUT, lw=1.2, ls="--", label=bench_nm)
    ax.axvline(IS1, color="#2b333c", lw=1)
    ax.set_yscale("log")
    ax.set_title(f"growth of $1 — gross returns (no costs) · {hero['name']} vs benchmark")
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    return b64(fig)


def plot_net_equity(books: list, bench: pd.Series | None, bench_nm: str, net_bps: float = DEFAULT_NET_BPS) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    cols = [ACC, BLU, "#c08adf", "#e8c468", "#4a8fd4", "#2f6f57"]
    for i, bk in enumerate(books):
        n = cnet(bk["gross"], bk["turn"], net_bps).dropna()
        ax.plot((1 + n).cumprod(), color=cols[i % len(cols)], lw=1.6 if bk.get("hl") else 1.1,
                label=bk["name"])
    if bench is not None:
        ax.plot((1 + bench).cumprod(), color=MUT, lw=1.1, ls="--", label=bench_nm)
    ax.axvline(IS1, color="#2b333c", lw=1)
    ax.set_yscale("log")
    ax.set_title(f"growth of $1 — net of {net_bps:.0f} bps/side")
    ax.legend(frameon=False, fontsize=7.5)
    clean(ax)
    return b64(fig)


def plot_leg_equity(book: dict) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 3.2))
    ll, ls = book.get("leg_long"), book.get("leg_short")
    if ll is not None:
        ax.plot((1 + ll.dropna()).cumprod(), color=ACC, lw=1.5, label="long leg")
    if ls is not None:
        ax.plot((1 + ls.dropna()).cumprod(), color=RED, lw=1.5, label="short leg")
    ax.plot((1 + book["gross"].dropna()).cumprod(), color=BLU, lw=1.8, ls="-", label="combined gross")
    ax.axvline(IS1, color="#2b333c", lw=1)
    ax.set_yscale("log")
    ax.set_title(f"leg attribution — {book['name']} (gross, gross exposure = 1.0)")
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    return b64(fig)


def plot_beta_exposure(books: list, mkt: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 2.8))
    cols = [ACC, BLU, ORG, "#c08adf"]
    for i, bk in enumerate(books):
        if mkt is None:
            continue
        b = rolling_beta(bk["gross"], mkt, 63)
        ax.plot(b, color=cols[i % len(cols)], lw=1.2, label=bk["name"])
    ax.axhline(0, color="#3a424c", lw=0.7)
    ax.axvline(IS1, color="#2b333c", lw=1)
    ax.set_title("rolling 63-day market beta exposure vs EW BIST50")
    ax.legend(frameon=False, fontsize=7.5)
    clean(ax)
    return b64(fig)


def plot_is_oos_bars(hero: dict, bench: pd.Series | None, net_bps: float = DEFAULT_NET_BPS) -> str:
    net = cnet(hero["gross"], hero["turn"], net_bps).dropna()
    metrics = ["Sharpe", "Ann return", "Vol", "Max DD"]
    h_is = [stats_block(net, "IS")["sharpe"], stats_block(net, "IS")["ann"] * 100,
            stats_block(net, "IS")["vol"] * 100, stats_block(net, "IS")["mdd"] * 100]
    h_oos = [stats_block(net, "OOS")["sharpe"], stats_block(net, "OOS")["ann"] * 100,
             stats_block(net, "OOS")["vol"] * 100, stats_block(net, "OOS")["mdd"] * 100]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.8))
    x = np.arange(2)
    for ax, m, vi, vo in zip(axes, metrics, h_is, h_oos):
        ax.bar(x, [vi, vo], color=[BLU, ACC], width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(["IS", "OOS"], fontsize=8)
        ax.set_title(m, fontsize=9)
        ax.axhline(0, color="#3a424c", lw=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"IS vs OOS — {hero['name']} (net {net_bps:.0f} bps/side)", fontsize=10, y=1.02)
    plt.tight_layout()
    return b64(fig)


def plot_dd_roll(book: dict, net_bps: float = DEFAULT_NET_BPS) -> str:
    n = cnet(book["gross"], book["turn"], net_bps).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 2.8))
    eq = (1 + n).cumprod()
    dd = eq / eq.cummax() - 1
    axes[0].fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.5)
    axes[0].plot(dd * 100, color=RED, lw=1)
    axes[0].set_title(f"drawdown (max {dd.min() * 100:.0f}%)")
    clean(axes[0])
    rs = n.rolling(63).mean() / n.rolling(63).std() * np.sqrt(252)
    axes[1].plot(rs, color=ACC, lw=1.3)
    axes[1].axhline(0, color="#3a424c", lw=0.7)
    axes[1].axvline(IS1, color="#2b333c", lw=1)
    axes[1].fill_between(rs.index, 0, rs, where=(rs > 0), color=ACC, alpha=0.15)
    axes[1].set_title("rolling 63d Sharpe")
    clean(axes[1])
    return b64(fig)


def plot_yearly(book: dict, net_bps: float = DEFAULT_NET_BPS) -> str:
    n = cnet(book["gross"], book["turn"], net_bps).dropna()
    yrs = sorted(set(n.index.year))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 2.9))
    sv = [sh(n[n.index.year == y]) for y in yrs]
    av = [n[n.index.year == y].mean() * 252 * 100 for y in yrs]
    axes[0].bar(range(len(yrs)), sv, color=ACC, width=0.6)
    axes[1].bar(range(len(yrs)), av, color=BLU, width=0.6)
    axes[0].set_title("Sharpe by year (net)")
    axes[1].set_title("annual return by year (net)")
    for ax, vv, fmt in ((axes[0], sv, "{:+.1f}"), (axes[1], av, "{:+.0f}%")):
        for i, v in enumerate(vv):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_xticks(range(len(yrs)))
        ax.set_xticklabels(yrs)
        ax.axhline(0, color="#3a424c", lw=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
    return b64(fig)


def plot_turnover(books: list) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 2.6))
    cols = [ACC, BLU, "#c08adf", "#e8c468"]
    for i, bk in enumerate(books):
        ax.plot(bk["turn"].rolling(21).mean() * 100, color=cols[i % len(cols)], lw=1.2, label=bk["name"])
    ax.set_title("turnover — 21d avg (% of book replaced / day)")
    ax.legend(frameon=False, fontsize=7.5)
    clean(ax)
    return b64(fig)


CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;margin:40px auto;max-width:1140px;color:#c9d1d9;background:#0e1116;padding:0 22px}
h1{font-size:22px;margin-bottom:2px}.sub{color:#7d8794;font-size:13px;margin-bottom:18px;line-height:1.55}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#7d8794;margin:34px 0 8px;border-bottom:1px solid #1f262e;padding-bottom:6px}
h3{font-size:11px;color:#8a93a0;margin:18px 0 6px;text-transform:uppercase;letter-spacing:.08em}
table{border-collapse:collapse;font-size:12px;width:100%;margin:12px 0;background:#0c1014;border:1px solid #1f262e;border-radius:10px;overflow:hidden}
td,th{padding:9px 12px;border-bottom:1px solid #1a212a;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
thead th,tr:first-child th{background:#12171e;border-bottom:1px solid #2b333c}
th{color:#8a93a0;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.04em}
th.per,td.per{border-left:1px solid #1a212a;text-align:center}
td:first-child,th:first-child{text-align:left;font-weight:500;color:#c9d1d9}
tr:last-child td{border-bottom:none}table tr:hover td{background:#141b23}
tr.hl td{color:#3fd99a;font-weight:600;background:#10231b}
.cell{display:flex;flex-direction:column;align-items:flex-end;gap:2px;line-height:1.12}
.cell .full{font-size:13px;font-weight:600}.cell .oos{font-size:9.5px;color:#7d8794}
img{max-width:100%;margin:8px 0 14px;border-radius:6px}
.note{background:#161b22;border-left:3px solid #3fd99a;padding:10px 14px;margin:12px 0;font-size:12.5px;line-height:1.55}
.lgnd{color:#7d8794;font-size:11px;margin:4px 0 10px;line-height:1.45}
.section{margin-top:28px;padding-top:8px;border-top:1px solid #1a212a}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 20px}
.card{flex:1;min-width:118px;padding:11px 13px;background:#12171e;border:1px solid #1f262e;border-radius:8px}
.card .k{font-size:10px;color:#7d8794;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:20px;font-weight:600;color:#3fd99a;margin:2px 0}
.card .b{font-size:10px;color:#7d8794}
"""

GROSS_NOTE = ("<b>Gross exposure = 1.0 always</b> on every L/S book: 0.5 long + 0.5 short "
              "(Σ|w| = 1.0). Benchmark EW BIST50 is long-only gross 1.0.")


def render_enhanced_report(
    title: str,
    sub: str,
    books: list,
    bench: pd.Series,
    bench_nm: str,
    out_path: str,
    *,
    hero_name: str | None = None,
    leg_hero_name: str | None = None,
    hero_leg_detail: bool = True,
    beta_books: list | None = None,
    extra_note: str = "",
    net_bps: float = DEFAULT_NET_BPS,
    cost_grid_books: list | None = None,
    reference_rows: list | None = None,
    cost_grid_bps: tuple = (0, 5, 10),
    gross_note: str | None = None,
    stock_analytics: dict | None = None,
    phase_analytics: dict | None = None,
):
    hero = next((b for b in books if b.get("hl")), None)
    if hero_name:
        hero = next((b for b in books if b["name"] == hero_name), hero)
    if hero is None:
        hero = books[0]

    leg_hero = hero
    if leg_hero_name:
        leg_hero = next((b for b in books if b["name"] == leg_hero_name), hero)

    mkt = bench
    for b in books:
        if "mkt" in b:
            mkt = b["mkt"]
            break

    cmp_rows = "".join(
        comparison_row(b["name"], b["gross"], b["turn"], mkt, b.get("hl"), net_bps) for b in books
    )
    if reference_rows:
        for label, series, hl in reference_rows:
            cmp_rows += reference_row(label, series, hl)
    z_turn = pd.Series(0.0, index=bench.index)
    cmp_rows += comparison_row(bench_nm, bench, z_turn, bench, False, net_bps)

    cost = "".join(cost_row(b["gross"], b["turn"], b["name"], b.get("hl")) for b in books
                   if not b.get("skip_cost_grid"))
    grid_books = cost_grid_books if cost_grid_books is not None else [b for b in books if not b.get("skip_cost_grid")]
    cost_grid = cost_grid_table(grid_books, cost_grid_bps)
    cost_grid_ann = cost_grid_ann_table(grid_books, cost_grid_bps)
    beta_list = beta_books if beta_books is not None else [b for b in books if b.get("hl") or b["name"] == hero["name"]][:3]

    leg_html = ""
    if hero_leg_detail and leg_hero.get("leg_long") is not None:
        leg_html = f"""
<h2>__SEC_LEG__ · leg attribution — {leg_hero['name']}</h2>
<p class='lgnd'>Long leg = 0.5 × mean(long basket return). Short leg = −0.5 × mean(short basket return). Combined gross = long + short.</p>
<table>
<tr><th>component</th><th class='per'>IS Sh</th><th>OOS Sh</th><th class='per'>IS ann</th><th>OOS ann</th>
<th class='per'>IS vol</th><th>OOS vol</th><th>IS MDD</th><th>OOS MDD</th><th>IS hit</th><th>OOS hit</th></tr>
{leg_detail_table(leg_hero, net_bps)}
</table>
<img src='data:image/png;base64,{plot_leg_equity(leg_hero)}'>
"""

    sec = 1
    gnote = gross_note if gross_note is not None else GROSS_NOTE
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1><div class='sub'>{sub}</div>
{kpi_cards_html(hero, bench, bench_nm, net_bps)}
{extra_note}
<h2>{sec} · strategy vs EW BIST50 — gross &amp; net (cells: FULL · OOS)</h2>
<p class='lgnd'>{gnote}<br>
<b>IS</b> = through {IS1.date()} · <b>OOS</b> after · net = gross − turnover × 2 × bps/side ·
β = rolling 63d vs EW.</p>
<table>
<tr><th>book</th><th class='per'>gross Sh</th><th class='per'>net Sh</th><th class='per'>net ann</th>
<th class='per'>net vol</th><th class='per'>net MDD</th><th class='per'>net hit</th><th>β IS</th><th>β OOS</th><th>turn/day</th></tr>
{cmp_rows}
</table>
"""
    sec += 1
    html += f"""
<h2>{sec} · cost sensitivity by basket — Sharpe (FULL · OOS)</h2>
<p class='lgnd'>Rows = top-N baskets · columns = per-side bps charged on entry <b>and</b> exit (turnover-weighted).
All books gross exposure 1.0.</p>
<table>{cost_grid}</table>
"""
    sec += 1
    html += f"""
<h2>{sec} · cost sensitivity by basket — net annual return (IS · OOS)</h2>
<table>{cost_grid_ann}</table>
"""
    sec += 1
    html += f"""
<h2>{sec} · growth of $1 — gross (hero) vs benchmark</h2>
<img src='data:image/png;base64,{plot_gross_equity(hero, bench, bench_nm)}'>
<img src='data:image/png;base64,{plot_eq_is_oos_split(hero, bench, bench_nm, net_bps)}'>
"""
    sec += 1
    html += f"""
<h2>{sec} · growth of $1 — net ({net_bps:.0f} bps/side) · all books</h2>
<img src='data:image/png;base64,{plot_net_equity(books, bench, bench_nm, net_bps)}'>
"""
    if leg_html:
        sec += 1
        html += leg_html.replace("__SEC_LEG__", str(sec))
    sec += 1
    html += f"""
<h2>{sec} · IS vs OOS snapshot — {hero['name']}</h2>
<img src='data:image/png;base64,{plot_is_oos_bars(hero, bench, net_bps)}'>
"""
    sec += 1
    html += f"""
<h2>{sec} · market beta exposure</h2>
<img src='data:image/png;base64,{plot_beta_exposure(beta_list, mkt)}'>
"""
    sec += 1
    html += f"""
<h2>{sec} · cost ladder (gross / 5 / 10 bps per side)</h2>
<table><tr><th>book</th><th class='per'>gross</th><th class='per'>5 bps/side</th><th class='per'>10 bps/side</th><th>gross 1×→</th></tr>{cost}</table>
"""
    sec += 1
    html += f"""
<h2>{sec} · drawdown &amp; rolling Sharpe — {hero['name']}</h2>
<img src='data:image/png;base64,{plot_dd_roll(hero, net_bps)}'>
"""
    sec += 1
    html += f"""
<h2>{sec} · year-by-year — {hero['name']} vs benchmark</h2>
<img src='data:image/png;base64,{plot_yearly_vs_bench(hero, bench, bench_nm, net_bps)}'>
<img src='data:image/png;base64,{plot_ret_dist_by_year(hero, bench, bench_nm, net_bps)}'>
"""
    sec += 1
    html += f"""
<h2>{sec} · robustness — Newey-West t (lag 5) &amp; autocorrelation</h2>
<p class='lgnd'>NW-t on daily net returns. Lag-1 autocorrelation &lt; 0.10 suggests Sharpe is not inflated by serial correlation.</p>
<table>
<tr><th>series</th><th>IS NW-t</th><th>OOS NW-t</th><th>FULL NW-t</th><th>ρ₁ (lag-1)</th></tr>
{robustness_table(hero, bench, net_bps)}
</table>
"""
    sec += 1
    html += f"""
<h2>{sec} · turnover</h2>
<img src='data:image/png;base64,{plot_turnover(books)}'>
"""
    if phase_analytics:
        from report_phase_sections import phase_sections_html
        frag, sec = phase_sections_html(
            phase_analytics,
            hero,
            bench_nm=bench_nm,
            long_lbl=stock_analytics.get("long_lbl", "LONG leg") if stock_analytics else "LONG leg",
            short_lbl=stock_analytics.get("short_lbl", "SHORT leg") if stock_analytics else "SHORT leg",
            sec_start=sec + 1,
        )
        html += frag

    if stock_analytics:
        from report_stock_sections import stock_sections_html
        stock_book = stock_analytics.get("hero_book") or leg_hero
        frag, _ = stock_sections_html(
            stock_analytics["picks"],
            stock_analytics["attribs"],
            stock_book,
            bench,
            long_lbl=stock_analytics.get("long_lbl", "LONG leg"),
            short_lbl=stock_analytics.get("short_lbl", "SHORT leg"),
            hero_n=stock_analytics.get("hero_n"),
            show_attrib_sizes=stock_analytics.get("show_attrib_sizes"),
            sec_start=sec + 1,
        )
        html += frag
    html += "</body></html>"
    Path(out_path).write_text(html, encoding="utf-8")
    print("written", out_path, len(html) // 1024, "KB")


# backward-compatible thin wrapper
def render_report(title, sub, books, bench, bench_nm, out_path, extra_note=""):
    render_enhanced_report(title, sub, books, bench, bench_nm, out_path, extra_note=extra_note)
