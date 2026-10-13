"""Extend intraday + daily execution pipelines to latest 1-min OHLC date.

Rebuilds (incremental where possible):
  - bist50_prices_daily.parquet (chain from OHLC auction)
  - f5_15_daily_records.parquet, f5_44_open_prices.parquet
  - returns_15min.parquet (append)
  - 15min P30 residuals (M05, BETA_126, M10, N02) + M15v2
  - book_top3.parquet, gap_panel_top1.csv

Then prints FINAL_LOCK-style metrics + extension-window diagnostics.
"""
from __future__ import annotations
from _rp import _RP  # portable-path shim (added when vendoring)

import shutil
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_RP(r""))
DATA = ROOT / "data"
F5 = ROOT / "Fable-5" / "results"
PX = ROOT / "claude_data" / "raw" / "bist50_prices_daily.parquet"
PANEL15 = ROOT / "claude_data" / "results" / "F_intraday" / "01_panels" / "returns_15min.parquet"
R15 = ROOT / "claude_data" / "results" / "F_intraday" / "03_residuals" / "15min" / "rank_l1"
CF = ROOT / "claude_data" / "results" / "cross_factor"
PKG = ROOT / "INTRADAY_DAILY" / "data"
CACHE = ROOT / "Daily_live" / "cache"
IS1 = pd.Timestamp("2025-01-02")
OLD_END = pd.Timestamp("2026-04-09")

sys.path.insert(0, str(ROOT / "claude_data" / "scripts"))
from shared.intraday_xsec import normalize_rank_l1, brogaard_xs_regression  # noqa: E402
from shared.intraday_panel import lag_factors_one_day, load_model_meta  # noqa: E402

MODELS4 = ["M05", "BETA_126", "M10", "N02_mom61_only"]


def backup(path: Path) -> None:
    if path.exists():
        dst = path.with_suffix(path.suffix + f".bak_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
        shutil.copy2(path, dst)
        print(f"  backup -> {dst.name}")


def latest_ohlc_date() -> pd.Timestamp:
    files = sorted((DATA / "THYAO.E" / "ohlc").glob("*.parquet"))
    return pd.to_datetime(files[-1].stem)


def bist50_tickers() -> list[str]:
    px = pd.read_parquet(PX, columns=["ticker"])
    return sorted(px.ticker.unique())


def ohlc_auction_close(tk: str, d: pd.Timestamp) -> float | None:
    fp = DATA / f"{tk}.E" / "ohlc" / f"{d.strftime('%Y%m%d')}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp, columns=["close", "volume"])
    df = df[df.volume > 0]
    if df.empty:
        return None
    late = df[df.index > d + pd.Timedelta(hours=18)]
    if len(late):
        return float(late["close"].iloc[0])
    cont = df[df.index <= d + pd.Timedelta(hours=18)]
    return float(cont["close"].iloc[-1]) if len(cont) else None


def extend_daily_prices(end: pd.Timestamp, tickers: list[str]) -> None:
    print("\n[1] extend bist50_prices_daily")
    backup(PX)
    px = pd.read_parquet(PX)
    px["date"] = pd.to_datetime(px["date"])
    template = px.iloc[-1].to_dict()
    new_rows = []
    for tk in tickers:
        sub = px[px.ticker == tk].sort_values("date")
        if sub.empty:
            continue
        last = sub.iloc[-1]
        last_d, last_adj = last.date, float(last.close_adj)
        last_raw = ohlc_auction_close(tk, last_d)
        if last_raw is None or last_raw <= 0:
            last_raw = float(last.close_adj)  # fallback
        d = last_d + pd.Timedelta(days=1)
        while d <= end:
            if d.weekday() >= 5:
                d += pd.Timedelta(days=1)
                continue
            raw = ohlc_auction_close(tk, d)
            if raw is None:
                d += pd.Timedelta(days=1)
                continue
            if ((px.ticker == tk) & (px.date == d)).any():
                d += pd.Timedelta(days=1)
                continue
            adj = last_adj * (raw / last_raw) if last_raw > 0 else last_adj
            row = template.copy()
            row.update(dict(ticker=tk, date=d, close_adj=adj, high_adj=adj, low_adj=adj, open_adj=adj))
            new_rows.append(row)
            last_adj, last_raw, last_d = adj, raw, d
            d += pd.Timedelta(days=1)
    if new_rows:
        px = pd.concat([px, pd.DataFrame(new_rows)], ignore_index=True)
        px = px.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        px.to_parquet(PX, index=False)
    print(f"  daily prices end: {px.date.max().date()} (+{len(new_rows)} rows)")


def extend_f5_15(end: pd.Timestamp, tickers: list[str]) -> pd.DataFrame:
    print("\n[2] extend f5_15_daily_records")
    fp = F5 / "f5_15_daily_records.parquet"
    backup(fp)
    rec = pd.read_parquet(fp)
    rec["date"] = pd.to_datetime(rec["date"])
    last = rec.date.max()
    new_dates = []
    d = last + pd.Timedelta(days=1)
    while d <= end:
        if d.weekday() < 5 and (DATA / "THYAO.E" / "ohlc" / f"{d:%Y%m%d}.parquet").exists():
            new_dates.append(d)
        d += pd.Timedelta(days=1)
    rows = []
    for tk in tickers:
        for day in new_dates:
            fpath = DATA / f"{tk}.E" / "ohlc" / f"{day:%Y%m%d}.parquet"
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath, columns=["close", "volume"])
            df = df[df.volume > 0]
            if not len(df):
                continue
            idx = df.index
            t18 = day + pd.Timedelta(hours=18)
            cont = df[idx <= t18]
            late = df[idx > t18]

            def px_at(hh, mm):
                s = cont["close"][cont.index <= day + pd.Timedelta(hours=hh, minutes=mm)]
                return float(s.iloc[-1]) if len(s) else np.nan

            rows.append(dict(
                ticker=tk, date=day,
                p1745=px_at(17, 45), p1759=px_at(17, 59), p1800=px_at(18, 0),
                p_auct=float(late["close"].iloc[0]) if len(late) else np.nan,
                v_auct=float(late["volume"].sum()) if len(late) else np.nan,
                v_morning=float(df[(idx > day + pd.Timedelta(hours=10)) & (idx <= day + pd.Timedelta(hours=10, minutes=30))]["volume"].sum()),
                med1m_day=float(cont["volume"].median()) if len(cont) else np.nan,
                v_total=float(df["volume"].sum()),
            ))
    if rows:
        rec = pd.concat([rec, pd.DataFrame(rows)], ignore_index=True)
        rec = rec.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
        rec.to_parquet(fp, index=False)
    print(f"  f5_15 end: {rec.date.max().date()} (+{len(rows)} ticker-days)")
    return rec


def extend_f5_44(rec: pd.DataFrame) -> None:
    print("\n[3] extend f5_44_open_prices")
    fp = F5 / "f5_44_open_prices.parquet"
    backup(fp)
    op = pd.read_parquet(fp)
    op["date"] = pd.to_datetime(op["date"])
    have = set(zip(op.ticker, op.date))
    rows = []
    for r in rec.itertuples():
        if (r.ticker, r.date) in have:
            continue
        fpath = DATA / f"{r.ticker}.E" / "ohlc" / f"{r.date:%Y%m%d}.parquet"
        po = np.nan
        if fpath.exists():
            df = pd.read_parquet(fpath, columns=["close", "volume"])
            df = df[df.volume > 0]
            w = df[(df.index >= r.date + pd.Timedelta(hours=9, minutes=55)) & (df.index <= r.date + pd.Timedelta(hours=10, minutes=10))]
            if len(w):
                po = float(w["close"].iloc[0])
        rows.append((r.ticker, r.date, po))
    if rows:
        op = pd.concat([op, pd.DataFrame(rows, columns=["ticker", "date", "p_open"])], ignore_index=True)
        op = op.drop_duplicates(["ticker", "date"], keep="last")
        op.to_parquet(fp, index=False)
    print(f"  f5_44 rows: {len(op):,} end {op.date.max().date()}")


def _last_close_before(bars: pd.DataFrame, t_target: time) -> float | None:
    if bars.empty:
        return None
    mask = bars.index.time <= t_target
    if not mask.any():
        return None
    return float(bars.loc[mask, "close"].iloc[-1])


def _vol_window(bars: pd.DataFrame, t_start: time | None, t_end: time) -> float:
    times = bars.index.time
    mask = times < t_end if t_start is None else (times >= t_start) & (times < t_end)
    return float(bars.loc[mask, "volume"].sum())


def aggregate_ticker_15min(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ohlc_dir = DATA / f"{ticker}.E" / "ohlc"
    if not ohlc_dir.exists():
        return pd.DataFrame()
    periods = [(0, None, time(10, 30), "overnight")]
    t = 10 * 60 + 30
    for i in range(30):
        nxt = t + 15
        periods.append((i + 1, time(t // 60, t % 60), time(nxt // 60, nxt % 60), ""))
        t = nxt
    rows, prev_close = [], None
    for f in sorted(ohlc_dir.glob("*.parquet")):
        d = pd.to_datetime(f.stem, format="%Y%m%d", errors="coerce")
        if pd.isna(d) or d < start or d > end:
            continue
        bars = pd.read_parquet(f, columns=["close", "volume"]).sort_index()
        bars = bars[bars.volume > 0]
        if bars.empty:
            continue
        if prev_close is not None:
            c0 = _last_close_before(bars, time(10, 31))
            if c0 is not None:
                rows.append(dict(ticker=ticker, date=d, period=0, close_start=prev_close, close_end=c0,
                                 ret=c0 / prev_close - 1, volume_30m=_vol_window(bars, time(10, 0), time(10, 30))))
        for p_id, t_start, t_end, _ in periods[1:]:
            probe = time(t_end.hour if t_end.minute < 59 else t_end.hour + 1,
                         (t_end.minute + 1) if t_end.minute < 59 else 0)
            c0 = _last_close_before(bars, t_start)
            c1 = _last_close_before(bars, probe)
            if c0 is None or c1 is None:
                continue
            rows.append(dict(ticker=ticker, date=d, period=p_id, close_start=c0, close_end=c1,
                             ret=c1 / c0 - 1, volume_30m=_vol_window(bars, t_start, t_end)))
        eod = _last_close_before(bars, time(18, 1))
        if eod is not None:
            prev_close = eod
    return pd.DataFrame(rows)


def extend_panel15(start: pd.Timestamp, end: pd.Timestamp, tickers: list[str]) -> None:
    print("\n[4] extend returns_15min")
    backup(PANEL15)
    old = pd.read_parquet(PANEL15)
    old["date"] = pd.to_datetime(old["date"])
    frames = [old]
    for i, tk in enumerate(tickers, 1):
        df = aggregate_ticker_15min(tk, start, end)
        if not df.empty:
            frames.append(df)
        if i % 12 == 0:
            print(f"  {i}/{len(tickers)} tickers", flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(["ticker", "date", "period"], keep="last").sort_values(["ticker", "date", "period"])
    lo, hi = panel["ret"].quantile([0.005, 0.995])
    panel["ret_winsor"] = panel["ret"].clip(lo, hi)
    panel.to_parquet(PANEL15, index=False)
    print(f"  panel end: {panel.date.max().date()} rows {len(panel):,}")


def load_factor_panel(end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    p = pd.read_parquet(CF / "anomaly_panel_daily.parquet")
    p["date"] = pd.to_datetime(p["date"])
    r17 = pd.read_parquet(CF / "anomaly_panel_daily_r17.parquet")
    r17["date"] = pd.to_datetime(r17["date"])
    new_cols = [c for c in r17.columns if c not in p.columns]
    p = p.merge(r17[["date", "ticker"] + new_cols], on=["date", "ticker"], how="left")
    pc = pd.read_parquet(F5 / "f5_37_pc_loadings_v2.parquet")
    pc["date"] = pd.to_datetime(pc["date"])
    pcc = [c for c in pc.columns if c.startswith("PC")]
    alld = sorted(set(p.date.unique()) | set(pd.date_range(p.date.max() + pd.Timedelta(days=1), end, freq="B")))
    alld = [d for d in alld if d.weekday() < 5]
    pieces = []
    for c in pcc:
        w = pc.pivot(index="date", columns="ticker", values=c).reindex(pd.to_datetime(alld)).ffill()
        pieces.append(w.stack().rename(c))
    pcl = pd.concat(pieces, axis=1).reset_index()
    pcl.columns = ["date", "ticker"] + pcc
    p = p.merge(pcl, on=["date", "ticker"], how="outer")
    p = p[p.date <= end].sort_values(["ticker", "date"])
    fcols = [c for c in p.columns if c not in ("date", "ticker")]
    p[fcols] = p.groupby("ticker", sort=False)[fcols].ffill()
    return p, pcc


def extend_residuals(start: pd.Timestamp, end: pd.Timestamp) -> None:
    print("\n[5] extend 15min P30 residuals")
    p30 = pd.read_parquet(PANEL15)
    p30["date"] = pd.to_datetime(p30["date"])
    p30 = p30[(p30.period == 30) & (p30.date >= start) & (p30.date <= end)]
    panel, pcc = load_factor_panel(end)
    fcols = [c for c in panel.columns if c not in ("date", "ticker")]
    panel = lag_factors_one_day(panel, fcols)
    sect = panel["R27_sector"].fillna(-1).astype(int)
    dum = pd.get_dummies(sect, prefix="sec", drop_first=True)
    panel = pd.concat([panel, dum], axis=1)
    sect_cols = list(dum.columns)
    m = p30.merge(panel[["date", "ticker"] + fcols + sect_cols], on=["date", "ticker"], how="inner")
    metas = {mid: load_model_meta(mid)["factors"] for mid in MODELS4}
    for mid, fl in metas.items():
        fp = R15 / f"{mid}_residuals.parquet"
        backup(fp)
        old = pd.read_parquet(fp)
        old["date"] = pd.to_datetime(old["date"])
        old = old[~((old.date >= start) & (old.period == 30))]
        vals = []
        need = [f for f in fl if f in m.columns]
        for d, g in m.groupby("date"):
            g = g.dropna(subset=["ret_winsor"] + need)
            g = g[g["volume_30m"].fillna(0) >= 1]
            if len(g) < 20:
                continue
            X = normalize_rank_l1(g[need].values.astype(float))
            X = np.column_stack([X, g[sect_cols].values.astype(float)])
            xs = brogaard_xs_regression(g["ret_winsor"].values.astype(float), X, min_stocks=20)
            if xs is None:
                continue
            for tk, ret, risk, resid, vol in zip(
                g.ticker.values, g.ret.values, xs["risk"], xs["residual"], g.volume_30m.values
            ):
                vals.append(dict(ticker=tk, date=d, period=30, ret=ret, risk=risk, residual=resid, volume_30m=vol))
        new = pd.DataFrame(vals)
        out = pd.concat([old, new], ignore_index=True).drop_duplicates(["ticker", "date", "period"], keep="last")
        out = out.sort_values(["ticker", "date", "period"])
        out.to_parquet(fp, index=False)
        print(f"  {mid}: +{len(new)} P30 rows -> end {out[out.period==30].date.max().date()}")


def extend_m15v2(start: pd.Timestamp, end: pd.Timestamp) -> None:
    print("\n[6] extend M15v2 residuals")
    fp = F5 / "f5_37_M15v2_residuals.parquet"
    backup(fp)
    p30 = pd.read_parquet(PANEL15)
    p30["date"] = pd.to_datetime(p30["date"])
    p30 = p30[(p30.period == 30) & (p30.date >= start) & (p30.date <= end)]
    panel, pcc = load_factor_panel(end)
    panel = lag_factors_one_day(panel, pcc + ["R27_sector"])
    sect = panel["R27_sector"].fillna(-1).astype(int)
    dum = pd.get_dummies(sect, prefix="sec", drop_first=True)
    panel = pd.concat([panel, dum], axis=1)
    sect_cols = list(dum.columns)
    m = p30.merge(panel[["date", "ticker"] + pcc + sect_cols], on=["date", "ticker"], how="inner")
    old = pd.read_parquet(fp)
    old["date"] = pd.to_datetime(old["date"])
    old = old[old.date < start]
    vals = []
    for d, g in m.groupby("date"):
        g = g.dropna(subset=["ret_winsor"] + pcc)
        g = g[g["volume_30m"].fillna(0) >= 1]
        if len(g) < 20:
            continue
        X = normalize_rank_l1(g[pcc].values.astype(float))
        X = np.column_stack([X, g[sect_cols].values.astype(float)])
        xs = brogaard_xs_regression(g["ret_winsor"].values.astype(float), X, min_stocks=20)
        if xs is None:
            continue
        for tk, resid in zip(g.ticker.values, xs["residual"]):
            vals.append(dict(ticker=tk, date=d, residual=resid))
    new = pd.DataFrame(vals)
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    out.to_parquet(fp, index=False)
    print(f"  M15v2 end: {out.date.max().date()} (+{len(new)} rows)")


def rebuild_books() -> None:
    print("\n[7] rebuild book_top3 / book_top5")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "INTRADAY_DAILY" / "scripts" / "f5_93a_build_books.py")], check=True)


def rebuild_gap_panel() -> None:
    print("\n[8] rebuild gap_panel_top1")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "Daily_live"))
    from execution_core import load_package
    import gap_analysis as GA
    backup(CACHE / "gap_panel_top1.csv")
    pkg = load_package(use_cap=False, baskets=(1,))
    GA.build_gap_panel(pkg, 1)
    gp = pd.read_csv(CACHE / "gap_panel_top1.csv", parse_dates=["date"])
    print(f"  gap_panel end: {gp.date.max().date()} rows {len(gp)}")


def print_final_lock() -> None:
    print("\n[9] FINAL_LOCK metrics")
    subprocess = __import__("subprocess")
    subprocess.run([sys.executable, str(ROOT / "INTRADAY_DAILY" / "scripts" / "final_lock.py")], check=True)


def extension_window_stats() -> None:
    print("\n[10] Extension window only (days after 2026-04-09)")
    IS1 = pd.Timestamp("2025-01-02")

    def sh(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        return x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 5 and x.std(ddof=1) > 0 else np.nan

    def zser(g):
        return (g - g.rolling(63, min_periods=30).mean().shift(1)) / g.rolling(63, min_periods=30).std().shift(1)

    B = pd.read_parquet(PKG / "book_top3.parquet")
    B["dow"] = B.index.dayofweek
    lon, sid, mkt = 0.5 * B.L_on, 0.5 * B.S_id, B.mkt
    intra = ((1 + mkt) / (1 + B.gap) - 1).shift(1)
    skz = (zser(B.gap) > 0.5).fillna(False)
    mtf = B.dow.isin([0, 3, 4])
    w = (1 + 0.5 * zser(intra)).clip(0.3, 2.0).fillna(1.0).where(mtf, 1.0)
    pmx = pd.Series(w * lon + np.where(skz.values, 0, sid), index=B.index)
    bt = (pd.concat([pmx.rename("s"), mkt.rename("m")], axis=1).dropna()
          .pipe(lambda d: (d.s.rolling(63, 40).cov(d.m) / d.m.rolling(63, 40).var()).shift(1).reindex(pmx.index)))
    intranet = pmx - bt * mkt - bt.abs() * 2 * 5 / 1e4 - 2 * 5 / 1e4 * (0.5 * w + np.where(skz.values, 0.0, 0.5))
    ext_i = intranet[intranet.index > OLD_END]
    print(f"  Intraday locked (hedged net5): n={len(ext_i)} mean={ext_i.mean()*100:+.3f}%/day Sharpe={sh(ext_i):+.2f} hit={(ext_i>0).mean():.0%}")

    p = pd.read_csv(CACHE / "gap_panel_top1.csv", parse_dates=["date"]).set_index("date")
    LON, SID, zg = p.LON.astype(float), p.SID.astype(float), pd.to_numeric(p.gap_z63, errors="coerce")
    PMd = 0.5 * LON + 0.5 * SID
    skd = (zg > 0.5).fillna(False)
    cpd = (0.925 - 0.693) / 252
    daily_g = pd.Series(np.where(skd, LON, PMd), index=p.index) - np.where(skd, cpd * 0.5, cpd)
    ext_d = daily_g[daily_g.index > OLD_END]
    print(f"  Daily locked (unhedged net5):    n={len(ext_d)} mean={ext_d.mean()*100:+.3f}%/day Sharpe={sh(ext_d):+.2f} hit={(ext_d>0).mean():.0%}")

    bk = pd.read_parquet(PKG / "book_top3.parquet")
    gp = pd.read_csv(CACHE / "gap_panel_top1.csv", parse_dates=["date"])
    print(f"\n  book_top3 end: {bk.index.max().date()} ({len(bk)} days)")
    print(f"  gap_panel end: {gp.date.max().date()} ({len(gp)} days)")


def main():
    end = latest_ohlc_date()
    tickers = bist50_tickers()
    start = OLD_END + pd.Timedelta(days=1)
    print(f"Target end: {end.date()} | extension from {start.date()} | tickers={len(tickers)}")
    extend_daily_prices(end, tickers)
    rec = extend_f5_15(end, tickers)
    extend_f5_44(rec)
    extend_panel15(start - pd.Timedelta(days=5), end, tickers)  # small overlap for overnight P0
    extend_residuals(start, end)
    extend_m15v2(start, end)
    rebuild_books()
    rebuild_gap_panel()
    print_final_lock()
    extension_window_stats()


if __name__ == "__main__":
    main()
