"""Daily ENS5v2 run (basic + S10c) on Yahoo + Is Yatirim data, using the friend's pipeline.

    python ens5/run_ens5.py                      # live: Yahoo 1-min, recompute last 2 sessions
    python ens5/run_ens5.py --minute-dir DIR --full   # rebuild every day found in DIR (validation)

Pipeline (vendored, unedited except portable paths; see ens5/README.md):
  Is Yatirim prices + fundamentals (02_fetch_prices_v2, 01_fetch_fundamentals_v3)
  -> 13_build_bist_ff3 -> 14_build_anomaly_panel -> 45_build_attention_paper_factors (r17 := r16)
  -> PC loadings (f5_37 build_loadings) -> Daily_live/extend_to_latest (records, open prices,
     15-min returns, 4-model + M15v2 residuals) -> f5_55 -> f5_93a -> _final_audit_intra
State that must persist between runs lives in ens5/state/ (committed by the workflow).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RD = HERE / "Research_done"
STATE = HERE / "state"
OUT = HERE / "out"
DATA = HERE / "data"                      # per-ticker per-day minute files (rebuilt each run)
RAW = RD / "claude_data" / "raw"
CF = RD / "claude_data" / "results" / "cross_factor"
F5 = RD / "Fable-5" / "results"
P15 = RD / "claude_data" / "results" / "F_intraday" / "01_panels" / "returns_15min.parquet"
R15 = RD / "claude_data" / "results" / "F_intraday" / "03_residuals" / "15min" / "rank_l1"
PKG = RD / "INTRADAY_DAILY"
MODELS4 = ["M05", "BETA_126", "M10", "N02_mom61_only"]
BANKS = ["AKBNK", "GARAN", "HALKB", "ISCTR", "TSKB", "VAKBN", "YKBNK"]
UNIVERSE = [
    "AEFES", "AKBNK", "AKSEN", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "CIMSA",
    "DOAS", "DOHOL", "EKGYO", "ENJSA", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HALKB",
    "HEKTS", "ISCTR", "KCHOL", "KONTR", "KRDMD", "MGROS", "ODAS", "OYAKC", "PETKM", "PGSUS",
    "SAHOL", "SASA", "SISE", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TRALT",
    "TRMET", "TSKB", "TTKOM", "TUPRS", "ULKER", "VAKBN", "VESTL", "YKBNK",
]
STATE_FILES = {                            # state name -> pipeline location
    "f5_15_daily_records.parquet": F5 / "f5_15_daily_records.parquet",
    "f5_44_open_prices.parquet": F5 / "f5_44_open_prices.parquet",
    "returns_15min.parquet": P15,
    "f5_37_M15v2_residuals.parquet": F5 / "f5_37_M15v2_residuals.parquet",
    "bist50_fundamentals_long.parquet": RAW / "bist50_fundamentals_long.parquet",
    **{f"{m}_residuals.parquet": R15 / f"{m}_residuals.parquet" for m in MODELS4},
}


def log(*a):
    print("[ens5]", *a, flush=True)


# ------------------------------------------------------------------ setup
def setup_tree():
    for d in (RAW, CF, F5, P15.parent, R15, PKG / "data", PKG / "results", OUT, STATE, DATA):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("claude_data", "Fable-5", "INTRADAY_DAILY", "Daily_live"):
        link, target = HERE / name, RD / name
        if link.exists() or link.is_symlink():
            continue
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    os.environ["ENS5_ROOT"] = str(HERE)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))


def run_script(path: Path, cwd: Path):
    log("run", path.name)
    r = subprocess.run([sys.executable, "-W", "ignore", str(path)], cwd=str(cwd),
                       capture_output=True, text=True, env=os.environ.copy())
    if r.returncode:
        print((r.stdout or "")[-2000:], (r.stderr or "")[-4000:], sep="\n")
        raise RuntimeError(f"{path.name} failed")


# ------------------------------------------------------------------ Is Yatirim
def _patch_timeout():
    if getattr(requests.Session.request, "_ens5", False):
        return
    orig = requests.Session.request

    def req(self, *a, **kw):
        kw["timeout"] = 120
        return orig(self, *a, **kw)
    req._ens5 = True
    requests.Session.request = req


def _exec_with_subs(path: Path, subs):
    src = path.read_text(encoding="utf-8")
    # the ingest scripts re-wrap sys.stdout; doing that twice in one process closes the console
    src = src.replace("sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')",
                      "pass  # (stdout re-wrap removed by run_ens5)")
    for old, new in subs:
        assert old in src, f"{path.name}: '{old}' not found"
        src = src.replace(old, new)
    ns = {"__name__": "ens5_ingest", "__file__": str(path)}
    stdout = sys.stdout
    exec(compile(src, str(path), "exec"), ns)
    return ns, stdout


ISY_URL = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
# same renaming / column selection as 02_fetch_prices_v2.py main()
ISY_RENAME = {"HGDG_HS_KODU": "ticker", "HGDG_TARIH": "date", "HGDG_KAPANIS": "close_adj", "HGDG_AOF": "vwap_adj",
              "HGDG_MIN": "low_adj", "HGDG_MAX": "high_adj", "HGDG_HACIM": "volume", "HG_KAPANIS": "close_raw",
              "HG_MIN": "low_raw", "HG_MAX": "high_raw", "SERMAYE": "shares_out", "PD": "mktcap_try",
              "PD_USD": "mktcap_usd", "HAO_PD": "mktcap_float_try", "HAO_PD_USD": "mktcap_float_usd",
              "DOLAR_BAZLI_FIYAT": "close_usd"}
ISY_KEEP = ["ticker", "date", "close_adj", "vwap_adj", "low_adj", "high_adj", "volume", "close_raw", "shares_out",
            "mktcap_try", "mktcap_usd", "mktcap_float_try", "mktcap_float_usd", "close_usd"]


def _isy_chunked(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for y in range(start.year, end.year + 1):
        a, b = max(start, pd.Timestamp(y, 1, 1)), min(end, pd.Timestamp(y, 12, 31))
        for attempt in range(3):
            try:
                r = requests.get(ISY_URL, params={"hisse": ticker, "startdate": f"{a:%d-%m-%Y}", "enddate": f"{b:%d-%m-%Y}"},
                                 timeout=120, headers={"User-Agent": "Mozilla/5.0"})
                frames.append(pd.DataFrame(r.json().get("value", [])))
                break
            except Exception:
                if attempt == 2:
                    log(f"  {ticker} {y}: failed")
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if df.empty:
        return df
    df = df.rename(columns=ISY_RENAME)[[c for c in ISY_KEEP if c in df.rename(columns=ISY_RENAME).columns]]
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    return df.drop_duplicates(["ticker", "date"]).sort_values("date")


def fetch_is_yatirim(end: date, fundamentals: bool):
    _patch_timeout()
    ing = RD / "claude_data" / "scripts" / "pipeline" / "ingest"
    ns, _ = _exec_with_subs(ing / "02_fetch_prices_v2.py", [
        ("END_DATE   = '06-05-2026'", f"END_DATE   = '{end:%d-%m-%Y}'"),
        ("assert len(BIST50) == 50", f"BIST50 = {UNIVERSE!r}\nassert len(BIST50) == {len(UNIVERSE)}")])
    try:
        ns["main"]()
    except ValueError as e:                      # the coverage printout can hit a closed stdout
        if "closed file" not in str(e):
            raise
    px = pd.read_parquet(RAW / "bist50_prices_daily.parquet")
    missing = sorted(set(UNIVERSE) - set(px.ticker))
    if missing:
        # Is Yatirim sometimes returns nothing for a long multi-year request on one name
        # (seen for TRMET); fetch those names year by year with the same columns as the script.
        log(f"Is prices missing for {missing}: fetching year by year")
        extra = [_isy_chunked(t, pd.Timestamp("2020-01-01"), pd.Timestamp(end)) for t in missing]
        extra = [e for e in extra if len(e)]
        if extra:
            px = pd.concat([px] + extra, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
            px.to_parquet(RAW / "bist50_prices_daily.parquet", index=False)
    if set(UNIVERSE) - set(px.ticker):
        raise RuntimeError(f"Is Yatirim prices missing for {sorted(set(UNIVERSE) - set(px.ticker))}")
    px["open_adj"] = px["close_adj"]            # Daily_live/extend_to_latest.py convention
    px.to_parquet(RAW / "bist50_prices_daily.parquet", index=False)
    log("Is prices", px.date.min().date(), "->", px.date.max().date(), "tickers", px.ticker.nunique())
    if fundamentals:
        f, _ = _exec_with_subs(ing / "01_fetch_fundamentals_v3.py",
                               [("END_YEAR   = '2025'", f"END_YEAR   = '{end.year}'")])
        frames, errors = f["fetch_universe"]([t for t in UNIVERSE if t not in BANKS], BANKS)
        if frames:
            f["save_panel"](frames)
        log("Is fundamentals errors:", errors)
    return px


# ------------------------------------------------------------------ Yahoo
def yahoo_fx_rf():
    import yfinance as yf
    fx = yf.download("TRY=X", start="2019-12-01", interval="1d", auto_adjust=False, progress=False)
    s = fx["Close"].squeeze().dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated()]
    pd.DataFrame({"Tarih": s.index.strftime("%d-%m-%Y"), "TP_DK_USD_A_YTL": s.values, "date": s.index}) \
        .to_parquet(RAW / "usdtry_daily.parquet", index=False)
    bd = pd.bdate_range("2020-01-01", pd.Timestamp.today().normalize())
    pd.DataFrame({"date": bd, "rf_annual_pct": 0.0, "rf_daily": 0.0}).to_parquet(RAW / "rf_daily.parquet", index=False)


def yahoo_minutes(px: pd.DataFrame) -> list[pd.Timestamp]:
    """Yahoo 1-min bars -> per-day files stamped at bar END, plus auction rows.
    Opening auction = 09:55 single-price bar; closing auction = Is Yatirim raw close
    (HG_KAPANIS), falling back to Yahoo's daily close if Is has not published the day yet."""
    import yfinance as yf
    tz = "Europe/Istanbul"
    m1 = yf.download([t + ".IS" for t in UNIVERSE], period="7d", interval="1m", auto_adjust=False,
                     group_by="column", progress=False, threads=True)
    d1 = yf.download([t + ".IS" for t in UNIVERSE], period="1mo", interval="1d", auto_adjust=False,
                     group_by="column", progress=False, threads=True)
    for f in (m1, d1):
        idx = pd.DatetimeIndex(f.index)
        f.index = idx.tz_convert(tz).tz_localize(None) if idx.tz is not None else idx
    d1.index = d1.index.normalize()
    isy = px.pivot_table(index="date", columns="ticker", values="close_raw")
    days = sorted(set(m1.index.normalize()))
    shutil.rmtree(DATA, ignore_errors=True)
    complete = []
    for d in days:
        n_ok = 0
        for tk in UNIVERSE:
            y = tk + ".IS"
            b = pd.DataFrame({c.lower(): m1[c][y] for c in ["Open", "High", "Low", "Close", "Volume"]})
            b = b[(b.index.normalize() == d)].dropna(subset=["close"])
            if b.empty:
                continue
            auct_open = b[b.index.time == pd.Timestamp("09:55").time()]["close"]
            b = b[b.index.time >= pd.Timestamp("10:00").time()]
            b.index = b.index + pd.Timedelta(minutes=1)
            close = isy[tk].get(d, np.nan) if tk in isy.columns else np.nan
            if not np.isfinite(close):
                close = d1["Close"][y].get(d, np.nan)
            rows = [b]
            o = float(auct_open.iloc[0]) if len(auct_open) else d1["Open"][y].get(d, np.nan)
            if np.isfinite(o) and o > 0:
                rows.append(pd.DataFrame({"open": o, "high": o, "low": o, "close": o, "volume": 1.0},
                                         index=[d + pd.Timedelta(hours=9, minutes=55)]))
            if np.isfinite(close) and close > 0:
                rows.append(pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
                                         index=[d + pd.Timedelta(hours=18, minutes=10)]))
                n_ok += 1
            out = DATA / f"{tk}.E" / "ohlc"
            out.mkdir(parents=True, exist_ok=True)
            pd.concat(rows).sort_index().to_parquet(out / f"{d:%Y%m%d}.parquet")
        last_bar = m1["Close"][m1.index.normalize() == d].dropna(how="all").index.max()
        now_ist = pd.Timestamp.now(tz=tz).tz_localize(None)
        if d >= now_ist.normalize() and now_ist < d + pd.Timedelta(hours=18, minutes=30):
            for tk in UNIVERSE:                  # today's session is still running
                (DATA / f"{tk}.E" / "ohlc" / f"{d:%Y%m%d}.parquet").unlink(missing_ok=True)
            continue
        if n_ok >= 0.9 * len(UNIVERSE) and last_bar.hour >= 17:
            complete.append(d)
        else:
            log(f"session {d.date()} incomplete (closing auctions {n_ok}/{len(UNIVERSE)}, last bar {last_bar}) - not used")
            for tk in UNIVERSE:
                (DATA / f"{tk}.E" / "ohlc" / f"{d:%Y%m%d}.parquet").unlink(missing_ok=True)
    return complete


# ------------------------------------------------------------------ factors
def build_factors():
    s = RD / "claude_data" / "scripts" / "analysis"
    run_script(s / "13_build_bist_ff3.py", RD)
    run_script(s / "14_build_anomaly_panel.py", RD)
    run_script(RD / "_ARCHIVE" / "legacy_scripts" / "45_build_attention_paper_factors.py", RD)
    shutil.copy2(CF / "anomaly_panel_daily_r16.parquet", CF / "anomaly_panel_daily_r17.parquet")
    # PC loadings: verbatim f5_37_pc_pipeline_v2.build_loadings
    NPC, WIN, MINOBS = 10, 252, 200
    px = pd.read_parquet(RAW / "bist50_prices_daily.parquet").pivot_table(index="date", columns="ticker", values="close_adj")
    px.index = pd.to_datetime(px.index); px = px.sort_index()
    ret = px.pct_change(); dates = ret.index
    rows = []
    for d in ret.groupby([dates.year, dates.month]).tail(1).index:
        w = ret.loc[ret.index <= d].tail(WIN)
        if len(w) < WIN:
            continue
        ok = w.columns[w.notna().sum() >= MINOBS]
        X = w[ok].fillna(0.0).values; X = X - X.mean(axis=0, keepdims=True)
        ev, V = np.linalg.eigh(X.T @ X / len(X)); V = V[:, np.argsort(ev)[::-1][:NPC]]
        for k in range(V.shape[1]):
            j = np.argmax(np.abs(V[:, k]))
            if V[j, k] < 0:
                V[:, k] = -V[:, k]
        for i, tk in enumerate(ok):
            rows.append((d, tk, *V[i, :]))
    pd.DataFrame(rows, columns=["date", "ticker"] + [f"PC{k}_loading_252d" for k in range(1, NPC + 1)]) \
        .to_parquet(F5 / "f5_37_pc_loadings_v2.parquet", index=False)
    ew = px.pct_change().mean(axis=1).rename("r").reset_index().rename(columns={"index": "date"})
    ew.to_csv(PKG / "data" / "EW_BIST50_daily.csv", index=False)


# ------------------------------------------------------------------ intraday chain
def load_ext():
    spec = importlib.util.spec_from_file_location("ens5_ext", RD / "Daily_live" / "extend_to_latest.py")
    ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    ext.backup = lambda p: None
    ext.DATA = DATA                                  # minute files built by this runner
    return ext


def restore_state():
    for name, dst in STATE_FILES.items():
        if (STATE / name).exists():
            shutil.copy2(STATE / name, dst)


def save_state():
    for name, src in STATE_FILES.items():
        if src.exists():
            shutil.copy2(src, STATE / name)


def intraday(days: list[pd.Timestamp], full: bool):
    ext = load_ext()
    restore_state() if not full else None
    tickers = UNIVERSE
    rec_fp = F5 / "f5_15_daily_records.parquet"
    rec = pd.read_parquet(rec_fp) if (rec_fp.exists() and not full) else pd.DataFrame(columns=["ticker", "date"])
    rec["date"] = pd.to_datetime(rec["date"])
    have = sorted(rec.date.unique())
    # (re)process: every available day newer than the 2nd-last stored session
    redo_from = have[-2] if len(have) >= 2 else (days[0] if days else None)
    todo = [d for d in days if redo_from is None or d >= redo_from]
    if not todo:
        log("no new sessions"); return False
    start, end = todo[0], todo[-1]
    log(f"processing sessions {start.date()} -> {end.date()} ({len(todo)})")
    keep = rec[rec.date < start]
    seed = start - pd.offsets.BDay(1)
    (keep if len(keep) else pd.DataFrame({"ticker": ["_SEED"], "date": [seed]})).to_parquet(rec_fp, index=False)
    new = ext.extend_f5_15(end, tickers)
    new = new[new.ticker != "_SEED"]; new.to_parquet(rec_fp, index=False)
    op_fp = F5 / "f5_44_open_prices.parquet"
    op = pd.read_parquet(op_fp) if (op_fp.exists() and not full) else pd.DataFrame(columns=["ticker", "date", "p_open"])
    op["date"] = pd.to_datetime(op["date"])
    op[op.date < start].to_parquet(op_fp, index=False)
    ext.extend_f5_44(new)
    p15 = pd.read_parquet(P15) if (P15.exists() and not full) else pd.DataFrame(
        columns=["ticker", "date", "period", "close_start", "close_end", "ret", "volume_30m", "ret_winsor"])
    p15["date"] = pd.to_datetime(p15["date"])
    p15[p15.date < start].to_parquet(P15, index=False)
    ext.extend_panel15(start, end, tickers)
    for mid in MODELS4:
        fp = R15 / f"{mid}_residuals.parquet"
        if full or not fp.exists():
            pd.DataFrame(columns=["ticker", "date", "period", "ret", "risk", "residual", "volume_30m"]).astype(
                {"date": "datetime64[ns]", "period": "int64"}).to_parquet(fp, index=False)
    ext.extend_residuals(start, end)
    m15 = F5 / "f5_37_M15v2_residuals.parquet"
    if full or not m15.exists():
        pd.DataFrame(columns=["ticker", "date", "residual"]).astype({"date": "datetime64[ns]"}).to_parquet(m15, index=False)
    ext.extend_m15v2(start, end)
    return True


def strategy_layer():
    s = PKG / "scripts"
    run_script(s / "f5_55_execution_levers.py", s)
    run_script(s / "f5_93a_build_books.py", s)
    run_script(s / "_final_audit_intra.py", s)


# ------------------------------------------------------------------ export for the website
def export():
    sys.path.insert(0, str(PKG / "scripts")); sys.path.insert(0, str(RD))
    import test_intra_daily_parity_battery as T            # same gates as _final_audit_intra
    series = pd.read_csv(PKG / "results" / "FINAL_daily_series.csv", index_col=0, parse_dates=True)
    p = T.build_pick_panel()
    Le = p[["long1_eps", "long2_eps", "long3_eps"]]
    cz = lambda s: (s - s.rolling(63, min_periods=30).mean().shift(1)) / s.rolling(63, min_periods=30).std().shift(1)
    flags = pd.DataFrame({
        "gap_skip_short": (T.zser(p["gap"]) > 0.5).fillna(False),
        "a3_long_top1_only": (cz(Le.std(axis=1)) < -0.5).fillna(False),
        "b4_extend_long": (cz(Le.mean(axis=1)) > 0.5).fillna(False),
        "short_halved": p["mom_streak"].fillna(False).astype(bool),
        "long_weight": T.e42_weight(p),
    }, index=p.index)
    series.join(flags).to_csv(OUT / "ens5_daily.csv", index_label="date")
    shutil.copy2(PKG / "full_picks.csv", OUT / "ens5_picks.csv")
    # today's signal (exits not known yet): same selection rule as f5_55 lines 91-95
    recs = pd.read_parquet(F5 / "f5_15_daily_records.parquet"); recs["date"] = pd.to_datetime(recs["date"])
    recs = recs.sort_values(["ticker", "date"])
    recs["official"] = recs["p_auct"].where(recs["p_auct"].notna(), recs["p1800"])
    recs["move"] = recs.p1800 / recs.groupby("ticker")["official"].shift(1) - 1.0
    last = recs.date.max()
    ranks = []
    for mid in MODELS4:
        rr = pd.read_parquet(R15 / f"{mid}_residuals.parquet", columns=["ticker", "date", "period", "residual"])
        rr["date"] = pd.to_datetime(rr.date); rr = rr[(rr.period == 30) & rr.residual.notna() & (rr.date == last)]
        ranks.append(rr.set_index("ticker")["residual"].rank(pct=True))
    m15 = pd.read_parquet(F5 / "f5_37_M15v2_residuals.parquet"); m15["date"] = pd.to_datetime(m15["date"])
    ranks.append(m15[m15.date == last].set_index("ticker")["residual"].rank(pct=True))
    ens = pd.concat(ranks, axis=1); ens = ens.mean(axis=1).where(ens.notna().sum(axis=1) >= 3)
    g = recs[recs.date == last].set_index("ticker").join(ens.rename("s"))
    g = g[g.s.notna() & g.p_auct.notna()].sort_values("s")
    L = g[~(g.move >= 0.094).fillna(False)].head(3); S = g[~(g.move <= -0.094).fillna(False)].tail(3)
    pend = pd.concat([L.assign(leg="LONG"), S.assign(leg="SHORT")]).reset_index()[["ticker", "leg", "s", "p_auct", "move"]]
    pend.insert(0, "date", last)
    pend.to_csv(OUT / "ens5_pending.csv", index=False)
    log("exported", len(series), "book days; pending signal", last.date(), "longs", list(L.index), "shorts", list(S.index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute-dir", help="use existing minute files instead of Yahoo (validation)")
    ap.add_argument("--full", action="store_true", help="rebuild every session from scratch")
    ap.add_argument("--skip-fetch", action="store_true", help="reuse Is Yatirim files already in raw/")
    args = ap.parse_args()
    setup_tree()
    if args.full:
        restore_state()                           # keeps fundamentals if present
    end = date.today()
    os.environ["ENS5_END"] = f"{end:%Y-%m-%d}"
    if not args.skip_fetch:
        meta_fp = STATE / "meta.json"
        meta = pd.read_json(meta_fp, typ="series") if meta_fp.exists() else pd.Series(dtype=object)
        last_fund = pd.Timestamp(meta.get("fundamentals_fetched", "2000-01-01"))
        stale = (not (STATE / "bist50_fundamentals_long.parquet").exists()) or (pd.Timestamp(end) - last_fund).days >= 7
        restore_state()
        px = fetch_is_yatirim(end, fundamentals=stale)
        if stale:
            meta["fundamentals_fetched"] = f"{end:%Y-%m-%d}"
            meta.to_json(meta_fp)
        yahoo_fx_rf()
    else:
        px = pd.read_parquet(RAW / "bist50_prices_daily.parquet")
    if args.minute_dir:
        global DATA
        DATA = Path(args.minute_dir)
        days = sorted(pd.to_datetime(f.stem) for f in (DATA / "THYAO.E" / "ohlc").glob("*.parquet"))
    else:
        days = yahoo_minutes(px)
    build_factors()
    if intraday(days, args.full):
        strategy_layer()
        save_state()
        export()


if __name__ == "__main__":
    main()
