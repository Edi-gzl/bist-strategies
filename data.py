"""All market data comes through this file (Yahoo Finance today).

Verified on Yahoo BIST data (Oct 2026):
  * daily Open  == the 09:55 opening-auction print (single-price 1m bar)
  * daily Close == the closing-auction print (differs from the 17:59 last trade)
  * 1-minute bars are labelled by bar START time, 09:55, 10:00 ... 17:59
  * 1-minute history is limited to the last ~8 calendar days

Swap this module for Matriks / BIST Datastore later; strategies only see the
dict-of-DataFrames shape returned here (date x ticker).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# BIST 48 research universe (core of BIST 50), from the original project config.
UNIVERSE = [
    "AEFES", "AKBNK", "AKSEN", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS",
    "BRSAN", "CIMSA", "DOAS", "DOHOL", "EKGYO", "ENJSA", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GUBRF", "HALKB", "HEKTS", "ISCTR", "KCHOL", "KONTR",
    "KRDMD", "MGROS", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA",
    "SISE", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TRALT", "TRMET",
    "TOASO", "TSKB", "TTKOM", "TUPRS", "VAKBN", "VESTL", "YKBNK", "ULKER",
]
# All 582 names active on the original panel's final date (Fund Liquidity 80 universe),
# from Strat-main/Long_only/results/broad_universe_ticker_coverage.csv.
UNIVERSE_ALL = [
    t.strip() for t in
    (Path(__file__).resolve().parent / "universe_all.txt").read_text(encoding="utf-8-sig").splitlines()
    if t.strip()
]
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TZ = "Europe/Istanbul"


def _download(tickers, period: str, interval: str, attempts: int = 3) -> dict[str, pd.DataFrame]:
    """Download with retries: Yahoo sometimes rate-limits cloud runners."""
    raw = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                [t + ".IS" for t in tickers], period=period, interval=interval,
                auto_adjust=False, group_by="column", progress=False, threads=True,
            )
        except Exception as exc:  # network / rate-limit errors
            print(f"  Yahoo {interval} attempt {attempt} failed: {exc}")
            raw = None
        if raw is not None and not raw.empty and raw["Close"].notna().any().mean() > 0.9:
            break
        if attempt < attempts:
            time.sleep(30 * attempt)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no data ({interval}, {period})")
    out = {}
    for f in FIELDS:
        if f not in raw.columns.get_level_values(0):
            continue
        df = raw[f].rename(columns=lambda c: c.replace(".IS", ""))
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(TZ).tz_localize(None)
        df.index = idx
        out[f] = df.reindex(columns=list(tickers)).astype(float)
    return out


def daily(tickers=UNIVERSE, period: str = "3y") -> dict[str, pd.DataFrame]:
    """Daily bars. Close = closing auction (raw); Adj Close = split/dividend adjusted."""
    d = _download(tickers, period, "1d")
    for f in d:
        d[f].index = d[f].index.normalize()
        d[f] = d[f][~d[f].index.duplicated(keep="last")].sort_index()
    return d


ISY_URL = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"


def _isy_days(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import requests
    for attempt in range(3):
        try:
            r = requests.get(ISY_URL, params={"hisse": ticker, "startdate": f"{start:%d-%m-%Y}",
                                              "enddate": f"{end:%d-%m-%Y}"},
                             timeout=90, headers={"User-Agent": "Mozilla/5.0"})
            v = pd.DataFrame(r.json().get("value", []))
            if v.empty:
                return v
            v["date"] = pd.to_datetime(v["HGDG_TARIH"], format="%d-%m-%Y")
            return v.set_index("date")
        except Exception:
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def fill_missing_sessions(D: dict, sessions, M: dict | None = None, workers: int = 16) -> list:
    """Yahoo's daily feed occasionally drops a whole BIST session (e.g. 2026-10-12)
    even though its minute feed has it. Rebuild those rows from Is Yatirim:
    Close = official close (HG_KAPANIS, == Yahoo close on normal days), High/Low,
    Volume = TRY value / close; Adj Close = Close x previous day's adjustment factor;
    Open = 09:55 opening-auction minute bar when minute data is supplied."""
    from concurrent.futures import ThreadPoolExecutor
    idx = D["Close"].index
    missing = [pd.Timestamp(s) for s in sessions if pd.Timestamp(s) not in idx and pd.Timestamp(s) > idx[0]]
    if not missing:
        return []
    tickers = list(D["Close"].columns)
    lo, hi = min(missing), max(missing)
    with ThreadPoolExecutor(workers) as ex:
        got = dict(zip(tickers, ex.map(lambda t: _isy_days(t, lo, hi), tickers)))
    new_idx = idx.union(pd.DatetimeIndex(missing))
    for f in D:
        D[f] = D[f].reindex(new_idx)
    filled = []
    for d in missing:
        prev = new_idx[new_idx.get_loc(d) - 1]
        n = 0
        for t in tickers:
            v = got.get(t)
            if v is None or v.empty or d not in v.index:
                continue
            c = float(v.at[d, "HG_KAPANIS"])
            if not c > 0:
                continue
            fac = D["Adj Close"].at[prev, t] / D["Close"].at[prev, t]
            D["Close"].at[d, t] = c
            D["Adj Close"].at[d, t] = c * (fac if np.isfinite(fac) and fac > 0 else 1.0)
            D["High"].at[d, t] = float(v.at[d, "HG_MAX"])
            D["Low"].at[d, t] = float(v.at[d, "HG_MIN"])
            D["Volume"].at[d, t] = float(v.at[d, "HGDG_HACIM"]) / c
            if M is not None and t in M["Open"].columns:
                o = M["Open"][t].get(pd.Timestamp(f"{d.date()} 09:55"), np.nan)
                D["Open"].at[d, t] = o
            n += 1
        if n >= 0.9 * len(tickers):
            filled.append(str(d.date()))
        else:                                   # not enough coverage: leave the session out
            for f in D:
                D[f] = D[f].drop(index=d)
    return filled


def minute(tickers=UNIVERSE, period: str = "7d") -> dict[str, pd.DataFrame]:
    """1-minute bars indexed by bar START time (Istanbul local)."""
    return _download(tickers, period, "1m")


def intraday_5m(tickers=UNIVERSE) -> dict[str, pd.DataFrame]:
    """5-minute bars (last ~60 days), labelled by bar START like the 1-minute data.
    The 17:55 bar's close is the last trade before 18:00; the 09:55 bar is the opening auction."""
    return _download(tickers, "60d", "5m")


def minute_dates(M) -> list[pd.Timestamp]:
    return sorted(set(M["Close"].index.normalize()))


def last_close_between(M, date, start: str, end: str) -> pd.Series:
    """Close of the last bar whose label is in [start, end] (per ticker)."""
    lo, hi = pd.Timestamp(f"{date.date()} {start}"), pd.Timestamp(f"{date.date()} {end}")
    win = M["Close"].loc[(M["Close"].index >= lo) & (M["Close"].index <= hi)]
    return win.ffill().iloc[-1] if len(win) else pd.Series(float("nan"), index=M["Close"].columns)


def first_open_between(M, date, start: str, end: str, inclusive_start=True) -> pd.Series:
    """Open of the first bar whose label is in [start, end] (or (start, end])."""
    lo, hi = pd.Timestamp(f"{date.date()} {start}"), pd.Timestamp(f"{date.date()} {end}")
    idx = M["Open"].index
    sel = ((idx >= lo) if inclusive_start else (idx > lo)) & (idx <= hi)
    win = M["Open"].loc[sel]
    return win.bfill().iloc[0] if len(win) else pd.Series(float("nan"), index=M["Open"].columns)
