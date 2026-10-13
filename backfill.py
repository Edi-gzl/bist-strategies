"""Fill each strategy's public log with the longest EXACT backtest the data allows.

    python backfill.py            # all strategies
    python backfill.py focus_long fund_liquidity_80

Only sessions BEFORE the first day already in docs/data/<id>.json are added, marked
as backtest ("reconstructed"); live days are never touched. Depth per strategy:
  focus_long         Yahoo daily history (as far back as the 48 names have data)
  fund_liquidity_80  2020-01-02, the original's start (its TRY cash-rate table starts in 2020)
  daily_beta_pm      ~60 sessions: Yahoo 5-min bars give the exact 18:00 price + auctions
  ens5_basic/_s10c   every session in ens5/out (Yahoo 5-min history, ~60 sessions)
  eodr_short, morning_gap_short: no backfill - they need exact 16:01 / 10:01 one-minute prices
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import pandas as pd

import data
import strategies as S
from run import ROOT

OUT = ROOT / "docs" / "data"
KEEP_TRADES_DAYS = 120          # backtest trade lists are trimmed to recent sessions


def merge(sid: str, rets: pd.Series, trades: list, note: str) -> None:
    path = OUT / f"{sid}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    have = {d["date"] for d in doc["days"]}
    first = min(have) if have else "9999-12-31"
    rets = rets.dropna()
    cutoff = str((rets.index.max() - pd.Timedelta(days=KEEP_TRADES_DAYS)).date()) if len(rets) else first
    counts = Counter(t["date"] for t in trades)
    add = [{"date": str(d.date()), "ret": float(v), "reconstructed": True, "n_trades": counts.get(str(d.date()), 0)}
           for d, v in rets.items() if str(d.date()) < first]
    old_tr = {(t["date"], t["ticker"], t["side"]) for t in doc.get("trades", [])}
    add_tr = [t for t in trades if t["date"] < first and t["date"] >= cutoff
              and (t["date"], t["ticker"], t["side"]) not in old_tr]
    doc["days"] = sorted(doc["days"] + add, key=lambda d: d["date"])
    doc["trades"] = sorted(doc.get("trades", []) + add_tr, key=lambda t: (t["date"], t["side"], t["ticker"]))
    doc["backtest_note"] = note
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{sid:20s} +{len(add)} backtest days ({add[0]['date'] if add else '-'} .. {add[-1]['date'] if add else '-'}), "
          f"+{len(add_tr)} trades")


def main(which):
    if "focus_long" in which:
        D = data.daily(period="max")
        res = S.focus_long(D, None, list(D["Close"].index[-KEEP_TRADES_DAYS:]))
        merge("focus_long", res.returns, res.trades,
              "Backtest on Yahoo daily history from the first date at least 80% of the 48 names traded "
              "(the original engine's coverage rule).")
    if "fund_liquidity_80" in which:
        Dall = data.daily(data.UNIVERSE_ALL, period="10y")
        res = S.fund_liquidity_80(Dall, None, None, live_start="2020-01-02")
        merge("fund_liquidity_80", res.returns, res.trades,
              "Backtest from 2020-01-02, the original research start (its TRY cash-rate table begins in 2020).")
    if "daily_beta_pm" in which or "ens5" in which:
        D = data.daily(period="3y")
        M5 = data.intraday_5m()
        data.fill_missing_sessions(D, data.minute_dates(M5)[-10:], M5)
        if "daily_beta_pm" in which:
            W5 = [d for d in data.minute_dates(M5) if d in D["Close"].index]
            res = S.daily_beta_pm(D, M5, W5)
            merge("daily_beta_pm", res.returns, res.trades,
                  "Backtest over Yahoo's 60-day 5-minute window (exact 18:00 price and auctions).")
        if "ens5" in which:
            for v in ("basic", "s10c"):
                res = S._ens5(D, v)
                merge(f"ens5_{v}", res.returns, res.trades,
                      "Backtest over Yahoo's 60-day 5-minute window, computed with the original pipeline.")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"focus_long", "fund_liquidity_80", "daily_beta_pm", "ens5"})
