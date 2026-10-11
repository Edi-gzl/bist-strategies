"""Daily live run: fetch Yahoo data, run strategies, append to the public log.

    python run.py                 # normal daily run (writes docs/data)
    python run.py --out tmp_dir   # dry run into another folder

Logging rules
  * Only sessions on/after LIVE_START are recorded (no backtest).
  * A session is written once and re-checked on the next run (to pick up
    late Yahoo corrections and overnight exits), then frozen forever.
  * If a run is missed, the next run back-fills any session still inside
    Yahoo's ~7-day 1-minute window.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import data
import strategies

LIVE_START = "2026-10-02"
# Sessions before this date were computed after the fact (on 2026-10-11) with the
# same code and data; they are labelled "reconstructed" on the site, not live.
RECORDED_FROM = "2026-10-12"
ROOT = Path(__file__).resolve().parent
# strategies whose return on day d comes from a position opened on the previous session
OVERNIGHT = {"focus_long", "daily_beta_pm", "fund_liquidity_80"}


def load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def stats(days: list[dict]) -> dict:
    if not days:
        return {"days": 0}
    r = pd.Series([d["ret"] for d in days], dtype=float)
    eq = (1 + r).cumprod()
    active = r[r != 0]
    return {
        "days": int(r.size),
        "active_days": int(active.size),
        "total_return": float(eq.iloc[-1] - 1),
        "max_drawdown": float(min(0.0, (eq / eq.cummax() - 1).min())),
        "hit_rate": float((active > 0).mean()) if active.size else None,
        "best_day": float(r.max()), "worst_day": float(r.min()),
        "first": days[0]["date"], "last": days[-1]["date"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "data"))
    ap.add_argument("--live-start", default=LIVE_START)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    live_start = pd.Timestamp(args.live_start)

    D = data.daily(period="3y")
    M = data.minute(period="7d")
    W = [d for d in data.minute_dates(M) if d in D["Close"].index]
    # Guard: don't record a session whose data Yahoo hasn't fully published yet.
    # It stays un-recorded and the next run back-fills it.
    incomplete = []
    for d in list(W[-2:]):
        closes = D["Close"].loc[d].notna().mean()
        bars = M["Close"].loc[M["Close"].index.normalize() == d]
        last_bar = bars.dropna(how="all").index.max()
        full_day = last_bar is not pd.NaT and last_bar.hour >= 17
        half_day = last_bar is not pd.NaT and last_bar.hour < 14   # e.g. holiday eves
        if closes < 0.9 or not (full_day or half_day):
            incomplete.append(str(d.date()))
            W.remove(d)
    if incomplete:
        print("INCOMPLETE Yahoo data, not recorded this run:", incomplete)
    if len(W) < 2:
        print("not enough 1-minute sessions from Yahoo; nothing to do")
        return 1
    last_session = W[-1]
    print(f"1-minute sessions available: {W[0].date()} .. {last_session.date()} ({len(W)})")

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "live_start": str(live_start.date()), "recorded_from": RECORDED_FROM,
               "last_session": str(last_session.date()),
               "data_source": "Yahoo Finance (unofficial; may be delayed or corrected)",
               "strategies": [], "failures": {},
               "incomplete_sessions": incomplete}

    D_all = data.daily(data.UNIVERSE_ALL, period="2y")

    for fn in strategies.ALL:
        try:
            if fn is strategies.fund_liquidity_80:
                res = fn(D_all, M, W, live_start=live_start)
            else:
                res = fn(D, M, W)
        except Exception as exc:          # one broken strategy must not stop the rest
            summary["failures"][fn.__name__] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            continue

        path = out / f"{res.id}.json"
        prev = load(path, {})
        old_days = {d["date"]: d for d in prev.get("days", [])}
        frozen_before = prev.get("last_session")         # sessions before this are final

        # sessions this run can evaluate: W[1:] so overnight books have their entry day
        candidates = [d for d in W[1:] if d >= live_start]
        if res.id in OVERNIGHT:
            candidates = [d for d in candidates if W[W.index(d) - 1] >= live_start]

        days = dict(old_days)
        for d in candidates:
            key = str(d.date())
            if frozen_before and key < frozen_before and key in old_days:
                continue
            days[key] = {"date": key, "ret": float(res.returns.get(d, 0.0)),
                         "reconstructed": key < RECORDED_FROM}
        days = [days[k] for k in sorted(days)]

        recomputed = {d["date"] for d in days
                      if not (frozen_before and d["date"] < frozen_before and d["date"] in old_days)}
        trades = [t for t in prev.get("trades", []) if t["date"] not in recomputed]
        trades += [t for t in res.trades
                   if t["date"] >= str(live_start.date()) and
                   (t["date"] in recomputed or t["date"] == str(last_session.date()))]
        trades.sort(key=lambda t: (t["date"], t["side"], t["ticker"]))
        for d in days:
            d["n_trades"] = sum(1 for t in trades if t["date"] == d["date"])

        today = str(last_session.date())
        doc = {"id": res.id, "name": res.name, "spec": res.spec, "deviations": res.deviations,
               "live_start": str(live_start.date()), "last_session": today,
               "stats": stats(days), "days": days, "trades": trades,
               "holdings": getattr(res, "holdings", None)}
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")

        today_row = next((d for d in days if d["date"] == today), None)
        summary["strategies"].append({
            "id": res.id, "name": res.name, "stats": doc["stats"],
            "today": {"date": today, "ret": today_row["ret"] if today_row else None,
                      "trades": [t for t in trades if t["date"] == today]},
        })
        s = doc["stats"]
        print(f"  {res.id:<20} live days {s['days']:>3}  total {s.get('total_return', 0):+.2%}  "
              f"trades today {len(summary['strategies'][-1]['today']['trades'])}")

    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if summary["failures"]:
        print("FAILURES:", summary["failures"])
    if not summary["strategies"]:
        return 1
    # exit 2 = data written, but something needs attention (workflow turns this red)
    return 2 if (summary["failures"] or incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
