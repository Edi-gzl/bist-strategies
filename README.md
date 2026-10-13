# BIST Strategy Live Log

Seven BIST strategies from the original research, run every weekday on Yahoo Finance
and İş Yatırım data. Each session's trades and returns are appended to a public log
with charts; sessions before 2026-10-12 are a clearly labelled backtest.

Site: https://edi-gzl.github.io/bist-strategies/

| File | Role |
|---|---|
| `data.py` | Yahoo daily + 1-minute bars; rebuilds sessions missing from Yahoo's daily feed from İş Yatırım. |
| `strategies.py` | The strategies, with line references to the original research code. |
| `ens5/` | ENS5v2: the original research pipeline, bundled and run daily (see `ens5/README.md`). |
| `universe_all.txt` | The 582-name universe used by Fund Liquidity 80. |
| `run.py` | Daily run: computes the latest sessions and appends them to `docs/data/*.json`. |
| `backfill.py` | One-off: adds the longest exact backtest each strategy's data allows. |
| `docs/index.html` | Website: combined chart, per-strategy charts, live vs backtest stats, daily log, trades. |
| `.github/workflows/daily.yml` | Runs ENS5 + `run.py` after the close (several times, GitHub starts late) and commits. |

## Strategies, history depth and fidelity

| Strategy | Backtest from | Checked against original results |
|---|---|---|
| Fund Liquidity 80 (FL26131) | 2020-01-02 (original start; cash-rate table begins 2020) | 0.975 daily corr. 2020-2026, every year within a few points |
| Focus Long | 2008-05-16 (first date with 80% of the 48 names) | 0.94 corr. 2020-2026; 0.998 on 2025-26 |
| Daily BETA PM | ~60 sessions (Yahoo 5-min gives the exact 18:00 price) | rules line-matched |
| EODR short | 7 sessions (needs exact 16:01 one-minute prices) | rules line-matched |
| Morning gap short (10:01 Core) | 7 sessions (needs exact 10:01 one-minute prices) | rules line-matched |
| ENS5v2 basic / S10c | ~60 sessions (Yahoo 5-min history) | bundle reproduces the original pipeline exactly on the same inputs |

## Log rules

- `RECORDED_FROM` in `run.py` (2026-10-12): first session recorded live. Earlier = backtest.
- A session is recorded only after it has closed (after 18:30 Istanbul) with complete data;
  it is re-checked on the next run, then frozen forever.
- Missed runs are back-filled automatically while the session is inside Yahoo's
  ~7-day 1-minute window.

## Run locally

    pip install -r requirements.txt
    python ens5/run_ens5.py
    python run.py
    python -m http.server 8765 --directory docs     # open http://localhost:8765
