# BIST Strategy Live Log

Five BIST strategies from the original research, reimplemented rule-for-rule and
run live every weekday on Yahoo Finance data. Each session's trades and returns
are appended to a public log with charts. No backtest, no local data files.

| File | Role |
|---|---|
| `data.py` | Only file that talks to the data source (Yahoo: daily bars + 1-minute bars). |
| `strategies.py` | The five strategies, with line references to the original research code. |
| `universe_all.txt` | The 582-name universe used by Fund Liquidity 80. |
| `run.py` | Daily run: computes the latest sessions and appends them to `docs/data/*.json`. |
| `docs/index.html` | Website: combined chart, per-strategy charts (1M/3M/6M/1Y/All), daily log, trades. |
| `.github/workflows/daily.yml` | Runs `run.py` Mon-Fri 19:00 Istanbul and commits the log. |

## Strategies and fidelity

| Strategy | Clock times used | Checked against original results |
|---|---|---|
| Fund Liquidity 80 (FL26131) | closing auction, biweekly | 0.994 daily correlation, Dec 2025-Jun 2026 |
| Focus Long | closing auction, daily | 221 of 234 days identical (<1 bp) |
| Daily BETA PM | 17:59 price, closing + next opening auction | rules line-matched; no minute history to compare |
| EODR short | 16:00 signal, 16:01 entry, closing auction | rules line-matched; no minute history to compare |
| Morning gap short (10:01 Core) | opening auction, 10:01 entry, closing auction | rules line-matched; no minute history to compare |

ENS5v2 is not included: its signal is an ensemble of five residual models from
the original research pipeline that cannot be rebuilt from public data.

## Log rules

- `LIVE_START` in `run.py` (2026-10-12) is the first live session.
- Each session is written once, re-checked on the next run (late Yahoo
  corrections, overnight exits), then frozen forever.
- A missed run is back-filled automatically while the session is still in
  Yahoo's ~7-day 1-minute window.

## Run locally

    pip install -r requirements.txt
    python run.py
    python -m http.server 8765 --directory docs     # open http://localhost:8765

## Put it online (one time)

1. Create a new **public** GitHub repo, e.g. `bist-strategies`.
2. From this folder:

        git init
        git add .
        git commit -m "initial"
        git branch -M main
        git remote add origin https://github.com/YOURNAME/bist-strategies.git
        git push -u origin main

3. Repo **Settings -> Pages -> Deploy from a branch -> main / docs -> Save**.
4. Repo **Actions** tab -> enable workflows. It then runs by itself every weekday.

Site: `https://YOURNAME.github.io/bist-strategies/`
