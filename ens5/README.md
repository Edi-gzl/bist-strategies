# ENS5v2 — the original research pipeline, run daily

`Research_done/` holds the original scripts from the Strat research repo, copied with
their folder layout. The **only** source changes are:

1. Hard-coded `C:\Research...` path literals replaced by `_RP(r"...")` (see `_rp.py`),
   which resolves them under this folder on any OS.
2. `shared/configs.py`: the research end date (`PERIODS['FULL']`) reads `ENS5_END`
   (set to today by `run_ens5.py`; the original fixed date is the default).

Verified: running this bundle on the same inputs reproduced the original
`C:\Research` run exactly (all 57 days, every config, identical picks).

`run_ens5.py` each day:

| Step | Code | Data |
|---|---|---|
| Prices, market cap | `02_fetch_prices_v2.py` (end date = today) | İş Yatırım |
| Statements (weekly) | `01_fetch_fundamentals_v3.py` | İş Yatırım |
| USD/TRY, risk-free | Yahoo `TRY=X`; risk-free 0 | Yahoo |
| Minute bars | Yahoo 1-min, stamped at bar end; 09:55 opening auction; closing auction = İş official close | Yahoo + İş |
| Factors | `13_build_bist_ff3` → `14_build_anomaly_panel` → `45_build_attention_paper_factors`, PC loadings (`f5_37`) | |
| Signal | `Daily_live/extend_to_latest.py` (records, 15-min returns, 5 residual models) | |
| Book / rules | `f5_55` → `f5_93a` → `_final_audit_intra` (basic = `top3-both`, S10c) | |

`state/` keeps the per-day records, 15-minute returns and residuals between runs
(sessions before 2026-10-13 were built from Yahoo 5-minute bars). `out/` is what the
website reads.
