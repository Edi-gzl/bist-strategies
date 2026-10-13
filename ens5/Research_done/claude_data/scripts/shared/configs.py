from _rp import _RP  # portable-path shim (added when vendoring)
"""
shared.configs — Single source of truth for ALL parameters.

Every parameter that controls the BIST → Brogaard pipeline lives here.
No script should hard-code a frequency, lag, window, sample size, or cost
threshold. Instead, import from this module:

    from shared.configs import CFG, PERIODS, SAMPLES, INTRADAY

Why this matters: when we want to test a factor with a different lag, holding
period, or cost assumption, we change ONE number here and re-run. No grep
across 18 scripts.
"""
from pathlib import Path
import pandas as pd

# ====================================================================
# 1. PATHS — reorganized 2026-05-08 per factor_methodology_guide.html
# ====================================================================
# Results are organized by METHOD (the question being asked), not by script.
# This mirrors the four methods in the guide:
#   A. Cross-sectional       → Test A (R²) + Test B (FM premium)  ← PRIMARY
#   B. Time-series           → Method 1: CAPM/FF3 per stock
#   C. Portfolios            → Method 4: L/S quintile sorts
#   D. Residual decomposition→ The Brogaard step (cross-sec on lagged X)
#   E. Diagnostics           → Hygiene, coverage, sample audit
#
# Each method directory contains a per_factor/<F>/ subfolder so every test
# always produces output in the same predictable location.
ROOT = Path(_RP(r"\Research_done"))
DATA_DIR     = ROOT / 'claude_data' / 'raw'
RESULTS_ROOT = ROOT / 'claude_data' / 'results'

# Primary result directories — one per Method
RESULTS_A_XSEC      = RESULTS_ROOT / 'A_cross_sectional'   # Test A + Test B
RESULTS_B_TIMESER   = RESULTS_ROOT / 'B_time_series'       # Method 1
RESULTS_C_PORTFOLIO = RESULTS_ROOT / 'C_portfolios'        # Method 4
RESULTS_D_RESIDUAL  = RESULTS_ROOT / 'D_residual_decomposition'  # Brogaard daily (Round 18)
RESULTS_E_DIAG      = RESULTS_ROOT / 'E_diagnostics'       # hygiene, coverage
RESULTS_F_INTRADAY  = RESULTS_ROOT / 'F_intraday'          # Brogaard 30-min + tick-ready

FIGURES_DIR  = ROOT / 'claude_data' / 'figures' / 'cross_factor'
LOGS_DIR     = ROOT / 'claude_data' / 'logs'
INTRADAY_DIR = ROOT / 'data'  # 1-min OHLC bars (override with --data-root)

# Legacy alias — keep working for now (scripts 17/18 still write here)
RESULTS_DIR  = RESULTS_ROOT / 'cross_factor'

# F_intraday subdirs (mirror D_residual_decomposition numbering)
F_INTRADAY_PANELS   = RESULTS_F_INTRADAY / '01_panels'
F_INTRADAY_MODELS   = RESULTS_F_INTRADAY / '02_factor_models'
F_INTRADAY_RESIDUAL = RESULTS_F_INTRADAY / '03_residuals'
F_INTRADAY_PREDICT  = RESULTS_F_INTRADAY / '04_predictability'
F_INTRADAY_PORTFOLIO = RESULTS_F_INTRADAY / '05_portfolios'
F_INTRADAY_SUMMARY  = RESULTS_F_INTRADAY / 'summary'

for d in (RESULTS_A_XSEC, RESULTS_B_TIMESER, RESULTS_C_PORTFOLIO,
          RESULTS_D_RESIDUAL, RESULTS_E_DIAG, RESULTS_F_INTRADAY,
          F_INTRADAY_PANELS, F_INTRADAY_MODELS, F_INTRADAY_RESIDUAL,
          F_INTRADAY_PREDICT, F_INTRADAY_PORTFOLIO, F_INTRADAY_SUMMARY,
          RESULTS_A_XSEC / 'per_factor', RESULTS_A_XSEC / 'summary',
          RESULTS_B_TIMESER / 'per_factor', RESULTS_B_TIMESER / 'summary',
          RESULTS_C_PORTFOLIO / 'per_factor', RESULTS_C_PORTFOLIO / 'summary',
          RESULTS_D_RESIDUAL / 'summary',
          RESULTS_E_DIAG,
          FIGURES_DIR, LOGS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def per_factor_dir(method, factor_name):
    """Return the standardized output directory for a single factor under a method.

    Args:
        method: 'A' (cross-sectional), 'B' (time-series), 'C' (portfolios)
        factor_name: e.g. 'R4_mom_12_1'

    Returns:
        Path to <results>/<method>/per_factor/<factor>/
        with tables/ and figures/ subdirs created.
    """
    method_map = {
        'A': RESULTS_A_XSEC, 'B': RESULTS_B_TIMESER,
        'C': RESULTS_C_PORTFOLIO, 'D': RESULTS_D_RESIDUAL,
    }
    base = method_map[method] / 'per_factor' / factor_name
    (base / 'tables').mkdir(parents=True, exist_ok=True)
    (base / 'figures').mkdir(parents=True, exist_ok=True)
    return base


# ====================================================================
# 2. SAMPLE PERIODS
# ====================================================================
# Windows we always run for regime sensitivity:
#   FULL: full available sample (data foundation)              ─ 65 months
#   PRE : pre-TAS29 hyperinflation accounting era (2021-2022)  ─ 24 months
#   POST: post-TAS29 inflation accounting clean period         ─ 40 months
#   OOS : reserved out-of-sample tail                          ─ 5 months
#
# TAS-29 ("Financial Reporting in Hyperinflationary Economies") became
# mandatory for Turkish firms starting Q1 2023. Pre-2023 fundamentals
# are nominal-TRY; post-2023 are inflation-restated. A factor that works
# in one regime but not the other tells us whether the signal is real
# economics or accounting-regime artifact.
PERIODS = {
    'FULL': ('2021-01-01', __import__('os').environ.get('ENS5_END', '2026-06-22')),   # live runs set ENS5_END
    'PRE' : ('2021-01-01', '2022-12-31'),   # pre-TAS29
    'POST': ('2023-01-01', '2026-05-05'),   # post-TAS29
    'OOS' : ('2026-01-01', '2026-05-05'),   # held-out tail
}
DEFAULT_PERIOD = 'FULL'

PERIOD_LABELS = {
    'FULL': 'Full sample (2021-2026)',
    'PRE' : 'Pre-TAS29 hyperinflation (2021-2022)',
    'POST': 'Post-TAS29 clean (2023-2026)',
    'OOS' : 'OOS tail (Jan-May 2026)',
}

# Per-period viable inference thresholds (number of monthly cross-sections):
#   FULL ~63 months → all factors fine
#   POST ~40 months → all but 36m beta fine
#   PRE  ~24 months → only 21d / 60d / 252d windows; skip 36m beta, fund.TTM
#   OOS  ~5 months  → diagnostic-only, no Test B inference
PERIOD_VIABILITY = {
    'FULL': {'all_methods': True},
    'PRE' : {'all_methods': True,  'skip_factors': ['R3_beta_36m', 'MKT_beta_36m']},
    'POST': {'all_methods': True},
    'OOS' : {'all_methods': False, 'note': 'too few months for inference; for OOS validation only'},
}


# ====================================================================
# 3. PIT LAGS
# ====================================================================
# Filing lag: from period_end to "info available" date.
# 60 days is conservative for BIST KAP filings (usually 45 day deadline +
# safety buffer for revisions).
LAG_FUNDAMENTALS_DAYS = 60
LAG_RETURNS_DAYS      = 1   # signal at t-1 → trade at t


# ====================================================================
# 4. ROLLING WINDOWS — for each anomaly that uses a rolling computation
# ====================================================================
WINDOWS = {
    # Beta windows (multiple horizons, picked in script 15)
    'beta_36m'        : 756,   # ~36 trading-day months
    'beta_12m'        : 252,
    'beta_6m'         : 126,
    # Idiosyncratic vol
    'ivol_60d'        : 60,
    'ivol_252d'       : 252,
    # Amihud illiquidity
    'amh_60d'         : 60,
    # Lottery preference (max daily ret over)
    'max_21d'         : 21,
    # Reversal & momentum (Brogaard standard)
    'rev_1m'          : 21,           # cumret last 21d
    'mom_12_1_start'  : 252,          # cumret_{t-252, t-22}
    'mom_12_1_skip'   : 22,
    # USDTRY / inflation betas
    'try_beta_60d'    : 60,
    'inf_beta_24m'    : 504,
}


# ====================================================================
# 5. INTRADAY STRUCTURE (BIST 30-min periods)
# ====================================================================
# BIST continuous trading: 10:00 → 18:00 Istanbul time.
# We drop 10:00–10:30 (open vol). Use 10:30 → 18:00 + overnight = 16 periods.
#
# Period schema:
#   P00 = overnight (close_yesterday → open_today, ≡ 18:00→10:30 next day)
#   P01 = 10:30–11:00
#   P02 = 11:00–11:30
#   ...
#   P15 = 17:30–18:00
INTRADAY = {
    'tz'              : 'Europe/Istanbul',
    'session_start'   : '10:30',
    'session_end'     : '18:00',
    'period_minutes'  : 30,
    'skip_open'       : True,           # drop 10:00-10:30
    'include_overnight': True,
    'n_periods_per_day': 16,            # 15 intraday + 1 overnight
    'use_midpoint'    : False,          # set True if BBO data feeds this
    'winsor_pct'      : 0.005,          # 0.5% per tail per Brogaard
}


# ====================================================================
# 6. CROSS-SECTIONAL REGRESSION (the Brogaard step)
# ====================================================================
XSEC = {
    'normalization'      : 'kns',          # rank → de-mean → /sum|dev|
    'min_stocks'         : 20,             # skip regressions below this
    'use_sectors'        : True,           # add sector dummies
    'intercept_in_resid' : True,           # per Brogaard Eq. 3
    'standardize_returns': False,          # raw 30-min returns
}

# Default factor set for the risk model in script 17.
# Edit this list to test different model specifications.
#
# Choice rationale for the BIST 2021-2026 sample (5.3 years):
# - R3b_beta_1y (NOT R3_beta_36m): 36-month rolling needs ~3yr warmup, leaving
#   only ~27 usable months. 1-year (252d) gives 51 usable months at 79% coverage.
# - R3c_beta_6m available as alternative (89% coverage, more responsive).
# - R5_rev_1m: short-horizon reversal, robust to small sample.
# - R12_acc passes Test A+B at t=3.1/2.9 — strongest accounting signal.
# - R23_try_beta_60d: USDTRY is critical BIST risk; 94% coverage.
DEFAULT_RISK_FACTORS = [
    'R1_size',          # log market cap
    'R3b_beta_1y',      # 1-year market beta (DEFAULT for short-sample work)
    'R4_mom_12_1',      # 12-1 momentum
    'R5_rev_1m',        # 1-month reversal
    'R15_ivol_60d',     # idiosyncratic vol (60d FF3 residual)
    'R16_amh_60d',      # Amihud illiquidity (60d)
    'R2_bm',            # book-to-market
    'R8_roa',           # ROA, TTM NI / lagged TA
    'R12_acc',          # accruals — strongest in Test A+B
    'R23_try_beta_60d', # USDTRY exposure
]
# Legacy 36-month beta version, kept for paper-replication comparisons:
DEFAULT_RISK_FACTORS_36M = [
    'R1_size', 'R3_beta_36m', 'R4_mom_12_1', 'R5_rev_1m',
    'R15_ivol_60d', 'R16_amh_60d', 'R2_bm', 'R8_roa', 'R6_ag',
]
# Optional additions when data quality permits:
EXTENDED_RISK_FACTORS = DEFAULT_RISK_FACTORS + [
    'R7_gp',             # gross profitability (non-banks)
    'R9_nsi',            # net stock issues
    'R6_ag',             # asset growth
    'R17_max_21d',       # lottery preference
]

# Round 18 daily champion — Phase-1 intraday default (daily-lagged X, 30-min y)
M05_INTRADAY_MODEL_ID = 'M05'
M05_INTRADAY_FACTORS = [
    'R5c_rev_5d', 'R46_mom_x_vol', 'R4b_mom_6_1', 'R21_turnover_60d',
    'R41_earnings_stability', 'R23b_try_beta_252d', 'R50_sector_xs_beta',
    'R35_try_dn_beta_60d', 'R5_rev_1m', 'R22c_coskew_60d',
]

# Top 10 factor models by daily rank_l1 / momentum / quintile Sharpe (Round 18 POST).
INTRADAY_TOP10_MODELS = [
    'M05', 'M16', 'M23', 'M20', 'M21', 'M17', 'M06', 'M10', 'M09', 'M15',
]
PERIODS_PER_YEAR_INTRADAY = INTRADAY['n_periods_per_day'] * 252  # 4032


# ====================================================================
# 7. PORTFOLIO SORT (the trade)
# ====================================================================
SORTS = {
    # Default sort granularity. BIST50 supports quintiles; BIST100 = deciles.
    'n_buckets'        : 5,        # quintiles (override to 10 for BIST100)
    'weighting'        : 'equal',  # 'equal' or 'value'
    'min_per_bucket'   : 4,        # require at least 4 stocks per bucket
    'sort_direction'   : 'asc',    # ascending → Q1=lowest, QN=highest
    'long_leg'         : 'Q1',     # for residual reversal: long low residual
    'short_leg'        : 'Q5',
}

# Rebalance frequencies for Method 4 portfolio sorts.
# Default is monthly (Brogaard/LYZ standard); weekly is more responsive
# but multiplies turnover; quarterly cuts turnover ~3x but loses signal
# updates between rebalances.
REBAL_FREQUENCIES = {
    'D': {'pandas_freq': 'B',     'n_per_year': 252, 'label': 'Daily (business day)'},
    'W': {'pandas_freq': 'W-FRI', 'n_per_year': 52,  'label': 'Weekly (Fri)'},
    'M': {'pandas_freq': 'ME',     'n_per_year': 12,  'label': 'Monthly'},
    'Q': {'pandas_freq': 'QE',     'n_per_year':  4,  'label': 'Quarterly'},
}
DEFAULT_REBAL = 'M'

# For LYZ systematic momentum, flip the legs:
LYZ_SORT_DIRECTION = {
    'long_leg' : 'Q5',     # long high systematic component
    'short_leg': 'Q1',
}


# ====================================================================
# 8. HOLDING PERIODS (for decay analysis)
# ====================================================================
# K = how many periods after formation we still hold.
# At intraday 30-min frequency, K=1 is "next 30 min", K=12 is "next 6 hours".
HOLDING_PERIODS_INTRADAY = [1, 2, 3, 6, 12, 16, 32]

# At daily frequency for monthly L/S sorts:
HOLDING_PERIODS_DAILY = [1, 5, 21, 63]   # 1d, 1w, 1mo, 3mo

# At monthly frequency for Stambaugh-style alpha tests:
HOLDING_PERIODS_MONTHLY = [1]   # Stambaugh standard


# ====================================================================
# 9. COST SCENARIOS (round-trip basis points per rebalance)
# ====================================================================
COSTS_BPS = [0, 5, 10, 15, 20, 25, 50]
DEFAULT_COST_BPS = 15   # BIST realistic execution


# ====================================================================
# 10. STATISTICAL TESTING
# ====================================================================
STATS = {
    'hac_method'          : 'newey_west',
    'hac_lag'             : 'auto',          # Andrews (1991) optimal
    't_stat_pass_thresh'  : 2.0,
    't_stat_borderline'   : 1.5,
    'bootstrap_iterations': 10_000,
    'bootstrap_block_size': 60,              # days
    'bootstrap_alpha'     : 0.05,
    # Test 3 sampling
    'n_sample_dates'      : 200,
    # Spanning regression test
    'spanning_min_obs'    : 100,
}


# ====================================================================
# 11. STAGE A — Stambaugh-style monthly tests (script 15)
# ====================================================================
STAGE_A = {
    'frequency'          : 'M',          # monthly L/S sort
    'n_buckets'          : 5,            # quintiles
    'benchmarks'         : ['CAPM', 'FF3'],   # alpha tests against
    'min_months'         : 24,
    'expected_sign_check': True,
}

# Pass thresholds for Test 1
PASS_T1 = {
    't_alpha_min'   : 2.0,    # |t(α)| ≥ 2 in Stambaugh-style FF3 test
    'sharpe_min'    : 0.50,   # annualized SR
    'sign_correct'  : True,
}

# Pass thresholds for Test 2 (Fama-MacBeth)
PASS_T2 = {
    't_slope_min'   : 2.0,    # |t(γ̄)| ≥ 2
    'sign_correct'  : True,
}

# Pass thresholds for Test 3 (risk model contribution)
PASS_T3 = {
    'delta_adj_r2'      : 0.005,   # R² gain ≥ 0.5pp
    'delta_resid_corr'  : -0.005,  # residual corr drops ≥ 0.5pp
    'either_suffices'   : True,    # OR semantics
}


# ====================================================================
# 12. UNIVERSE
# ====================================================================
UNIVERSE = {
    'name'        : 'BIST50',
    'reconstitute': 'end_of_june',     # Brogaard standard
    'min_price'   : 1.0,               # TRY, drop sub-1 lira (delisted)
    'min_volume'  : 100_000,           # TRY/day, liquidity floor
    'exclude'     : [],                # explicit exclusions
}


# ====================================================================
# 13. STANDARD REPORT SCHEMA
# ====================================================================
# Every factor goes through this. See shared.factor_report.
# Per-factor standardized report schema — aligned to guide Parts VI sections 23-29
REPORT_TABLES_A = [   # Test A + Test B outputs
    'T1_summary_stats',
    'T2_xsec_R2_per_period',    # R² time series + average
    'T3_xsec_slope_per_period', # b̄ time series
    'T4_FM_premium',            # λ̄ with controls
    'T5_IC_per_period',         # Spearman ρ
    'T6_decision_matrix',       # Test A pass + Test B pass → verdict
]

REPORT_FIGURES_A = [
    'F1_R2_time_series',
    'F2_slope_time_series',
    'F3_IC_time_series',
    'F4_R2_vs_slope_scatter',
    'F5_premium_with_CI',
]

REPORT_TABLES_C = [   # Long-short portfolio outputs (Method 4)
    'T1_quintile_returns',
    'T2_LS_stats',              # mean, t, SR, MDD, Calmar, etc.
    'T3_FF3_alpha',
    'T4_year_by_year',
    'T5_holding_decay',
    'T6_cost_sensitivity',
    'T7_double_sort',
    'T8_robustness',
]

REPORT_FIGURES_C = [
    'F1_cumulative_LS',
    'F2_quintile_bars',
    'F3_year_bars',
    'F4_rolling_sharpe',
    'F5_cost_survival',
    'F6_drawdown',
]

# Backwards-compat (the per-factor reporter from before)
REPORT_TABLES = REPORT_TABLES_C
REPORT_FIGURES = REPORT_FIGURES_C


# ====================================================================
# 14. ENGINEERING DEFAULTS
# ====================================================================
ENG = {
    'parquet_compression'   : 'snappy',
    'float_dtype'           : 'float32',
    'verbose'               : True,
    'random_seed'           : 42,
    'n_jobs'                : -1,
}


# ====================================================================
# Convenience accessor — single dict for backwards compat
# ====================================================================
# ====================================================================
# 15. TEST A + TEST B — the primary validation tests per guide Part IV
# ====================================================================
TEST_A = {
    # Method 2 cross-sectional regression, one per period (month).
    # R_{s,t} = a_t + b_t * X_{s,t-1} + eps_{s,t}
    'frequency'        : 'M',        # monthly cross-sections
    'min_stocks'       : 20,
    'r2_pass_min'      : 0.02,       # avg R² >= 2% per guide section 13
    'slope_t_min'      : 1.5,        # |t(b̄)| >= 1.5 (acceptable for BIST small N)
    'slope_t_strong'   : 2.0,        # >= 2.0 is stronger
    'use_lagged_X'     : True,       # X at t-1 predicts R at t
}

TEST_B = {
    # Fama-MacBeth with controls per guide section 13
    # Step 1: each month — R_{s,t} = lambda_0 + lambda_1 * X + Controls + eps
    # Step 2: average lambdas across months
    # Step 3: t-stat
    'frequency'        : 'M',
    'controls'         : ['R1_size', 'R2_bm'],  # size + B/M minimum, per guide
    'controls_extended': ['R1_size', 'R2_bm', 'R4_mom_12_1', 'R3_beta_36m'],
    't_pass_min'       : 1.5,        # |t| >= 1.5 acceptable (BIST small N caveat)
    't_pass_strong'    : 2.0,        # >= 2.0 standard
    'sign_check'       : True,
}

# Decision matrix per guide section 14
#   Pass A + Pass B → RISK FACTOR (use in Eq.1 + claim premium)
#   Pass A + Fail B → COMMON       (use in Eq.1 only; no premium claim)
#   Fail A + Pass B → SUSPICIOUS   (investigate; likely alpha/mispricing)
#   Fail A + Fail B → DROP


CFG = {
    'PERIODS'  : PERIODS,
    'WINDOWS'  : WINDOWS,
    'INTRADAY' : INTRADAY,
    'XSEC'     : XSEC,
    'SORTS'    : SORTS,
    'COSTS_BPS': COSTS_BPS,
    'STATS'    : STATS,
    'STAGE_A'  : STAGE_A,
    'TEST_A'   : TEST_A,
    'TEST_B'   : TEST_B,
    'UNIVERSE' : UNIVERSE,
    'REPORT_TABLES_A': REPORT_TABLES_A,
    'REPORT_FIGURES_A': REPORT_FIGURES_A,
    'REPORT_TABLES_C': REPORT_TABLES_C,
    'REPORT_FIGURES_C': REPORT_FIGURES_C,
    'DEFAULT_RISK_FACTORS' : DEFAULT_RISK_FACTORS,
}


def print_config_summary():
    """Print all config values for the run log."""
    print('=' * 70)
    print('CONFIG SNAPSHOT')
    print('=' * 70)
    print(f'Sample periods: {PERIODS}')
    print(f'Fundamental lag: {LAG_FUNDAMENTALS_DAYS}d')
    print(f'Intraday: {INTRADAY["n_periods_per_day"]} periods/day, '
          f'{INTRADAY["period_minutes"]}-min')
    print(f'Cross-section: KNS norm, min_stocks={XSEC["min_stocks"]}, '
          f'use_sectors={XSEC["use_sectors"]}')
    print(f'Sort: {SORTS["n_buckets"]} buckets, {SORTS["weighting"]}-weight')
    print(f'Costs (bps): {COSTS_BPS}')
    print(f'HAC: {STATS["hac_method"]}, lag={STATS["hac_lag"]}')
    print(f'Pass t-stat: ≥{STATS["t_stat_pass_thresh"]}')
    print(f'Default risk factors ({len(DEFAULT_RISK_FACTORS)}): '
          f'{DEFAULT_RISK_FACTORS}')
    print('=' * 70)


if __name__ == '__main__':
    print_config_summary()
