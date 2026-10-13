"""
shared.test_AB — Test A (cross-sectional R²) + Test B (Fama-MacBeth premium)

Aligned to factor_methodology_guide.html Part IV, section 13.

TEST A (Method 2 — pure cross-sectional regression):
  For each period t (month):
    R_{s,t} = a_t + b_t · X_{s,t-1} + eps_{s,t}
  Output: per-period R², per-period slope b_t, then average across T months.
  Pass: avg(R²) > 2%  AND  |t(b̄)| > 1.5

TEST B (Method 3 — Fama-MacBeth with controls):
  For each period t:
    R_{s,t} = λ_0,t + λ_1,t · X_{s,t-1} + γ_t · Controls + eps_{s,t}
  Output: average λ̄_1, t-stat = λ̄_1 / (std(λ_1,t)/√T)
  Pass: |t(λ̄_1)| >= 1.5, with predicted sign
  Controls: size + B/M minimum (per guide)

Decision matrix:
   Pass A, Pass B  → RISK FACTOR
   Pass A, Fail B  → COMMON (still usable in Eq.1)
   Fail A, Pass B  → SUSPICIOUS (alpha-like)
   Fail A, Fail B  → DROP

Usage:
    from shared.test_AB import run_test_A, run_test_B, decision

    test_A = run_test_A(signal_panel, ret_monthly, factor='R4_mom_12_1')
    test_B = run_test_B(signal_panel, ret_monthly, factor='R4_mom_12_1',
                        controls=['R1_size', 'R2_bm'])
    verdict = decision(test_A, test_B)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as scstats

from .stats import newey_west_var
from . import configs as CFG


# ─────────────────────────────────────────────────────────────────────
# Frequency helpers — enable testing at D / W / M / Q
# ─────────────────────────────────────────────────────────────────────
# We allow Test A and Test B to run at different frequencies. The signal
# panel is automatically resampled to that frequency, returns are
# compounded into the same period buckets, then strict next-period
# semantics apply (signal at period P predicts return at P+1).
_FREQ_MAP = {
    'D': ('D', 1),       # daily — cross-section every trading day
    'W': ('W-FRI', 1),   # weekly, Friday-anchored
    'M': ('ME', 1),      # month-end
    'Q': ('QE', 1),      # quarter-end
}
_FREQ_LABEL = {
    'D': 'daily', 'W': 'weekly', 'M': 'monthly', 'Q': 'quarterly',
}
_FREQ_ANNUAL_PERIODS = {
    'D': 252, 'W': 52, 'M': 12, 'Q': 4,
}


def _to_period_offset(frequency):
    """Return the pandas period frequency string for a given freq code."""
    return {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q'}[frequency]


def _resample_signal_to_freq(sp, frequency, period_col, ticker_col, factor):
    """For W/M/Q: take the last available signal in each (ticker, period)."""
    if frequency == 'D':
        # Already daily; just ensure types
        sp[period_col] = pd.to_datetime(sp[period_col])
        return sp
    sp[period_col] = pd.to_datetime(sp[period_col])
    pcode = _to_period_offset(frequency)
    sp['_period'] = sp[period_col].dt.to_period(pcode)
    out = (sp.sort_values(period_col)
             .groupby([ticker_col, '_period'])[factor]
             .last()
             .reset_index())
    # period end as the canonical date
    if frequency == 'W':
        out[period_col] = out['_period'].dt.to_timestamp(how='end').dt.normalize()
    else:
        out[period_col] = out['_period'].dt.to_timestamp(how='end').dt.normalize()
    return out[[period_col, ticker_col, factor]]


def _compound_returns(ret_daily, frequency):
    """Compound a wide daily-returns DataFrame into the target frequency."""
    if frequency == 'D':
        return ret_daily.copy()
    pandas_freq = _FREQ_MAP[frequency][0]
    return (1 + ret_daily).resample(pandas_freq).prod() - 1


# ─────────────────────────────────────────────────────────────────────
# Test A: Cross-sectional R² + slope (Method 2)
# ─────────────────────────────────────────────────────────────────────
def run_test_A(signal_panel: pd.DataFrame,
               ret_monthly: pd.DataFrame,
               factor: str,
               min_stocks: int = None,
               period_col: str = 'date',
               ticker_col: str = 'ticker',
               frequency: str = 'M') -> dict:
    """Run Test A — cross-sectional regression of returns on X_{t-1} at the
    chosen frequency (D, W, M, Q).

    Parameters
    ----------
    signal_panel : DataFrame [date, ticker, <factor>]
        Daily signal panel — will be resampled to `frequency` if needed.
    ret_monthly : DataFrame (wide, date × ticker)
        Either daily or monthly returns. If `frequency != 'M'`, daily is
        required; this function will compound to the target frequency.
    factor : str
    frequency : str
        'D', 'W', 'M', 'Q'. Default 'M' (monthly, the legacy behavior).
    """
    if min_stocks is None:
        min_stocks = CFG.TEST_A['min_stocks']
    r2_pass = CFG.TEST_A['r2_pass_min']
    t_pass = CFG.TEST_A['slope_t_min']

    if frequency not in _FREQ_MAP:
        raise ValueError(f'frequency must be one of {list(_FREQ_MAP)}, got {frequency!r}')

    sp = signal_panel[[period_col, ticker_col, factor]].copy()
    sp[period_col] = pd.to_datetime(sp[period_col])
    sp = sp.dropna(subset=[factor])

    # If signal is daily and we want a coarser freq, resample (take last)
    sp = _resample_signal_to_freq(sp, frequency, period_col, ticker_col, factor)

    # Returns: compound to target frequency if not already there
    ret_monthly = ret_monthly.copy()
    ret_monthly.index = pd.to_datetime(ret_monthly.index).normalize()
    ret_resampled = _compound_returns(ret_monthly, frequency)

    rows = []
    ic_rows = []
    dates = sorted(sp[period_col].unique())

    # Strict next-period semantics at this frequency.
    # For W/M/Q we can use Period arithmetic. For D we must NOT use `Period('D')+1`
    # because that returns the next CALENDAR day (Saturday for a Friday signal),
    # which is absent from a trading-day-only index → all Friday signals skipped.
    # Instead we positionally take the next trading day with `searchsorted`.
    pcode = _to_period_offset(frequency)
    ret_periods = ret_resampled.copy()
    ret_periods.index = ret_periods.index.to_period(pcode)
    ret_dt_idx = pd.DatetimeIndex(ret_resampled.index)

    for d in dates:
        if frequency == 'D':
            next_pos = ret_dt_idx.searchsorted(pd.Timestamp(d), side='right')
            if next_pos >= len(ret_dt_idx):
                continue
            next_d = ret_dt_idx[next_pos]
        else:
            sig_period = pd.Timestamp(d).to_period(pcode)
            next_period = sig_period + 1
            if next_period not in ret_periods.index:
                continue
            next_pos = ret_periods.index.get_loc(next_period)
            next_d = next_period.to_timestamp(how='end').normalize()

        x = sp[sp[period_col] == d].set_index(ticker_col)[factor]
        y = ret_resampled.iloc[next_pos].reindex(x.index)
        valid = x.notna() & y.notna()
        x_v = x[valid].values
        y_v = y[valid].values
        n = len(x_v)
        if n < min_stocks:
            continue

        # Cross-sectional OLS: y = a + b*x + e
        x_c = x_v - x_v.mean()
        y_c = y_v - y_v.mean()
        ssx = (x_c ** 2).sum()
        if ssx == 0:
            continue
        b = (x_c * y_c).sum() / ssx
        a = y_v.mean() - b * x_v.mean()
        yhat = a + b * x_v
        resid = y_v - yhat
        ss_tot = (y_c ** 2).sum()
        ss_res = (resid ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # adj R² with K=1 regressor
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else np.nan

        # SE on slope
        se_resid = np.sqrt(ss_res / (n - 2)) if n > 2 else np.nan
        se_b = se_resid / np.sqrt(ssx) if ssx > 0 else np.nan
        t_b = b / se_b if (se_b is not None and se_b > 0) else np.nan

        # Spearman rank correlation = IC
        try:
            ic = scstats.spearmanr(x_v, y_v).statistic
        except Exception:
            ic = np.nan

        rows.append({
            'date': next_d,
            'R2': r2, 'R2_adj': r2_adj,
            'slope_b': b, 't_slope': t_b,
            'n_stocks': n,
        })
        ic_rows.append({'date': next_d, 'IC': ic})

    if not rows:
        return {
            'frequency': frequency,
            'per_period': pd.DataFrame(), 'avg_R2': np.nan, 'avg_R2_adj': np.nan,
            'avg_slope': np.nan, 't_avg_slope': np.nan, 'n_months': 0,
            'ic_per_period': pd.Series(dtype=float), 'avg_IC': np.nan,
            'ICIR': np.nan, 'hit_rate': np.nan,
            'pass_R2': False, 'pass_slope': False, 'pass_A': False,
        }

    df = pd.DataFrame(rows).set_index('date').sort_index()
    ic_s = pd.DataFrame(ic_rows).set_index('date').sort_index()['IC']

    # Aggregate stats
    avg_R2 = df['R2'].mean()
    avg_R2_adj = df['R2_adj'].mean()
    avg_slope = df['slope_b'].mean()

    # NW t-stat on average slope
    slope_centered = df['slope_b'].values - avg_slope
    nw_var = newey_west_var(slope_centered)
    t_avg = avg_slope / np.sqrt(nw_var / len(df)) if nw_var > 0 else np.nan

    avg_IC = ic_s.mean()
    icir = avg_IC / ic_s.std() * np.sqrt(len(ic_s)) if ic_s.std() > 0 else np.nan
    hit_rate = (np.sign(df['slope_b']) == np.sign(avg_slope)).mean() if avg_slope != 0 else np.nan

    pass_R2 = bool(avg_R2 > r2_pass) if not np.isnan(avg_R2) else False
    pass_slope = bool(abs(t_avg) > t_pass) if not np.isnan(t_avg) else False
    pass_A = pass_R2 and pass_slope

    return {
        'frequency': frequency,
        'per_period': df,
        'avg_R2': avg_R2,
        'avg_R2_adj': avg_R2_adj,
        'avg_slope': avg_slope,
        't_avg_slope': t_avg,
        'n_months': len(df),
        'ic_per_period': ic_s,
        'avg_IC': avg_IC,
        'ICIR': icir,
        'hit_rate': hit_rate,
        'pass_R2': pass_R2,
        'pass_slope': pass_slope,
        'pass_A': pass_A,
    }


# ─────────────────────────────────────────────────────────────────────
# Test B: Fama-MacBeth with controls (Method 3)
# ─────────────────────────────────────────────────────────────────────
def run_test_B(signal_panel: pd.DataFrame,
               ret_monthly: pd.DataFrame,
               factor: str,
               controls: list = None,
               period_col: str = 'date',
               ticker_col: str = 'ticker',
               frequency: str = 'M') -> dict:
    """Run Test B — Fama-MacBeth two-step with controls at the chosen frequency.

    Step 1: Each period — R_{s,t} = λ_0 + λ_1·X + γ·Controls + eps
    Step 2: λ̄_1 = (1/T) Σ λ_1,t
    Step 3: t = λ̄_1 / NW SE
    """
    if controls is None:
        controls = CFG.TEST_B['controls']
    # Remove factor from controls if it's there (self-collinearity)
    controls = [c for c in controls if c != factor and c in signal_panel.columns]
    t_pass = CFG.TEST_B['t_pass_min']

    if frequency not in _FREQ_MAP:
        raise ValueError(f'frequency must be one of {list(_FREQ_MAP)}, got {frequency!r}')

    cols_needed = [factor] + controls
    sp = signal_panel[[period_col, ticker_col] + cols_needed].copy()
    sp[period_col] = pd.to_datetime(sp[period_col])
    sp = sp.dropna(subset=cols_needed)

    # Resample signal to target frequency — take last value per (ticker, period)
    if frequency != 'D':
        pcode = _to_period_offset(frequency)
        sp['_period'] = sp[period_col].dt.to_period(pcode)
        sp = (sp.sort_values(period_col)
                .groupby([ticker_col, '_period'])[cols_needed]
                .last().reset_index())
        sp[period_col] = sp['_period'].dt.to_timestamp(how='end').dt.normalize()
        sp = sp.drop(columns=['_period'])

    ret_monthly = ret_monthly.copy()
    ret_monthly.index = pd.to_datetime(ret_monthly.index).normalize()
    ret_resampled = _compound_returns(ret_monthly, frequency)

    lambdas_factor = []
    lambdas_controls = {c: [] for c in controls}
    r2s = []
    r2_adj_per_period = []
    dates_used = []

    # See note in run_test_A: D mode uses positional next-trading-day shift to
    # avoid losing Friday signals (Period('D')+1 = Saturday ∉ index).
    pcode = _to_period_offset(frequency)
    ret_periods = ret_resampled.copy()
    ret_periods.index = ret_periods.index.to_period(pcode)
    ret_dt_idx = pd.DatetimeIndex(ret_resampled.index)

    for d in sorted(sp[period_col].unique()):
        if frequency == 'D':
            next_pos = ret_dt_idx.searchsorted(pd.Timestamp(d), side='right')
            if next_pos >= len(ret_dt_idx):
                continue
            next_d = ret_dt_idx[next_pos]
        else:
            sig_period = pd.Timestamp(d).to_period(pcode)
            next_period = sig_period + 1
            if next_period not in ret_periods.index:
                continue
            next_pos = ret_periods.index.get_loc(next_period)
            next_d = ret_resampled.index[next_pos]

        grp = sp[sp[period_col] == d].set_index(ticker_col)
        y = ret_resampled.iloc[next_pos].reindex(grp.index)
        merged = grp.copy()
        merged['_y'] = y
        merged = merged.dropna()
        n = len(merged)
        if n < (len(controls) + 5):
            continue

        Xm = np.column_stack([np.ones(n)] +
                              [merged[factor].values] +
                              [merged[c].values for c in controls])
        ym = merged['_y'].values
        try:
            beta = np.linalg.lstsq(Xm, ym, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        lambdas_factor.append(beta[1])
        for i, c in enumerate(controls):
            lambdas_controls[c].append(beta[2 + i])
        resid = ym - Xm @ beta
        ss_tot = ((ym - ym.mean()) ** 2).sum()
        ss_res = (resid ** 2).sum()
        r2_t = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # Adjusted R^2: penalty for k regressors (factor + len(controls))
        k = 1 + len(controls)
        r2_adj_t = (1 - (1 - r2_t) * (n - 1) / (n - k - 1)
                    if (not np.isnan(r2_t) and n > k + 1) else np.nan)
        r2s.append(r2_t)
        r2_adj_per_period.append(r2_adj_t)
        dates_used.append(next_d)

    if not lambdas_factor:
        return {
            'frequency': frequency,
            'lambdas': pd.DataFrame(), 'avg_lambda_factor': np.nan,
            't_lambda_factor': np.nan, 'avg_R2': np.nan,
            'avg_R2_adj': np.nan, 'n_factors_in_reg': 1 + len(controls),
            'n_months': 0, 'controls_used': controls,
            'pass_B': False, 'pass_B_strong': False,
        }

    lam_arr = np.array(lambdas_factor)
    mu = lam_arr.mean()
    nw_var = newey_west_var(lam_arr - mu)
    t_mu = mu / np.sqrt(nw_var / len(lam_arr)) if nw_var > 0 else np.nan

    # Per-control summary
    ctrl_summary = {}
    for c in controls:
        arr = np.array(lambdas_controls[c])
        c_mu = arr.mean()
        c_nw = newey_west_var(arr - c_mu)
        c_t = c_mu / np.sqrt(c_nw / len(arr)) if c_nw > 0 else np.nan
        ctrl_summary[c] = {'mean': c_mu, 't_NW': c_t}

    df = pd.DataFrame({'date': dates_used, 'lambda_factor': lambdas_factor})
    for c in controls:
        df[f'lambda_{c}'] = lambdas_controls[c]
    df['R2_with_controls'] = r2s
    df['R2_adj_with_controls'] = r2_adj_per_period
    df = df.set_index('date').sort_index()

    pass_B = bool(abs(t_mu) > t_pass) if not np.isnan(t_mu) else False
    pass_B_strong = bool(abs(t_mu) > CFG.TEST_B['t_pass_strong']) if not np.isnan(t_mu) else False

    return {
        'frequency': frequency,
        'lambdas': df,
        'avg_lambda_factor': mu,
        't_lambda_factor': t_mu,
        'avg_R2': float(np.mean(r2s)) if r2s else np.nan,
        'avg_R2_adj': float(np.nanmean(r2_adj_per_period)) if r2_adj_per_period else np.nan,
        'n_factors_in_reg': 1 + len(controls),
        'n_months': len(dates_used),
        'controls_used': controls,
        'controls_summary': ctrl_summary,
        'pass_B': pass_B,
        'pass_B_strong': pass_B_strong,
    }


# ─────────────────────────────────────────────────────────────────────
# Test C: Fama-MacBeth WITHOUT controls (pure univariate FM premium)
# ─────────────────────────────────────────────────────────────────────
# Mathematically equivalent to Test A's slope test (no R² gate). Reported in
# premium form (lambda_bar + t-stat) to pair with Test B and Test D.
def run_test_C(signal_panel: pd.DataFrame,
               ret_monthly: pd.DataFrame,
               factor: str,
               period_col: str = 'date',
               ticker_col: str = 'ticker',
               frequency: str = 'M') -> dict:
    """Test C — Fama-MacBeth premium with NO controls.

    Per period t:  R_{s,t+1} = lambda_0 + lambda_1 * X_{s,t} + eps
    Average lambda_1 across T periods, NW HAC t-stat.
    Pass: |t(lambda_bar)| > TEST_B.t_pass_min.

    Use this when you don't trust the standard size+B/M controls (e.g. in
    markets where those US-style controls don't behave normally — like BIST).
    """
    # Reuses run_test_B with empty controls — algebra is identical.
    res = run_test_B(signal_panel, ret_monthly, factor=factor,
                     controls=[], period_col=period_col,
                     ticker_col=ticker_col, frequency=frequency)
    # Rename pass fields to Test C semantics
    res = dict(res)
    res['pass_C'] = res.pop('pass_B')
    res['pass_C_strong'] = res.pop('pass_B_strong')
    return res


# ──────────────────────────────────────────────────────────────────
def incremental_R2_B_over_C(test_B_result: dict,
                            test_C_result: dict) -> dict:
    """Compute the incremental average R^2 from adding controls to Test C.

    Test C: R = lambda_0 + lambda_X * X + eps              (avg R^2 = R2_C)
    Test B: R = lambda_0 + lambda_X * X + gamma * Z + eps  (avg R^2 = R2_B)

    Returns delta_R2_B_over_C = R2_B - R2_C and delta_R2_adj similarly.

    Interpretation:
      - Large positive delta → controls (size, BM) add independent explanatory power.
      - Near-zero delta   → controls are redundant given the focal factor.
      - Negative delta_adj→ controls only inflate raw R^2; on an adj basis they
                            don't justify their degrees-of-freedom cost.
    """
    rB     = test_B_result.get('avg_R2',     np.nan)
    rB_adj = test_B_result.get('avg_R2_adj', np.nan)
    rC     = test_C_result.get('avg_R2',     np.nan)
    rC_adj = test_C_result.get('avg_R2_adj', np.nan)
    return {
        'avg_R2_C':     rC,
        'avg_R2_B':     rB,
        'delta_R2_B_over_C': (rB - rC) if (not np.isnan(rB) and not np.isnan(rC)) else np.nan,
        'avg_R2_adj_C': rC_adj,
        'avg_R2_adj_B': rB_adj,
        'delta_R2_adj_B_over_C':
            (rB_adj - rC_adj) if (not np.isnan(rB_adj) and not np.isnan(rC_adj)) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────
# Test D: Sequential Fama-MacBeth with already-accepted factors as controls
# ─────────────────────────────────────────────────────────────────────
ALIAS_GROUPS = [
    {'R3b_beta_1y', 'MKT_beta', 'MKT_beta_1y'},
    {'R3c_beta_6m', 'MKT_beta_6m'},
    {'R3_beta_36m', 'MKT_beta_36m'},
    {'R4_mom_12_1', 'F03_MOM_12_1'},
    {'R2_bm',       'F01_BM'},
]


def _alias_set(factor: str):
    """Return the alias-group containing `factor` (or {factor} alone)."""
    for grp in ALIAS_GROUPS:
        if factor in grp:
            return grp
    return {factor}


def _cross_sectional_corr(signal_panel: pd.DataFrame,
                          factors: list,
                          period_col: str = 'date',
                          ticker_col: str = 'ticker') -> pd.DataFrame:
    """Average cross-sectional Pearson correlation between every pair of factors.

    Done one period at a time (so we capture cross-sectional, not time-series,
    similarity). Used to detect data-driven near-duplicates not covered by the
    manual ALIAS_GROUPS list — e.g. two slightly differently-windowed beta
    columns that happen to be identical in this dataset.
    """
    sp = signal_panel[[period_col, ticker_col] + factors].copy()
    sp[period_col] = pd.to_datetime(sp[period_col])
    corrs = []
    for d, grp in sp.groupby(period_col):
        sub = grp[factors].dropna(how='any')
        if len(sub) < 10:
            continue
        c = sub.corr(method='pearson').values
        corrs.append(c)
    if not corrs:
        return pd.DataFrame(np.nan, index=factors, columns=factors)
    mean_c = np.nanmean(np.stack(corrs, axis=0), axis=0)
    return pd.DataFrame(mean_c, index=factors, columns=factors)


def _dedupe_factor_list(factors: list,
                        univariate_t: dict) -> list:
    """Return factors with ALIAS_GROUPS duplicates removed.

    `univariate_t` maps factor → |t(lambda_C)| from Test C. Within each alias
    group present in `factors`, keep the member with the largest |t_C|; ties
    broken alphabetically.
    """
    seen_keys = set()
    keep = []
    for f in factors:
        a = _alias_set(f)
        key = min(a)
        if key in seen_keys:
            continue
        members_in_input = [g for g in factors if g in a]
        if len(members_in_input) <= 1:
            keep.append(f)
            seen_keys.add(key)
            continue
        # multiple aliases present — keep best |t_C|
        ranked = sorted(members_in_input,
                        key=lambda g: (-abs(univariate_t.get(g, 0.0)), g))
        keep.append(ranked[0])
        seen_keys.add(key)
    return keep


def run_test_D_stepwise(signal_panel: pd.DataFrame,
                        ret_monthly: pd.DataFrame,
                        factors: list,
                        frequency: str = 'M',
                        t_pass: float = None,
                        delta_r2_min: float = 0.003,
                        corr_dup_thresh: float = 0.85,
                        max_rounds: int = 12,
                        do_backward: bool = True,
                        period_col: str = 'date',
                        ticker_col: str = 'ticker') -> dict:
    """Test D — stepwise FM with empirical controls (forward + backward).

    Improvement over `run_test_D_sequential`:

    1. PRE-FILTER
       - drop ALIAS_GROUPS duplicates (keep member with best univariate |t_C|)
       - drop data-driven duplicates (mean cross-sectional |corr| > corr_dup_thresh,
         keep the one with higher univariate |t_C|)

    2. FORWARD selection on INCREMENTAL R^2
       - at each round, for every candidate F not in selected set S,
         fit  R = lambda_0 + lambda_F * F + gamma * S + eps
         record joint avg R^2 and |t(lambda_F)|
       - winner = candidate that MAXIMIZES joint avg R^2 (not |t|),
         subject to (joint R^2 - baseline R^2) > delta_r2_min  AND  |t(lambda_F)| > t_pass
       - stop when no candidate satisfies both criteria

    3. BACKWARD elimination
       - for each F in S, refit without F and check whether
         (a) joint R^2 drops by less than delta_r2_min, OR
         (b) some other member of S now has |t| < t_pass
       - if (a), drop F (it's redundant given the rest)
       - if (b), keep all (the spec is jointly identified)

    Returns:
        {
          'selected_order' : final selected factors in pick order (post-backward),
          'pre_filter_drops': [(factor, reason), ...],
          'path'           : DataFrame of every (round, candidate) evaluated,
          'backward'       : DataFrame of every (factor, action_taken) in backward step,
          'method'         : 'stepwise_R2',
          'frequency'      : freq,
          't_pass'         : t_pass,
          'delta_r2_min'   : delta_r2_min,
        }
    """
    if t_pass is None:
        t_pass = CFG.TEST_B['t_pass_min']

    # ──────────────────────────────────────────────────────────────────
    # Step 0: filter to factors that exist in the panel
    # ──────────────────────────────────────────────────────────────────
    factors_in = [f for f in factors if f in signal_panel.columns]

    # ──────────────────────────────────────────────────────────────────
    # Step 1: compute univariate Test C |t| for ranking aliases/dup-pairs
    # ──────────────────────────────────────────────────────────────────
    univariate_t = {}
    for f in factors_in:
        try:
            tc = run_test_C(signal_panel, ret_monthly, factor=f,
                            period_col=period_col, ticker_col=ticker_col,
                            frequency=frequency)
            univariate_t[f] = tc.get('t_lambda_factor', np.nan)
        except Exception:
            univariate_t[f] = np.nan

    pre_filter_drops = []

    # ──────────────────────────────────────────────────────────────────
    # Step 2a: alias deduplication
    # ──────────────────────────────────────────────────────────────────
    deduped = _dedupe_factor_list(factors_in, univariate_t)
    for f in factors_in:
        if f not in deduped:
            kept = [g for g in deduped if g in _alias_set(f)]
            pre_filter_drops.append({
                'factor': f, 'reason': 'alias',
                'kept_instead': kept[0] if kept else '',
                'corr_to_kept': np.nan,
            })

    # ──────────────────────────────────────────────────────────────────
    # Step 2b: correlation-based dedup
    # ──────────────────────────────────────────────────────────────────
    if len(deduped) >= 2:
        c_mat = _cross_sectional_corr(signal_panel, deduped,
                                      period_col=period_col,
                                      ticker_col=ticker_col)
        to_drop_corr = set()
        pairs_seen = set()
        for i, a in enumerate(deduped):
            for j in range(i + 1, len(deduped)):
                b = deduped[j]
                if (a, b) in pairs_seen or (b, a) in pairs_seen:
                    continue
                pairs_seen.add((a, b))
                if a in to_drop_corr or b in to_drop_corr:
                    continue
                rho = c_mat.loc[a, b] if a in c_mat.index and b in c_mat.columns else np.nan
                if not np.isnan(rho) and abs(rho) > corr_dup_thresh:
                    ta, tb = abs(univariate_t.get(a, 0.0)), abs(univariate_t.get(b, 0.0))
                    losing = b if ta >= tb else a
                    keeping = a if losing == b else b
                    to_drop_corr.add(losing)
                    pre_filter_drops.append({
                        'factor': losing, 'reason': f'corr>{corr_dup_thresh}',
                        'kept_instead': keeping,
                        'corr_to_kept': float(rho),
                    })
        deduped = [f for f in deduped if f not in to_drop_corr]

    candidate_pool = list(deduped)

    # ──────────────────────────────────────────────────────────────────
    # Step 3: FORWARD selection on incremental R^2
    # ──────────────────────────────────────────────────────────────────
    selected = []
    baseline_r2 = 0.0
    path_rows = []

    for round_num in range(1, max_rounds + 1):
        remaining = [f for f in candidate_pool if f not in selected]
        if not remaining:
            break

        round_results = []
        for f in remaining:
            try:
                tb = run_test_B(signal_panel, ret_monthly,
                                factor=f, controls=list(selected),
                                period_col=period_col, ticker_col=ticker_col,
                                frequency=frequency)
            except Exception:
                continue
            t_lam = tb.get('t_lambda_factor', np.nan)
            joint_r2 = tb.get('avg_R2', np.nan)
            joint_r2_adj = tb.get('avg_R2_adj', np.nan)
            d_r2 = joint_r2 - baseline_r2 if not np.isnan(joint_r2) else np.nan
            round_results.append({
                'round': round_num,
                'factor': f,
                'controls': '|'.join(selected) if selected else '(none)',
                'n_controls': len(selected),
                'k_total_in_reg': 1 + len(selected),
                'lambda': tb.get('avg_lambda_factor', np.nan),
                't_lambda': t_lam,
                'joint_R2': joint_r2,
                'joint_R2_adj': joint_r2_adj,
                'delta_R2': d_r2,
                'baseline_R2': baseline_r2,
                'n_periods': tb.get('n_months', 0),
            })

        if not round_results:
            break

        rr = pd.DataFrame(round_results)
        # Eligibility: joint pass + sufficient incremental R^2
        rr['t_ok']   = rr['t_lambda'].abs() > t_pass
        rr['dr2_ok'] = rr['delta_R2'] > delta_r2_min
        rr['eligible'] = rr['t_ok'] & rr['dr2_ok']

        eligibles = rr[rr['eligible']].copy()
        if eligibles.empty:
            for _, r in rr.iterrows():
                path_rows.append({**r.to_dict(),
                                  'selected': False,
                                  'note': 'forward_stop_no_eligible'})
            break

        # Winner = max delta_R2 among eligibles (R^2-greedy)
        eligibles = eligibles.sort_values('delta_R2', ascending=False)
        winner = eligibles.iloc[0]
        winner_factor = winner['factor']

        for _, r in rr.iterrows():
            path_rows.append({**r.to_dict(),
                              'selected': (r['factor'] == winner_factor),
                              'note': f'forward_round_{round_num}'})

        selected.append(winner_factor)
        baseline_r2 = float(winner['joint_R2'])
        # also track adj for diagnostics; not used as the selection criterion
        baseline_r2_adj = float(winner.get('joint_R2_adj', np.nan))

    forward_selected = list(selected)

    # ──────────────────────────────────────────────────────────────────
    # Step 4: BACKWARD elimination
    # ──────────────────────────────────────────────────────────────────
    backward_rows = []
    if do_backward and len(selected) >= 2:
        # Reference: joint R^2 of the FULL selected set
        # Use Test B on the *last-picked* factor with all others as controls
        # for the reference; same R^2 as any rearrangement.
        last = selected[-1]
        rest = selected[:-1]
        try:
            ref = run_test_B(signal_panel, ret_monthly,
                             factor=last, controls=list(rest),
                             period_col=period_col, ticker_col=ticker_col,
                             frequency=frequency)
            full_r2 = ref.get('avg_R2', np.nan)
        except Exception:
            full_r2 = np.nan

        for f in list(selected):
            others = [g for g in selected if g != f]
            if not others:
                backward_rows.append({
                    'factor': f, 'action': 'kept_singleton',
                    'r2_with_f': full_r2, 'r2_without_f': np.nan,
                    'r2_loss': np.nan, 't_in_full': np.nan,
                })
                continue
            try:
                tb_full = run_test_B(signal_panel, ret_monthly,
                                     factor=f, controls=[g for g in selected if g != f],
                                     period_col=period_col, ticker_col=ticker_col,
                                     frequency=frequency)
                t_in_full = tb_full.get('t_lambda_factor', np.nan)
                r2_with = tb_full.get('avg_R2', np.nan)

                # baseline without f: refit any remaining factor with others as controls
                ref_remain = run_test_B(
                    signal_panel, ret_monthly,
                    factor=others[-1], controls=others[:-1],
                    period_col=period_col, ticker_col=ticker_col,
                    frequency=frequency,
                )
                r2_without = ref_remain.get('avg_R2', np.nan)
            except Exception:
                t_in_full = np.nan
                r2_with = np.nan
                r2_without = np.nan

            r2_loss = (r2_with - r2_without) if (not np.isnan(r2_with) and
                                                  not np.isnan(r2_without)) else np.nan

            # Drop rule (conservative AND): drop only if factor fails BOTH
            # criteria — its own |t| is now insignificant AND removing it loses
            # less than delta_r2_min of joint R^2. This avoids ejecting a
            # factor that explains substantial variance just because its
            # marginal slope is borderline (and vice versa).
            drop_for_t  = (not np.isnan(t_in_full)) and abs(t_in_full) < t_pass
            drop_for_r2 = (not np.isnan(r2_loss))   and r2_loss        < delta_r2_min

            if drop_for_t and drop_for_r2:
                action = 'dropped_redundant'
                selected = [g for g in selected if g != f]
            else:
                action = 'kept'

            backward_rows.append({
                'factor': f,
                'action': action,
                'r2_with_f': r2_with,
                'r2_without_f': r2_without,
                'r2_loss': r2_loss,
                't_in_full': t_in_full,
            })

    path_df = pd.DataFrame(path_rows)
    backward_df = pd.DataFrame(backward_rows)
    pre_filter_df = pd.DataFrame(pre_filter_drops) if pre_filter_drops else pd.DataFrame(
        columns=['factor', 'reason', 'kept_instead', 'corr_to_kept'])

    # Final-set joint R^2 (raw + adj) — fit one Test B with the LAST selected
    # factor and the rest as controls. Same R^2 regardless of which is focal.
    final_R2 = np.nan
    final_R2_adj = np.nan
    if selected:
        try:
            last = selected[-1]
            rest = selected[:-1]
            ref = run_test_B(signal_panel, ret_monthly,
                             factor=last, controls=list(rest),
                             period_col=period_col, ticker_col=ticker_col,
                             frequency=frequency)
            final_R2 = ref.get('avg_R2', np.nan)
            final_R2_adj = ref.get('avg_R2_adj', np.nan)
        except Exception:
            pass

    return {
        'method': 'stepwise_R2',
        'selected_order': selected,                  # post-backward
        'selected_order_pre_backward': forward_selected,
        'pre_filter_drops': pre_filter_df,
        'path': path_df,
        'backward': backward_df,
        'frequency': frequency,
        't_pass': t_pass,
        'delta_r2_min': delta_r2_min,
        'corr_dup_thresh': corr_dup_thresh,
        'final_joint_R2': final_R2,
        'final_joint_R2_adj': final_R2_adj,
        'final_n_factors': len(selected),
    }


def run_test_D_sequential(signal_panel: pd.DataFrame,
                          ret_monthly: pd.DataFrame,
                          factors: list,
                          frequency: str = 'M',
                          t_pass: float = None,
                          max_rounds: int = 12,
                          period_col: str = 'date',
                          ticker_col: str = 'ticker') -> dict:
    """Test D — Stepwise/greedy FM with empirical controls.

    Procedure:
        Round 1: run univariate FM (Test C) on every factor in `factors`.
                 Pick the factor with the largest |t(lambda)| that passes |t|>t_pass.
        Round k: for every remaining factor F, run FM with F as the test and the
                 already-selected factors as controls. Pick the strongest passer.
        Stop when no remaining factor passes |t| > t_pass.

    Aliases (identical columns like R3b_beta_1y == MKT_beta) are deduplicated:
    once any alias is picked, all aliases are dropped from `remaining` to avoid
    perfect collinearity in subsequent rounds.

    Returns:
        {
          'selected_order': [factor1, factor2, ...],  # in pick order
          'path': DataFrame with one row per (round, candidate_factor) showing
                  lambda, t_lambda, controls used, and selected/not-selected,
          'final_round': int,
        }
    """
    if t_pass is None:
        t_pass = CFG.TEST_B['t_pass_min']

    # Deduplicate input factors: keep canonical name from each alias group
    seen_aliases = set()
    canonical = []
    for f in factors:
        a = _alias_set(f)
        key = min(a)  # canonical = alphabetically first member
        if key in seen_aliases:
            continue
        if f in a:
            canonical.append(f)
            seen_aliases.add(key)

    selected = []
    remaining = list(canonical)
    path_rows = []
    final_round = 0

    for round_num in range(1, max_rounds + 1):
        if not remaining:
            break
        round_results = []
        for f in remaining:
            try:
                tb = run_test_B(signal_panel, ret_monthly,
                                factor=f, controls=list(selected),
                                period_col=period_col, ticker_col=ticker_col,
                                frequency=frequency)
            except Exception:
                continue
            t_lam = tb.get('t_lambda_factor', np.nan)
            round_results.append({
                'round': round_num,
                'factor': f,
                'controls': '|'.join(selected) if selected else '(none)',
                'n_controls': len(selected),
                'lambda': tb.get('avg_lambda_factor', np.nan),
                't_lambda': t_lam,
                'n_periods': tb.get('n_months', 0),
                'passes': bool((not np.isnan(t_lam)) and abs(t_lam) > t_pass),
            })
        if not round_results:
            break
        rr = pd.DataFrame(round_results)
        # Rank by |t_lambda| descending (NaN goes last)
        rr['abs_t'] = rr['t_lambda'].abs()
        rr = rr.sort_values('abs_t', ascending=False)

        passers = rr[rr['passes']]
        if len(passers) == 0:
            # Record this round but no winner; mark all as not-selected and stop
            for _, r in rr.iterrows():
                path_rows.append({**r.to_dict(), 'selected': False,
                                   'note': 'no_passer_round_stop'})
            final_round = round_num - 1
            break

        winner = passers.iloc[0]
        winner_factor = winner['factor']
        for _, r in rr.iterrows():
            path_rows.append({
                **r.to_dict(),
                'selected': (r['factor'] == winner_factor),
                'note': f'round_{round_num}',
            })

        # Drop the winner's entire alias group from remaining
        alias_grp = _alias_set(winner_factor)
        remaining = [f for f in remaining if f not in alias_grp]
        selected.append(winner_factor)
        final_round = round_num

    path_df = pd.DataFrame(path_rows)
    if 'abs_t' in path_df.columns:
        path_df = path_df.drop(columns=['abs_t'])

    return {
        'selected_order': selected,
        'path': path_df,
        'final_round': final_round,
        'frequency': frequency,
        't_pass': t_pass,
    }


# ─────────────────────────────────────────────────────────────────────
# Decision matrix
# ─────────────────────────────────────────────────────────────────────
def decision(test_A_result: dict, test_B_result: dict) -> dict:
    """Apply the decision matrix per guide section 14."""
    pA = test_A_result.get('pass_A', False)
    pB = test_B_result.get('pass_B', False)

    if pA and pB:
        verdict = 'RISK_FACTOR'
        action = 'Use as regressor in Eq.1 AND claim risk premium'
    elif pA and not pB:
        verdict = 'COMMON'
        action = 'Useful in Eq.1 to soak up co-movement; no premium claim'
    elif not pA and pB:
        verdict = 'SUSPICIOUS'
        action = 'Predicts returns but no co-movement — likely alpha/mispricing or measurement issue'
    else:
        verdict = 'DROP'
        action = 'No useful information about Turkish stock returns'

    return {
        'verdict': verdict,
        'action': action,
        'pass_A': pA,
        'pass_B': pB,
        'avg_R2': test_A_result.get('avg_R2', np.nan),
        't_slope_A': test_A_result.get('t_avg_slope', np.nan),
        'avg_lambda_B': test_B_result.get('avg_lambda_factor', np.nan),
        't_lambda_B': test_B_result.get('t_lambda_factor', np.nan),
    }


def decision_AC(test_A_result: dict, test_C_result: dict) -> dict:
    """BIST-native decision matrix using Test A + Test C (no controls).

    Justification: standard size+B/M controls used by Test B are a US-Fama-French
    convention. In BIST50, R2_bm is on the drop list (the value premium INVERTS)
    and R1_size has POST-period sign issues. Imposing them as controls
    contaminates Test B's verdict. Test C tests the raw FM premium without
    those imported controls.

    Rule (same logic, different second test):
        Pass A + Pass C → RISK_FACTOR
        Pass A + Fail C → COMMON
        Fail A + Pass C → SUSPICIOUS
        Fail A + Fail C → DROP

    This is the HEADLINE verdict for the post-Round-13 pipeline; decision()
    (A+B) is reported as a robustness check only.
    """
    pA = test_A_result.get('pass_A', False)
    pC = test_C_result.get('pass_C', False)

    if pA and pC:
        verdict = 'RISK_FACTOR'
        action = 'BIST-native risk factor: explains XS variance + earns FM premium without imported controls'
    elif pA and not pC:
        verdict = 'COMMON'
        action = 'Co-moves with returns but no clean FM premium — use as control, not premium'
    elif not pA and pC:
        verdict = 'SUSPICIOUS'
        action = 'Predicts returns but does not explain XS variance — likely alpha/mispricing'
    else:
        verdict = 'DROP'
        action = 'No useful information about Turkish stock returns'

    return {
        'verdict_AC': verdict,
        'action_AC': action,
        'pass_A': pA,
        'pass_C': pC,
        'avg_R2': test_A_result.get('avg_R2', np.nan),
        'avg_R2_adj': test_A_result.get('avg_R2_adj', np.nan),
        't_slope_A': test_A_result.get('t_avg_slope', np.nan),
        'avg_lambda_C': test_C_result.get('avg_lambda_factor', np.nan),
        't_lambda_C': test_C_result.get('t_lambda_factor', np.nan),
    }
