"""
shared.stats — Performance metrics, HAC standard errors, bootstrap, spanning regression.

Pure functions, no dependencies on data layout. Used by all factor and analysis scripts.

Functions:
    perf(returns)                          → dict of metrics
    newey_west_var(residuals, lag=None)    → HAC variance
    block_bootstrap(returns, ...)          → array of bootstrap Sharpes
    gate3_spanning(factor_ret, mkt_ret, label)  → dict of regression output
    multi_factor_regression(F_new, F_others_dict)  → dict of regression output
    quintile_returns(signal_wide, ret_wide, rebal_dates)  → DataFrame Q1-Q5
"""
import numpy as np
import pandas as pd

TDAYS = 252


def perf(r):
    """Performance summary with HAC-corrected t-stat."""
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return {'ann_ret': np.nan, 'vol': np.nan, 'sharpe': np.nan,
                'max_dd': np.nan, 'win_rate': np.nan, 'n': len(r),
                't_hac': np.nan, 'skew': np.nan, 'kurt': np.nan}
    cum = (1 + r).cumprod()
    tot = cum.iloc[-1] - 1
    yrs = len(r) / TDAYS
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(TDAYS)
    sharpe = ann / vol if vol > 0 else np.nan
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    win = (r > 0).mean()
    nw_var = newey_west_var(r - r.mean())
    t_hac = r.mean() / np.sqrt(nw_var / len(r)) if nw_var > 0 else np.nan
    return {
        'ann_ret':  ann,
        'vol':      vol,
        'sharpe':   sharpe,
        'max_dd':   mdd,
        'win_rate': win,
        'n':        len(r),
        't_hac':    t_hac,
        'skew':     r.skew(),
        'kurt':     r.kurtosis(),
    }


def newey_west_var(residuals, lag=None):
    """Newey-West HAC variance estimator (long-run variance of residuals).
    
    Returns LRV = E[u_t^2] + 2*sum_l w_l*E[u_t u_{t-l}], adjusted for autocorrelation.
    Uses Andrews (1991) optimal lag: lag = 4 × (T/100)^(2/9).
    
    Note: this returns the per-observation LRV. To get variance of the *mean*,
    divide by T. To get HAC standard error of an OLS coefficient β_i, use
    sqrt(LRV × (X'X)^{-1}_{ii})  — see gate3_spanning() and multi_factor_regression().
    """
    u = pd.Series(residuals).dropna().values
    T = len(u)
    if T < 2:
        return np.nan
    if lag is None:
        lag = max(1, int(4 * (T / 100) ** (2 / 9)))
    nw = (u ** 2).mean()
    for k in range(1, lag + 1):
        if k < T:
            nw += 2 * (1 - k / (lag + 1)) * (u[k:] * u[:-k]).mean()
    return nw


def block_bootstrap(returns, block_size=60, n_boot=10000, seed=42):
    """Moving block bootstrap of Sharpe ratio."""
    rng = np.random.default_rng(seed)
    arr = pd.Series(returns).dropna().values
    T = len(arr)
    if T < block_size + 1:
        return np.array([np.nan])
    nb = int(np.ceil(T / block_size))
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block_size + 1, size=nb)
        sample = np.concatenate([arr[s:s+block_size] for s in starts])[:T]
        mu, sigma = sample.mean(), sample.std(ddof=1)
        sharpes[i] = (mu * TDAYS) / (sigma * np.sqrt(TDAYS)) if sigma > 0 else np.nan
    return sharpes


def gate3_spanning(factor_ret, mkt_ret, label='factor'):
    """Spanning regression: factor_ret = α + β·market_ret + ε. Newey-West t-stats.
    
    HAC FORMULA (corrected 2026-05-07):
        Var(β̂)_HAC = LRV × (X'X)^{-1}
    where LRV = newey_west_var(residuals).
    
    SE(β̂_i) = sqrt(LRV × (X'X)^{-1}_{ii})
    
    For an intercept α with mean-zero regressors, (X'X)^{-1}_{0,0} ≈ 1/T,
    so SE(α̂) = sqrt(LRV/T), matching the perf() formula for t_HAC.
    
    PRIOR BUG: code had `sigma2_nw = nw_var * T` which gave SE that was √T too big.
    """
    f_clean = pd.Series(factor_ret).dropna()
    m_clean = pd.Series(mkt_ret).dropna()
    common = f_clean.index.intersection(m_clean.index)
    if len(common) < 30:
        return None
    f = f_clean.loc[common].values
    m = m_clean.loc[common].values
    valid = np.isfinite(f) & np.isfinite(m)
    f, m = f[valid], m[valid]
    if len(f) < 30:
        return None
    
    X = np.column_stack([np.ones(len(m)), m])
    beta, *_ = np.linalg.lstsq(X, f, rcond=None)
    resid = f - X @ beta
    T = len(f)
    
    # Newey-West long-run variance of residuals
    nw_var = newey_west_var(resid - resid.mean())
    if nw_var is None or not np.isfinite(nw_var) or nw_var <= 0:
        return None
    
    # HAC variance-covariance: LRV × (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_hac = nw_var * XtX_inv  # NO * T
    se_alpha = np.sqrt(cov_hac[0, 0])
    se_beta  = np.sqrt(cov_hac[1, 1])
    
    return {
        'config':       label,
        'n_days':       T,
        'alpha_daily':  beta[0],
        'alpha_annual': beta[0] * TDAYS,
        'beta_mkt':     beta[1],
        't_alpha':      beta[0] / se_alpha if se_alpha > 0 else np.nan,
        't_beta':       beta[1] / se_beta if se_beta > 0 else np.nan,
        'ir':           beta[0] / pd.Series(resid).std() * np.sqrt(TDAYS),
        'r_squared':    1 - resid.var() / f.var(),
    }


def multi_factor_regression(F_new, F_others_dict, mkt_ret=None, label='factor'):
    """Regress F_new on (1, market, F_others...) — uniqueness test.
    
    HAC FORMULA (corrected 2026-05-07):
        Var(β̂)_HAC = LRV × (X'X)^{-1}
    
    Args:
        F_new: pd.Series of factor returns to test
        F_others_dict: dict {name: pd.Series} of existing factors
        mkt_ret: optional market return series (added as control)
        label: identifier for output
    
    Returns dict with alpha, t_alpha (Newey-West), betas, R², per-regressor t-stats.
    """
    cols = {'__y__': F_new}
    if mkt_ret is not None:
        cols['mkt'] = mkt_ret
    for name, s in F_others_dict.items():
        cols[name] = s
    
    df = pd.concat(cols, axis=1).dropna()
    if len(df) < 30:
        return None
    
    y = df['__y__'].values
    X_names = ['const']
    X_cols = [np.ones(len(df))]
    if mkt_ret is not None:
        X_cols.append(df['mkt'].values)
        X_names.append('mkt')
    for name in F_others_dict:
        X_cols.append(df[name].values)
        X_names.append(name)
    X = np.column_stack(X_cols)
    
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[valid], X[valid]
    if len(y) < 30:
        return None
    
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    T = len(y)
    
    nw_var = newey_west_var(resid - resid.mean())
    if nw_var is None or not np.isfinite(nw_var) or nw_var <= 0:
        return None
    
    # HAC variance-covariance: LRV × (X'X)^{-1}
    cov_hac = nw_var * XtX_inv  # NO * T
    se = np.sqrt(np.maximum(np.diag(cov_hac), 0))
    
    out = {
        'config':       label,
        'n_days':       T,
        'alpha_daily':  beta[0],
        'alpha_annual': beta[0] * TDAYS,
        't_alpha':      beta[0] / se[0] if se[0] > 0 else np.nan,
        'r_squared':    1 - resid.var() / y.var() if y.var() > 0 else np.nan,
        'ir':           beta[0] / pd.Series(resid).std() * np.sqrt(TDAYS),
    }
    for i, name in enumerate(X_names[1:], start=1):
        out[f'beta_{name}'] = beta[i]
        out[f't_{name}'] = beta[i] / se[i] if se[i] > 0 else np.nan
    return out


def quintile_returns(sig_wide, ret_wide, rebal_dates, period_dates, n_quintiles=5):
    """Form Q1-Q5 portfolios at rebal_dates, compute mean returns over period_dates."""
    pos_per_q = {q: {} for q in range(1, n_quintiles + 1)}
    
    for rd in rebal_dates:
        if rd not in sig_wide.index:
            continue
        sig = sig_wide.loc[rd].dropna()
        if len(sig) < 2 * n_quintiles:
            continue
        ranks = sig.rank(method='first')
        q_size = len(sig) / n_quintiles
        q_assign = ((ranks - 1) // q_size + 1).clip(upper=n_quintiles).astype(int)
        for q in range(1, n_quintiles + 1):
            members = sig[q_assign == q].index.tolist()
            pos_per_q[q][rd] = members
    
    q_returns = {}
    for q in range(1, n_quintiles + 1):
        pos_dict = pos_per_q[q]
        if not pos_dict:
            q_returns[f'Q{q}'] = pd.Series(dtype=float)
            continue
        mask = pd.DataFrame(False, index=period_dates, columns=ret_wide.columns)
        rebal_keys = sorted(pos_dict.keys())
        for i, rk in enumerate(rebal_keys):
            next_rk = rebal_keys[i+1] if i+1 < len(rebal_keys) else period_dates[-1] + pd.Timedelta(days=1)
            in_period = (mask.index >= rk) & (mask.index < next_rk)
            for t in pos_dict[rk]:
                if t in mask.columns:
                    mask.loc[in_period, t] = True
        q_ret = ret_wide.reindex(period_dates).where(mask).mean(axis=1)
        q_returns[f'Q{q}'] = q_ret
    
    return pd.DataFrame(q_returns)


def correlation_matrix(returns_dict):
    """Pairwise correlation of factor returns."""
    df = pd.DataFrame(returns_dict)
    return df.corr()


if __name__ == '__main__':
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.0005, 0.015, 1000),
                   index=pd.date_range('2020-01-01', periods=1000, freq='B'))
    s = perf(r)
    print(f"perf test: Sharpe={s['sharpe']:.2f}, t_HAC={s['t_hac']:.2f}, MaxDD={s['max_dd']:.1%}")
    
    # Sanity check: gate3_spanning vs perf for intercept-only with zero market
    mkt = pd.Series(np.zeros(1000),
                     index=pd.date_range('2020-01-01', periods=1000, freq='B'))
    g3 = gate3_spanning(r, mkt + np.random.normal(0, 0.0001, 1000), 'self_test')
    print(f"gate3 test: alpha_daily={g3['alpha_daily']:.6f} (vs r.mean={r.mean():.6f})")
    print(f"gate3 test: t_alpha={g3['t_alpha']:.2f} (should be close to perf t_HAC={s['t_hac']:.2f})")
    
    bs = block_bootstrap(r, n_boot=1000)
    print(f"bootstrap: 5%={np.percentile(bs,5):.2f}, median={np.percentile(bs,50):.2f}, 95%={np.percentile(bs,95):.2f}")
    print("OK")
