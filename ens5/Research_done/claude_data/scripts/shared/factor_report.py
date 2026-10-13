"""
shared.factor_report — Standardized table + figure generation for any factor.

The contract: given a factor's signal panel (date × ticker → score) plus
returns, produce the 10 standard tables and 5 standard figures listed in
shared.configs.REPORT_TABLES / REPORT_FIGURES.

This is what enables the iteration loop:
  add new factor in script 14 → call FactorReport(factor_name, ...) → done

Usage:
    from shared.factor_report import FactorReport
    fr = FactorReport(
        factor_name='R4_mom_12_1',
        signal_panel=panel,        # cols: date, ticker, factor
        returns=ret_d,             # wide df: date × ticker
        returns_monthly=ret_m,
        ff_factors=ff3_d,          # daily MKT, SMB, HML
        out_dir=RESULTS_DIR / 'factors' / 'R4_mom_12_1',
    )
    fr.run_all()       # produces every table + figure
    fr.summary()       # one-row pass/fail verdict
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .stats import newey_west_var, perf
from .plotting import setup_plot_style, PALETTE, VERDICT_COLORS, verdict_color
from . import configs as CFG

ANNUAL_FACTOR_DAILY = 252
ANNUAL_FACTOR_MONTHLY = 12


# ====================================================================
# Helper: HAC t-stat
# ====================================================================
def hac_tstat(series, lag='auto'):
    """Newey-West t-stat for a series's mean."""
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return np.nan, np.nan, np.nan
    mu = s.mean()
    if lag == 'auto':
        lag = max(1, int(np.floor(4 * (len(s) / 100) ** (2/9))))
    var = newey_west_var(s.values - mu, lag)
    se = np.sqrt(var / len(s))
    t = mu / se if se > 0 else np.nan
    return mu, se, t


def hac_ols(y, X, lag='auto'):
    """OLS with HAC (Newey-West) covariance for the coefficient vector.

    Returns (beta, se, t) where each is np.ndarray of length k+1
    (intercept first when X already includes a constant column).

    Uses the sandwich form:
        Var(beta_hat) = (X'X)^-1 . S . (X'X)^-1
    where S = sum_{ell=-L}^{L} w(ell) * X'_t e_t e_{t+ell} X_{t+ell}
    and w(ell) = 1 - |ell|/(L+1)  (Bartlett kernel).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k + 2:
        return None, None, None
    if lag == 'auto':
        lag = max(1, int(np.floor(4 * (n / 100) ** (2/9))))

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None, None, None
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Build S = X'X scaled by residuals' long-run variance
    # Compute sum_{ell=-L}^{L} w(ell) * Gamma(ell)
    # Gamma(ell) = sum_t (X_t e_t)(X_{t+ell} e_{t+ell})'  for ell >= 0
    Xe = X * resid[:, None]   # shape (n, k)
    S = Xe.T @ Xe             # ell=0
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)
        Gamma_ell = Xe[ell:].T @ Xe[:-ell]  # shape (k, k)
        S += w * (Gamma_ell + Gamma_ell.T)

    var_beta = XtX_inv @ S @ XtX_inv
    diag = np.diag(var_beta)
    se = np.sqrt(np.maximum(diag, 0))
    t = beta / se
    return beta, se, t


def annualize_sr(daily_returns, freq=252):
    r = pd.Series(daily_returns).dropna()
    if len(r) < 5 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(freq)


# ====================================================================
# THE STANDARDIZED REPORT
# ====================================================================
class FactorReport:
    """
    Produces 10 standardized tables + 5 standardized figures for a factor.

    Parameters
    ----------
    factor_name : str
        e.g. 'R4_mom_12_1' or 'F03'
    signal_panel : DataFrame
        long format: date, ticker, <factor_name>
    returns : DataFrame
        wide daily returns: date × ticker
    returns_monthly : DataFrame, optional
        wide monthly returns; computed if omitted
    ff_factors : DataFrame, optional
        daily df with cols MKT, SMB, HML (and 'rf' if available)
    out_dir : Path
        where tables & figs are written
    sort_direction : str
        'asc' (default, Brogaard reversal: long Q1) or 'desc' (LYZ momentum: long Q5)
    """

    def __init__(self, factor_name, signal_panel, returns,
                 returns_monthly=None, ff_factors=None,
                 out_dir=None, sort_direction='asc',
                 n_buckets=None, costs_bps=None,
                 holding_periods_daily=None,
                 period_label=None,
                 rebal_freq='M'):
        """
        rebal_freq : str
            'W' weekly, 'M' monthly (default, Brogaard standard), or 'Q' quarterly.
            Drives both the signal-resampling frequency and the L/S return
            compounding step. Cost-per-rebalance is scaled accordingly.
        """
        self.name = factor_name
        self.signal = signal_panel.copy()
        self.signal['date'] = pd.to_datetime(self.signal['date'])
        self.ret_d = returns

        # Resample returns to chosen rebal frequency
        if rebal_freq not in CFG.REBAL_FREQUENCIES:
            raise ValueError(f'rebal_freq must be one of {list(CFG.REBAL_FREQUENCIES)}, got {rebal_freq!r}')
        self.rebal_freq = rebal_freq
        self.rebal_meta = CFG.REBAL_FREQUENCIES[rebal_freq]

        # Compound daily returns into the chosen rebalance bucket
        pandas_freq = self.rebal_meta['pandas_freq']
        if returns_monthly is not None and rebal_freq == 'M':
            self.ret_periodic = returns_monthly
        else:
            self.ret_periodic = (1 + returns).resample(pandas_freq).prod() - 1
        # Keep `self.ret_m` for backwards compat — points to ret_periodic
        self.ret_m = self.ret_periodic

        self.ff = ff_factors
        self.out = Path(out_dir) if out_dir else CFG.RESULTS_DIR / 'factors' / factor_name
        self.fig_out = self.out / 'figures'
        self.tab_out = self.out / 'tables'
        self.fig_out.mkdir(parents=True, exist_ok=True)
        self.tab_out.mkdir(parents=True, exist_ok=True)

        self.sort_dir = sort_direction
        self.n_buckets = n_buckets or CFG.SORTS['n_buckets']
        self.costs = costs_bps or CFG.COSTS_BPS
        self.holding_d = holding_periods_daily or CFG.HOLDING_PERIODS_DAILY
        self.period_label = period_label or CFG.DEFAULT_PERIOD
        # Annualization factor for the chosen frequency
        self.ann_factor = self.rebal_meta['n_per_year']

        # Filter to declared period
        s, e = CFG.PERIODS[self.period_label]
        self.signal = self.signal[(self.signal['date'] >= s) & (self.signal['date'] <= e)]
        self.ret_d = self.ret_d.loc[s:e] if isinstance(self.ret_d.index, pd.DatetimeIndex) else self.ret_d
        self.ret_periodic = self.ret_periodic.loc[s:e] if isinstance(self.ret_periodic.index, pd.DatetimeIndex) else self.ret_periodic
        self.ret_m = self.ret_periodic

        setup_plot_style()
        self.results = {}

    # ─────────────────────────────────────────────────────────────────
    # T1: Summary stats
    # ─────────────────────────────────────────────────────────────────
    def t1_summary_stats(self):
        col = self.name
        s = self.signal[col].dropna()
        # Cross-sectional autocorr (per stock, then averaged)
        ac1 = (self.signal.groupby('ticker')[col]
               .apply(lambda x: x.autocorr(lag=1)).mean())
        # Coverage (% of (ticker, date) pairs non-null)
        total = self.signal.shape[0]
        cov = s.shape[0] / total if total > 0 else np.nan

        out = pd.DataFrame({
            'metric'  : ['n_obs', 'mean', 'std', 'min', 'p25', 'median',
                         'p75', 'max', 'skew', 'kurt', 'autocorr_lag1',
                         'coverage_pct', 'n_tickers', 'n_dates',
                         'period_start', 'period_end'],
            'value'   : [int(s.shape[0]), s.mean(), s.std(), s.min(),
                         s.quantile(0.25), s.median(), s.quantile(0.75),
                         s.max(), s.skew(), s.kurt(), ac1,
                         100 * cov, self.signal['ticker'].nunique(),
                         self.signal['date'].nunique(),
                         str(self.signal['date'].min().date()),
                         str(self.signal['date'].max().date())],
        })
        out.to_csv(self.tab_out / f'T1_summary_stats_{self.name}.csv', index=False)
        self.results['T1'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # T2: Cross-sectional correlation (this factor vs all others in panel)
    # ─────────────────────────────────────────────────────────────────
    def t2_correlations(self, other_factors=None):
        if other_factors is None or len(other_factors) == 0:
            self.results['T2'] = pd.DataFrame()
            return self.results['T2']
        cols = [self.name] + [f for f in other_factors if f != self.name]
        wide = self.signal.pivot_table(index=['date', 'ticker'],
                                        values=cols).dropna(how='all')
        pearson = wide.corr(method='pearson')
        spearman = wide.corr(method='spearman')
        out = pd.DataFrame({
            'pair': [f'{a}|{b}' for a in cols for b in cols if a < b],
            'pearson':  [pearson.loc[a, b] for a in cols for b in cols if a < b],
            'spearman': [spearman.loc[a, b] for a in cols for b in cols if a < b],
        })
        out.to_csv(self.tab_out / f'T2_correlations_{self.name}.csv', index=False)
        self.results['T2'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # T3: Cross-sectional regression of next-period return on this factor
    # Per-date univariate FM, then averaged over time.
    # ─────────────────────────────────────────────────────────────────
    def _resample_signal(self):
        """Resample signal panel to chosen rebal frequency.
        Takes LAST value per (ticker, period). NO LOOK-AHEAD: signal at period P
        is used for period P+1's trade."""
        col = self.name
        pcode = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q'}[self.rebal_freq]
        sig = self.signal.copy()
        sig['_p'] = sig['date'].dt.to_period(pcode)
        sig = (sig.sort_values('date')
                  .groupby(['ticker', '_p'])[col].last()
                  .reset_index())
        sig['date'] = sig['_p'].dt.to_timestamp(how='end').dt.normalize()
        return sig[['date', 'ticker', col]]

    def t3_xsec_regression(self):
        col = self.name
        sig = self._resample_signal()
        ret_per = self.ret_periodic.copy()
        ret_per.index = pd.to_datetime(ret_per.index).normalize()
        sig['date'] = pd.to_datetime(sig['date']).dt.normalize()
        pcode = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q'}[self.rebal_freq]
        ret_p_idx = ret_per.copy()
        ret_p_idx.index = ret_p_idx.index.to_period(pcode)
        # For daily, also keep a sorted DatetimeIndex of trading days so we can
        # take the *next trading day* (skipping weekends / TR holidays) without
        # relying on calendar 'D' period arithmetic.
        ret_dt_idx = pd.DatetimeIndex(ret_per.index)

        slopes, r2s, ns = [], [], []
        for d in sorted(sig['date'].unique()):
            if self.rebal_freq == 'D':
                pos = ret_dt_idx.searchsorted(pd.Timestamp(d), side='right')
                if pos >= len(ret_dt_idx):
                    continue
                next_pos = pos
            else:
                sig_p = pd.Timestamp(d).to_period(pcode)
                nxt = sig_p + 1
                if nxt not in ret_p_idx.index:
                    continue
                next_pos = ret_p_idx.index.get_loc(nxt)
            fwd = ret_per.iloc[next_pos]
            grp = sig[sig['date'] == d].set_index('ticker')[col]
            x = grp.dropna()
            y = fwd.reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(y) < 10:
                continue
            x_d = x - x.mean()
            y_d = y - y.mean()
            denom = (x_d ** 2).sum()
            if denom == 0:
                continue
            slope = (x_d * y_d).sum() / denom
            yhat = x_d * slope
            r2 = 1 - ((y_d - yhat) ** 2).sum() / ((y_d ** 2).sum() + 1e-12)
            slopes.append(slope)
            r2s.append(r2)
            ns.append(len(y))
        if not slopes:
            self.results['T3'] = pd.DataFrame()
            return self.results['T3']
        slopes = np.array(slopes)
        mu, se, t = hac_tstat(slopes)
        out = pd.DataFrame({
            'metric': ['mean_slope', 'se_NW', 't_NW', 'n_periods',
                       'mean_xsec_R2', 'mean_n_stocks', 'rebal_freq'],
            'value' : [mu, se, t, len(slopes), np.mean(r2s), np.mean(ns),
                       self.rebal_freq],
        })
        out.to_csv(self.tab_out / f'T3_xsec_regression_{self.name}.csv', index=False)
        self.results['T3'] = out
        self.results['_T3_slopes'] = pd.Series(slopes)
        return out

    # ─────────────────────────────────────────────────────────────────
    # T4: Quintile portfolio sort (monthly), L/S premium + FF3 alpha
    # ─────────────────────────────────────────────────────────────────
    def t4_quintile_sort(self):
        col = self.name
        Q = self.n_buckets
        sig = self._resample_signal()
        ret_per = self.ret_periodic.copy()
        ret_per.index = pd.to_datetime(ret_per.index).normalize()
        sig['date'] = pd.to_datetime(sig['date']).dt.normalize()
        pcode = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q'}[self.rebal_freq]
        ret_p_idx = ret_per.copy()
        ret_p_idx.index = ret_p_idx.index.to_period(pcode)
        ret_dt_idx = pd.DatetimeIndex(ret_per.index)

        rows = []
        for d, grp in sig.groupby('date'):
            x = grp.set_index('ticker')[col].dropna()
            if len(x) < Q * CFG.SORTS['min_per_bucket']:
                continue
            if self.rebal_freq == 'D':
                pos = ret_dt_idx.searchsorted(pd.Timestamp(d), side='right')
                if pos >= len(ret_dt_idx):
                    continue
                next_pos = pos
            else:
                sig_p = pd.Timestamp(d).to_period(pcode)
                nxt = sig_p + 1
                if nxt not in ret_p_idx.index:
                    continue
                next_pos = ret_p_idx.index.get_loc(nxt)
            next_d = ret_per.index[next_pos]
            fwd = ret_per.iloc[next_pos]
            ranks = x.rank(method='first', ascending=True)
            buckets = np.ceil(Q * ranks / len(x)).astype(int)
            for q in range(1, Q + 1):
                tickers_q = buckets[buckets == q].index
                r_q = fwd.reindex(tickers_q).dropna().mean()
                rows.append({'date': next_d, 'quintile': q, 'ret': r_q,
                             'n_stocks': len(tickers_q)})
        df = pd.DataFrame(rows)
        if df.empty:
            self.results['T4'] = pd.DataFrame()
            return self.results['T4']

        wide = df.pivot_table(index='date', columns='quintile', values='ret')
        # L/S based on sort direction
        long_q = 1 if self.sort_dir == 'asc' else Q
        short_q = Q if self.sort_dir == 'asc' else 1
        wide['LS'] = wide[long_q] - wide[short_q]

        # Per-quintile stats
        rows_out = []
        for q in list(range(1, Q + 1)) + ['LS']:
            r = wide[q].dropna()
            mu, se, t = hac_tstat(r)
            ann_ret = mu * self.ann_factor
            ann_sr = annualize_sr(r, self.ann_factor)
            rows_out.append({
                'bucket': f'Q{q}' if q != 'LS' else 'LS',
                'mean_per_period': mu, 'se_NW': se, 't_NW': t,
                'ann_ret': ann_ret, 'ann_sr': ann_sr, 'n_obs': len(r),
                'rebal_freq': self.rebal_freq,
            })
        out = pd.DataFrame(rows_out)

        # FF3 alpha for L/S — uses proper HAC sandwich SE on the intercept
        if self.ff is not None and 'MKT' in self.ff.columns:
            ff_per = self.ff[['MKT', 'SMB', 'HML']].resample(
                self.rebal_meta['pandas_freq']).apply(lambda x: (1 + x).prod() - 1)
            ls = wide['LS'].dropna()
            joined = pd.concat([ls.rename('y'), ff_per.reindex(ls.index)], axis=1).dropna()
            if len(joined) >= 12:
                X = joined[['MKT', 'SMB', 'HML']].values
                y = joined['y'].values
                X1 = np.column_stack([np.ones(len(X)), X])
                beta_hat, se_hat, t_hat = hac_ols(y, X1)
                if beta_hat is not None:
                    alpha = beta_hat[0]
                    se_alpha = se_hat[0]
                    t_alpha = t_hat[0]
                    ann_alpha = alpha * self.ann_factor
                    out.loc[len(out)] = {
                        'bucket': 'LS_FF3_alpha',
                        'mean_per_period': alpha, 'se_NW': se_alpha, 't_NW': t_alpha,
                        'ann_ret': ann_alpha, 'ann_sr': np.nan, 'n_obs': len(joined),
                        'rebal_freq': self.rebal_freq,
                    }

        out.to_csv(self.tab_out / f'T4_quintile_sort_{self.name}.csv', index=False)
        self.results['T4'] = out
        self.results['_T4_panel'] = wide
        return out

    # ─────────────────────────────────────────────────────────────────
    # T5: Year-by-year stability of L/S
    # ─────────────────────────────────────────────────────────────────
    def t5_year_by_year(self):
        if '_T4_panel' not in self.results:
            self.t4_quintile_sort()
        wide = self.results.get('_T4_panel')
        if wide is None or wide.empty:
            self.results['T5'] = pd.DataFrame()
            return self.results['T5']
        ls = wide['LS'].dropna()
        rows = []
        for y, grp in ls.groupby(ls.index.year):
            mu, se, t = hac_tstat(grp)
            rows.append({
                'year': y, 'n_periods': len(grp),
                'mean_per_period': mu, 't_NW': t,
                'ann_ret': mu * self.ann_factor,
                'ann_sr': annualize_sr(grp, self.ann_factor),
                'hit_rate': (grp > 0).mean(),
                'rebal_freq': self.rebal_freq,
            })
        out = pd.DataFrame(rows)
        out.to_csv(self.tab_out / f'T5_year_by_year_{self.name}.csv', index=False)
        self.results['T5'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # T6: Period-by-period stability (intraday only — overridden externally)
    # ─────────────────────────────────────────────────────────────────
    def t6_period_by_period(self, intraday_returns=None):
        """Pass-through for daily factors. Intraday version provided in
        the residual-reversal report subclass."""
        if intraday_returns is None:
            self.results['T6'] = pd.DataFrame()
            return self.results['T6']
        # Expects: DataFrame with cols (date, period, ticker, ret)
        rows = []
        for p, grp in intraday_returns.groupby('period'):
            r = grp.groupby('date')['ret'].mean()  # cross-sectional avg per day
            mu, se, t = hac_tstat(r)
            rows.append({'period': int(p), 'n_obs': len(r),
                         'mean': mu, 't_NW': t,
                         'sr_ann': annualize_sr(r, 252 * 16)})
        out = pd.DataFrame(rows)
        out.to_csv(self.tab_out / f'T6_period_by_period_{self.name}.csv', index=False)
        self.results['T6'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # T7: Holding-period decay
    # ─────────────────────────────────────────────────────────────────
    def t7_holding_decay(self):
        col = self.name
        Q = self.n_buckets
        sig = self.signal.copy()
        sig['date_m'] = sig['date'].dt.to_period('M').dt.to_timestamp()
        sig = (sig.sort_values('date')
                  .groupby(['ticker', 'date_m'])[col].last()
                  .reset_index().rename(columns={'date_m': 'date'}))
        ret_m = self.ret_m.copy()
        ret_m.index = pd.to_datetime(ret_m.index).normalize()
        sig['date'] = pd.to_datetime(sig['date']).dt.normalize()

        out_rows = []
        for K in [1, 2, 3, 6, 12]:
            ls_returns = []
            for d, grp in sig.groupby('date'):
                x = grp.set_index('ticker')[col].dropna()
                if len(x) < Q * CFG.SORTS['min_per_bucket']:
                    continue
                idx = ret_m.index.get_indexer([d])[0]
                if idx < 0 or idx + K >= len(ret_m):
                    continue
                fwd = ret_m.iloc[idx + K]
                ranks = x.rank(method='first', ascending=True)
                buckets = np.ceil(Q * ranks / len(x)).astype(int)
                long_q = 1 if self.sort_dir == 'asc' else Q
                short_q = Q if self.sort_dir == 'asc' else 1
                long_r = fwd.reindex(buckets[buckets == long_q].index).mean()
                short_r = fwd.reindex(buckets[buckets == short_q].index).mean()
                if not np.isnan(long_r) and not np.isnan(short_r):
                    ls_returns.append(long_r - short_r)
            if ls_returns:
                arr = np.array(ls_returns)
                mu, se, t = hac_tstat(arr)
                out_rows.append({
                    'K': K, 'n_obs': len(arr),
                    'mean_ls': mu, 't_NW': t,
                    'ann_ret': mu * ANNUAL_FACTOR_MONTHLY,
                })
        out = pd.DataFrame(out_rows)
        out.to_csv(self.tab_out / f'T7_holding_decay_{self.name}.csv', index=False)
        self.results['T7'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # T8: Double sort with auxiliary signal (placeholder — needs partner)
    # ─────────────────────────────────────────────────────────────────
    def t8_double_sort(self, aux_panel=None, aux_name='aux'):
        if aux_panel is None or aux_panel.empty:
            self.results['T8'] = pd.DataFrame()
            return self.results['T8']
        Q = self.n_buckets
        a = self.signal.merge(aux_panel.rename(columns={aux_name: 'aux'}),
                              on=['date', 'ticker'])
        a['date_m'] = a['date'].dt.to_period('M').dt.to_timestamp()
        a = (a.sort_values('date')
              .groupby(['ticker', 'date_m']).last().reset_index()
              .rename(columns={'date_m': 'date'}))
        ret_m = self.ret_m
        out_rows = []
        for d, grp in a.groupby('date'):
            x1 = grp[self.name].dropna()
            x2 = grp['aux'].dropna()
            common = x1.index.intersection(x2.index)
            if len(common) < Q * Q * 2:
                continue
            try:
                next_d = ret_m.index[ret_m.index.get_indexer([d])[0] + 1]
                fwd = ret_m.loc[next_d]
            except (KeyError, IndexError):
                continue
            r1 = grp.loc[common, self.name].rank(pct=True)
            r2 = grp.loc[common, 'aux'].rank(pct=True)
            b1 = np.ceil(Q * r1).astype(int).clip(1, Q)
            b2 = np.ceil(Q * r2).astype(int).clip(1, Q)
            tickers = grp.loc[common, 'ticker'].values
            for i, t in enumerate(tickers):
                out_rows.append({
                    'date': next_d, 'ticker': t,
                    'q_main': b1.iloc[i], 'q_aux': b2.iloc[i],
                    'ret': fwd.get(t, np.nan),
                })
        if not out_rows:
            self.results['T8'] = pd.DataFrame()
            return self.results['T8']
        df = pd.DataFrame(out_rows)
        grid = df.groupby(['q_main', 'q_aux'])['ret'].mean().unstack()
        grid.to_csv(self.tab_out / f'T8_double_sort_{self.name}.csv')
        self.results['T8'] = grid
        return grid

    # ─────────────────────────────────────────────────────────────────
    # T9: Cost sensitivity
    # ─────────────────────────────────────────────────────────────────
    def t9_cost_sensitivity(self):
        if '_T4_panel' not in self.results:
            self.t4_quintile_sort()
        wide = self.results.get('_T4_panel')
        if wide is None or wide.empty:
            self.results['T9'] = pd.DataFrame()
            return self.results['T9']
        ls = wide['LS'].dropna()
        # Turnover: compute from actual quintile membership churn.
        # Default 1.0 (full rebalance) is pessimistic for daily (real ~0.71)
        # and optimistic for monthly (real ~0.30).  For honest cost numbers,
        # use EXP_lib.cost_stress() with weighted turnover from the portfolio builder.
        if 'turnover' in wide.columns:
            turnover = wide['turnover'].dropna().mean()
        elif 'turnover_weighted' in wide.columns:
            turnover = wide['turnover_weighted'].dropna().mean()
        else:
            turnover = 1.0  # fallback: assume full rebalance
        rows = []
        for c_bps in self.costs:
            cost_per_period = c_bps / 10_000 * turnover * 2  # round-trip
            net = ls - cost_per_period
            mu, se, t = hac_tstat(net)
            rows.append({
                'cost_bps': c_bps,
                'mean_per_period_net': mu,
                'ann_ret_net': mu * self.ann_factor,
                'ann_sr_net': annualize_sr(net, self.ann_factor),
                't_NW_net': t,
                'rebal_freq': self.rebal_freq,
            })
        out = pd.DataFrame(rows)
        out.to_csv(self.tab_out / f'T9_cost_sensitivity_{self.name}.csv', index=False)
        self.results['T9'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # T10: Robustness — leave-one-year-out
    # ─────────────────────────────────────────────────────────────────
    def t10_robustness(self):
        if '_T4_panel' not in self.results:
            self.t4_quintile_sort()
        wide = self.results.get('_T4_panel')
        if wide is None or wide.empty:
            self.results['T10'] = pd.DataFrame()
            return self.results['T10']
        ls = wide['LS'].dropna()
        rows = []
        for excl_year in sorted(ls.index.year.unique()):
            sub = ls[ls.index.year != excl_year]
            mu, se, t = hac_tstat(sub)
            rows.append({
                'excluded_year': excl_year,
                'mean_per_period': mu,
                'ann_ret': mu * self.ann_factor,
                'ann_sr': annualize_sr(sub, self.ann_factor),
                't_NW': t, 'n_obs': len(sub),
                'rebal_freq': self.rebal_freq,
            })
        out = pd.DataFrame(rows)
        out.to_csv(self.tab_out / f'T10_robustness_{self.name}.csv', index=False)
        self.results['T10'] = out
        return out

    # ─────────────────────────────────────────────────────────────────
    # FIGURES
    # ─────────────────────────────────────────────────────────────────
    def f1_cumulative_ls(self):
        wide = self.results.get('_T4_panel')
        if wide is None or wide.empty:
            return
        ls = wide['LS'].dropna()
        cum = (1 + ls).cumprod() - 1
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cum.index, cum.values * 100, color=PALETTE['factor'], linewidth=1.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'F1 — Cumulative L/S Return ({self.name}, {self.period_label})',
                     fontweight='bold')
        ax.set_ylabel('Cumulative return (%)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(self.fig_out / f'F1_cumulative_LS_{self.name}.png', dpi=130)
        plt.close()

    def f2_quintile_bars(self):
        out = self.results.get('T4')
        if out is None or out.empty:
            return
        df = out[out['bucket'].str.startswith('Q')].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [PALETTE['short'] if x == 'Q1' else
                  (PALETTE['long'] if x.endswith(str(self.n_buckets)) else PALETTE['neutral_gray'])
                  for x in df['bucket']]
        ax.bar(df['bucket'], df['ann_ret'] * 100, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'F2 — Quintile Annualized Returns ({self.name})', fontweight='bold')
        ax.set_ylabel('Annualized return (%)')
        plt.tight_layout()
        plt.savefig(self.fig_out / f'F2_quintile_bars_{self.name}.png', dpi=130)
        plt.close()

    def f3_year_bars(self):
        out = self.results.get('T5')
        if out is None or out.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [verdict_color(t) for t in out['t_NW']]
        ax.bar(out['year'].astype(str), out['ann_sr'], color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'F3 — Annual L/S Sharpe ({self.name})', fontweight='bold')
        ax.set_ylabel('Annualized SR')
        plt.tight_layout()
        plt.savefig(self.fig_out / f'F3_year_bars_{self.name}.png', dpi=130)
        plt.close()

    def f4_rolling_sharpe(self):
        wide = self.results.get('_T4_panel')
        if wide is None or wide.empty:
            return
        ls = wide['LS'].dropna()
        roll_n = self.ann_factor   # 1-year rolling window
        roll = ls.rolling(roll_n, min_periods=max(6, roll_n // 2)).apply(
            lambda x: x.mean() / x.std() * np.sqrt(self.ann_factor) if x.std() > 0 else np.nan)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(roll.index, roll.values, color=PALETTE['primary_blue'], linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(0.5, color=PALETTE['neutral_gray'], linewidth=0.5, linestyle='--')
        ax.set_title(f'F4 — 12-Month Rolling Sharpe ({self.name})', fontweight='bold')
        ax.set_ylabel('Rolling SR (annualized)')
        plt.tight_layout()
        plt.savefig(self.fig_out / f'F4_rolling_sharpe_{self.name}.png', dpi=130)
        plt.close()

    def f5_cost_survival(self):
        out = self.results.get('T9')
        if out is None or out.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out['cost_bps'], out['ann_sr_net'], 'o-',
                color=PALETTE['primary_orange'], linewidth=2, markersize=8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(0.5, color=PALETTE['neutral_gray'], linewidth=0.5, linestyle='--', label='SR=0.5')
        ax.set_xlabel('Round-trip cost (bps/rebalance)')
        ax.set_ylabel('Net annualized SR')
        ax.set_title(f'F5 — Cost Survival ({self.name})', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.fig_out / f'F5_cost_survival_{self.name}.png', dpi=130)
        plt.close()

    # ─────────────────────────────────────────────────────────────────
    # ORCHESTRATION
    # ─────────────────────────────────────────────────────────────────
    def run_all(self, other_factors=None, intraday_returns=None,
                aux_panel=None, aux_name=None):
        print(f'\n{"=" * 60}')
        print(f'FactorReport: {self.name} (period={self.period_label})')
        print('=' * 60)

        steps = [
            ('T1 summary stats',     lambda: self.t1_summary_stats()),
            ('T2 correlations',      lambda: self.t2_correlations(other_factors)),
            ('T3 xsec regression',   lambda: self.t3_xsec_regression()),
            ('T4 quintile sort',     lambda: self.t4_quintile_sort()),
            ('T5 year-by-year',      lambda: self.t5_year_by_year()),
            ('T6 period-by-period',  lambda: self.t6_period_by_period(intraday_returns)),
            ('T7 holding decay',     lambda: self.t7_holding_decay()),
            ('T8 double sort',       lambda: self.t8_double_sort(aux_panel, aux_name) if aux_panel is not None else None),
            ('T9 cost sensitivity',  lambda: self.t9_cost_sensitivity()),
            ('T10 robustness',       lambda: self.t10_robustness()),
        ]
        for name, fn in steps:
            try:
                fn()
                print(f'  [OK] {name}')
            except Exception as e:
                print(f'  [FAIL] {name}: {e}')

        for name, fn in [('F1', self.f1_cumulative_ls),
                          ('F2', self.f2_quintile_bars),
                          ('F3', self.f3_year_bars),
                          ('F4', self.f4_rolling_sharpe),
                          ('F5', self.f5_cost_survival)]:
            try:
                fn()
                print(f'  [OK] figure {name}')
            except Exception as e:
                print(f'  [FAIL] figure {name}: {e}')

        return self.summary()

    def summary(self):
        """One-row pass/fail verdict for the master classification table."""
        # Pull key statistics from the tables we built
        t3 = self.results.get('T3', pd.DataFrame())
        t4 = self.results.get('T4', pd.DataFrame())

        t_fm = np.nan
        if not t3.empty:
            mask = t3['metric'] == 't_NW'
            if mask.any():
                t_fm = t3.loc[mask, 'value'].iloc[0]

        t_alpha_ff3 = np.nan
        ann_ret_ls = np.nan
        ann_sr_ls = np.nan
        if not t4.empty:
            ls = t4[t4['bucket'] == 'LS']
            if not ls.empty:
                ann_ret_ls = ls['ann_ret'].iloc[0]
                ann_sr_ls = ls['ann_sr'].iloc[0]
            ff3 = t4[t4['bucket'] == 'LS_FF3_alpha']
            if not ff3.empty:
                t_alpha_ff3 = ff3['t_NW'].iloc[0]

        # Pass logic
        pass_t1 = (not np.isnan(t_alpha_ff3)
                   and abs(t_alpha_ff3) >= CFG.PASS_T1['t_alpha_min'])
        pass_t2 = (not np.isnan(t_fm)
                   and abs(t_fm) >= CFG.PASS_T2['t_slope_min'])

        out = pd.DataFrame([{
            'factor': self.name,
            'period': self.period_label,
            'n_obs': len(self.signal),
            'ann_ret_LS': ann_ret_ls,
            'ann_sr_LS': ann_sr_ls,
            't_FF3_alpha': t_alpha_ff3,
            't_FM_slope': t_fm,
            'pass_T1_FF3': pass_t1,
            'pass_T2_FM': pass_t2,
        }])
        out.to_csv(self.tab_out / f'_summary_{self.name}.csv', index=False)
        return out


def run_factor(factor_name, signal_panel, returns, returns_monthly=None,
               ff_factors=None, **kwargs):
    """Convenience: build report and run all in one call."""
    fr = FactorReport(factor_name=factor_name,
                       signal_panel=signal_panel,
                       returns=returns,
                       returns_monthly=returns_monthly,
                       ff_factors=ff_factors,
                       **kwargs)
    return fr.run_all()
