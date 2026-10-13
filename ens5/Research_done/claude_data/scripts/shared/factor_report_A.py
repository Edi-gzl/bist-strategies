"""
shared.factor_report_A — Standardized Test A + Test B report per factor.

This produces the tables/figures listed in CFG.REPORT_TABLES_A and
REPORT_FIGURES_A, organized under:

    results/A_cross_sectional/per_factor/<factor>/
        tables/
            T1_summary_stats.csv
            T2_xsec_R2_per_period.csv
            T3_xsec_slope_per_period.csv
            T4_FM_premium.csv
            T5_IC_per_period.csv
            T6_decision_matrix.csv
        figures/
            F1_R2_time_series.png
            F2_slope_time_series.png
            F3_IC_time_series.png
            F4_R2_vs_slope_scatter.png
            F5_premium_with_CI.png

The headline file is `T6_decision_matrix.csv` — one row, machine-readable
verdict. A master summary aggregating every factor's verdict lives at
`results/A_cross_sectional/summary/master_verdicts.csv`.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .test_AB import run_test_A, run_test_B, decision
from .plotting import setup_plot_style, PALETTE
from . import configs as CFG


def _fmt_pct(v):
    return f'{v*100:.2f}%' if pd.notna(v) else 'NaN'


def _fmt_n(v, d=3):
    return f'{v:.{d}f}' if pd.notna(v) else 'NaN'


class FactorReportA:
    """Test A + Test B per-factor reporter.

    Parameters
    ----------
    factor_name : str
    signal_panel : DataFrame [date, ticker, <factor>, ...]
    ret_monthly : DataFrame (wide, date × ticker) of monthly returns
    controls : list of column names (default: configs.TEST_B['controls'])
    period_label : str, key in CFG.PERIODS
    """

    def __init__(self, factor_name, signal_panel, ret_monthly,
                 controls=None, period_label='FULL'):
        self.factor = factor_name
        self.signal = signal_panel.copy()
        self.signal['date'] = pd.to_datetime(self.signal['date'])
        self.ret_m = ret_monthly.copy()
        self.ret_m.index = pd.to_datetime(self.ret_m.index)
        self.controls = controls if controls is not None else CFG.TEST_B['controls']
        self.period_label = period_label

        # Filter to period
        start, end = CFG.PERIODS[period_label]
        self.signal = self.signal[
            (self.signal['date'] >= start) & (self.signal['date'] <= end)
        ]
        self.ret_m = self.ret_m.loc[start:end]

        self.out = CFG.per_factor_dir('A', factor_name)
        self.tab = self.out / 'tables'
        self.fig = self.out / 'figures'

        self.results = {}
        setup_plot_style()

    # ───────────────────────────────────────────────────────────────
    def run(self):
        """Run Test A + Test B and produce all tables and figures."""
        print(f'\n{"─" * 60}\nFactor: {self.factor}  ({self.period_label})\n{"─" * 60}')

        # Run tests
        ta = run_test_A(self.signal, self.ret_m, factor=self.factor)
        tb = run_test_B(self.signal, self.ret_m, factor=self.factor,
                         controls=self.controls)
        verdict = decision(ta, tb)

        self.results['A'] = ta
        self.results['B'] = tb
        self.results['verdict'] = verdict

        # Build tables
        self._table_T1(ta, tb, verdict)
        self._table_T2(ta)
        self._table_T3(ta)
        self._table_T4(tb)
        self._table_T5(ta)
        self._table_T6(verdict, ta, tb)

        # Build figures
        self._fig_F1_R2(ta)
        self._fig_F2_slope(ta)
        self._fig_F3_IC(ta)
        self._fig_F4_scatter(ta)
        self._fig_F5_premium_CI(tb)

        # Console summary
        self._print_summary(ta, tb, verdict)
        return verdict

    # ── Tables ─────────────────────────────────────────────────────
    def _table_T1(self, ta, tb, verdict):
        """T1 — summary stats: one-row headline per factor.

        Merges with existing T1 file by (factor, period) so multi-period runs
        accumulate instead of overwriting.
        """
        row = {
            'factor': self.factor,
            'period': self.period_label,
            'n_months': ta['n_months'],
            'avg_R2': ta['avg_R2'],
            'avg_R2_adj': ta['avg_R2_adj'],
            'avg_slope_b': ta['avg_slope'],
            't_slope_NW': ta['t_avg_slope'],
            'avg_IC_spearman': ta['avg_IC'],
            'ICIR': ta['ICIR'],
            'hit_rate_slope_sign': ta['hit_rate'],
            'pass_R2_gt_2pct': ta['pass_R2'],
            'pass_slope_t_gt_1.5': ta['pass_slope'],
            'pass_A': ta['pass_A'],
            'avg_lambda_FM': tb['avg_lambda_factor'],
            't_lambda_FM': tb['t_lambda_factor'],
            'FM_controls': '|'.join(tb['controls_used']),
            'FM_n_months': tb['n_months'],
            'pass_B': tb['pass_B'],
            'verdict': verdict['verdict'],
            'action': verdict['action'],
        }
        out_path = self.tab / 'T1_summary_stats.csv'
        df_new = pd.DataFrame([row])
        if out_path.exists():
            existing = pd.read_csv(out_path)
            existing = existing[
                ~((existing['factor'] == self.factor) &
                  (existing['period'] == self.period_label))
            ]
            df_new = pd.concat([existing, df_new], ignore_index=True)
        df_new.to_csv(out_path, index=False)

    def _table_T2(self, ta):
        """T2 — per-period R² (time series). Period-suffixed filename so all
        three regime runs coexist."""
        df = ta['per_period'].copy()
        if df.empty:
            return
        df[['R2', 'R2_adj', 'n_stocks']].to_csv(
            self.tab / f'T2_xsec_R2_per_period_{self.period_label}.csv')
        # Default symlink-style file (latest run): also keep without suffix
        df[['R2', 'R2_adj', 'n_stocks']].to_csv(
            self.tab / 'T2_xsec_R2_per_period.csv')

    def _table_T3(self, ta):
        df = ta['per_period'].copy()
        if df.empty:
            return
        df[['slope_b', 't_slope', 'n_stocks']].to_csv(
            self.tab / f'T3_xsec_slope_per_period_{self.period_label}.csv')
        df[['slope_b', 't_slope', 'n_stocks']].to_csv(
            self.tab / 'T3_xsec_slope_per_period.csv')

    def _table_T4(self, tb):
        """T4 — FM premium: λ̄ and t-stat for factor + controls.
        Period-suffixed plus default-name copy."""
        rows = [{
            'variable': 'factor',
            'mean_lambda': tb['avg_lambda_factor'],
            't_NW': tb['t_lambda_factor'],
            'n_months': tb['n_months'],
            'period': self.period_label,
        }]
        for c, summary in tb.get('controls_summary', {}).items():
            rows.append({
                'variable': c,
                'mean_lambda': summary['mean'],
                't_NW': summary['t_NW'],
                'n_months': tb['n_months'],
                'period': self.period_label,
            })
        df_premium = pd.DataFrame(rows)
        df_premium.to_csv(
            self.tab / f'T4_FM_premium_{self.period_label}.csv', index=False)
        df_premium.to_csv(self.tab / 'T4_FM_premium.csv', index=False)
        # Also save the per-period lambda time series (period-suffixed)
        if not tb['lambdas'].empty:
            tb['lambdas'].to_csv(
                self.tab / f'T4_FM_lambdas_per_period_{self.period_label}.csv')
            tb['lambdas'].to_csv(
                self.tab / 'T4_FM_lambdas_per_period.csv')

    def _table_T5(self, ta):
        if ta['ic_per_period'].empty:
            return
        ic_df = pd.DataFrame({'IC': ta['ic_per_period']})
        ic_df.to_csv(self.tab / f'T5_IC_per_period_{self.period_label}.csv')
        ic_df.to_csv(self.tab / 'T5_IC_per_period.csv')

    def _table_T6(self, verdict, ta, tb):
        """T6 — the decision matrix row. THIS IS THE HEADLINE.

        Merges by (factor, period) so multi-period runs accumulate. Also
        writes a per-period sidecar `T6_decision_matrix_<period>.csv` so each
        period can be inspected in isolation.
        """
        row = {
            'factor': self.factor,
            'period': self.period_label,
            # Test A
            'A_avg_R2': ta['avg_R2'],
            'A_avg_slope': ta['avg_slope'],
            'A_t_slope': ta['t_avg_slope'],
            'A_pass_R2': ta['pass_R2'],
            'A_pass_slope': ta['pass_slope'],
            'A_PASS': ta['pass_A'],
            # Test B
            'B_lambda_premium': tb['avg_lambda_factor'],
            'B_t_lambda': tb['t_lambda_factor'],
            'B_controls': '|'.join(tb['controls_used']),
            'B_PASS': tb['pass_B'],
            # Verdict
            'VERDICT': verdict['verdict'],
            'ACTION': verdict['action'],
        }
        df_new = pd.DataFrame([row])
        # Per-period sidecar (always overwrites itself — it's just THIS period)
        sidecar = self.tab / f'T6_decision_matrix_{self.period_label}.csv'
        df_new.to_csv(sidecar, index=False)
        # Main file: merge by (factor, period)
        out_path = self.tab / 'T6_decision_matrix.csv'
        if out_path.exists():
            existing = pd.read_csv(out_path)
            existing = existing[
                ~((existing['factor'] == self.factor) &
                  (existing['period'] == self.period_label))
            ]
            df_new = pd.concat([existing, df_new], ignore_index=True)
        df_new.to_csv(out_path, index=False)

    # ── Figures ────────────────────────────────────────────────────
    def _fig_F1_R2(self, ta):
        df = ta['per_period']
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df.index, df['R2'] * 100, color=PALETTE['primary_blue'],
                linewidth=1.2, alpha=0.85)
        ax.axhline(ta['avg_R2'] * 100, color=PALETTE['primary_orange'],
                    linestyle='--', linewidth=1.5,
                    label=f"avg = {ta['avg_R2']*100:.2f}%")
        ax.axhline(2.0, color=PALETTE['neutral_gray'], linestyle=':',
                    linewidth=1, label='2% pass threshold')
        ax.set_ylabel('Cross-sectional R² (%)')
        ax.set_title(f'F1 — Cross-sectional R² over time ({self.factor})',
                     fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(self.fig / f'F1_R2_time_series_{self.period_label}.png', dpi=130)
        plt.savefig(self.fig / 'F1_R2_time_series.png', dpi=130)
        plt.close()

    def _fig_F2_slope(self, ta):
        df = ta['per_period']
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df.index, df['slope_b'], color=PALETTE['primary_blue'],
                linewidth=1.0, alpha=0.7)
        ax.axhline(ta['avg_slope'], color=PALETTE['primary_orange'],
                    linestyle='--', linewidth=1.5,
                    label=f"avg b̄ = {ta['avg_slope']:.4f}  "
                          f"t = {ta['t_avg_slope']:.2f}")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Slope b_t')
        ax.set_title(f'F2 — Cross-sectional slope over time ({self.factor})',
                     fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(self.fig / f'F2_slope_time_series_{self.period_label}.png', dpi=130)
        plt.savefig(self.fig / 'F2_slope_time_series.png', dpi=130)
        plt.close()

    def _fig_F3_IC(self, ta):
        ic = ta['ic_per_period']
        if ic.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(ic.index, ic.values,
               width=20,
               color=[PALETTE['long'] if v > 0 else PALETTE['short']
                       for v in ic.values],
               edgecolor='none', alpha=0.7)
        ax.axhline(ta['avg_IC'], color='black', linestyle='--',
                    linewidth=1.5, label=f"avg IC = {ta['avg_IC']:.4f}  "
                                          f"ICIR = {ta['ICIR']:.2f}")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('IC (Spearman ρ)')
        ax.set_title(f'F3 — Information Coefficient ({self.factor})',
                     fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(self.fig / f'F3_IC_time_series_{self.period_label}.png', dpi=130)
        plt.savefig(self.fig / 'F3_IC_time_series.png', dpi=130)
        plt.close()

    def _fig_F4_scatter(self, ta):
        df = ta['per_period']
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(df['R2'] * 100, df['slope_b'],
                    s=20, alpha=0.5, color=PALETTE['primary_blue'])
        ax.axhline(ta['avg_slope'], color=PALETTE['primary_orange'],
                    linestyle='--', linewidth=1)
        ax.axvline(ta['avg_R2'] * 100, color=PALETTE['primary_orange'],
                    linestyle='--', linewidth=1)
        ax.set_xlabel('R² (%)')
        ax.set_ylabel('slope b_t')
        ax.set_title(f'F4 — R² vs slope per period ({self.factor})',
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.fig / f'F4_R2_vs_slope_scatter_{self.period_label}.png', dpi=130)
        plt.savefig(self.fig / 'F4_R2_vs_slope_scatter.png', dpi=130)
        plt.close()

    def _fig_F5_premium_CI(self, tb):
        if tb['lambdas'].empty:
            return
        lam = tb['lambdas']['lambda_factor']
        fig, ax = plt.subplots(figsize=(10, 4.5))
        # Rolling 12-month mean of lambda with 1σ band
        roll_mean = lam.rolling(12, min_periods=6).mean()
        roll_std = lam.rolling(12, min_periods=6).std() / np.sqrt(12)
        ax.fill_between(lam.index, roll_mean - 2*roll_std,
                         roll_mean + 2*roll_std,
                         color=PALETTE['primary_blue'], alpha=0.2,
                         label='95% CI')
        ax.plot(lam.index, roll_mean, color=PALETTE['primary_blue'],
                linewidth=1.8, label='12m rolling mean λ')
        ax.axhline(tb['avg_lambda_factor'], color=PALETTE['primary_orange'],
                    linestyle='--', linewidth=1.5,
                    label=f"avg λ̄ = {tb['avg_lambda_factor']:.4f}  "
                          f"t = {tb['t_lambda_factor']:.2f}")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('FM premium λ_t')
        ax.set_title(f'F5 — Fama-MacBeth premium with rolling CI '
                      f'({self.factor})', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        plt.savefig(self.fig / f'F5_premium_with_CI_{self.period_label}.png', dpi=130)
        plt.savefig(self.fig / 'F5_premium_with_CI.png', dpi=130)
        plt.close()

    # ── Console summary ────────────────────────────────────────────
    def _print_summary(self, ta, tb, verdict):
        print(f"  Test A (cross-sectional R² + slope):")
        print(f"    avg R²:    {_fmt_pct(ta['avg_R2']):>10}    "
              f"(pass ≥ 2%)   → {'PASS' if ta['pass_R2'] else 'FAIL'}")
        print(f"    avg slope: {_fmt_n(ta['avg_slope'], 5):>10}    "
              f"t = {_fmt_n(ta['t_avg_slope'], 2):>6}    "
              f"(pass |t| ≥ 1.5) → {'PASS' if ta['pass_slope'] else 'FAIL'}")
        print(f"    avg IC:    {_fmt_n(ta['avg_IC'], 4):>10}    "
              f"ICIR = {_fmt_n(ta['ICIR'], 2):>5}")
        print(f"    hit rate:  {_fmt_n(ta['hit_rate'], 3):>10}    "
              f"n_months = {ta['n_months']}")
        print(f"    → Test A:  {'PASS' if ta['pass_A'] else 'FAIL'}")

        print(f"\n  Test B (Fama-MacBeth, controls = {tb.get('controls_used')}):")
        print(f"    λ̄ premium: {_fmt_n(tb['avg_lambda_factor'], 5):>10}    "
              f"t = {_fmt_n(tb['t_lambda_factor'], 2):>6}    "
              f"n_months = {tb['n_months']}")
        print(f"    → Test B:  {'PASS' if tb['pass_B'] else 'FAIL'}")

        print(f"\n  VERDICT:   {verdict['verdict']}")
        print(f"  ACTION:    {verdict['action']}")
