"""
shared.plotting — Hedge-fund-quality plot utilities.

Single source of truth for all visualization. Every script calls setup_plot_style()
at the top, then uses the helper functions below.

See docs/PLOT_STANDARDS.md for design philosophy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import warnings


PALETTE = {
    'primary_blue':   '#1565C0',
    'primary_red':    '#C62828',
    'primary_green':  '#2E7D32',
    'primary_orange': '#F57C00',
    'neutral_gray':   '#616161',
    'light_gray':     '#BDBDBD',
    'background':     '#FFFFFF',
    'panel_bg':       '#F8F9FA',
    'long':           '#1565C0',
    'short':          '#C62828',
    'factor':         '#2E7D32',
    'mkt':            '#9E9E9E',
    'tas29':          '#C62828',
    'oos':            '#1976D2',
    'F01':            '#9E9E9E',
    'F02':            '#1565C0',
    'F03':            '#F57C00',
    'F04':            '#2E7D32',
    'F05':            '#7B1FA2',
    'F06':            '#00796B',
    'F07':            '#5D4037',
    'F08':            '#E64A19',
}

VERDICT_COLORS = {
    'pass':       '#2E7D32',
    'borderline': '#F57C00',
    'fail':       '#C62828',
}


def verdict_color(t_stat, pass_thresh=2.0, borderline_thresh=1.5):
    if t_stat is None or (isinstance(t_stat, float) and np.isnan(t_stat)):
        return PALETTE['neutral_gray']
    if abs(t_stat) >= pass_thresh:
        return VERDICT_COLORS['pass']
    if abs(t_stat) >= borderline_thresh:
        return VERDICT_COLORS['borderline']
    return VERDICT_COLORS['fail']


def setup_plot_style():
    """Apply institutional style. Call at start of every plotting script."""
    plt.rcParams.update({
        'figure.facecolor':  'white',
        'figure.dpi':         100,
        'savefig.facecolor': 'white',
        'savefig.dpi':        130,
        'savefig.bbox':      'tight',
        'axes.facecolor':    PALETTE['panel_bg'],
        'axes.edgecolor':    PALETTE['light_gray'],
        'axes.linewidth':     1.0,
        'axes.grid':          True,
        'grid.alpha':         0.4,
        'grid.linestyle':    '--',
        'grid.color':        PALETTE['light_gray'],
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.titlesize':     12,
        'axes.titleweight':  'bold',
        'axes.labelsize':     10,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'legend.fontsize':    9,
        'font.size':          10,
        'lines.linewidth':    1.6,
    })


DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'rg', [PALETTE['primary_red'], '#FFFFFF', PALETTE['primary_green']], N=256
)


def save_figure(fig, path, dpi=130):
    """Save with standardized settings, suppressing date locator warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='AutoDateLocator was unable to pick.*')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def format_metrics(stats, include=None):
    if include is None:
        include = ['sharpe', 't_hac', 'ann_ret', 'vol', 'max_dd', 'win_rate', 'n']
    fmt = {
        'ann_ret':  ('Ann. Return', lambda v: f'{v:+6.1%}'),
        'vol':      ('Volatility',  lambda v: f'{v:6.1%}'),
        'sharpe':   ('Sharpe',      lambda v: f'{v:+6.2f}'),
        't_hac':    ('t-HAC',       lambda v: f'{v:+6.2f}'),
        'max_dd':   ('Max DD',      lambda v: f'{v:6.1%}'),
        'win_rate': ('Win %',       lambda v: f'{v:6.1%}'),
        'n':        ('N',           lambda v: f'{int(v):>6d}'),
        'skew':     ('Skewness',    lambda v: f'{v:+6.2f}'),
        'kurt':     ('Kurtosis',    lambda v: f'{v:+6.2f}'),
    }
    lines = []
    for k in include:
        if k in stats and k in fmt:
            label, formatter = fmt[k]
            v = stats[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                lines.append(f'{label:<12}  n/a')
            else:
                try:
                    lines.append(f'{label:<12}  {formatter(v)}')
                except Exception:
                    lines.append(f'{label:<12}  {v}')
    return '\n'.join(lines)


def add_metrics_box(ax, stats, loc='upper left', include=None, fontsize=8):
    text = format_metrics(stats, include=include)
    if loc == 'upper left':
        ha, va, x, y = 'left', 'top', 0.02, 0.98
    elif loc == 'upper right':
        ha, va, x, y = 'right', 'top', 0.98, 0.98
    elif loc == 'lower left':
        ha, va, x, y = 'left', 'bottom', 0.02, 0.02
    elif loc == 'lower right':
        ha, va, x, y = 'right', 'bottom', 0.98, 0.02
    else:
        ha, va, x, y = 'left', 'top', 0.02, 0.98
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
             family='monospace', ha=ha, va=va,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor=PALETTE['light_gray'], alpha=0.92))


def format_date_axis(ax, freq='auto'):
    """Smart date formatting that adapts to range length.
    
    BUG FIX (2026-05-07): Inspect actual axis range; pick locator based on span
    to avoid 'AutoDateLocator was unable to pick' warning for short ranges.
    """
    try:
        x_min, x_max = ax.get_xlim()
        span_days = x_max - x_min  # matplotlib stores dates as float days
    except Exception:
        span_days = 365
    
    if freq == 'auto':
        if span_days <= 90:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif span_days <= 365 * 1.5:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif span_days <= 365 * 4:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif freq == 'year':
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif freq == 'quarter':
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')


def mark_tas29(ax, label='TAS-29', y_text=None):
    """Mark the TAS-29 break (Q1 2023)."""
    ax.axvline(pd.Timestamp('2023-01-01'), color=PALETTE['tas29'],
                lw=1.0, linestyle=':', alpha=0.7, label=label)


def mark_oos(ax, start='2026-01-01', end='2026-05-05', label='OOS holdout'):
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                color=PALETTE['oos'], alpha=0.10, label=label)


def plot_cumulative_with_dd(returns_dict, title=None, log_scale=True,
                             stats_dict=None, mark_regimes=True,
                             figsize=(14, 8)):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    ax_cum, ax_dd = axes
    for label, r in returns_dict.items():
        r = r.dropna()
        if r.empty: continue
        cum = (1 + r).cumprod()
        col = PALETTE.get(label, None)
        if col is None:
            for k in PALETTE:
                if k.startswith('F') and k in label:
                    col = PALETTE[k]; break
        if col is None:
            col = next(iter(PALETTE.values()))
        ax_cum.plot(cum.index, cum, color=col, lw=2.0, label=label)
        ax_cum.annotate(f' {cum.iloc[-1]:.2f}×',
                         xy=(cum.index[-1], cum.iloc[-1]),
                         xytext=(5, 0), textcoords='offset points',
                         fontsize=8, color=col, fontweight='bold', va='center')
    if log_scale:
        ax_cum.set_yscale('log')
    ax_cum.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}×'))
    ax_cum.axhline(1.0, color='black', lw=0.6, ls=':', alpha=0.7)
    ax_cum.set_ylabel('Cumulative Return' + (' (log)' if log_scale else ''))
    ax_cum.legend(loc='upper left', ncol=min(4, len(returns_dict)), fontsize=9)
    if mark_regimes:
        mark_tas29(ax_cum)
        mark_oos(ax_cum)
    for label, r in returns_dict.items():
        r = r.dropna()
        if r.empty: continue
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        col = PALETTE.get(label, PALETTE['short'])
        if col is None:
            for k in PALETTE:
                if k.startswith('F') and k in label:
                    col = PALETTE[k]; break
        if col is None:
            col = PALETTE['short']
        ax_dd.fill_between(dd.index, dd, 0, color=col, alpha=0.25)
        ax_dd.plot(dd.index, dd, color=col, lw=0.9)
    ax_dd.axhline(0, color='black', lw=0.6)
    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    if mark_regimes:
        mark_tas29(ax_dd)
        mark_oos(ax_dd)
    format_date_axis(ax_dd)
    if stats_dict:
        text_lines = []
        for label, stats in stats_dict.items():
            text_lines.append(f'── {label} ──')
            text_lines.append(format_metrics(stats,
                include=['sharpe', 't_hac', 'ann_ret', 'max_dd', 'win_rate']))
            text_lines.append('')
        ax_cum.text(0.99, 0.02, '\n'.join(text_lines).strip(),
                     transform=ax_cum.transAxes, fontsize=7.5,
                     family='monospace', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                               edgecolor=PALETTE['light_gray'], alpha=0.92))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout()
    return fig


def plot_bar_with_threshold(values, labels, t_stats=None,
                              title=None, xlabel=None,
                              thresholds=(2.0, 1.5),
                              reference_line=None, figsize=(12, 6),
                              annotations=None):
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(values))
    if t_stats is not None:
        colors = [verdict_color(t, *thresholds) for t in t_stats]
    else:
        colors = [verdict_color(v, *thresholds) for v in values]
    ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=1.2)
    if annotations is not None:
        for i, (v, ann) in enumerate(zip(values, annotations)):
            x_text = v + (0.02 * max(abs(np.array(values))) if v >= 0 else -0.02 * max(abs(np.array(values))))
            ax.text(x_text, i, ann, va='center',
                     ha='left' if v >= 0 else 'right', fontsize=8.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.7)
    if reference_line is not None:
        ax.axvline(reference_line, color=PALETTE['primary_green'],
                    lw=1.5, ls='--', alpha=0.7,
                    label=f'Pass threshold ({reference_line})')
        ax.legend(loc='lower right', fontsize=9)
    if xlabel: ax.set_xlabel(xlabel)
    if title: ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_matrix, title=None, figsize=(8, 7), vmin=-1, vmax=1):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix.values, cmap=DIVERGING_CMAP,
                    vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index, fontsize=10)
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                     fontsize=10, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation')
    if title: ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_quintile_monotonicity(quintile_returns, title=None, figsize=(10, 6), cum=True):
    fig, ax = plt.subplots(figsize=figsize)
    quintile_cols = [c for c in quintile_returns.columns if c.startswith('Q')]
    if cum:
        values = [(1 + quintile_returns[c].dropna()).prod() - 1 for c in quintile_cols]
        ylabel = 'Cumulative Return'
    else:
        values = [quintile_returns[c].dropna().mean() * 252 for c in quintile_cols]
        ylabel = 'Annualized Return'
    n = len(quintile_cols)
    colors = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.5:
            r = 0xC6/255 + t*2*(0xBD-0xC6)/255; g = 0x28/255 + t*2*(0xBD-0x28)/255; b = 0x28/255 + t*2*(0xBD-0x28)/255
        else:
            r = 0xBD/255 + (t-0.5)*2*(0x2E-0xBD)/255; g = 0xBD/255 + (t-0.5)*2*(0x7D-0xBD)/255; b = 0xBD/255 + (t-0.5)*2*(0x32-0xBD)/255
        colors.append((r, g, b))
    x = np.arange(n)
    ax.bar(x, [v*100 for v in values], color=colors, edgecolor='white', linewidth=1.5)
    for i, v in enumerate(values):
        ax.text(i, v*100 + 0.5 if v >= 0 else v*100 - 1.5,
                 f'{v*100:+.1f}%', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=10, fontweight='bold')
    hl = values[-1] - values[0]
    ax.set_xticks(x); ax.set_xticklabels(quintile_cols)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_ylabel(ylabel + ' (%)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    is_inc = all(values[i] <= values[i+1] for i in range(n-1))
    is_dec = all(values[i] >= values[i+1] for i in range(n-1))
    if is_inc:
        verdict = 'Monotonic INCREASING ✓'; col_v = PALETTE['primary_green']
    elif is_dec:
        verdict = 'Monotonic DECREASING ✓'; col_v = PALETTE['primary_green']
    else:
        verdict = 'Non-monotonic ⚠'; col_v = PALETTE['primary_orange']
    ax.text(0.02, 0.97, f'{verdict}\nH−L spread: {hl*100:+.1f}%',
             transform=ax.transAxes, fontsize=10, ha='left', va='top',
             color=col_v, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor=col_v, alpha=0.9))
    if title: ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(returns_dict, window=252, title=None, figsize=(14, 5),
                          mark_regimes=True):
    fig, ax = plt.subplots(figsize=figsize)
    for label, r in returns_dict.items():
        r = r.dropna()
        if len(r) < window: continue
        roll_sr = r.rolling(window).mean() * 252 / (r.rolling(window).std() * np.sqrt(252))
        col = PALETTE.get(label, None)
        if col is None:
            for k in PALETTE:
                if k.startswith('F') and k in label:
                    col = PALETTE[k]; break
        if col is None:
            col = next(iter(PALETTE.values()))
        ax.plot(roll_sr.index, roll_sr, color=col, lw=1.6, label=label)
    ax.axhline(0, color='black', lw=0.7)
    ax.axhline(1.0, color=PALETTE['primary_green'], lw=0.8, ls=':', alpha=0.5, label='SR=1')
    ax.axhline(-1.0, color=PALETTE['primary_red'], lw=0.8, ls=':', alpha=0.5, label='SR=-1')
    ax.set_ylabel(f'Rolling {window}-Day Sharpe Ratio')
    if mark_regimes:
        mark_tas29(ax); mark_oos(ax)
    format_date_axis(ax)
    ax.legend(loc='upper left', fontsize=9, ncol=4)
    if title: ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_factor_panel_heatmap(sig_wide, ticker_sector=None, title=None,
                                 figsize=(20, 12), sample_freq='M',
                                 vmax_pct=95, mark_regimes=True):
    if sample_freq == 'M':
        df = pd.DataFrame({'date': sig_wide.index})
        df['ym'] = df['date'].dt.to_period('M').astype(str)
        sample_dates = pd.DatetimeIndex(df.groupby('ym')['date'].min().values)
        sig = sig_wide.loc[sig_wide.index.isin(sample_dates)]
    else:
        sig = sig_wide
    sig_z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1), axis=0)
    if ticker_sector:
        sorted_tickers = sorted(sig_z.columns,
                                  key=lambda t: (ticker_sector.get(t, 'Z'), t))
    else:
        sorted_tickers = sorted(sig_z.columns)
    sig_z = sig_z[sorted_tickers]
    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.nanpercentile(np.abs(sig_z.values), vmax_pct)
    if vmax == 0 or np.isnan(vmax): vmax = 1.0
    im = ax.imshow(sig_z.values.T, cmap=DIVERGING_CMAP,
                    vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_yticks(range(len(sorted_tickers)))
    ax.set_yticklabels(sorted_tickers, fontsize=6)
    n_cols = len(sig_z.index)
    tick_step = max(1, n_cols // 12)
    tick_idx = list(range(0, n_cols, tick_step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([sig_z.index[i].strftime('%Y-%m') for i in tick_idx],
                       rotation=45, fontsize=8)
    if ticker_sector:
        prev_sec = None
        for i, t in enumerate(sorted_tickers):
            sec = ticker_sector.get(t, 'Z')
            if sec != prev_sec:
                ax.axhline(i - 0.5, color='black', lw=0.4, alpha=0.5)
                prev_sec = sec
    if mark_regimes:
        try:
            tas29_pos = sig_z.index.searchsorted(pd.Timestamp('2023-01-01'))
            ax.axvline(tas29_pos - 0.5, color=PALETTE['tas29'], lw=2, alpha=0.7)
        except Exception: pass
        try:
            oos_pos = sig_z.index.searchsorted(pd.Timestamp('2026-01-01'))
            if 0 <= oos_pos <= n_cols:
                ax.axvline(oos_pos - 0.5, color=PALETTE['oos'], lw=2, alpha=0.7)
        except Exception: pass
    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.7)
    if title: ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_annual_returns(returns_dict, title=None, figsize=(14, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    annual = {}
    all_years = set()
    for label, r in returns_dict.items():
        r = r.dropna()
        ann = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual[label] = ann
        all_years.update(ann.index)
    years = sorted(all_years)
    n_strats = len(returns_dict)
    width = 0.8 / n_strats
    x = np.arange(len(years))
    for i, (label, ann) in enumerate(annual.items()):
        col = PALETTE.get(label, None)
        if col is None:
            for k in PALETTE:
                if k.startswith('F') and k in label:
                    col = PALETTE[k]; break
        if col is None:
            col = next(iter(PALETTE.values()))
        vals = [ann.get(y, 0) * 100 for y in years]
        ax.bar(x + (i - n_strats/2 + 0.5) * width, vals, width,
                color=col, label=label, edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.set_ylabel('Annual Return (%)')
    ax.legend(loc='best', fontsize=9, ncol=min(4, n_strats))
    if title: ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig
