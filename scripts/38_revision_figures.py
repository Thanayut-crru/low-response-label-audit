"""Figures added in the first revision (Reviewer 1, comment 4: presentation of the simulation,
ablation and replication results), in the style of 08_manuscript_figures.py.

- fig_r1_simulation: the known-truth comparisons of former Table 1 together with the zero-forecast
  comparisons from the same Holm family, which the table did not show;
- fig_r1_ablation: the eleven RDFL component cells, their loss relative to the zero forecast and their
  prediction dispersion side by side;
- fig_r1_robustness: RDFL_SOFT against the zero forecast in both markets under the locked analysis and
  every robustness analysis added in revision.
"""
from __future__ import annotations

import importlib.util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

spec = importlib.util.spec_from_file_location('figs', common.ROOT / 'scripts/08_manuscript_figures.py')
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)

R = common.ROOT / 'results'
OUT = R / 'figures_revision'
MARGIN = 2.0
PRIMARY = 20


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    print('wrote', name)


def minus(text):
    return text.replace('-', '−')


def simulation():
    s = pd.read_csv(R / 'report/simulation_paired_inference.csv')
    order = ['strong_signal', 'null_effect', 'high_noise', 'cancellation_050', 'cancellation_085', 'cancellation_100',
             'anticipation', 'stale_price', 'informative_missing', 'regime_shift']
    labels = {k: v.replace('\n', ' ').replace('$c=', 'c = ').replace('$', '') for k, v in figs.SCENARIO_LABELS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(figs.COL_DOUBLE, 2.9), sharey=True)
    for ax, comp, title in zip(axes, ['DIRECT', 'ZERO'],
                               ['(a) Direct ridge MSE minus RDFL MSE', '(b) Zero-forecast MSE minus RDFL MSE']):
        d = s[s.competitor == comp].set_index('scenario').reindex(order)
        y = np.arange(len(order))[::-1]
        ax.axvline(0, color=figs.GREY, lw=0.8)
        for yi, (name, row) in zip(y, d.iterrows()):
            colour = figs.GREEN if row.loss_difference_baseline_minus_rdfl > 0 else figs.VERMILLION
            ax.plot([row.ci95_low, row.ci95_high], [yi, yi], color=colour, lw=2.2, solid_capstyle='butt')
            ax.plot([row.loss_difference_baseline_minus_rdfl], [yi], 'o', color=colour, ms=4,
                    markeredgecolor='white', markeredgewidth=0.6)
        ax.set_title(title, loc='left')
        ax.set_yticks(y)
        ax.set_yticklabels([labels[k] for k in order])
        ax.set_xlabel('Latent MSE difference (positive favours RDFL)')
        ax.set_ylim(-0.7, len(order) - 0.3)
    axes[1].text(0.99, 0.02, 'bars: 95% Monte Carlo intervals, 100 replications', transform=axes[1].transAxes,
                 ha='right', fontsize=6.4, color=figs.GREY, style='italic')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_r1_simulation')


CELL_LABELS = {
    'RDFL|retained|confidence': 'RDFL rule, retained, confidence (RDFL_SOFT)',
    'RDFL|retained|uniform': 'RDFL rule, retained, uniform',
    'RDFL|eligible|confidence': 'RDFL rule, eligible, confidence',
    'RDFL|eligible|uniform': 'RDFL rule, eligible, uniform',
    'ORD|retained|confidence': 'Ordinary rule, retained, confidence',
    'ORD|retained|uniform': 'Ordinary rule, retained, uniform',
    'ORD|eligible|confidence': 'Ordinary rule, eligible, confidence',
    'ORD|eligible|uniform': 'Ordinary rule, eligible, uniform',
    'RDFL|retained_without_low|confidence': 'RDFL rule, low response dropped',
    'RDFL|random_size_matched|confidence': 'RDFL rule, random size-matched',
    'ORD|ordinary_eligible|uniform': 'Ordinary rule, all cases (ORDINARY_SOFT)',
}


def ablation():
    c = pd.read_csv(R / 'rdfl_component_ablation/cells_vs_zero.csv')
    c = c[c.block_length == PRIMARY].set_index('cell')
    p = pd.read_csv(R / 'rdfl_component_ablation/predictions.csv')
    sd = p.groupby('cell').prediction.std() * 100
    outcome_sd = p.drop_duplicates(['Date', 'Symbol']).actual.std() * 100
    order = sd.reindex(c.index).sort_values().index.tolist()
    y = np.arange(len(order))[::-1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figs.COL_DOUBLE, 3.1), sharey=True,
                                 gridspec_kw={'width_ratios': [2.2, 1]})
    ax.axvspan(-MARGIN, MARGIN, color=figs.GREY, alpha=0.12, lw=0)
    ax.axvline(0, color=figs.GREY, lw=0.8)
    for yi, cell in zip(y, order):
        row = c.loc[cell]
        colour = figs.BLUE if cell.startswith('ORD') else figs.VERMILLION
        ax.plot([row.family_ci_low * 100, row.family_ci_high * 100], [yi, yi], color=colour, lw=2.0,
                solid_capstyle='butt')
        ax.plot([row.relative_improvement * 100], [yi], 'o', color=colour, ms=4, markeredgecolor='white',
                markeredgewidth=0.6)
        bx.barh(yi, sd[cell], color=colour, height=0.6, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([CELL_LABELS[k] for k in order])
    ax.set_xlabel('Relative loss improvement over the zero forecast (%)')
    ax.set_title('(a) Loss relative to the zero forecast', loc='left')
    bx.axvline(outcome_sd, color=figs.GREY, lw=0.8, ls=':')
    bx.text(outcome_sd, len(order) - 0.45, f' outcome SD {outcome_sd:.2f}%', fontsize=6.4, color=figs.GREY,
            va='bottom', ha='right')
    bx.set_xlabel('Prediction SD (% of daily return)')
    bx.set_title('(b) Forecast dispersion', loc='left')
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.text(0, -0.24, 'bars in (a): family-adjusted intervals over the eleven cells; shaded band: ±2% margin; '
            'cells ordered by prediction SD', transform=ax.transAxes, fontsize=6.3, color=figs.GREY, style='italic')
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig_r1_ablation')


ROWS = [('locked', 'Locked analysis (TF-IDF, uncalibrated)'),
        ('shift1', 'News moved one session later'),
        ('drop_ambiguous', 'Ambiguous timestamps dropped'),
        ('encoder_fixed', 'Transformer encoder, locked settings'),
        ('encoder_tuned', 'Transformer encoder, validated regularization'),
        ('high_volatility', 'High market volatility'),
        ('low_volatility', 'Low market volatility'),
        ('rising_market', 'Rising market'),
        ('falling_market', 'Falling market')]


def robustness_values():
    v = {}
    th = pd.read_csv(R / 'report/purged_pilot_paired_inference.csv')
    th = th[(th.block_length == PRIMARY) & (th.competitor == 'ZERO')].iloc[0]
    v[('thai', 'locked')] = (th.relative_improvement, th.relative_family_equivalence_ci_low,
                             th.relative_family_equivalence_ci_high)
    us = pd.read_csv(R / 'fnspid_replication/paired_inference.csv')
    us = us[(us.block_length == PRIMARY) & (us.analysis == 'uncalibrated') & (us.competitor == 'ZERO')].iloc[0]
    v[('us', 'locked')] = (us.relative_improvement, us.family_ci_low, us.family_ci_high)
    pick = lambda d: d[(d.block_length == PRIMARY) & (d.baseline == 'ZERO') & (d.variant == 'RDFL_SOFT')]
    # In availability_shift/paired_inference.csv the 'variant' column holds the mapping variant
    # (shift1, drop_ambiguous); RDFL_SOFT is the compared method in every row of that file.
    a = pd.read_csv(R / 'availability_shift/paired_inference.csv')
    for r in a[(a.block_length == PRIMARY) & (a.baseline == 'ZERO')].itertuples():
        v[(r.market, r.variant)] = (r.relative_improvement, r.family_ci_low, r.family_ci_high)
    for name, folder in [('encoder_fixed', 'encoder_sensitivity'), ('encoder_tuned', 'encoder_tuned')]:
        e = pd.read_csv(R / f'{folder}/paired_inference.csv')
        if 'family' in e:
            e = e[e.family == 'rdfl_vs_comparators']
        for r in pick(e).itertuples():
            v[(r.market, name)] = (r.relative_improvement, r.family_ci_low, r.family_ci_high)
    for r in pick(pd.read_csv(R / 'market_regimes/paired_inference.csv')).itertuples():
        v[(r.market, r.state)] = (r.relative_improvement, r.family_ci_low, r.family_ci_high)
    return v


def robustness():
    v = robustness_values()
    fig, axes = plt.subplots(1, 2, figsize=(figs.COL_DOUBLE, 3.0), sharey=True)
    y = np.arange(len(ROWS))[::-1]
    for ax, market, title in zip(axes, ['thai', 'us'], ['(a) Thai pilot', '(b) U.S. replication']):
        ax.axvspan(-MARGIN, MARGIN, color=figs.GREY, alpha=0.12, lw=0)
        ax.axvline(0, color=figs.GREY, lw=0.8)
        for yi, (key, _) in zip(y, ROWS):
            if (market, key) not in v:
                ax.text(-1, yi, 'not applicable (date-only records)', fontsize=6.3, color=figs.GREY, va='center',
                        ha='right', style='italic')
                continue
            est, lo, hi = (x * 100 for x in v[(market, key)])
            colour = figs.BLUE if key == 'locked' else figs.VERMILLION
            ax.plot([lo, hi], [yi, yi], color=colour, lw=2.0, solid_capstyle='butt')
            ax.plot([est], [yi], 'o', color=colour, ms=4, markeredgecolor='white', markeredgewidth=0.6)
            ax.text(max(hi, MARGIN) + 1.2, yi, minus(f'{est:+.1f}%'), fontsize=6.3, color=colour, va='center')
        ax.set_title(title, loc='left')
        ax.set_xlabel('RDFL_SOFT vs zero forecast (%)')
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in ROWS])
        ax.set_ylim(-0.7, len(ROWS) - 0.3)
        ax.axhline(len(ROWS) - 1.5, color=figs.GREY, lw=0.4, ls=':')
    axes[0].set_xlim(-100, 16)
    axes[1].set_xlim(-90, 16)
    axes[1].text(0.99, 0.02, 'negative favours zero; bars: family-adjusted intervals; band: ±2%', transform=axes[1].transAxes,
                 ha='right', fontsize=6.3, color=figs.GREY, style='italic')
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig_r1_robustness')


def main():
    figs.style()
    simulation()
    ablation()
    robustness()


if __name__ == '__main__':
    main()
