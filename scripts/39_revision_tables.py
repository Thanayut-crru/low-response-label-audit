"""Markdown tables for the revised manuscript and supplement, written straight from the result files
so that no number is transcribed by hand. Output: results/revision_tables/*.md
"""
from __future__ import annotations

import json

import pandas as pd

import common

R = common.ROOT / 'results'
OUT = R / 'revision_tables'
PRIMARY = 20
NAMES = {'ZERO': 'Zero forecast', 'PRICE_RIDGE': 'Price-only forecast', 'DIRECT_MARKET': 'Direct, market',
         'DIRECT_MARKET_TEXT': 'Direct, market + text', 'RDFL_MARKET': 'RDFL, market',
         'RDFL_MARKET_TEXT': 'RDFL, market + text', 'DIRECT_HUBER': 'DIRECT_HUBER', 'ORDINARY_SOFT': 'ORDINARY_SOFT',
         'HARD_RDFL': 'HARD_RDFL', 'RDFL_SOFT': 'RDFL_SOFT'}
MARKET = {'thai': 'Thai', 'us': 'U.S.'}


def pct(x, sign=True):
    s = f'{100 * x:+.2f}%' if sign else f'{100 * x:.2f}%'
    return s.replace('+', '+') if sign else s


def interval(lo, hi):
    return f'[{100 * lo:.2f}%, {100 * hi:.2f}%]'


def p(x):
    return f'{x:.4f}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---' if i < 2 else '---:' for i in range(len(header))]) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(lines)


def market_vs_text():
    d = pd.read_csv(R / 'market_vs_text/contrasts.csv')
    d = d[d.block_length == PRIMARY]
    rows = [[MARKET[r.analysis], NAMES[r.baseline], NAMES[r.variant], pct(r.relative_improvement),
             interval(r.family_ci_low, r.family_ci_high), p(r.holm_p), r.decision_2pct_family] for r in d.itertuples()]
    return table(['Market', 'Baseline', 'Variant', 'Variant vs Baseline', 'Family-Adjusted 98% Interval',
                  'Holm-Adjusted p', 'Decision'], rows)


def simulation_table():
    s = pd.read_csv(R / 'report/simulation_paired_inference.csv')
    order = ['anticipation', 'cancellation_050', 'cancellation_085', 'cancellation_100', 'high_noise',
             'informative_missing', 'null_effect', 'regime_shift', 'stale_price', 'strong_signal']
    label = {'anticipation': 'Anticipation', 'cancellation_050': 'Cancellation 0.50', 'cancellation_085': 'Cancellation 0.85',
             'cancellation_100': 'Cancellation 1.00', 'high_noise': 'High noise', 'informative_missing': 'Informative missingness',
             'null_effect': 'Null effect', 'regime_shift': 'Regime shift', 'stale_price': 'Stale price', 'strong_signal': 'Strong signal'}
    rows = []
    for sc in order:
        dr = s[(s.scenario == sc) & (s.competitor == 'DIRECT')].iloc[0]
        zr = s[(s.scenario == sc) & (s.competitor == 'ZERO')].iloc[0]
        rows.append([label[sc], f'{dr.loss_difference_baseline_minus_rdfl:.4f}', f'[{dr.ci95_low:.4f}, {dr.ci95_high:.4f}]',
                     p(dr.holm_p), f'{zr.loss_difference_baseline_minus_rdfl:.4f}', f'[{zr.ci95_low:.4f}, {zr.ci95_high:.4f}]',
                     p(zr.holm_p)])
    return table(['Scenario', 'Direct − RDFL', '95% MC Interval', 'Holm p', 'Zero − RDFL', '95% MC Interval', 'Holm p'], rows)


def source_timing():
    t = pd.read_csv(R / 'provenance_checks/timing_by_source.csv').set_index('Source')
    z = pd.read_csv(R / 'provenance_checks/timezone_scenarios.csv')
    risk = z.pivot_table(index='Source', columns='offset_hypothesis', values='articles_possibly_after_decision_time')
    zone = {'BangkokPost': 'Unknown (date only)', 'Reuters': 'UTC'}
    rows = []
    for src, r in t.iterrows():
        kind = 'Date only' if r.date_only_share == 1 else 'Date and clock time'
        med = '—' if r.date_only_share == 1 else f'{int(r.clock_median):02d}:{int(round(60 * (r.clock_median % 1))):02d}'
        rows.append([src, int(r.articles), kind, zone.get(src, 'ICT (UTC+7)'), med,
                     f'{100 * r.share_clock_00_06:.1f}% / {100 * r.share_clock_06_18:.1f}% / {100 * r.share_clock_18_24:.1f}%',
                     int(risk.loc[src, 'UTC']), int(risk.loc[src, 'UTC-3']), int(risk.loc[src, 'US_Eastern_standard_UTC-5'])])
    rows.append(['Total', int(t.articles.sum()), '', '', '', '', int(risk['UTC'].sum()), int(risk['UTC-3'].sum()),
                 int(risk['US_Eastern_standard_UTC-5'].sum())])
    return table(['Source', 'Mapped Articles', 'Timestamp', 'Inferred Time Zone', 'Median Clock Time',
                  'Clock 00–06 / 06–18 / 18–24', 'At Risk if UTC', 'At Risk if UTC−3', 'At Risk if UTC−5'], rows)


def family_rows(d, label_col_values):
    rows = []
    for r in d.itertuples():
        rows.append([*label_col_values(r), NAMES[r.baseline], pct(r.relative_improvement),
                     interval(r.family_ci_low, r.family_ci_high), p(r.holm_p), r.decision_2pct_family])
    return rows


def availability():
    d = pd.read_csv(R / 'availability_shift/paired_inference.csv')
    d = d[d.block_length == PRIMARY]
    v = pd.read_csv(R / 'availability_shift/variants.csv').set_index(['market', 'variant'])
    vname = {'shift1': 'One session later', 'drop_ambiguous': 'Ambiguous times dropped'}
    rows = family_rows(d, lambda r: [MARKET[r.market], vname[r.variant],
                                     f'{int(v.loc[(r.market, r.variant), "evaluated_stock_days"])}'])
    return table(['Market', 'Mapping', 'Stock-Days', 'Comparator', 'RDFL vs Comparator', 'Family-Adjusted 97.5% Interval',
                  'Holm-Adjusted p', 'Decision'], rows)


def encoder():
    fixed = pd.read_csv(R / 'encoder_sensitivity/paired_inference.csv')
    fixed = fixed[fixed.block_length == PRIMARY]
    tuned = pd.read_csv(R / 'encoder_tuned/paired_inference.csv')
    tuned = tuned[tuned.block_length == PRIMARY]
    rows = family_rows(fixed, lambda r: [MARKET[r.market], 'Locked settings', 'RDFL_SOFT'])
    t1 = tuned[tuned.family == 'rdfl_vs_comparators']
    rows += family_rows(t1, lambda r: [MARKET[r.market], 'Validated penalty', 'RDFL_SOFT'])
    t2 = tuned[(tuned.family == 'learner_vs_zero') & (tuned.variant != 'RDFL_SOFT')]   # RDFL_SOFT row given above
    for r in t2.itertuples():
        rows.append([MARKET[r.market], 'Validated penalty', NAMES[r.variant], 'Zero forecast', pct(r.relative_improvement),
                     interval(r.family_ci_low, r.family_ci_high), p(r.holm_p), r.decision_2pct_family])
    return table(['Market', 'Encoder Setting', 'Learner', 'Comparator', 'Learner vs Comparator',
                  'Family-Adjusted 97.5% Interval', 'Holm-Adjusted p', 'Decision'], rows)


def regimes():
    d = pd.read_csv(R / 'market_regimes/paired_inference.csv')
    d = d[d.block_length == PRIMARY]
    n = pd.read_csv(R / 'market_regimes/state_dates.csv').set_index(['market', 'state']).dates
    sname = {'high_volatility': 'High volatility', 'low_volatility': 'Low volatility', 'rising_market': 'Rising market',
             'falling_market': 'Falling market'}
    rows = family_rows(d, lambda r: [MARKET[r.market], sname[r.state], int(n[(r.market, r.state)])])
    return table(['Market', 'State', 'Dates', 'Comparator', 'RDFL vs Comparator', 'Family-Adjusted 97.5% Interval',
                  'Holm-Adjusted p', 'Decision'], rows)


def sim_robust():
    s = pd.read_csv(R / 'simulation_robustness/summary.csv')
    label = {'baseline': 'Baseline (Gaussian, c = 0.85, n = 2400)', 'noise_t5': 'Student-t noise, 5 df',
             'noise_t3': 'Student-t noise, 3 df', 'noise_hetero': 'Heteroskedastic noise', 'cancel_0': 'c = 0',
             'cancel_0.25': 'c = 0.25', 'cancel_0.5': 'c = 0.50', 'cancel_0.75': 'c = 0.75', 'cancel_1': 'c = 1.00',
             'n_600': 'n = 600', 'n_1200': 'n = 1200', 'n_4800': 'n = 4800', 'n_9600': 'n = 9600',
             'volume_sd_0.25': 'Volume noise SD 0.25', 'volume_sd_1.5': 'Volume noise SD 1.5',
             'rho_0.5': 'Equicorrelated features, ρ = 0.5'}
    rows = [[label[r.setting], f'{r.weighted_component_absence_error:.4f}', f'{r.component_prevalence:.4f}',
             f'{r.gap_error_minus_prevalence:+.4f}', f'{r.retained_fraction:.3f}', f'{r.rdfl_soft_latent_mse:.4f}',
             f'{r.direct_latent_mse:.4f}',
             f'{r.direct_minus_rdfl_mse:.4f} [{r.direct_minus_rdfl_mse_ci95_low:.4f}, {r.direct_minus_rdfl_mse_ci95_high:.4f}]']
            for r in s.itertuples()]
    return table(['Setting', 'Component-Absence Error', 'Prevalence', 'Error − Prevalence', 'Retained Fraction',
                  'RDFL Latent MSE', 'Direct Latent MSE', 'Direct − RDFL MSE [95% MC]'], rows)


def oracle():
    o = pd.read_csv(R / 'simulation_robustness/oracle_summary.csv')
    rows = [[f'{r.rule_accuracy:g}', 'Independent' if r.error_model == 'independent' else 'Concentrated on small net responses',
             f'{r.realized_accuracy_on_retained_low_response:.3f}',
             f'{r.weighted_component_absence_error_after:.4f} [{r.weighted_component_absence_error_after_ci95_low:.4f}, '
             f'{r.weighted_component_absence_error_after_ci95_high:.4f}]',
             f'{r.retained_fraction_after:.3f}', f'{r.rdfl_soft_latent_mse_after:.4f}'] for r in o.itertuples()]
    return table(['Accuracy q', 'Error Model', 'Realized Accuracy on Low-Response Cases', 'Component-Absence Error [95% MC]',
                  'Retained Fraction', 'RDFL Latent MSE'], rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {'table6_market_vs_text': market_vs_text(), 'S19_simulation': simulation_table(),
             'S20_source_timing': source_timing(), 'S21_availability': availability(), 'S22_encoder': encoder(),
             'S23_regimes': regimes(), 'S24_sim_robustness': sim_robust(), 'S25_oracle': oracle()}
    for name, text in parts.items():
        (OUT / f'{name}.md').write_text(text + '\n', encoding='utf-8')
    alphas = pd.read_csv(R / 'encoder_tuned/chosen_alphas.csv')
    (OUT / 'encoder_alphas.json').write_text(alphas.to_json(orient='records', indent=1), encoding='utf-8')
    print('\n\n'.join(f'## {k}\n{v}' for k, v in parts.items()))


if __name__ == '__main__':
    main()
