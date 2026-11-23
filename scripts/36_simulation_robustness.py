"""Reviewer 1, comment 3, and Reviewer 2, comment 4: robustness of the known-truth results.

Part A varies one assumption at a time around the cancellation-0.85 generator of
01_simulation.py (which the baseline setting reproduces exactly, checked first):
  - observation noise: Gaussian, Student-t with 5 and 3 degrees of freedom (unit
    variance), and heteroskedastic noise whose variance depends on a feature;
  - cancellation strength c in {0, 0.25, 0.5, 0.75, 0.85, 1};
  - training size n in {600, 1200, 2400, 4800, 9600};
  - volume noise SD in {0.25, 0.75, 1.5};
  - equicorrelated features (rho = 0.5), with the signal rescaled to unit variance.
Each setting has 100 replications on the seeds of the main simulation; the outputs are
the weighted component-absence error at delta = 0.35 against the component prevalence,
and held-out latent MSE of RDFL soft supervision, direct ridge and the zero forecast.

Part B repeats the component-flag oracle of 18_simulation_extensions.py (accuracy q)
with errors that are no longer independent: the error probability is proportional to
exp(-|S| / SD(S)), so flags are least reliable where the net response is small, with
the same average error 1 - q.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import common

CONFIG = common.CONFIG
DELTA = 0.35
BASE = dict(noise='gaussian', cancel=0.85, n_train=2400, volume_sd=0.75, rho=0.0)
SETTINGS = [('baseline', {})]
SETTINGS += [(f'noise_{k}', {'noise': k}) for k in ['t5', 't3', 'hetero']]
SETTINGS += [(f'cancel_{c:g}', {'cancel': c}) for c in [0.0, 0.25, 0.5, 0.75, 1.0]]
SETTINGS += [(f'n_{n}', {'n_train': n}) for n in [600, 1200, 4800, 9600]]
SETTINGS += [(f'volume_sd_{v:g}', {'volume_sd': v}) for v in [0.25, 1.5]]
SETTINGS += [('rho_0.5', {'rho': 0.5})]
ACCURACIES = [0.7, 0.9]


def features(rng, n, p, rho):
    z = rng.normal(size=(n, p))
    if rho == 0:
        return z
    return np.sqrt(1 - rho) * z + np.sqrt(rho) * rng.normal(size=(n, 1))


def noise(rng, n, kind, x):
    if kind == 'gaussian':
        return rng.normal(size=n)
    if kind in ('t5', 't3'):
        df = int(kind[1:])
        return rng.standard_t(df, size=n) * np.sqrt((df - 2) / df)
    if kind == 'hetero':
        return rng.normal(size=n) * np.sqrt(0.5 + 0.5 * x[:, 6] ** 2)
    raise ValueError(kind)


def generate(seed, n_test, p, noise_kind, cancel, n_train, volume_sd, rho):
    """Mirror of 01_simulation.generate for cancellation scenarios, with one option changed at a time."""
    rng = np.random.default_rng(seed)
    xtr = features(rng, n_train, p, rho)
    xte = features(rng, n_test, p, rho)
    beta = np.zeros(p)
    beta[:3] = [0.7, -0.5, 0.4]
    beta /= np.linalg.norm(beta)
    if rho:
        sigma = (1 - rho) * np.eye(p) + rho
        beta /= np.sqrt(beta @ sigma @ beta)

    def parts(x):
        first = x @ beta
        second = -cancel * first + (0.3 * x[:, 5] if cancel > 0 else 0.0)
        return first, second, first + second

    a, b, t = parts(xtr)
    _, _, te = parts(xte)
    y = t + 0.7 * noise(rng, n_train, noise_kind, xtr)
    ye = te + 0.7 * noise(rng, n_test, noise_kind, xte)
    volume = 0.8 * np.abs(t) + rng.normal(scale=volume_sd, size=n_train)
    return {'x': xtr, 'xt': xte, 'y': y, 'yt': ye, 'theta': t, 'thetat': te, 'a': a, 'b': b,
            'component': np.maximum(np.abs(a), np.abs(b)), 'volume': volume}


def labels(legacy, d):
    v, _, _, _ = legacy.robust_standardize(d['volume'], d['volume'])
    return legacy.rdfl_labels(d['y'], np.abs(v)), np.abs(v)


def low_mass(f, keep=None):
    keep = f['retain'] if keep is None else keep
    return f['probabilities'][:, 1] * f['confidence'] * keep


def absence_error(weights, d):
    mass = weights.sum()
    return float(weights @ (d['component'] > DELTA) / mass) if mass > 0 else np.nan


def soft_predict(legacy, x, xt, q, w, y):
    model = legacy.fit_soft_classifier(x, q, w)
    return legacy.predict_probabilities(model, xt) @ legacy.class_centers(y, q, w)


def replication(legacy, seed, name, options, cfg):
    s = {**BASE, **options}
    d = generate(seed, cfg['n_test'], cfg['features'], s['noise'], s['cancel'], s['n_train'], s['volume_sd'], s['rho'])
    f, _ = labels(legacy, d)
    sel = f['retain']
    rdfl = soft_predict(legacy, d['x'][sel], d['xt'], f['probabilities'][sel], f['confidence'][sel], d['y'][sel])
    direct = Ridge(alpha=1.0).fit(d['x'], d['y']).predict(d['xt'])
    mse = lambda p: float(np.mean((p - d['thetat']) ** 2))
    return {'setting': name, 'seed': seed, **s,
            'weighted_component_absence_error': absence_error(low_mass(f), d),
            'component_prevalence': float((d['component'] > DELTA).mean()),
            'retained_fraction': float(sel.mean()),
            'rdfl_soft_latent_mse': mse(rdfl), 'direct_latent_mse': mse(direct),
            'zero_latent_mse': mse(np.zeros(len(d['thetat'])))}


def oracle_replication(legacy, seed, q, model, cfg):
    d = generate(seed, cfg['n_test'], cfg['features'], **{k: BASE[k] for k in ['cancel', 'n_train', 'rho']},
                 noise_kind=BASE['noise'], volume_sd=BASE['volume_sd'])
    f, _ = labels(legacy, d)
    opposing = (d['a'] * d['b'] < 0) & (d['component'] > DELTA)
    rng = np.random.default_rng(seed + 7919)
    if model == 'independent':
        error = np.full(len(opposing), 1 - q)
    else:
        w = np.exp(-np.abs(d['theta']) / d['theta'].std(ddof=1))
        error = np.clip((1 - q) * w / w.mean(), 0, 1)
    # Same draw as 18_simulation_extensions.rule_replication: with independent errors this is identical.
    flagged = np.where(rng.random(len(opposing)) < 1 - error, opposing, ~opposing)
    keep = f['retain'] & ~(flagged & (f['hard'] == 1))
    pred = soft_predict(legacy, d['x'][keep], d['xt'], f['probabilities'][keep], f['confidence'][keep], d['y'][keep])
    low = f['retain'] & (f['hard'] == 1)
    return {'seed': seed, 'rule_accuracy': q, 'error_model': model,
            'realized_accuracy_on_retained_low_response': float((flagged == opposing)[low].mean()),
            'weighted_component_absence_error_after': absence_error(low_mass(f, keep), d),
            'retained_fraction_after': float(keep.mean()),
            'rdfl_soft_latent_mse_after': float(np.mean((pred - d['thetat']) ** 2))}


def mc(frame, by, columns):
    rows = []
    for key, group in frame.groupby(by, sort=False):
        row = dict(zip(by, key)) if isinstance(key, tuple) else {by[0]: key}
        row['replications'] = len(group)
        for column in columns:
            values = group[column].to_numpy(float)
            boot = values[np.random.default_rng(5).integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
            row[column] = float(values.mean())
            row[f'{column}_ci95_low'], row[f'{column}_ci95_high'] = common.interval(boot)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    out = common.start_run('simulation_robustness')
    cfg = CONFIG['simulation']
    legacy = common.load_legacy('06_run_known_truth_simulation.py')
    seeds = list(range(cfg['seed_start'], cfg['seed_start'] + cfg['replications']))

    # Reproduction gate: the baseline setting equals the archived cancellation-0.85 scenario.
    reference = pd.read_csv(common.ROOT / 'results/simulation/absence_diagnostics.csv')
    reference = reference[(reference.scenario == 'cancellation_085') & (reference.threshold_definition == 'fixed_0.35')]
    for seed in seeds[:5]:
        d = generate(seed, cfg['n_test'], cfg['features'], **{k if k != 'noise' else 'noise_kind': v for k, v in BASE.items()})
        f, _ = labels(legacy, d)
        archived = float(reference.loc[reference.seed == seed, 'weighted_component_absence_error'].iloc[0])
        if abs(absence_error(low_mass(f), d) - archived) > 1e-12:
            raise ValueError(f'baseline does not reproduce the archived diagnostic for seed {seed}')
    print('Reproduction gate passed', flush=True)

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('error', category=ConvergenceWarning)
    rows, oracle = [], []
    with threadpool_limits(limits=1):
        for name, options in SETTINGS:
            began = time.time()
            rows += [replication(legacy, s, name, options, cfg) for s in seeds]
            print(f'{name}: done ({time.time() - began:.0f}s)', flush=True)
        for q in ACCURACIES:
            for model in ['independent', 'correlated']:
                oracle += [oracle_replication(legacy, s, q, model, cfg) for s in seeds]
                print(f'oracle q={q} {model}: done', flush=True)
    rows, oracle = pd.DataFrame(rows), pd.DataFrame(oracle)
    rows['gap_error_minus_prevalence'] = rows.weighted_component_absence_error - rows.component_prevalence
    rows['direct_minus_rdfl_mse'] = rows.direct_latent_mse - rows.rdfl_soft_latent_mse
    rows['zero_minus_rdfl_mse'] = rows.zero_latent_mse - rows.rdfl_soft_latent_mse
    rows.to_csv(out / 'replications.csv', index=False)
    oracle.to_csv(out / 'oracle_replications.csv', index=False)
    summary = mc(rows, ['setting'], ['weighted_component_absence_error', 'component_prevalence',
                                     'gap_error_minus_prevalence', 'retained_fraction', 'rdfl_soft_latent_mse',
                                     'direct_latent_mse', 'direct_minus_rdfl_mse', 'zero_minus_rdfl_mse'])
    summary.to_csv(out / 'summary.csv', index=False)
    osum = mc(oracle, ['rule_accuracy', 'error_model'], ['realized_accuracy_on_retained_low_response',
                                                         'weighted_component_absence_error_after',
                                                         'retained_fraction_after', 'rdfl_soft_latent_mse_after'])
    osum.to_csv(out / 'oracle_summary.csv', index=False)
    lines = ['# Simulation robustness (reviewer request, post-lock)', '', common.markdown(summary[[
        'setting', 'weighted_component_absence_error', 'component_prevalence', 'gap_error_minus_prevalence',
        'retained_fraction', 'rdfl_soft_latent_mse', 'direct_latent_mse', 'direct_minus_rdfl_mse',
        'direct_minus_rdfl_mse_ci95_low', 'direct_minus_rdfl_mse_ci95_high']]), '',
        '## Component-flag oracle with correlated errors', '', common.markdown(osum[[
            'rule_accuracy', 'error_model', 'realized_accuracy_on_retained_low_response',
            'weighted_component_absence_error_after', 'retained_fraction_after', 'rdfl_soft_latent_mse_after']]), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, status='POST_LOCK_REVIEWER_REQUESTED', settings=[s for s, _ in SETTINGS],
                    replications=cfg['replications'], delta=DELTA)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
