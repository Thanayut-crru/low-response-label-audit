"""Shared helpers for the analyses added in the first revision (reviewer requests).

Everything here mirrors the locked pipeline: the fold rule, the five-session purge,
exact-overlap exclusion, the loss scale, the paired moving-block bootstrap, the
family-adjusted interval, Holm adjustment, and the +/-2% interval decision rule.
These analyses were run after all locked results were known and are logged as
reviewer-requested deviations in research/PREREGISTRATION_DEVIATIONS.md.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re

import numpy as np
import pandas as pd

import common

CONFIG = common.CONFIG
PURGE = 5
SCALE_PURGE = 5
MARGIN = 0.02
MARKET_FEATURES = ['x_log_ret', 'x_vol_5d', 'x_vol_22d', 'x_mom_5d', 'x_mom_22d', 'x_hl_range',
                   'x_turnover_rel', 'x_beta_60d', 'x_set_ret', 'x_set_vol_22d']
LABEL_DIRS = {'thai': common.ROOT / 'provenance/historical/p1_pilot_labels',
              'us': common.ROOT / 'results/fnspid_prepared/p1_labels'}
REFERENCE_PREDICTIONS = {'thai': common.ROOT / 'results/purged_pilot/predictions.csv',
                         'us': common.ROOT / 'results/fnspid_replication/predictions.csv'}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(text):
    normalized = re.sub(r'\s+', ' ', str(text)).strip().casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def outer_starts(market, m):
    if market == 'thai':
        return m.OUTER_STARTS
    prepared = json.loads((common.ROOT / 'results/fnspid_prepared/COMPLETE.json').read_text(encoding='utf-8'))
    replication = load_module(common.ROOT / 'scripts/21_fnspid_replication.py', 'replication')
    return replication.block_starts(prepared['window_start'], prepared['window_end'])


def legacy(label_dir):
    m = common.load_legacy('04_run_tfidf_label_methods.py')
    m.P1 = label_dir
    m.SEEDS = CONFIG['real_data']['seeds']
    return m


def attach_hashes(m, bags):
    articles = pd.read_csv(m.P1 / 'article_to_stock_day_pilot.csv', low_memory=False)
    articles['Date'] = pd.to_datetime(articles['decision_date'])
    articles['text_hash'] = articles['model_text'].fillna('').map(fingerprint)
    groups = articles.groupby(['Date', 'Symbol'])['text_hash'].agg(set).to_dict()
    bags = bags.copy()
    bags['text_hashes'] = [groups.get((r.Date, r.Symbol), set()) for r in bags.itertuples()]
    return bags


def load_market(market, label_dir=None):
    m = legacy(label_dir or LABEL_DIRS[market])
    panel, bags = m.load_bags()
    bags = attach_hashes(m, bags)
    m.OUTER_STARTS = outer_starts(market, m)
    return m, panel, bags


def folds(m, panel, bags, purge=PURGE, exclude=True):
    """Yield (block, train, test, scales) exactly as 13_factorial_ablation.train_config does."""
    valid = panel[panel.outcome_data_valid.astype(str).str.lower().eq('true')]
    dates = np.sort(valid.Date.unique())
    for start, stop in zip(m.OUTER_STARTS[:-1], m.OUTER_STARTS[1:]):
        prior = dates[dates < start.to_datetime64()]
        if len(prior) <= max(purge, SCALE_PURGE):
            continue
        boundary = pd.Timestamp(prior[-purge]) if purge else pd.Timestamp(start)
        scale_boundary = pd.Timestamp(prior[-SCALE_PURGE])
        train = bags[bags.Date < boundary].copy()
        test = bags[(bags.Date >= start) & (bags.Date < stop)].copy()
        if test.empty:
            continue
        if exclude:
            test_hashes = set().union(*test.text_hashes)
            train = train.loc[~train.text_hashes.map(lambda s: bool(s & test_hashes))]
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        retained = train[train.retain_for_text_training]
        ordinary = train[np.isfinite(train[['residual_z', 'volume_deviation']].to_numpy(float)).all(axis=1)]
        if len(retained) < 100 or len(ordinary) < 100:
            raise RuntimeError('Insufficient fold data; do not silently omit the fold')
        scales = valid[valid.Date < scale_boundary].groupby('Symbol').target_open_close.std(ddof=1).clip(lower=1e-6)
        yield str(start.date()), train, test, scales


def five_methods(m, x, xt, train):
    """The five locked learners on an arbitrary training matrix x and test matrix xt."""
    retained = train[train.retain_for_text_training]
    ordinary = train[np.isfinite(train[['residual_z', 'volume_deviation']].to_numpy(float)).all(axis=1)]
    y_ret = retained.target_open_close.to_numpy(float)
    q = retained[['p_negative', 'p_low_response', 'p_positive']].to_numpy(float)
    w = retained.label_confidence.to_numpy(float)
    rdfl = m.fit_soft_ensemble(x[retained.index], q, w, xt) @ m.class_centers(y_ret, q, w)
    hard = retained.weak_label.map(m.LABEL_TO_INT).to_numpy(int)
    hpred = m.fit_hard_ensemble(x[retained.index], hard, xt) @ m.class_centers(y_ret, np.eye(3)[hard])
    oq = m.ordinary_probabilities(ordinary)
    opred = (m.fit_soft_ensemble(x[ordinary.index], oq, np.ones(len(ordinary)), xt)
             @ m.class_centers(ordinary.target_open_close.to_numpy(float), oq))
    return {'ZERO': np.zeros(xt.shape[0]),
            'DIRECT_HUBER': m.fit_direct_ensemble(x, train.target_open_close.to_numpy(float), xt),
            'HARD_RDFL': hpred, 'ORDINARY_SOFT': opred, 'RDFL_SOFT': rdfl}


def frame_for(test, scales, block, method, prediction, **extra):
    frame = test[['Date', 'Symbol', 'target_open_close']].rename(columns={'target_open_close': 'actual'}).copy()
    frame['prediction'] = prediction
    frame['method'] = method
    frame['block'] = block
    frame['stock_training_scale'] = frame.Symbol.map(scales)
    frame['loss'] = ((frame.actual - frame.prediction) / frame.stock_training_scale) ** 2
    for key, value in extra.items():
        frame[key] = value
    return frame


def daily(frame, key='method'):
    if frame.duplicated(['Date', 'Symbol', key]).any():
        raise ValueError('Duplicate unit rows; inference would pseudo-replicate')
    wide = frame.pivot_table(index='Date', columns=key, values='loss', aggfunc='mean').sort_index()
    counts = frame.groupby(key).size()
    if counts.nunique() != 1:
        raise ValueError('Methods are not evaluated on identical stock-days')
    return wide


def paired(a, b, rng, block, family_size):
    """Relative improvement of b over a, (mean a - mean b) / mean a; same estimand as 13_factorial_ablation."""
    iterations = int(CONFIG['inference']['bootstrap_iterations'])
    diff, rel = common.bootstrap_paired(a, b, iterations, rng, block)
    observed = float(np.mean(a) - np.mean(b))
    finite = rel[np.isfinite(rel)]
    ci95 = common.interval(rel)
    family = common.interval(rel, .1 / family_size)
    return {'relative_improvement': observed / float(np.mean(a)), 'bootstrap_se': float(finite.std(ddof=1)),
            'relative_ci95_low': ci95[0], 'relative_ci95_high': ci95[1],
            'family_ci_low': family[0], 'family_ci_high': family[1],
            'raw_p': float((1 + np.sum(np.abs(diff - observed) >= abs(observed))) / (iterations + 1))}


def decision(holm_p, low, high, margin=MARGIN):
    """Interval reading of 'variant relative to baseline' (W/D/E/I codes of Table 3 and Table 4)."""
    if high < -margin:
        return 'W'   # variant worse beyond margin
    if low > margin:
        return 'B'   # variant better beyond margin
    if -margin < low and high < margin:
        return 'E'   # equivalent within margin
    if holm_p < .05:
        return 'D'   # difference detected, equivalence not shown
    return 'I'       # inconclusive


def contrast_family(wide, pairs, seed_offset, label, block_lengths=None):
    """pairs: list of (baseline, variant). One Holm family per block length."""
    rows = []
    for block in block_lengths or CONFIG['inference']['block_lengths']:
        group = []
        for k, (base, variant) in enumerate(pairs):
            rng = np.random.default_rng(CONFIG['inference']['seed'] + seed_offset + 17 * block + k)
            row = paired(wide[base].to_numpy(), wide[variant].to_numpy(), rng, block, len(pairs))
            row.update(analysis=label, baseline=base, variant=variant, block_length=block, n_dates=len(wide))
            group.append(row)
        for row, p in zip(group, common.holm([r['raw_p'] for r in group])):
            row['holm_p'] = float(p)
            row['decision_2pct_family'] = decision(p, row['family_ci_low'], row['family_ci_high'])
            row['smallest_margin_met'] = float(max(abs(row['family_ci_low']), abs(row['family_ci_high'])))
        rows += group
    return rows


def rdfl_vs_comparators(wide, seed_offset, label):
    """The locked primary family: RDFL_SOFT against the four comparators (RDFL is the variant)."""
    return contrast_family(wide, [(c, 'RDFL_SOFT') for c in ['ZERO', 'DIRECT_HUBER', 'ORDINARY_SOFT', 'HARD_RDFL']],
                           seed_offset, label)
