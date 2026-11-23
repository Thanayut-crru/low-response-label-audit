"""Reviewer 1, comment 4 (continued): transformer embeddings with inner-validated regularization.

35_encoder_sensitivity.py keeps every hyperparameter at its locked TF-IDF value. Dense
embeddings may need stronger regularization, so here each learner chooses its L2
penalty alpha from {1e-4, 1e-3, 1e-2, 1e-1} inside each training window: the last 20%
of training dates form a validation segment, separated from the inner training segment
by a five-session gap; the alpha with the lowest validation loss (normalized squared
error on the scale of Equation (5)) is refitted on the whole training window. No test
information enters the choice. Learners, labels, class centers, folds and inference are
otherwise the locked ones, and every learner is also compared with the zero forecast.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import revision_common as rc

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1]
GAP = 5
EMB = common.ROOT / 'results' / 'transformer_embeddings'
encoder = rc.load_module(common.ROOT / 'scripts/35_encoder_sensitivity.py', 'encoder')


def aligned(model, x):
    raw = model.predict_proba(x)
    out = np.zeros((x.shape[0], 3))
    for pos, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, pos]
    return out


def soft(x, q, w, xt, alpha, seeds):
    ex = sparse.vstack([x, x, x], format='csr')
    labels = np.repeat([0, 1, 2], x.shape[0])
    weights = np.concatenate([q[:, k] * w for k in range(3)])
    keep = weights > 1e-10
    preds = []
    for seed in seeds:
        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha, max_iter=3000, tol=1e-4, average=True,
                              random_state=seed)
        model.fit(ex[keep], labels[keep], sample_weight=weights[keep])
        preds.append(aligned(model, xt))
    return np.mean(preds, axis=0)


def hard(x, labels, xt, alpha, seeds):
    preds = []
    for seed in seeds:
        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha, max_iter=3000, tol=1e-4, average=True,
                              random_state=seed)
        model.fit(x, labels)
        preds.append(aligned(model, xt))
    return np.mean(preds, axis=0)


def direct(x, y, xt, alpha, seeds):
    center, scale = float(np.mean(y)), float(np.std(y, ddof=1))
    preds = []
    for seed in seeds:
        model = SGDRegressor(loss='huber', epsilon=1.35, penalty='l2', alpha=alpha, max_iter=3000, tol=1e-4,
                             average=True, random_state=seed)
        model.fit(x, (y - center) / scale)
        preds.append(model.predict(xt) * scale + center)
    return np.mean(preds, axis=0)


def learners(m, x, xt, train, alphas):
    seeds = m.SEEDS
    retained = train[train.retain_for_text_training]
    ordinary = train[np.isfinite(train[['residual_z', 'volume_deviation']].to_numpy(float)).all(axis=1)]
    yr = retained.target_open_close.to_numpy(float)
    q = retained[['p_negative', 'p_low_response', 'p_positive']].to_numpy(float)
    w = retained.label_confidence.to_numpy(float)
    h = retained.weak_label.map(m.LABEL_TO_INT).to_numpy(int)
    oq = m.ordinary_probabilities(ordinary)
    return {
        'DIRECT_HUBER': lambda: direct(x, train.target_open_close.to_numpy(float), xt, alphas['DIRECT_HUBER'], seeds),
        'RDFL_SOFT': lambda: soft(x[retained.index], q, w, xt, alphas['RDFL_SOFT'], seeds) @ m.class_centers(yr, q, w),
        'HARD_RDFL': lambda: hard(x[retained.index], h, xt, alphas['HARD_RDFL'], seeds) @ m.class_centers(yr, np.eye(3)[h]),
        'ORDINARY_SOFT': lambda: (soft(x[ordinary.index], oq, np.ones(len(ordinary)), xt, alphas['ORDINARY_SOFT'], seeds)
                                  @ m.class_centers(ordinary.target_open_close.to_numpy(float), oq)),
    }


def choose_alphas(m, matrix, train, scales):
    """Validation = last 20% of the dates that carry retained labels (labels start after the
    builder's warm-up); inner training ends five sessions before it. Too few inner labels ->
    the locked alpha is kept and recorded."""
    names = ['DIRECT_HUBER', 'RDFL_SOFT', 'HARD_RDFL', 'ORDINARY_SOFT']
    dates = np.sort(train.Date.unique())
    labelled = np.sort(train.loc[train.retain_for_text_training, 'Date'].unique())
    cut = labelled[int(np.floor(0.8 * len(labelled)))]
    inner_end = dates[max(0, np.searchsorted(dates, cut) - GAP)]
    inner = train[train.Date < inner_end].reset_index(drop=True)
    val = train[train.Date >= cut].reset_index(drop=True)
    counts = inner.loc[inner.retain_for_text_training, 'weak_label'].value_counts()
    if counts.sum() < 100 or any(counts.get(k, 0) < 10 for k in m.LABEL_TO_INT):
        return {name: 1e-4 for name in names} | {'fallback_locked_alpha': True}, {}
    scaler = StandardScaler()
    xi = sparse.csr_matrix(scaler.fit_transform(matrix[inner.row.to_numpy()]))
    xv = sparse.csr_matrix(scaler.transform(matrix[val.row.to_numpy()]))
    scale = val.Symbol.map(scales).to_numpy(float)
    losses = {}
    for alpha in ALPHAS:
        fits = learners(m, xi, xv, inner, {k: alpha for k in ['DIRECT_HUBER', 'RDFL_SOFT', 'HARD_RDFL', 'ORDINARY_SOFT']})
        for name, fit in fits.items():
            pred = fit()
            losses[(name, alpha)] = float(np.mean(((val.target_open_close.to_numpy(float) - pred) / scale) ** 2))
    chosen = {name: min(ALPHAS, key=lambda a: losses[(name, a)]) for name in names}
    return chosen | {'fallback_locked_alpha': False}, losses


def run(market):
    m, panel, bags = rc.load_market(market)
    matrix = encoder.bag_matrix(m, bags, market)
    bags = bags.copy()
    bags['row'] = np.arange(len(bags))
    parts, choices = [], []
    for block, train, test, scales in rc.folds(m, panel, bags):
        began = time.time()
        chosen, _ = choose_alphas(m, matrix, train, scales)
        scaler = StandardScaler()
        x = sparse.csr_matrix(scaler.fit_transform(matrix[train.row.to_numpy()]))
        xt = sparse.csr_matrix(scaler.transform(matrix[test.row.to_numpy()]))
        predictions = {'ZERO': np.zeros(len(test))}
        predictions.update({k: f() for k, f in learners(m, x, xt, train, chosen).items()})
        for method, prediction in predictions.items():
            parts.append(rc.frame_for(test, scales, block, method, prediction, market=market))
        choices.append({'market': market, 'block': block, **chosen})
        print(f'{market} {block}: alphas={chosen} ({time.time() - began:.0f}s)', flush=True)
    return pd.concat(parts, ignore_index=True), choices


def main():
    out = common.start_run('encoder_tuned')
    rows, frames, choices = [], [], []
    with threadpool_limits(limits=1):
        for offset, market in [(99000, 'thai'), (99500, 'us')]:
            frame, chosen = run(market)
            frames.append(frame)
            choices += chosen
            wide = rc.daily(frame)
            rows += [dict(r, market=market, family='rdfl_vs_comparators')
                     for r in rc.rdfl_vs_comparators(wide, offset, market)]
            rows += [dict(r, market=market, family='learner_vs_zero')
                     for r in rc.contrast_family(wide, [('ZERO', k) for k in ['DIRECT_HUBER', 'ORDINARY_SOFT', 'HARD_RDFL', 'RDFL_SOFT']],
                                                 offset + 250, market)]
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(out / 'predictions.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'paired_inference.csv', index=False)
    pd.DataFrame(choices).to_csv(out / 'chosen_alphas.csv', index=False)
    sd = predictions[predictions.method != 'ZERO'].groupby(['market', 'method']).prediction.std().unstack(0)
    sd.to_csv(out / 'prediction_sd.csv')
    primary = result[result.block_length == common.CONFIG['inference']['primary_block_length']]
    lines = ['# Encoder sensitivity with inner-validated regularization (reviewer request, post-lock)', '',
             common.markdown(pd.DataFrame(choices)), '',
             common.markdown(primary[['family', 'analysis', 'baseline', 'variant', 'relative_improvement',
                                      'family_ci_low', 'family_ci_high', 'holm_p', 'decision_2pct_family']]), '',
             'Prediction SD (return units):', '', common.markdown(sd.reset_index()), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, status='POST_LOCK_REVIEWER_REQUESTED', alphas=ALPHAS, inner_gap_sessions=GAP)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
