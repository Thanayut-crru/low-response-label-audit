"""Audit correction of 35b_encoder_tuned.py: a validation criterion that matches the evaluation estimand.

An external audit (2026-11-26), made after the 35b results were known, found two differences between the
penalty choice of 35b and the evaluation it feeds: (1) 35b averages the validation loss over stock-days,
whereas the evaluation averages losses within dates first and weights dates equally; (2) 35b scales the
validation loss by each stock's outcome standard deviation over the whole outer training window, which
includes the validation dates. Neither version uses the outer test block. This script changes only
these two points:

  * the scale is each stock's outcome standard deviation over valid dates before the end of the
    inner training segment, so no validation outcome enters it;
  * the validation loss is the mean over dates of the within-date mean scaled squared error.

Everything else (embeddings, learners, seeds, folds, outer scales, inference offsets) is 35b's. Its results
replace those of 35b in Table S22 and Figure 9; 35b and results/encoder_tuned are kept for traceability
(deviation 9). Output: results/encoder_tuned_corrected.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import revision_common as rc

tuned = rc.load_module(common.ROOT / 'scripts/35b_encoder_tuned.py', 'tuned')
encoder = tuned.encoder
NAMES = ['DIRECT_HUBER', 'RDFL_SOFT', 'HARD_RDFL', 'ORDINARY_SOFT']


def choose_alphas_matched(m, matrix, train, panel):
    dates = np.sort(train.Date.unique())
    labelled = np.sort(train.loc[train.retain_for_text_training, 'Date'].unique())
    cut = labelled[int(np.floor(0.8 * len(labelled)))]
    inner_end = dates[max(0, np.searchsorted(dates, cut) - tuned.GAP)]
    inner = train[train.Date < inner_end].reset_index(drop=True)
    val = train[train.Date >= cut].reset_index(drop=True)
    counts = inner.loc[inner.retain_for_text_training, 'weak_label'].value_counts()
    if counts.sum() < 100 or any(counts.get(k, 0) < 10 for k in m.LABEL_TO_INT):
        return {name: 1e-4 for name in NAMES} | {'fallback_locked_alpha': True}, {}
    valid = panel[panel.outcome_data_valid.astype(str).str.lower().eq('true')]
    inner_scales = (valid[valid.Date < inner_end].groupby('Symbol').target_open_close.std(ddof=1)
                    .clip(lower=1e-6))
    scale = val.Symbol.map(inner_scales).to_numpy(float)
    if not np.isfinite(scale).all():
        raise RuntimeError('A validation stock has no inner-window scale')
    scaler = StandardScaler()
    xi = sparse.csr_matrix(scaler.fit_transform(matrix[inner.row.to_numpy()]))
    xv = sparse.csr_matrix(scaler.transform(matrix[val.row.to_numpy()]))
    y = val.target_open_close.to_numpy(float)
    losses = {}
    for alpha in tuned.ALPHAS:
        fits = tuned.learners(m, xi, xv, inner, {k: alpha for k in NAMES})
        for name, fit in fits.items():
            unit = ((y - fit()) / scale) ** 2
            losses[(name, alpha)] = float(pd.Series(unit).groupby(val.Date.to_numpy()).mean().mean())
    chosen = {name: min(tuned.ALPHAS, key=lambda a: losses[(name, a)]) for name in NAMES}
    return chosen | {'fallback_locked_alpha': False}, losses


def run(market):
    m, panel, bags = rc.load_market(market)
    matrix = encoder.bag_matrix(m, bags, market)
    bags = bags.copy()
    bags['row'] = np.arange(len(bags))
    parts, choices = [], []
    for block, train, test, scales in rc.folds(m, panel, bags):
        began = time.time()
        chosen, _ = choose_alphas_matched(m, matrix, train, panel)
        scaler = StandardScaler()
        x = sparse.csr_matrix(scaler.fit_transform(matrix[train.row.to_numpy()]))
        xt = sparse.csr_matrix(scaler.transform(matrix[test.row.to_numpy()]))
        predictions = {'ZERO': np.zeros(len(test))}
        predictions.update({k: f() for k, f in tuned.learners(m, x, xt, train, chosen).items()})
        for method, prediction in predictions.items():
            parts.append(rc.frame_for(test, scales, block, method, prediction, market=market))
        choices.append({'market': market, 'block': block, **chosen})
        print(f'{market} {block}: alphas={chosen} ({time.time() - began:.0f}s)', flush=True)
    return pd.concat(parts, ignore_index=True), choices


def main():
    out = common.start_run('encoder_tuned_corrected')
    rows, frames, choices = [], [], []
    with threadpool_limits(limits=1):
        for offset, market in [(99000, 'thai'), (99500, 'us')]:   # the offsets of 35b
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
    lines = ['# Encoder sensitivity with inner-validated regularization, audit correction (post-lock)', '',
             'Validation scale from the inner training segment only; validation loss averaged within dates first.', '',
             common.markdown(pd.DataFrame(choices)), '',
             common.markdown(primary[['family', 'analysis', 'baseline', 'variant', 'relative_improvement',
                                      'family_ci_low', 'family_ci_high', 'holm_p', 'decision_2pct_family']]), '',
             'Prediction SD (return units):', '', common.markdown(sd.reset_index()), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, status='POST_LOCK_AUDIT_CORRECTION', alphas=tuned.ALPHAS, inner_gap_sessions=tuned.GAP,
                    change='inner-window scale and equal-date validation loss', replaces='encoder_tuned')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
