"""Reviewer 2, comment 3: market-feature model versus the same model augmented with text.

On the locked folds of both markets (five-session purge, exact-overlap exclusion,
identical evaluation stock-days and loss scale), each learner is fitted on

- MARKET: the ten lagged market features of the RDFL price-only forecast
  (standardized and median-imputed inside each training window);
- MARKET_TEXT: the same features plus the fold-fitted character TF-IDF matrix.

Two learners are used, the direct Huber regression (DIRECT) and RDFL soft
supervision (RDFL_SOFT), so that each pair differs only in whether text is added.
PRICE_RIDGE is the per-stock forward ridge forecast of RDFL step 2, which uses every
trading day and is available for every evaluated stock-day. The text-only direct
learner is refitted as a reproduction gate against the locked predictions.

Contrast family (relative improvement of the variant over the baseline, family of 5):
  DIRECT_MARKET vs ZERO, PRICE_RIDGE vs ZERO, DIRECT_MARKET_TEXT vs DIRECT_MARKET,
  RDFL_MARKET_TEXT vs RDFL_MARKET, RDFL_MARKET_TEXT vs DIRECT_MARKET_TEXT.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import revision_common as rc

CONTRASTS = [('ZERO', 'DIRECT_MARKET'), ('ZERO', 'PRICE_RIDGE'), ('DIRECT_MARKET', 'DIRECT_MARKET_TEXT'),
             ('RDFL_MARKET', 'RDFL_MARKET_TEXT'), ('DIRECT_MARKET_TEXT', 'RDFL_MARKET_TEXT')]


def soft_and_direct(m, x, xt, train):
    retained = train[train.retain_for_text_training]
    y_ret = retained.target_open_close.to_numpy(float)
    q = retained[['p_negative', 'p_low_response', 'p_positive']].to_numpy(float)
    w = retained.label_confidence.to_numpy(float)
    rdfl = m.fit_soft_ensemble(x[retained.index], q, w, xt) @ m.class_centers(y_ret, q, w)
    direct = m.fit_direct_ensemble(x, train.target_open_close.to_numpy(float), xt)
    return direct, rdfl


def run(market):
    m, panel, bags = rc.load_market(market)
    parts, gate = [], []
    for block, train, test, scales in rc.folds(m, panel, bags):
        began = time.time()
        market_model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        xm = sparse.csr_matrix(market_model.fit_transform(train[rc.MARKET_FEATURES].to_numpy(float)))
        xmt = sparse.csr_matrix(market_model.transform(test[rc.MARKET_FEATURES].to_numpy(float)))
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_df=.98,
                              max_features=20000, sublinear_tf=True, norm='l2', dtype=np.float32)
        x_text = vec.fit_transform(train.bag_text)
        xt_text = vec.transform(test.bag_text)
        x_both = sparse.hstack([x_text, xm], format='csr')
        xt_both = sparse.hstack([xt_text, xmt], format='csr')
        direct_market, rdfl_market = soft_and_direct(m, xm, xmt, train)
        direct_both, rdfl_both = soft_and_direct(m, x_both, xt_both, train)
        text_direct = m.fit_direct_ensemble(x_text, train.target_open_close.to_numpy(float), xt_text)
        price_ridge = test.price_only_prediction.to_numpy(float)
        if not np.isfinite(price_ridge).all():
            raise ValueError('Price-only forecast missing on an evaluated stock-day')
        predictions = {'ZERO': np.zeros(len(test)), 'PRICE_RIDGE': price_ridge,
                       'DIRECT_MARKET': direct_market, 'DIRECT_MARKET_TEXT': direct_both,
                       'RDFL_MARKET': rdfl_market, 'RDFL_MARKET_TEXT': rdfl_both}
        for method, prediction in predictions.items():
            parts.append(rc.frame_for(test, scales, block, method, prediction, market=market))
        gate.append(rc.frame_for(test, scales, block, 'DIRECT_HUBER', text_direct))
        print(f'{market} {block}: train={len(train)} test={len(test)} ({time.time() - began:.0f}s)', flush=True)
    frame = pd.concat(parts, ignore_index=True)
    gate = pd.concat(gate, ignore_index=True)
    reference = pd.read_csv(rc.REFERENCE_PREDICTIONS[market])
    reference = reference[reference.method == 'DIRECT_HUBER'].copy()
    reference['Date'] = pd.to_datetime(reference['Date'])
    merged = gate.merge(reference[['Date', 'Symbol', 'prediction']], on=['Date', 'Symbol'], suffixes=('', '_locked'),
                        validate='one_to_one')
    if len(merged) != len(gate) or len(gate) != len(reference):
        raise ValueError(f'{market}: evaluation units differ from the locked run')
    gap = float(np.max(np.abs(merged.prediction - merged.prediction_locked)))
    if gap > 1e-8:
        raise ValueError(f'{market}: text-only DIRECT_HUBER does not reproduce the locked run (gap {gap:.3g})')
    return frame, gap


def main():
    out = common.start_run('market_vs_text')
    rows, gaps, frames = [], {}, []
    with threadpool_limits(limits=1):
        for offset, market in [(91000, 'thai'), (92000, 'us')]:
            frame, gaps[market] = run(market)
            frames.append(frame)
            rows += rc.contrast_family(rc.daily(frame), CONTRASTS, offset, market)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(out / 'predictions.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'contrasts.csv', index=False)
    sd = predictions[predictions.method != 'ZERO'].groupby(['market', 'method']).prediction.std().unstack(0)
    sd.to_csv(out / 'prediction_sd.csv')
    primary = result[result.block_length == common.CONFIG['inference']['primary_block_length']]
    lines = ['# Market features versus market plus text (reviewer request, post-lock)', '',
             f'Reproduction gate (text-only DIRECT_HUBER vs locked predictions, max abs gap): {gaps}', '',
             common.markdown(primary[['analysis', 'baseline', 'variant', 'relative_improvement', 'family_ci_low',
                                      'family_ci_high', 'holm_p', 'decision_2pct_family', 'smallest_margin_met']]),
             '', 'Prediction SD (return units):', '', common.markdown(sd.reset_index()), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, reproduction_gap=gaps, status='POST_LOCK_REVIEWER_REQUESTED',
                    stock_days={k: int(v) for k, v in predictions[predictions.method == 'ZERO'].groupby('market').size().items()})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
