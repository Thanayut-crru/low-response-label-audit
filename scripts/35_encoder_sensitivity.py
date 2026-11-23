"""Reviewer 1, comment 4: does a pretrained transformer encoder change the conclusions?

The five locked learners are refitted on the locked folds (five-session purge,
exact-overlap exclusion, identical stock-days and loss scale) with the character
TF-IDF matrix replaced by frozen transformer embeddings from 34_extract_embeddings.py.
A bag embedding is the mean of its article embeddings; embeddings are standardized
inside each training window. Labels, learners, seeds, class-center mapping and
inference are unchanged, so the encoder is the only factor that varies.
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

EMB = common.ROOT / 'results' / 'transformer_embeddings'


def bag_matrix(m, bags, market):
    vectors = np.load(EMB / f'{market}_embeddings.npy')
    ids = pd.read_csv(EMB / f'{market}_article_ids.csv').article_id.tolist()
    index = {a: i for i, a in enumerate(ids)}
    articles = pd.read_csv(m.P1 / 'article_to_stock_day_pilot.csv', low_memory=False)
    articles['decision_date'] = pd.to_datetime(articles['decision_date'], errors='coerce')
    articles['model_text'] = articles['model_text'].fillna('').astype(str).str.strip()
    articles = articles.dropna(subset=['decision_date'])
    articles = articles[articles.model_text.ne('')]
    groups = articles.groupby(['Symbol', 'decision_date']).article_id.agg(list).to_dict()
    matrix = np.vstack([vectors[[index[a] for a in groups[(r.Symbol, r.Date)]]].mean(axis=0)
                        for r in bags.itertuples()])
    return matrix


def run(market):
    m, panel, bags = rc.load_market(market)
    matrix = bag_matrix(m, bags, market)
    bags = bags.copy()
    bags['row'] = np.arange(len(bags))
    parts = []
    for block, train, test, scales in rc.folds(m, panel, bags):
        began = time.time()
        scaler = StandardScaler()
        x = sparse.csr_matrix(scaler.fit_transform(matrix[train.row.to_numpy()]))
        xt = sparse.csr_matrix(scaler.transform(matrix[test.row.to_numpy()]))
        for method, prediction in rc.five_methods(m, x, xt, train).items():
            parts.append(rc.frame_for(test, scales, block, method, prediction, market=market))
        print(f'{market} {block}: train={len(train)} test={len(test)} ({time.time() - began:.0f}s)', flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    out = common.start_run('encoder_sensitivity')
    record = (EMB / 'COMPLETE.json').read_text(encoding='utf-8')
    rows, frames = [], []
    with threadpool_limits(limits=1):
        for offset, market in [(96000, 'thai'), (97000, 'us')]:
            frame = run(market)
            frames.append(frame)
            rows += [dict(r, market=market) for r in rc.rdfl_vs_comparators(rc.daily(frame), offset, market)]
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(out / 'predictions.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'paired_inference.csv', index=False)
    sd = predictions[predictions.method != 'ZERO'].groupby(['market', 'method']).prediction.std().unstack(0)
    sd.to_csv(out / 'prediction_sd.csv')
    primary = result[result.block_length == common.CONFIG['inference']['primary_block_length']]
    lines = ['# Encoder sensitivity: frozen transformer embeddings (reviewer request, post-lock)', '',
             f'Embeddings: {record}', '',
             common.markdown(primary[['analysis', 'baseline', 'relative_improvement', 'family_ci_low', 'family_ci_high',
                                      'holm_p', 'decision_2pct_family', 'smallest_margin_met']]), '',
             'Prediction SD (return units):', '', common.markdown(sd.reset_index()), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, status='POST_LOCK_REVIEWER_REQUESTED',
                    stock_days={k: int(v) for k, v in predictions[predictions.method == 'ZERO'].groupby('market').size().items()})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
