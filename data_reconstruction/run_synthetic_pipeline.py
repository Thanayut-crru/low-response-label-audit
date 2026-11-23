"""Run the complete Thai pipeline end to end on synthetic inputs (no restricted data needed).

usage: python run_synthetic_pipeline.py [work folder]   (default: data_reconstruction/synthetic_run)

Steps, all with the released code and unchanged settings:
1. make_synthetic_thai.py writes inputs with the schema of DATA_DICTIONARY.md;
2. the archived label builder (provenance/source_code/01_build_pilot_labels.py) maps articles to sessions and
   builds the RDFL labels;
3. the five learners are trained on the locked folds with the five-session purge and exact-overlap exclusion,
   and the locked family (RDFL_SOFT against four comparators) is estimated with the paired moving-block bootstrap.
The numbers describe synthetic data only; the run shows that the pipeline executes from raw files to decisions.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
import common              # noqa: E402
import revision_common as rc   # noqa: E402


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'synthetic_run'
    raw = work / 'raw'
    subprocess.run([sys.executable, str(HERE / 'make_synthetic_thai.py'), str(raw)], check=True)
    builder = common.load_legacy('01_build_pilot_labels.py')
    builder.RAW, builder.NEWS_PATH, builder.FIN_DIR = raw, raw / 'All_news_sentiment.csv', raw / 'financial'
    builder.OUT = work / 'p1_pilot_labels'
    builder.main()
    m, panel, bags = rc.load_market('thai', builder.OUT)
    parts = []
    for block, train, test, scales in rc.folds(m, panel, bags):
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_df=.98, max_features=20000,
                              sublinear_tf=True, norm='l2', dtype=np.float32)
        x, xt = vec.fit_transform(train.bag_text), vec.transform(test.bag_text)
        for method, prediction in rc.five_methods(m, x, xt, train).items():
            parts.append(rc.frame_for(test, scales, block, method, prediction))
        print(f'block {block}: train={len(train)} test={len(test)}', flush=True)
    frame = pd.concat(parts, ignore_index=True)
    result = pd.DataFrame(rc.contrast_family(rc.daily(frame), [(c, 'RDFL_SOFT') for c in
                                                              ['ZERO', 'DIRECT_HUBER', 'ORDINARY_SOFT', 'HARD_RDFL']],
                                             0, 'synthetic', [20]))
    result.to_csv(work / 'synthetic_inference.csv', index=False)
    print(result[['baseline', 'relative_improvement', 'family_ci_low', 'family_ci_high', 'holm_p',
                  'decision_2pct_family']].round(4).to_string(index=False))
    print(f'end-to-end run complete: {frame.Date.nunique()} dates, {(frame.method == "ZERO").sum()} stock-days')


if __name__ == '__main__':
    main()
