"""Write the redistributable files that stand in for the restricted Thai data (Reviewer 1, comment 2).

- date_level_losses/<analysis>.csv: for every analysis, the equal-weight date-level mean of the normalized squared
  loss (5) of every method or cell, with the number of stock-days behind each mean. Every paired interval, p-value
  and decision of the article is a function of these files (recompute_from_date_losses.py).
- thai_article_manifest.csv: one row per Thai article with source, URL, recorded time, stock and decision date,
  plus a SHA-256 of the normalized text, but no text.

Row-level predictions are not written: they contain realized returns of the restricted price panels.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / 'results'
OUT = HERE / 'date_level_losses'

SOURCES = {   # name: (file, key column, prediction column, loss column or None)
    'thai_fresh_pilot': (R / 'purged_pilot/predictions.csv', 'method', 'prediction', None),
    'thai_archived_calibrated': (ROOT / 'provenance/historical/p2c_forward_calibration/calibrated_predictions.csv',
                                 'method', 'calibrated_prediction', None),
    'thai_factorial_ablation': (R / 'factorial_ablation/predictions.csv', ['cell', 'method'], 'prediction', 'loss'),
    'thai_component_ablation': (R / 'rdfl_component_ablation/predictions.csv', 'cell', 'prediction', 'loss'),
    'label_threshold_sensitivity': (R / 'threshold_sensitivity/predictions.csv', ['dataset', 'cell'], 'prediction', 'loss'),
    'thai_near_duplicate': (R / 'near_duplicate_sensitivity/predictions.csv', 'method', 'prediction', None),
    'us_uncalibrated': (R / 'fnspid_replication/predictions.csv', 'method', 'prediction', None),
    'us_calibrated': (R / 'fnspid_replication/calibrated_predictions.csv', 'method', 'calibrated_prediction', None),
    'us_component_ablation': (R / 'fnspid_replication/component_predictions.csv', 'cell', 'prediction', None),
    'revision_market_vs_text': (R / 'market_vs_text/predictions.csv', ['market', 'method'], 'prediction', 'loss'),
    'revision_availability_shift': (R / 'availability_shift/predictions.csv', ['market', 'variant', 'method'],
                                    'prediction', 'loss'),
    'revision_encoder_locked': (R / 'encoder_sensitivity/predictions.csv', ['market', 'method'], 'prediction', 'loss'),
    # validated penalty: the audit-corrected run (35c); the original 35b run is kept for traceability (deviation 9)
    'revision_encoder_validated': (R / 'encoder_tuned_corrected/predictions.csv', ['market', 'method'], 'prediction',
                                   'loss'),
    'revision_encoder_validated_original': (R / 'encoder_tuned/predictions.csv', ['market', 'method'], 'prediction',
                                            'loss'),
}


def date_level(name, path, key, pred, loss):
    frame = pd.read_csv(path, low_memory=False)
    if key is None:   # threshold sensitivity: identify the setting columns present
        key = [c for c in ('market', 'setting', 'cell', 'method') if c in frame.columns]
    keys = [key] if isinstance(key, str) else list(key)
    if loss is None or loss not in frame:
        frame['loss'] = ((frame['actual'] - frame[pred]) / frame['stock_training_scale']) ** 2
    else:
        frame['loss'] = frame[loss]
    out = frame.groupby(keys + ['Date']).agg(mean_loss=('loss', 'mean'), stock_days=('loss', 'size')).reset_index()
    out.to_csv(OUT / f'{name}.csv', index=False, float_format='%.12g')
    return len(out)


def normalized_hash(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', str(text)).strip().casefold().encode('utf-8')).hexdigest()


def manifest():
    a = pd.read_csv(ROOT / 'provenance/historical/p1_pilot_labels/article_to_stock_day_pilot.csv', low_memory=False)
    a['text_sha256'] = a['model_text'].fillna('').map(normalized_hash)
    a['has_description'] = a['Description'].fillna('').astype(str).str.strip().ne('')
    keep = ['article_id', 'Source', 'Language', 'Symbol', 'published_raw', 'published_date', 'decision_date', 'URL',
            'has_description', 'text_sha256']
    a[keep].to_csv(HERE / 'thai_article_manifest.csv', index=False)
    return len(a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        if not spec[0].exists():
            print('missing, skipped:', name, file=sys.stderr)
            continue
        print(f'{name}: {date_level(name, *spec)} rows')
    print('manifest rows:', manifest())


if __name__ == '__main__':
    main()
