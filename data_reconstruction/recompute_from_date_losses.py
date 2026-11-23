"""Recompute the paired inference of the article from the released date-level losses alone.

usage: python recompute_from_date_losses.py <preset> [--block 20]
presets: table4_fresh, table4_archived, table5_uncalibrated, table5_calibrated, table6_market_vs_text,
         s21_availability, s22_encoder_locked, s22_encoder_validated

Every estimate is exact. The revision analyses (Table 6, S21, S22) use the seeds of their scripts and reproduce
the published intervals and p-values exactly; the locked analyses drew their bootstrap seeds in a different order,
so their intervals and p-values are reproduced up to bootstrap Monte Carlo error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
import revision_common as rc   # noqa: E402

D = HERE / 'date_level_losses'
LOCKED = [(c, 'RDFL_SOFT') for c in ['ZERO', 'DIRECT_HUBER', 'ORDINARY_SOFT', 'HARD_RDFL']]
MARKET = [('ZERO', 'DIRECT_MARKET'), ('ZERO', 'PRICE_RIDGE'), ('DIRECT_MARKET', 'DIRECT_MARKET_TEXT'),
          ('RDFL_MARKET', 'RDFL_MARKET_TEXT'), ('DIRECT_MARKET_TEXT', 'RDFL_MARKET_TEXT')]


def wide(file, key='method', **filters):
    d = pd.read_csv(D / file)
    for k, v in filters.items():
        d = d[d[k] == v]
    return d.pivot(index='Date', columns=key, values='mean_loss').sort_index()


def run(preset, block):
    out = []
    if preset == 'table4_fresh':
        out += rc.contrast_family(wide('thai_fresh_pilot.csv'), LOCKED, 0, preset, [block])
    elif preset == 'table4_archived':
        out += rc.contrast_family(wide('thai_archived_calibrated.csv'), LOCKED, 0, preset, [block])
    elif preset == 'table5_uncalibrated':
        out += rc.contrast_family(wide('us_uncalibrated.csv'), LOCKED, 0, preset, [block])
    elif preset == 'table5_calibrated':
        out += rc.contrast_family(wide('us_calibrated.csv'), LOCKED, 0, preset, [block])
    elif preset == 'table6_market_vs_text':
        for offset, market in [(91000, 'thai'), (92000, 'us')]:
            out += rc.contrast_family(wide('revision_market_vs_text.csv', market=market), MARKET, offset, market, [block])
    elif preset == 's21_availability':
        for k, (market, variant) in enumerate([('thai', 'shift1'), ('thai', 'drop_ambiguous'), ('us', 'shift1')]):
            out += rc.contrast_family(wide('revision_availability_shift.csv', market=market, variant=variant), LOCKED,
                                      93000 + 1000 * k, f'{market}_{variant}', [block])
    elif preset in ('s22_encoder_locked', 's22_encoder_validated'):
        file = 'revision_encoder_locked.csv' if preset.endswith('locked') else 'revision_encoder_validated.csv'
        offsets = {'s22_encoder_locked': (96000, 97000), 's22_encoder_validated': (99000, 99500)}[preset]
        for offset, market in zip(offsets, ['thai', 'us']):
            out += rc.contrast_family(wide(file, market=market), LOCKED, offset, market, [block])
    else:
        raise SystemExit(f'unknown preset {preset}')
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')
    parser.add_argument('--block', type=int, default=20)
    args = parser.parse_args()
    r = run(args.preset, args.block)
    cols = ['analysis', 'baseline', 'variant', 'relative_improvement', 'family_ci_low', 'family_ci_high', 'holm_p',
            'decision_2pct_family']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(r[cols].round(6).to_string(index=False))


if __name__ == '__main__':
    main()
