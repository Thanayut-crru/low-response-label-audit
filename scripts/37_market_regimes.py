"""Reviewer 1, comment 3: do the main comparisons depend on market conditions?

Evaluation dates of the locked uncalibrated runs (Thai fresh pilot, U.S. replication)
are split by two ex-ante market states, both computed only from information dated
before the session: high versus low market volatility (the lagged 22-session market
volatility feature, split at its median over evaluation dates) and rising versus
falling market (sign of the sum of the lagged market return over the previous 22
sessions). Within each state the locked family (RDFL_SOFT against four comparators)
is re-estimated with the same paired moving-block bootstrap on the dates of that state.
Dates within a state are not contiguous, so the block bootstrap is approximate there,
and the state-specific readings are descriptive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import revision_common as rc


def states(market):
    panel = pd.read_csv(rc.LABEL_DIRS[market] / 'stock_day_labels_pilot.csv', low_memory=False,
                        usecols=['Date', 'Symbol', 'x_set_ret', 'x_set_vol_22d'])
    panel['Date'] = pd.to_datetime(panel['Date'])
    by_date = panel.groupby('Date')[['x_set_ret', 'x_set_vol_22d']].median().sort_index()
    by_date['trend_22'] = by_date.x_set_ret.rolling(22, min_periods=22).sum()
    return by_date


def main():
    out = common.start_run('market_regimes')
    rows, shares = [], []
    for k, market in enumerate(['thai', 'us']):
        frame = pd.read_csv(rc.REFERENCE_PREDICTIONS[market])
        frame['Date'] = pd.to_datetime(frame['Date'])
        frame['loss'] = ((frame.actual - frame.prediction) / frame.stock_training_scale) ** 2
        wide = rc.daily(frame)
        s = states(market).reindex(wide.index)
        if s[['x_set_vol_22d', 'trend_22']].isna().any().any():
            raise ValueError(f'{market}: market state missing on an evaluation date')
        split = {'high_volatility': s.x_set_vol_22d > s.x_set_vol_22d.median(),
                 'low_volatility': s.x_set_vol_22d <= s.x_set_vol_22d.median(),
                 'rising_market': s.trend_22 > 0, 'falling_market': s.trend_22 <= 0}
        for j, (state, mask) in enumerate(split.items()):
            sub = wide.loc[mask.to_numpy()]
            shares.append({'market': market, 'state': state, 'dates': len(sub)})
            rows += [dict(r, market=market, state=state)
                     for r in rc.rdfl_vs_comparators(sub, 98000 + 1000 * k + 100 * j, f'{market}_{state}')]
    result = pd.DataFrame(rows)
    result.to_csv(out / 'paired_inference.csv', index=False)
    pd.DataFrame(shares).to_csv(out / 'state_dates.csv', index=False)
    primary = result[result.block_length == common.CONFIG['inference']['primary_block_length']]
    lines = ['# Market-state split (reviewer request, post-lock; descriptive)', '',
             common.markdown(pd.DataFrame(shares)), '',
             common.markdown(primary[['market', 'state', 'baseline', 'relative_improvement', 'family_ci_low',
                                      'family_ci_high', 'holm_p', 'decision_2pct_family']]), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, status='POST_LOCK_REVIEWER_REQUESTED_DESCRIPTIVE')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
