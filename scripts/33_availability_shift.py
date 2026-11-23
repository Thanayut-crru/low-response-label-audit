"""Reviewer 2, comment 1: sensitivity of the main comparisons to news availability.

The locked mapping assigns an article recorded on calendar date d to the first
observed session strictly after d. Two perturbations are evaluated:

- shift1 (both markets): every article is moved one further observed session later,
  so an article reaches its session only if it was available at least one full
  trading session plus the locked lag before the open;
- drop_ambiguous (Thai): articles whose recorded time cannot rule out arrival after
  the 09:00 ICT decision time are dropped: date-only records, and records that would
  arrive after the decision time under any alternative source time zone (UTC, UTC-3,
  UTC-4, UTC-5), with the rule of 14_provenance_checks.py.

The fuzzy memberships depend only on prices and volume, so they are unchanged; the
news-dependent retention rule (confidence >= 0.70, news present, valid outcome) is
re-applied after remapping, which is checked first on the locked mapping. The five
learners, folds, purge, exclusion and inference are the locked ones; the evaluated
stock-days change with the mapping and are reported.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

import common
import revision_common as rc

DECISION_HOUR_ICT = 9.0
OFFSETS = {'UTC': 0, 'UTC-3': -3, 'UTC-4': -4, 'UTC-5': -5}
CONFIDENCE_MIN = 0.70
LABELS = np.array(['NEGATIVE', 'LOW_RESPONSE', 'POSITIVE'], dtype=object)


def relabel(panel, articles):
    """Re-apply the news-dependent part of the label builder to a remapped article table."""
    panel = panel.copy()
    mapped = articles.dropna(subset=['decision_date'])
    counts = mapped.groupby(['Symbol', 'decision_date']).size().rename('news_count_new').reset_index()
    counts = counts.rename(columns={'decision_date': 'Date'})
    panel = panel.merge(counts, on=['Symbol', 'Date'], how='left', validate='one_to_one')
    panel['news_count'] = panel.pop('news_count_new').fillna(0).astype(int)
    panel['has_news'] = panel['news_count'] > 0
    p = panel[['p_negative', 'p_low_response', 'p_positive']].to_numpy(float)
    valid = panel['outcome_data_valid'].astype(str).str.lower().eq('true').to_numpy()
    conf = panel['label_confidence'].to_numpy(float)
    finite = np.isfinite(p).all(axis=1) & np.isfinite(conf)
    keep = finite & (np.nan_to_num(conf) >= CONFIDENCE_MIN) & panel['has_news'].to_numpy() & valid
    label = np.full(len(panel), 'ABSTAIN', dtype=object)
    label[keep] = LABELS[np.argmax(np.nan_to_num(p[keep]), axis=1)]
    panel['weak_label'] = label
    panel['retain_for_text_training'] = keep
    return panel


def calendars(panel):
    return {s: np.array(sorted(g.Date.unique()), dtype='datetime64[ns]') for s, g in panel.groupby('Symbol')}


def shift_one_session(articles, cal):
    articles = articles.copy()
    new = np.full(len(articles), np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
    for symbol, idx in articles.groupby('Symbol').groups.items():
        dates = articles.loc[idx, 'decision_date'].to_numpy(dtype='datetime64[ns]')
        calendar = cal[symbol]
        ok = ~np.isnat(dates)
        pos = np.searchsorted(calendar, dates[ok], side='right')
        mapped = np.full(ok.sum(), np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
        inside = pos < len(calendar)
        mapped[inside] = calendar[pos[inside]]
        block = np.full(len(idx), np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
        block[ok] = mapped
        new[articles.index.get_indexer(idx)] = block
    articles['decision_date'] = pd.to_datetime(new)
    return articles


def ambiguous(articles):
    stamp = pd.to_datetime(articles.published_raw.astype(str).str.slice(0, 19), errors='coerce')
    clock = stamp.dt.hour + stamp.dt.minute / 60 + stamp.dt.second / 3600
    date_only = (stamp.dt.hour == 0) & (stamp.dt.minute == 0) & (stamp.dt.second == 0)
    worst = np.where(date_only, 24.0, clock)
    gap = (articles.decision_date - pd.to_datetime(articles.published_date)).dt.days
    risk = np.zeros(len(articles), dtype=bool)
    for offset in OFFSETS.values():
        risk |= (worst + 7 - offset >= 24 * gap + DECISION_HOUR_ICT).to_numpy()
    return date_only.to_numpy() | risk, date_only.to_numpy()


def build_variant(market, variant, out):
    source = rc.LABEL_DIRS[market]
    panel = pd.read_csv(source / 'stock_day_labels_pilot.csv', low_memory=False)
    panel['Date'] = pd.to_datetime(panel['Date'])
    articles = pd.read_csv(source / 'article_to_stock_day_pilot.csv', low_memory=False)
    articles['decision_date'] = pd.to_datetime(articles['decision_date'], errors='coerce')
    # Gate: re-applying the retention rule to the locked mapping must reproduce the locked labels.
    check = relabel(panel, articles)
    locked = panel.retain_for_text_training.astype(str).str.lower().eq('true').to_numpy()
    if not (np.array_equal(check.retain_for_text_training.to_numpy(), locked)
            and (check.weak_label.astype(str).to_numpy() == panel.weak_label.astype(str).to_numpy()).all()):
        raise ValueError(f'{market}: relabelling the locked mapping does not reproduce the locked labels')
    info = {'market': market, 'variant': variant, 'articles': len(articles),
            'articles_mapped_locked': int(articles.decision_date.notna().sum())}
    if variant == 'shift1':
        articles = shift_one_session(articles, calendars(panel))
    elif variant == 'drop_ambiguous':
        flag, date_only = ambiguous(articles)
        info['articles_dropped'] = int((flag & articles.decision_date.notna()).sum())
        info['dropped_date_only'] = int((date_only & articles.decision_date.notna()).sum())
        articles.loc[flag, 'decision_date'] = pd.NaT
    else:
        raise ValueError(variant)
    info['articles_mapped_variant'] = int(articles.decision_date.notna().sum())
    relabelled = relabel(panel, articles)
    info['retained_labels_locked'] = int(locked.sum())
    info['retained_labels_variant'] = int(relabelled.retain_for_text_training.sum())
    target = out / f'{market}_{variant}'
    target.mkdir(parents=True, exist_ok=True)
    relabelled.to_csv(target / 'stock_day_labels_pilot.csv', index=False, encoding='utf-8-sig')
    articles.to_csv(target / 'article_to_stock_day_pilot.csv', index=False, encoding='utf-8-sig')
    return target, info


def evaluate(market, label_dir):
    m, panel, bags = rc.load_market(market, label_dir)
    parts = []
    for block, train, test, scales in rc.folds(m, panel, bags):
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_df=.98,
                              max_features=20000, sublinear_tf=True, norm='l2', dtype=np.float32)
        x = vec.fit_transform(train.bag_text)
        xt = vec.transform(test.bag_text)
        for method, prediction in rc.five_methods(m, x, xt, train).items():
            parts.append(rc.frame_for(test, scales, block, method, prediction))
    return pd.concat(parts, ignore_index=True)


def main():
    out = common.start_run('availability_shift')
    rows, infos, frames = [], [], []
    with threadpool_limits(limits=1):
        for k, (market, variant) in enumerate([('thai', 'shift1'), ('thai', 'drop_ambiguous'), ('us', 'shift1')]):
            began = time.time()
            label_dir, info = build_variant(market, variant, out)
            frame = evaluate(market, label_dir)
            frame['market'], frame['variant'] = market, variant
            frames.append(frame)
            wide = rc.daily(frame)
            info.update(evaluated_stock_days=int((frame.method == 'ZERO').sum()), evaluated_dates=len(wide))
            infos.append(info)
            rows += [dict(r, market=market, variant=variant) for r in rc.rdfl_vs_comparators(wide, 93000 + 1000 * k, f'{market}_{variant}')]
            print(f'{market} {variant}: {info} ({time.time() - began:.0f}s)', flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / 'predictions.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'paired_inference.csv', index=False)
    pd.DataFrame(infos).to_csv(out / 'variants.csv', index=False)
    primary = result[result.block_length == common.CONFIG['inference']['primary_block_length']]
    lines = ['# News-availability sensitivity (reviewer request, post-lock)', '',
             common.markdown(pd.DataFrame(infos)), '',
             common.markdown(primary[['analysis', 'baseline', 'relative_improvement', 'family_ci_low', 'family_ci_high',
                                      'holm_p', 'decision_2pct_family']]), '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    common.complete(out, status='POST_LOCK_REVIEWER_REQUESTED', variants=infos)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
