"""Synthetic Thai inputs with the exact schema of the restricted files (see DATA_DICTIONARY.md).

usage: python make_synthetic_thai.py <output folder>   (writes All_news_sentiment.csv and financial/panel_<SYMBOL>.csv)

Prices follow a random walk with a common market factor; volume is log-normal; the ten market features of the
label builder are computed from the synthetic prices exactly as their definitions state. News articles are
random strings drawn from a small vocabulary whose words carry a weak signal about the next session's return,
so that the pipeline has something to learn. Nothing here is derived from the real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ['BBL', 'KBANK', 'KTB', 'SCB', 'BAY', 'TTB', 'KKP', 'TISCO']
SOURCES = ['Kaohoon', 'Thunhoon', 'SET', 'Settrade', 'NationThailand', 'BangkokPost', 'Reuters']
START, END = '2023-04-24', '2026-04-24'
WORDS_UP = ['profit', 'growth', 'dividend', 'upgrade', 'record', 'expansion']
WORDS_DOWN = ['loss', 'provision', 'downgrade', 'default', 'decline', 'probe']
WORDS_NEUTRAL = ['bank', 'meeting', 'board', 'report', 'quarter', 'statement', 'market', 'branch', 'digital', 'loan']


def market(dates, rng):
    ret = rng.normal(0.0002, 0.009, len(dates))
    return pd.DataFrame({'Date': dates, 'set_ret': ret, 'set_vol_22d': pd.Series(ret).rolling(22).std(ddof=1)})


def panel(symbol, dates, mkt, rng):
    n = len(dates)
    beta = rng.uniform(0.7, 1.3)
    idio = rng.normal(0, 0.011, n)
    close = 30 * np.exp(np.cumsum(beta * mkt.set_ret.to_numpy() + idio))
    gap = rng.normal(0, 0.004, n)
    open_ = close * np.exp(-(beta * mkt.set_ret.to_numpy() + idio) + gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    volume = np.round(np.exp(rng.normal(15, 0.5, n)))
    f = pd.DataFrame({'Date': dates, 'Symbol': symbol, 'Open': open_, 'High': high, 'Low': low, 'Close': close,
                      'Volume': volume})
    f['log_ret'] = np.log(f.Close / f.Close.shift(1))
    f['vol_5d'] = f.log_ret.rolling(5).std(ddof=1)
    f['vol_22d'] = f.log_ret.rolling(22).std(ddof=1)
    f['mom_5d'] = f.log_ret.rolling(5).sum()
    f['mom_22d'] = f.log_ret.rolling(22).sum()
    f['hl_range'] = np.log(f.High / f.Low)
    f['turnover_rel'] = f.Volume / f.Volume.rolling(22).mean()
    f['set_ret'] = mkt.set_ret.to_numpy()
    f['set_vol_22d'] = mkt.set_vol_22d.to_numpy()
    f['beta_60d'] = f.log_ret.rolling(60).cov(f.set_ret) / f.set_ret.rolling(60).var()
    return f


def news(panels, rng):
    rows = []
    for symbol, f in panels.items():
        nxt = np.log(f.Close / f.Open).shift(-1).to_numpy()   # the next session's open-to-close return
        for i, date in enumerate(f.Date[:-1]):
            for _ in range(rng.poisson(0.55)):
                tilt = np.tanh(nxt[i] / 0.01) if np.isfinite(nxt[i]) else 0.0
                k = rng.integers(4, 9)
                words = []
                for _ in range(k):
                    u = rng.random()
                    if u < 0.15 * (1 + tilt):
                        words.append(rng.choice(WORDS_UP))
                    elif u < 0.30:
                        words.append(rng.choice(WORDS_DOWN))
                    else:
                        words.append(rng.choice(WORDS_NEUTRAL))
                clock = f'{rng.integers(6, 22):02d}:{rng.integers(0, 60):02d}:00'
                src = rng.choice(SOURCES)
                rows.append({'Date': f'{date.date()} {clock}', 'Symbol': symbol, 'Headline': f'{symbol} ' + ' '.join(words),
                             'Description': '' if src in ('SET', 'Settrade', 'Thunhoon') else ' '.join(rng.choice(WORDS_NEUTRAL, 6)),
                             'URL': f'https://example.invalid/{symbol}/{len(rows)}', 'Source': src, 'Language': 'TH'})
    return pd.DataFrame(rows)


def main():
    out = Path(sys.argv[1])
    (out / 'financial').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261119)
    dates = pd.bdate_range(START, END)
    mkt = market(dates, rng)
    panels = {s: panel(s, dates, mkt, rng) for s in SYMBOLS}
    for s, f in panels.items():
        f.to_csv(out / 'financial' / f'panel_{s}.csv', index=False)
    articles = news(panels, rng)
    articles.to_csv(out / 'All_news_sentiment.csv', index=False)
    print(f'synthetic inputs written to {out}: {len(dates)} sessions x {len(SYMBOLS)} stocks, {len(articles)} articles')


if __name__ == '__main__':
    main()
