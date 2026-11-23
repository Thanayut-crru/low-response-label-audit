# Data dictionary of the Thai inputs

The Thai news records and price panels are licensed from their providers and are not redistributed. This file
describes every input the pipeline reads, so that the files can be rebuilt from the providers or replaced by
the synthetic inputs of `make_synthetic_thai.py`, which follow the same schema.

## `data/raw/All_news_sentiment.csv` — one row per news record

| Column | Type | Meaning |
|---|---|---|
| `Date` | text `YYYY-MM-DD HH:MM:SS[.fff]` or `YYYY-MM-DD` | Recorded publication time as given by the source (Thai local time for Kaohoon, Thunhoon, SET, Settrade and Nation Thailand; UTC for Reuters; date only for Bangkok Post; Table S20) |
| `Symbol` | text | SET ticker of the bank the record is filed under: BAY, BBL, KBANK, KKP, KTB, SCB, TISCO, TTB |
| `Headline` | text | Headline |
| `Description` | text, may be empty | Lead or summary; empty for SET, Settrade and Thunhoon, which carry headlines only |
| `URL` | text | Source URL |
| `Source` | text | Kaohoon, Thunhoon, SET, Settrade, NationThailand, BangkokPost, Reuters |
| `Language` | text | `TH` or `EN` |

Further columns of the provider file (model scores of earlier projects) are ignored. Records are deduplicated on
(`Date`, `Symbol`, `Headline`, `URL`); the model text is `Headline` + `Description`.

## `data/raw/financial/panel_<SYMBOL>.csv` — one row per stock and trading date

| Column | Meaning |
|---|---|
| `Date` | Trading date (`YYYY-MM-DD`) |
| `Open`, `High`, `Low`, `Close` | Daily prices (back-adjusted by the provider; the target log(Close/Open) is invariant to a multiplicative adjustment) |
| `Volume` | Shares traded; rows with zero volume are invalid outcomes |
| `log_ret` | log(Close_t / Close_{t−1}) |
| `vol_5d`, `vol_22d` | Rolling standard deviation of `log_ret` over 5 and 22 sessions |
| `mom_5d`, `mom_22d` | Rolling sum of `log_ret` over 5 and 22 sessions |
| `hl_range` | log(High / Low) |
| `turnover_rel` | Volume relative to its 22-session mean |
| `beta_60d` | 60-session rolling beta of `log_ret` on `set_ret` |
| `set_ret` | Daily log return of the SET index |
| `set_vol_22d` | 22-session rolling standard deviation of `set_ret` |

The label builder lags every feature by one session (`x_<feature>`), so a stock-day's features use information up
to the previous close. Other provider columns are ignored.

## Derived files (written by the pipeline, not redistributed)

- `provenance/historical/p1_pilot_labels/stock_day_labels_pilot.csv`: one row per stock-day with the target, the
  lagged features, the price-only forecast and residual, the fuzzy memberships, the soft label, the confidence and
  the retention flag (Supplementary Section S11).
- `provenance/historical/p1_pilot_labels/article_to_stock_day_pilot.csv`: one row per deduplicated article with its
  mapped decision date; `thai_article_manifest.csv` is this file without text.

## Released stand-ins

- `thai_article_manifest.csv`: article id, source, language, stock, recorded time, decision date, URL, whether a
  description exists, and the SHA-256 of the normalized model text (whitespace collapsed, case folded), which lets
  a rebuilt corpus be matched article by article without redistributing text.
- `date_level_losses/*.csv`: the date-level mean loss (5) of every method or cell in every analysis, from which
  `recompute_from_date_losses.py` recomputes the paired inference.
