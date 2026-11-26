# Reproducing the Thai analysis without the restricted data

The Thai news and prices cannot be redistributed. This folder provides four routes to independent checking.

1. **Schema.** `DATA_DICTIONARY.md` documents every input file and column the pipeline reads.
2. **End-to-end run on synthetic inputs.** `python run_synthetic_pipeline.py` writes synthetic inputs with that
   schema (`make_synthetic_thai.py`), runs the archived label builder, trains the five learners on the locked folds
   with the five-session purge and exact-overlap exclusion, and estimates the locked family with the paired
   moving-block bootstrap. The synthetic text carries a planted signal, so the run is also a positive control for
   the empirical pipeline: RDFL_SOFT improves on the zero forecast by about 52% there.
3. **Inference from date-level losses.** `date_level_losses/` holds the date-level mean loss of every method or cell
   in the analyses behind Tables 3–6, S4, S8, S21 and S22 of the revised article and in the near-duplicate and U.S.
   component analyses. `python recompute_from_date_losses.py <preset>` recomputes the intervals, Holm-adjusted
   p-values and ±2% decisions of Tables 4–6, S21 and S22 from these files alone (presets are listed in the script).
   Estimates are exact; the revision analyses reproduce their published intervals exactly, and the locked analyses
   up to bootstrap Monte Carlo error, because their seeds were drawn in a different order. The validated-penalty
   rows of Table S22 come from the audit-corrected run (`revision_encoder_validated.csv`, deviation 9);
   `revision_encoder_validated_original.csv` and the preset `s22_encoder_validated_original` keep the original run.
4. **Rebuilding the corpus.** `thai_article_manifest.csv` lists every article with source, URL, recorded time,
   stock, decision date and a SHA-256 of the normalized text, but no text. Researchers with access to the
   providers can rebuild `All_news_sentiment.csv`, check it article by article against the hashes, and then run
   the full pipeline as described in the repository README.

`export_release_files.py` regenerates the date-level losses and the manifest from the restricted files.
