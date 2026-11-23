# Encoder sensitivity with inner-validated regularization (reviewer request, post-lock)

| market | block | DIRECT_HUBER | RDFL_SOFT | HARD_RDFL | ORDINARY_SOFT | fallback_locked_alpha |
| --- | --- | --- | --- | --- | --- | --- |
| thai | 2024-08-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| thai | 2024-11-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| thai | 2025-02-01 | 0.1 | 0.1 | 0.1 | 0.01 | False |
| thai | 2025-05-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| thai | 2025-08-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| thai | 2025-11-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| thai | 2026-02-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2018-10-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2019-01-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2019-04-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2019-07-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2019-10-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2020-01-01 | 0.1 | 0.1 | 0.1 | 0.1 | False |
| us | 2020-04-01 | 0.1 | 0.1 | 0.1 | 0.01 | False |

| family | analysis | baseline | variant | relative_improvement | family_ci_low | family_ci_high | holm_p | decision_2pct_family |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rdfl_vs_comparators | thai | ZERO | RDFL_SOFT | -0.119112 | -0.183475 | -0.0700071 | 0.00039996 | W |
| rdfl_vs_comparators | thai | DIRECT_HUBER | RDFL_SOFT | 0.00706459 | -0.062906 | 0.0658998 | 0.806719 | I |
| rdfl_vs_comparators | thai | ORDINARY_SOFT | RDFL_SOFT | -0.100079 | -0.158139 | -0.0562474 | 0.00039996 | W |
| rdfl_vs_comparators | thai | HARD_RDFL | RDFL_SOFT | 0.0095461 | 0.00311083 | 0.016467 | 0.00159984 | E |
| learner_vs_zero | thai | ZERO | DIRECT_HUBER | -0.127074 | -0.18713 | -0.0783805 | 0.00039996 | W |
| learner_vs_zero | thai | ZERO | ORDINARY_SOFT | -0.0173016 | -0.0305941 | -0.00523306 | 0.00429957 | D |
| learner_vs_zero | thai | ZERO | HARD_RDFL | -0.129898 | -0.19814 | -0.0788507 | 0.00039996 | W |
| learner_vs_zero | thai | ZERO | RDFL_SOFT | -0.119112 | -0.183427 | -0.0704809 | 0.00039996 | W |
| rdfl_vs_comparators | us | ZERO | RDFL_SOFT | -0.0346316 | -0.100422 | -0.00938828 | 0.00539946 | D |
| rdfl_vs_comparators | us | DIRECT_HUBER | RDFL_SOFT | 0.028563 | 0.000487975 | 0.0744882 | 0.0359964 | D |
| rdfl_vs_comparators | us | ORDINARY_SOFT | RDFL_SOFT | -0.0332115 | -0.0911201 | -0.0109638 | 0.00039996 | D |
| rdfl_vs_comparators | us | HARD_RDFL | RDFL_SOFT | 0.0052822 | -0.00109106 | 0.0170505 | 0.0668933 | E |
| learner_vs_zero | us | ZERO | DIRECT_HUBER | -0.0650528 | -0.175675 | -0.0201493 | 0.0019998 | W |
| learner_vs_zero | us | ZERO | ORDINARY_SOFT | -0.00137445 | -0.0164963 | 0.0087458 | 0.749325 | E |
| learner_vs_zero | us | ZERO | HARD_RDFL | -0.0401258 | -0.112668 | -0.010759 | 0.00269973 | D |
| learner_vs_zero | us | ZERO | RDFL_SOFT | -0.0346316 | -0.0973238 | -0.00902771 | 0.00269973 | D |

Prediction SD (return units):

| method | thai | us |
| --- | --- | --- |
| DIRECT_HUBER | 0.00388497 | 0.00499208 |
| HARD_RDFL | 0.00333084 | 0.00368012 |
| ORDINARY_SOFT | 0.00112521 | 0.00120033 |
| RDFL_SOFT | 0.00318583 | 0.00346804 |
