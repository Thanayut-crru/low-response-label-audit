# News-availability sensitivity (reviewer request, post-lock)

| market | variant | articles | articles_mapped_locked | articles_mapped_variant | retained_labels_locked | retained_labels_variant | evaluated_stock_days | evaluated_dates | articles_dropped | dropped_date_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thai | shift1 | 9616 | 9594 | 9553 | 1722 | 1685 | 2185 | 403 |  |  |
| thai | drop_ambiguous | 9616 | 9594 | 9254 | 1722 | 1690 | 2168 | 403 | 340 | 167 |
| us | shift1 | 8080 | 8080 | 8075 | 1656 | 1661 | 1874 | 426 |  |  |

| analysis | baseline | relative_improvement | family_ci_low | family_ci_high | holm_p | decision_2pct_family |
| --- | --- | --- | --- | --- | --- | --- |
| thai_shift1 | ZERO | -0.125444 | -0.186588 | -0.0794985 | 0.00039996 | W |
| thai_shift1 | DIRECT_HUBER | -0.11517 | -0.171833 | -0.0742228 | 0.00039996 | W |
| thai_shift1 | ORDINARY_SOFT | -0.117206 | -0.168696 | -0.0784835 | 0.00039996 | W |
| thai_shift1 | HARD_RDFL | 0.0187709 | 0.0111894 | 0.0264723 | 0.00039996 | D |
| thai_drop_ambiguous | ZERO | -0.159102 | -0.234597 | -0.0981671 | 0.00039996 | W |
| thai_drop_ambiguous | DIRECT_HUBER | -0.158251 | -0.236709 | -0.0900556 | 0.00039996 | W |
| thai_drop_ambiguous | ORDINARY_SOFT | -0.148503 | -0.213331 | -0.0938763 | 0.00039996 | W |
| thai_drop_ambiguous | HARD_RDFL | 0.0212451 | 0.0137643 | 0.0290148 | 0.00039996 | D |
| us_shift1 | ZERO | -0.0700183 | -0.165562 | -0.0312912 | 0.00039996 | W |
| us_shift1 | DIRECT_HUBER | -0.0644916 | -0.144095 | -0.0313859 | 0.00039996 | W |
| us_shift1 | ORDINARY_SOFT | -0.060325 | -0.142174 | -0.027487 | 0.00039996 | W |
| us_shift1 | HARD_RDFL | 0.0117354 | 0.00429306 | 0.0269165 | 0.00039996 | D |
