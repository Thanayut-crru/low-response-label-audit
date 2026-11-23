# Market-state split (reviewer request, post-lock; descriptive)

| market | state | dates |
| --- | --- | --- |
| thai | high_volatility | 201 |
| thai | low_volatility | 202 |
| thai | rising_market | 197 |
| thai | falling_market | 206 |
| us | high_volatility | 213 |
| us | low_volatility | 213 |
| us | rising_market | 270 |
| us | falling_market | 156 |

| market | state | baseline | relative_improvement | family_ci_low | family_ci_high | holm_p | decision_2pct_family |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thai | high_volatility | ZERO | -0.17597 | -0.277894 | -0.0967803 | 0.00039996 | W |
| thai | high_volatility | DIRECT_HUBER | -0.175227 | -0.272457 | -0.0959824 | 0.00039996 | W |
| thai | high_volatility | ORDINARY_SOFT | -0.164488 | -0.251251 | -0.0993775 | 0.00039996 | W |
| thai | high_volatility | HARD_RDFL | 0.0179894 | 0.00971267 | 0.0270562 | 0.00039996 | D |
| thai | low_volatility | ZERO | -0.143664 | -0.250345 | -0.054315 | 0.00059994 | W |
| thai | low_volatility | DIRECT_HUBER | -0.141537 | -0.250693 | -0.039131 | 0.00209979 | W |
| thai | low_volatility | ORDINARY_SOFT | -0.13107 | -0.223494 | -0.0518559 | 0.00059994 | W |
| thai | low_volatility | HARD_RDFL | 0.0220915 | 0.0133997 | 0.0336733 | 0.00039996 | D |
| thai | rising_market | ZERO | -0.162123 | -0.309209 | -0.0678032 | 0.00039996 | W |
| thai | rising_market | DIRECT_HUBER | -0.171658 | -0.331145 | -0.0679993 | 0.00039996 | W |
| thai | rising_market | ORDINARY_SOFT | -0.144384 | -0.268232 | -0.0650164 | 0.00039996 | W |
| thai | rising_market | HARD_RDFL | 0.0206868 | 0.010772 | 0.0330272 | 0.00039996 | D |
| thai | falling_market | ZERO | -0.159077 | -0.245583 | -0.0936732 | 0.00039996 | W |
| thai | falling_market | DIRECT_HUBER | -0.148657 | -0.229135 | -0.0762411 | 0.00039996 | W |
| thai | falling_market | ORDINARY_SOFT | -0.151871 | -0.223001 | -0.100328 | 0.00039996 | W |
| thai | falling_market | HARD_RDFL | 0.0192999 | 0.0118207 | 0.0280628 | 0.00039996 | D |
| us | high_volatility | ZERO | -0.0505046 | -0.178037 | -0.00254207 | 0.0235976 | D |
| us | high_volatility | DIRECT_HUBER | -0.0428609 | -0.146701 | -0.00467733 | 0.0235976 | D |
| us | high_volatility | ORDINARY_SOFT | -0.0403238 | -0.127961 | -0.00575503 | 0.0235976 | D |
| us | high_volatility | HARD_RDFL | 0.00779304 | 0.00123967 | 0.0228053 | 0.0235976 | D |
| us | low_volatility | ZERO | -0.227683 | -0.358825 | -0.120575 | 0.00039996 | W |
| us | low_volatility | DIRECT_HUBER | -0.21253 | -0.320155 | -0.118506 | 0.00039996 | W |
| us | low_volatility | ORDINARY_SOFT | -0.20808 | -0.307901 | -0.124915 | 0.00039996 | W |
| us | low_volatility | HARD_RDFL | 0.0237815 | 0.0116558 | 0.033676 | 0.00039996 | D |
| us | rising_market | ZERO | -0.152222 | -0.287952 | -0.0854005 | 0.00039996 | W |
| us | rising_market | DIRECT_HUBER | -0.141526 | -0.266879 | -0.080362 | 0.00039996 | W |
| us | rising_market | ORDINARY_SOFT | -0.131983 | -0.24517 | -0.0765745 | 0.00039996 | W |
| us | rising_market | HARD_RDFL | 0.0201136 | 0.0111168 | 0.0335895 | 0.00039996 | D |
| us | falling_market | ZERO | -0.0331773 | -0.20615 | 0.0143705 | 0.392061 | I |
| us | falling_market | DIRECT_HUBER | -0.0255894 | -0.155448 | 0.00938889 | 0.392061 | I |
| us | falling_market | ORDINARY_SOFT | -0.0265761 | -0.135877 | 0.00482368 | 0.371163 | I |
| us | falling_market | HARD_RDFL | 0.00413794 | -0.00201279 | 0.0206717 | 0.392061 | I |
