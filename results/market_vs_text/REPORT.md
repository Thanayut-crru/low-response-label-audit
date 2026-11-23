# Market features versus market plus text (reviewer request, post-lock)

Reproduction gate (text-only DIRECT_HUBER vs locked predictions, max abs gap): {'thai': 9.996344030316351e-17, 'us': 9.996344030316351e-17}

| analysis | baseline | variant | relative_improvement | family_ci_low | family_ci_high | holm_p | decision_2pct_family | smallest_margin_met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thai | ZERO | DIRECT_MARKET | -0.00754634 | -0.0351864 | 0.0125081 | 0.908309 | I | 0.0351864 |
| thai | ZERO | PRICE_RIDGE | -0.0373623 | -0.0663419 | -0.0151502 | 0.00319968 | D | 0.0663419 |
| thai | DIRECT_MARKET | DIRECT_MARKET_TEXT | -0.000495526 | -0.0210632 | 0.0157725 | 0.949205 | I | 0.0210632 |
| thai | RDFL_MARKET | RDFL_MARKET_TEXT | -0.054101 | -0.126814 | 0.00621802 | 0.0926907 | I | 0.126814 |
| thai | DIRECT_MARKET_TEXT | RDFL_MARKET_TEXT | -0.517487 | -0.730952 | -0.324956 | 0.00049995 | W | 0.730952 |
| us | ZERO | DIRECT_MARKET | -0.0465313 | -0.117022 | -0.000409143 | 0.10299 | I | 0.117022 |
| us | ZERO | PRICE_RIDGE | -0.0468828 | -0.105016 | -0.0145019 | 0.0859914 | I | 0.105016 |
| us | DIRECT_MARKET | DIRECT_MARKET_TEXT | -0.020659 | -0.041928 | -0.00335929 | 0.0859914 | I | 0.041928 |
| us | RDFL_MARKET | RDFL_MARKET_TEXT | -0.0174244 | -0.0721225 | 0.0120984 | 0.184382 | I | 0.0721225 |
| us | DIRECT_MARKET_TEXT | RDFL_MARKET_TEXT | -0.147865 | -0.320234 | -0.0794403 | 0.00049995 | W | 0.320234 |

Prediction SD (return units):

| method | thai | us |
| --- | --- | --- |
| DIRECT_MARKET | 0.00137857 | 0.00336257 |
| DIRECT_MARKET_TEXT | 0.00170361 | 0.0042383 |
| PRICE_RIDGE | 0.00225233 | 0.00426379 |
| RDFL_MARKET | 0.00554754 | 0.00855741 |
| RDFL_MARKET_TEXT | 0.00628192 | 0.00884388 |
