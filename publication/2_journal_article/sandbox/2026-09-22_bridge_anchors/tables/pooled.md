### Beta `sure`

| anchors | n_days | n_rpf | energy_iou | energy_precision | sure_day_recall | sure_day_uncertain_rate | day_precision | day_recall | day_f1 | rate_auto_correct | rate_uncertain | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nearest | 2305 | 470 | 0.8945 | 0.9407 | 0.8957 | 0.0532 | 0.9525 | 0.8957 | 0.9232 | 0.1918 | 0.0430 | 0.0088 |
| edge | 2305 | 470 | 0.5864 | 0.6924 | 0.6468 | 0.2064 | 0.9297 | 0.6468 | 0.7629 | 0.1419 | 0.1020 | 0.0200 |
| gap_edge | 2305 | 470 | 0.8892 | 0.9400 | 0.8851 | 0.0489 | 0.9498 | 0.8851 | 0.9163 | 0.1900 | 0.0438 | 0.0089 |

### Alpha

| anchors | n_days | n_rpf | energy_iou | energy_precision | sure_day_recall | sure_day_uncertain_rate | day_precision | day_recall | day_f1 | rate_auto_correct | rate_uncertain | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nearest | 10425 | 3381 | 0.8692 | 0.8886 | 0.9048 | 0.0497 | 0.9773 | 0.9048 | 0.9396 | 0.3002 | 0.0261 | 0.0218 |
| edge | 10425 | 3381 | 0.2670 | 0.3022 | 0.3144 | 0.4167 | 0.8292 | 0.3144 | 0.4559 | 0.1230 | 0.3369 | 0.0504 |
| gap_edge | 10425 | 3381 | 0.8692 | 0.8886 | 0.9048 | 0.0497 | 0.9773 | 0.9048 | 0.9396 | 0.3002 | 0.0261 | 0.0218 |

### Beta `unsure` (sensitivity only)

| anchors | n_days | n_rpf | energy_iou | energy_precision | sure_day_recall | sure_day_uncertain_rate | day_precision | day_recall | day_f1 | rate_auto_correct | rate_uncertain | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nearest | 617 | 159 | 0.3858 | 0.5338 | — | — | 0.5524 | 0.4969 | 0.5232 | 0.2318 | 0.1799 | — |
| edge | 617 | 159 | 0.2884 | 0.8386 | — | — | 0.8750 | 0.1321 | 0.2295 | 0.0389 | 0.1718 | — |
| gap_edge | 617 | 159 | 0.3834 | 0.5310 | — | — | 0.5448 | 0.4969 | 0.5197 | 0.2350 | 0.1799 | — |
