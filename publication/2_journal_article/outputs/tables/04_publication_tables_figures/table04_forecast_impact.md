| data_condition      | data_condition_label    | model                  | model_label            |   n_targets |   rmse_MW |   mae_MW | is_placeholder   | status                 |
|:--------------------|:------------------------|:-----------------------|:-----------------------|------------:|----------:|---------:|:-----------------|:-----------------------|
| m8_xgb_corrected    | m8_xgb-corrected data   | linear_regression      | Linear regression      |        2880 |  2.27131  | 1.311    | True             | placeholder_smoke_only |
| m8_xgb_corrected    | m8_xgb-corrected data   | perfect_model_baseline | Perfect-model baseline |        2880 |  0.476638 | 0.202497 | False            | complete               |
| m8_xgb_corrected    | m8_xgb-corrected data   | seasonal_naive         | Seasonal naive         |        2880 |  4.11318  | 2.37976  | True             | placeholder_smoke_only |
| m8_xgb_corrected    | m8_xgb-corrected data   | xgboost                | XGBoost                |        2880 |  1.73692  | 1.00275  | True             | placeholder_smoke_only |
| raw_uncorrected     | Uncorrected data        | linear_regression      | Linear regression      |        2880 |  5.20717  | 3.06293  | True             | placeholder_smoke_only |
| raw_uncorrected     | Uncorrected data        | perfect_model_baseline | Perfect-model baseline |        2880 |  4.76638  | 2.02497  | False            | complete               |
| raw_uncorrected     | Uncorrected data        | seasonal_naive         | Seasonal naive         |        2880 |  5.33835  | 3.14395  | True             | placeholder_smoke_only |
| raw_uncorrected     | Uncorrected data        | xgboost                | XGBoost                |        2880 |  4.96627  | 2.92114  | True             | placeholder_smoke_only |
| reference_corrected | Manually corrected data | linear_regression      | Linear regression      |        2880 |  1.51322  | 0.854908 | True             | placeholder_smoke_only |
| reference_corrected | Manually corrected data | perfect_model_baseline | Perfect-model baseline |        2880 |  0        | 0        | False            | complete               |
| reference_corrected | Manually corrected data | seasonal_naive         | Seasonal naive         |        2880 |  4.29565  | 2.42985  | True             | placeholder_smoke_only |
| reference_corrected | Manually corrected data | xgboost                | XGBoost                |        2880 |  1.04141  | 0.590333 | True             | placeholder_smoke_only |
