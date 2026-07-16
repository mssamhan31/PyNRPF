| regime          | confidence_scope   | aggregation      |   substation_id |   support |   positive_support |   tp |   fp |   fn |   tn |   precision |   recall |     f1 |
|:----------------|:-------------------|:-----------------|----------------:|----------:|-------------------:|-----:|-----:|-----:|-----:|------------:|---------:|-------:|
| beta_only       | beta_sure          | pooled           |             nan |      2310 |                471 |  394 |   79 |   77 | 1760 |      0.8330 |   0.8365 | 0.8347 |
| beta_only       | beta_sure          | macro_substation |             nan |      2310 |                471 |  394 |   79 |   77 | 1760 |      0.6589 |   0.6824 | 0.6470 |
| beta_plus_alpha | beta_sure          | pooled           |             nan |      2310 |                471 |  434 |  127 |   37 | 1712 |      0.7736 |   0.9214 | 0.8411 |
| beta_plus_alpha | beta_sure          | macro_substation |             nan |      2310 |                471 |  434 |  127 |   37 | 1712 |      0.5695 |   0.7408 | 0.6298 |
| alpha_only      | beta_sure          | pooled           |             nan |      2310 |                471 |  460 |  227 |   11 | 1612 |      0.6696 |   0.9766 | 0.7945 |
| alpha_only      | beta_sure          | macro_substation |             nan |      2310 |                471 |  460 |  227 |   11 | 1612 |      0.4943 |   0.8316 | 0.5970 |