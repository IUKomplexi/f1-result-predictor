# Hyperparameter tuning (walk-forward)

Objective: **mae** (quantize=on). Each candidate is a full walk-forward backtest; the model is re-trained per test season.

## Candidates

| rank | params | winner_hit | top3_overlap | top10_overlap | spearman | mae |
|---|---|---|---|---|---|---|
| baseline | `{"max_iter": 400, "learning_rate": 0.03, "max_depth": 3, "l2_regularization": 1.0, "min_samples_leaf": 20}` | 0.5343 | 0.6552 | 0.7745 | 0.6501 | 2.9478 | |
| 1 | `{"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "l2_regularization": 10.0, "min_samples_leaf": 50}` | 0.5441 | 0.6536 | 0.7779 | 0.6525 | 2.9150 | |
| 2 | `{"max_iter": 400, "learning_rate": 0.03, "max_depth": 4, "l2_regularization": 1.0, "min_samples_leaf": 50}` | 0.5196 | 0.6520 | 0.7770 | 0.6473 | 2.9376 | |
| 3 | `{"max_iter": 200, "learning_rate": 0.05, "max_depth": 4, "l2_regularization": 1.0, "min_samples_leaf": 50}` | 0.5392 | 0.6585 | 0.7770 | 0.6495 | 2.9402 | |
| 4 | `{"max_iter": 400, "learning_rate": 0.03, "max_depth": 3, "l2_regularization": 1.0, "min_samples_leaf": 20}` | 0.5343 | 0.6552 | 0.7745 | 0.6501 | 2.9478 | |
| 5 | `{"max_iter": 200, "learning_rate": 0.05, "max_depth": 5, "l2_regularization": 1.0, "min_samples_leaf": 50}` | 0.5441 | 0.6552 | 0.7760 | 0.6486 | 2.9493 | |
| 6 | `{"max_iter": 400, "learning_rate": 0.05, "max_depth": 4, "l2_regularization": 1.0, "min_samples_leaf": 20}` | 0.5049 | 0.6552 | 0.7657 | 0.6433 | 2.9562 | |
| 7 | `{"max_iter": 800, "learning_rate": 0.01, "max_depth": 2, "l2_regularization": 0.1, "min_samples_leaf": 5}` | 0.5441 | 0.6683 | 0.7745 | 0.6539 | 2.9589 | |
| 8 | `{"max_iter": 200, "learning_rate": 0.1, "max_depth": 2, "l2_regularization": 10.0, "min_samples_leaf": 5}` | 0.5343 | 0.6634 | 0.7755 | 0.6515 | 2.9596 | |
| 9 | `{"max_iter": 400, "learning_rate": 0.05, "max_depth": 5, "l2_regularization": 10.0, "min_samples_leaf": 20}` | 0.4755 | 0.6487 | 0.7691 | 0.6434 | 2.9674 | |
| 10 | `{"max_iter": 400, "learning_rate": 0.05, "max_depth": 2, "l2_regularization": 10.0, "min_samples_leaf": 20}` | 0.5343 | 0.6569 | 0.7765 | 0.6484 | 2.9691 | |
| 11 | `{"max_iter": 200, "learning_rate": 0.1, "max_depth": 5, "l2_regularization": 10.0, "min_samples_leaf": 20}` | 0.4804 | 0.6422 | 0.7676 | 0.6430 | 2.9804 | |
| 12 | `{"max_iter": 800, "learning_rate": 0.01, "max_depth": 5, "l2_regularization": 0.1, "min_samples_leaf": 5}` | 0.5147 | 0.6569 | 0.7760 | 0.6473 | 2.9829 | |
| 13 | `{"max_iter": 800, "learning_rate": 0.05, "max_depth": 3, "l2_regularization": 1.0, "min_samples_leaf": 20}` | 0.4902 | 0.6438 | 0.7686 | 0.6417 | 3.0046 | |
| 14 | `{"max_iter": 400, "learning_rate": 0.05, "max_depth": 3, "l2_regularization": 0.1, "min_samples_leaf": 5}` | 0.4853 | 0.6520 | 0.7721 | 0.6440 | 3.0098 | |
| 15 | `{"max_iter": 400, "learning_rate": 0.1, "max_depth": 5, "l2_regularization": 0.1, "min_samples_leaf": 50}` | 0.4902 | 0.6225 | 0.7637 | 0.6366 | 3.0115 | |
| 16 | `{"max_iter": 400, "learning_rate": 0.05, "max_depth": 3, "l2_regularization": 1.0, "min_samples_leaf": 5}` | 0.4853 | 0.6438 | 0.7730 | 0.6456 | 3.0129 | |
| 17 | `{"max_iter": 400, "learning_rate": 0.1, "max_depth": 5, "l2_regularization": 1.0, "min_samples_leaf": 50}` | 0.5245 | 0.6258 | 0.7681 | 0.6347 | 3.0375 | |
| 18 | `{"max_iter": 800, "learning_rate": 0.1, "max_depth": 5, "l2_regularization": 10.0, "min_samples_leaf": 20}` | 0.4118 | 0.6127 | 0.7627 | 0.6284 | 3.0787 | |
| 19 | `{"max_iter": 600, "learning_rate": 0.1, "max_depth": 5, "l2_regularization": 10.0, "min_samples_leaf": 5}` | 0.4314 | 0.6111 | 0.7686 | 0.6330 | 3.1051 | |
| 20 | `{"max_iter": 800, "learning_rate": 0.1, "max_depth": 4, "l2_regularization": 0.1, "min_samples_leaf": 5}` | 0.4608 | 0.6111 | 0.7627 | 0.6265 | 3.1148 | |
| 21 | `{"max_iter": 200, "learning_rate": 0.01, "max_depth": 5, "l2_regularization": 1.0, "min_samples_leaf": 50}` | 0.5490 | 0.6699 | 0.7789 | 0.6501 | 3.1994 | |
| 22 | `{"max_iter": 200, "learning_rate": 0.01, "max_depth": 5, "l2_regularization": 0.1, "min_samples_leaf": 5}` | 0.5392 | 0.6650 | 0.7789 | 0.6495 | 3.2158 | |
| 23 | `{"max_iter": 200, "learning_rate": 0.01, "max_depth": 5, "l2_regularization": 1.0, "min_samples_leaf": 5}` | 0.5392 | 0.6569 | 0.7770 | 0.6502 | 3.2381 | |
| 24 | `{"max_iter": 200, "learning_rate": 0.01, "max_depth": 4, "l2_regularization": 10.0, "min_samples_leaf": 20}` | 0.5392 | 0.6601 | 0.7784 | 0.6520 | 3.2457 | |
