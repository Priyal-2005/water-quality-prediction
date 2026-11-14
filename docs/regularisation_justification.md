
# Regularisation choices and evidence (Phase 2)

| Choice | Setting | What it does | Why we chose it | Validation Evidence |
|---|---|---|---|---|
| Dropout | 0.2 | Randomly disables 20% of neurons during training. | Helps prevent overfitting and improves generalisation. | Dropout 0.0 vs 0.2: RMSE 11.245 vs 11.231; MAE 7.665 vs 7.767. RMSE is nearly the same, while MAE is slightly lower without dropout. |
| Weight Decay | 1e-4 | Penalises large model weights during optimisation. | Helps control model complexity and reduce overfitting. | Weight decay 0.0 vs 1e-4: RMSE 11.245 vs 11.231, showing a negligible effect. |
| Early Stopping | Patience = 10 epochs | Stops training when validation performance stops improving. | Prevents unnecessary training and overfitting. | According to `results/dl/mlp_history.csv`, training stopped at epoch 29 and the best epoch was 19, preventing 10 additional epochs after the best epoch. |
| Batch Normalisation | Disabled initially | Normalises layer activations during training. | Kept optional so its effect can be evaluated separately. | Not tested; disabled in the final model. |

**Note:** Final evidence is based on the available validation and training results.

