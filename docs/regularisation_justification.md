# Regularisation choices and evidence (Phase 2)

| Choice | Setting | What it does | Why we chose it | Validation Evidence |
|---|---|---|---|---|
| Dropout | 0.2 | Randomly disables 20% of neurons during training. | Helps prevent overfitting and improves generalisation. | Pending ablation experiment. |
| Weight Decay | 1e-4 | Penalises large model weights during optimisation. | Helps control model complexity and reduce overfitting. | Pending validation RMSE comparison. |
| Early Stopping | Patience = 10 epochs | Stops training when validation performance stops improving. | Prevents unnecessary training and overfitting. | Pending training and validation loss comparison. |
| Batch Normalisation | Disabled initially | Normalises layer activations during training. | Kept optional so its effect can be evaluated separately. | Pending comparison if enabled in an ablation. |

**Note:** These are initial settings, not experimentally confirmed optimal values. Final choices will be supported by validation results.
