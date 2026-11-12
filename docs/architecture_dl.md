# MLP architecture (Phase 2)

Task: predict the CCME Water Quality Index (0 to 100) from the same 14 inputs, the same station split (`split`) and the same seed (42) as Phase 1. Code: `src/train_dl.py`, data: `src/dl_data.py`, regularisation switches: `src/regularisation.py`.

## 1. Final network

```text
input (30)  ->  Linear 30->128  -> ReLU -> Dropout 0.2
            ->  Linear 128->64  -> ReLU -> Dropout 0.2
            ->  Linear 64->1    (no activation)   -> predicted score
```

| Part | Choice | Why |
|---|---|---|
| Input | 30 columns | The 14 Phase 1 inputs after preprocessing: median imputation, missing-value flags, scaling, one-hot encoding of water type and country. The five chemicals the target is computed from stay hidden. |
| Depth | 2 hidden layers | The inputs are few and mostly low-dimensional. In our tuning, 2 layers were as good as 3 (best 3-layer trial: RMSE 11.33 vs 11.23). More depth adds parameters, not accuracy. |
| Width | 128, then 64 | A funnel shape (wide, then narrower) lets the first layer mix the inputs and the second compress them. The network has 12,289 weights for about 182k training rows, so it is small. |
| Activation | ReLU | Cheap, does not saturate, and works well on standardised inputs. Without a non-linear activation the layers would collapse into one linear model (that is Ridge). |
| Output head | 1 neuron, no activation | The score is a plain number, so we do not squash it. A sigmoid would flatten the many scores near 100. |
| Target scaling | (y - mean) / std during training | Keeps the loss on a stable scale. Predictions are converted back to the 0 to 100 scale. Mean and std come from the training rows only. |
| Loss | MSE | It is the quantity RMSE measures, which is our selection metric. Huber was tried (4 trials). It lowered MAE (about 7.5 vs 7.8) but raised RMSE (11.39 to 11.59 vs 11.23), so MSE was kept. |
| Optimiser | Adam, learning rate 0.001, batch size 512 | Adam gives each weight its own step size and needs little tuning. |
| Regularisation | Dropout 0.2, weight decay 1e-4, early stopping (patience 10), no batch norm | Values from `src/regularisation.py`. Training stopped at epoch 29 and the weights from epoch 19 were kept. |

## 2. Why an MLP for table data?

- It is the simplest neural network that is a fair opponent for XGBoost on a table: same inputs, same splits, only the model changes.
- Papers (for example Grinsztajn et al.) show trees often match or beat neural networks on tables, so we did not expect the MLP to win. The point of Phase 2 is a fair, honest comparison.
- A bigger architecture (CNN, transformer) has no structure to exploit here. Our rows are independent readings with a few columns.

## 3. How the setting was chosen

- 13 trials: the default setting first, then 12 random settings (fixed seed). Search space: widths (5 options), dropout (0 to 0.3), weight decay (0 to 1e-3), learning rate (3e-4 to 3e-3), loss (MSE or Huber), batch size (256 to 1024).
- Every trial uses early stopping on validation loss. The winner is the lowest validation RMSE. The test set is not scored in `train_dl.py`.
- All trials are logged in `results/dl/experiments_dl.csv`. The chosen settings are in `results/dl/mlp_config.json`.

## 4. Validation result (not the final test score)

| Model | Val MAE | Val RMSE | Val R2 |
|---|---|---|---|
| Mean baseline | 9.762 | 13.912 | -0.005 |
| Ridge | 8.688 | 12.263 | 0.219 |
| Random Forest | 7.773 | 11.247 | 0.343 |
| **MLP (this network)** | **7.767** | **11.231** | **0.345** |
| XGBoost | 7.729 | 11.205 | 0.348 |

The MLP lands between Random Forest and XGBoost on validation. The gaps are small (RMSE within 0.03), so the real comparison is the one-time test evaluation with confidence intervals (Anusha, `src/evaluate_dl.py`).

## 5. What the tuning showed

- 12 of 13 trials landed within RMSE 11.23 to 11.41 (the worst was 11.59). The default setting was the best. The network is not very sensitive to depth, width or learning rate, which suggests the limit is the information in the 14 inputs, not the model.
- Early stopping mattered: validation loss stopped improving after about epoch 19 while training loss kept falling, which is the start of overfitting.
- Limitation: 13 random trials is a small search, and the winner is the default setting, so we have not shown that tuning cannot help. Ablations (Sarthak) test one factor at a time.
