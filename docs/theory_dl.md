# Theoretical rigor (Phase 2, deep learning)

Sections 4.1 to 4.4 (how the network learns) are written by Priyal. Sections 4.5 to 4.8 (regularisation, bias and variance, trees vs networks, hybrids and out-of-fold predictions) are written by Anusha and added below. Every section uses our real settings: MLP 30 -> 128 -> 64 -> 1, ReLU, MSE, Adam (learning rate 0.001, batch size 512), dropout 0.2, weight decay 1e-4, seed 42.

## 4.1 Neuron, layer and activation

A neuron takes the inputs x, computes a weighted sum plus a bias, then applies an activation:

z = w . x + b,  a = f(z)

A layer is many neurons side by side, so a layer is a matrix multiplication followed by f. Our network has two hidden layers (128 and 64 neurons) and one output neuron.

- **Why we need an activation.** Without f, two layers compute W2(W1 x) = (W2 W1) x, which is still one linear model. Ridge already covers that case (validation R2 0.219). The non-linearity is what lets the network model curves and interactions, for example dissolved oxygen mattering differently at different temperatures.
- **Why ReLU.** f(z) = max(0, z). Its gradient is 1 for positive z and 0 otherwise, so it does not shrink gradients the way sigmoid or tanh do in deep stacks, and it is cheap to compute.
- **Why no activation on the output.** The target is a number from 0 to 100. A linear output can reach any value, so it is the standard head for regression. Many scores sit near 100 (44 percent are exactly 100), and a sigmoid would squash that region.
- **Size.** The weights count is 30x128+128 + 128x64+64 + 64x1+1 = 12,289, small next to 181,747 training rows.

## 4.2 Loss functions

The loss is the single number training minimises.

- **MSE** = mean((y - y_hat)^2). Squaring makes big misses count much more, and the function is smooth, so gradient descent works well. Its square root is RMSE, our main selection metric, so minimising MSE directly targets what we measure.
- **Huber** is quadratic for small errors and linear for large ones, so it is less pulled by outliers. It behaves more like MAE.
- **What we found.** In 4 Huber trials the validation MAE was lower (7.50 to 7.65 vs 7.77 for the chosen MSE model) but the validation RMSE was higher (11.39 to 11.59 vs 11.23). So the loss decides which error the model is good at: Huber is better at typical rows, MSE at the big misses. Because we select on RMSE, we kept MSE.
- **Target scaling.** During training the target is standardised, (y - mean) / std with the training mean and std, so the loss is of order 1. Predictions are converted back to the 0 to 100 scale before any metric is computed.

## 4.3 Backpropagation and gradient descent

Training means finding weights that make the loss small.

1. **Forward pass:** push a batch of rows through the layers to get predictions, then compute the loss.
2. **Backward pass (backpropagation):** apply the chain rule from the output backwards to get the gradient of the loss with respect to every weight. Each layer reuses the gradient of the layer after it, so all 12,289 gradients cost a small multiple of a forward pass (roughly twice as much), not one pass per weight.
3. **Update:** move each weight a small step against its gradient: w <- w - eta * dL/dw, where eta is the learning rate.

We do this on mini-batches of 512 rows (about 355 updates per epoch). A mini-batch gradient is a noisy but cheap estimate of the full gradient, and the noise helps escape poor spots. This is the same gradient-descent idea as in boosting, except here the steps change the weights, while XGBoost adds a tree at each step.

**Reading our curves** (`results/dl/mlp_history.csv`): training loss fell steadily (0.768 in epoch 1 to 0.666 in epoch 29, in standardised units) while the validation loss stopped improving at epoch 19 (0.784). That gap opening up is the start of overfitting, and early stopping keeps the epoch-19 weights.

## 4.4 Adam optimiser

Plain gradient descent uses one learning rate for every weight. Adam keeps two running averages per weight:

- the average of the gradient (momentum), which smooths the noisy mini-batch gradients;
- the average of the squared gradient, which scales the step, so weights with large, noisy gradients take smaller steps.

It also corrects these averages for the start of training. The result is that it usually trains faster and needs less tuning than plain gradient descent.

- **Settings.** Learning rate 0.001 (the most important knob), batch size 512, Adam's default averaging constants.
- **What we found.** Learning rates 3e-4, 1e-3 and 3e-3 all gave validation RMSE between 11.23 and 11.33 for the MSE models. The lowest rate often needed more epochs (best epoch 35 vs 19) to get there. Adam is not fragile here.
- **Weight decay** in `torch.optim.Adam` adds an L2 penalty (1e-4) to the gradient. It is covered with the other regularisation methods in 4.5.

## Summary: what to say in the viva

"A neuron is a weighted sum plus a non-linearity. Without the non-linearity many layers collapse into one linear model, which is just Ridge. Backpropagation uses the chain rule to get every weight's gradient, and Adam uses those gradients with per-weight step sizes. We use MSE because it matches RMSE, which is how we choose the model, and we tested Huber. It lowered MAE but raised RMSE."

(Sections 4.5 to 4.8: Anusha.)

## 4.5 Regularisation

A neural network can learn the training data too closely, including noise and patterns that do not generalise to unseen observations. Regularisation reduces this risk by discouraging unnecessarily complex solutions.

Our MLP uses three regularisation techniques:

- **Dropout (0.2):** During training, dropout randomly disables 20% of the hidden-layer activations. This prevents the network from relying too heavily on particular neurons and encourages it to learn more robust representations. Dropout is disabled during evaluation.
- **Weight decay (1e-4):** We use L2-style weight regularisation through Adam's weight-decay parameter. It penalises large weights and encourages a simpler model.
- **Early stopping:** Training monitors validation loss. If validation performance does not improve for 10 consecutive epochs, training stops and restores the best-performing model checkpoint. This avoids continuing to fit the training data after generalisation has stopped improving.

Our configuration uses dropout 0.2, weight decay 0.0001 and early-stopping patience 10. Batch normalisation is disabled.

These techniques address different aspects of overfitting: dropout reduces dependence between neurons, weight decay discourages large weights, and early stopping limits unnecessary training.

## 4.6 Bias, variance and overfitting

Bias and variance describe two different sources of prediction error.

- **High bias:** A model is too simple to capture important relationships in the data. It underfits and performs poorly on both training and unseen data.
- **High variance:** A model is too sensitive to its training observations. It may perform very well on training data but poorly on unseen data.
- **Overfitting:** The model learns patterns specific to the training data that do not generalise reliably.

We use separate validation data to monitor generalisation and select model configurations. The final test set is kept separate from model selection.

Our results show why validation and test performance must both be considered. On the temporal split, the selected hybrid achieved validation RMSE of approximately 8.02 and test RMSE of approximately 7.84. This result does not by itself prove that the model has no overfitting; it shows that its performance was maintained on the later-year test period.

We also examined errors by country and CCME water-quality category. This revealed that good overall performance can hide substantial subgroup errors. In particular, all three models overpredicted every polluted-water observation in the temporal test subset (766 observations with actual CCME scores below 60). This is an important limitation that aggregate metrics alone would not reveal.

## 4.7 Why trees often outperform neural networks on tabular data

Neural networks are powerful, but they are not automatically the best choice for structured tabular datasets.

Tree-based models such as XGBoost have several advantages:

- They naturally learn non-linear relationships and interactions between features.
- They can work effectively with mixed patterns in numerical and categorical-derived features.
- They do not require the same neural-network architecture and optimisation choices, such as layer widths, learning rates and regularisation settings.
- They can perform strongly on datasets where useful information is already represented in engineered features.

Our dataset contains structured water-quality measurements, station information, country and waterbody characteristics, and engineered features. XGBoost therefore provides a strong baseline.

On the station-based validation split, XGBoost achieved RMSE 11.2048, compared with 11.2311 for the MLP. The MLP did not outperform XGBoost on this validation comparison.

On the temporal validation split, XGBoost achieved RMSE 8.1267 and the MLP achieved 8.1451. Their performance was close, with XGBoost slightly ahead.

These results support using XGBoost as a strong tabular-data baseline rather than assuming that a neural network must perform better simply because it is a deep-learning model.

## 4.8 Hybrid models and out-of-fold predictions

A hybrid model combines predictions from different model types. The motivation is that models may capture different patterns and their errors may complement one another.

Our project combines XGBoost and an MLP using a weighted blend:

\[
\hat{y}_{hybrid} =
w\hat{y}_{XGB} + (1-w)\hat{y}_{MLP}
\]

The blending weight is selected using validation data only. The selected weight is then applied to the corresponding test predictions. Neither model is fitted on validation rows, so ordinary validation predictions are enough to choose the weight. Out-of-fold predictions are **not** used for the weighted blend.

For the station-based split, the selected blend used 55% XGBoost and 45% MLP. For the temporal split, the selected blend used 50% XGBoost and 50% MLP. The different weights reflect the fact that the two splits represent different generalisation settings.

### When out-of-fold predictions are needed (residual learning only)

Out-of-fold (OOF) predictions are needed for a different hybrid, residual learning, where a second model is trained on the training rows to predict XGBoost's remaining error. We tried it and did not select it, so it is not part of the final hybrid.

For the residual model we need XGBoost errors on the training rows that are not unrealistically small. XGBoost has already seen those rows, so its in-sample predictions are too good. We therefore:

1. Divide the training stations into five folds using GroupKFold.
2. For each fold, train XGBoost on the other four folds and predict the held-out fold.
3. Store those predictions in their original row positions.

Every training row then receives a prediction from a model that did not train on it, and the residual target is

\[
r_{train} = y_{train} - \hat{y}_{XGB,OOF}
\]

A residual MLP is trained to predict these residuals. At prediction time, its correction is added to predictions from an XGBoost model fitted on the full training split.

A weighted blend does not need this step, because the blend weight is chosen on validation rows that neither model was fitted on.

### What we found

We tested both weighted blending and residual learning.

Residual learning was not selected. Its validation residual corrections had very weak correlation with the actual residuals: approximately 0.0877 on the station-based split and approximately -0.0013 on the temporal split. This indicated that the residual MLP was not learning a useful correction signal.

Weighted blending was therefore selected as the final hybrid approach. It uses the Phase 1 saved XGBoost predictions and the saved MLP predictions, with the weight chosen on validation.

**Station-based split (main result).** The test set was scored once, with 95% station-bootstrap confidence intervals (2,000 draws) and paired differences (notebook 08).

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| XGBoost | 11.058 | 7.543 | 0.293 |
| MLP | 11.108 | 7.606 | 0.287 |
| Hybrid | 11.010 | 7.521 | 0.300 |

| Paired RMSE difference | Value | 95% CI | CI excludes 0? |
|---|---|---|---|
| MLP minus XGBoost | +0.049 | [-0.053, 0.147] | No |
| Hybrid minus XGBoost | -0.049 | [-0.098, -0.002] | Yes, barely |
| Hybrid minus MLP | -0.098 | [-0.150, -0.043] | Yes |

- The MLP and XGBoost cannot be separated on the test stations.
- The hybrid beats the MLP clearly.
- The hybrid's gain over XGBoost is about 0.05 RMSE (0.4%), and its interval only just excludes zero. We describe it as a small gain, not a large improvement.

**Temporal split (secondary check).** On the temporal test split, the hybrid achieved RMSE 7.8421, compared with 8.0605 for XGBoost and 7.8659 for the MLP. The hybrid achieved the best RMSE and R² among the three models, although the MLP achieved the lowest MAE. This is a single split without confidence intervals, so it is supporting evidence only.

The hybrid therefore gave a small overall improvement, but it did not eliminate the overprediction problem for polluted-water observations: all three models predict about 80 where the true CCME value averages about 50. This demonstrates why hybrid models must be evaluated using both overall metrics and subgroup error analysis.
