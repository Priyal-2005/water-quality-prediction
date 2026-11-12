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
2. **Backward pass (backpropagation):** apply the chain rule from the output backwards to get the gradient of the loss with respect to every weight. Each layer reuses the gradient of the layer after it, so all 12,289 gradients cost about as much as one extra forward pass.
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
