# Deep Learning Literature Review

**Water Quality Prediction via Gradient Boosting and Deep Neural Networks**

*Phase 2 \| Prepared from the project repository and the assigned literature brief*

# 1. Introduction

Phase 1 showed that gradient-boosted trees (XGBoost) can predict the CCME water-quality index from a restricted set of inputs on stations the model has never seen (test R² = 0.293). Phase 2 asks the next question: can a neural network do as well or better on the same inputs, the same splits and the same metrics? This review collects the deep-learning literature that bears on that question. It covers what a multilayer perceptron (MLP) can represent and how it is trained, how neural networks compare with tree models on tabular data, the regularisation tools we use, how predictions from two model families can be combined, and how deep learning has been applied to water quality and hydrology.

The project's Phase 2 model is a small MLP (30 inputs, hidden layers of 128 and 64 units, 12,289 parameters) trained with Adam, dropout and early stopping. It is compared with the Phase 1 XGBoost model on the station split and the year split, with paired station-cluster bootstrap intervals. An optional hybrid blends the XGBoost and MLP predictions. The literature below is used to justify these choices and to set honest expectations: most of it says that neural networks and trees are close on tabular data and that neither family wins everywhere.

# 2. Related work by theme

## 2.1 What an MLP can do, and how it is trained

Hornik, Stinchcombe and White (1989) showed that multilayer feedforward networks with enough hidden units can approximate a very wide class of functions. This is an existence result: it says a network of this kind is expressive enough, not that training will find the right weights or that a small network will generalise. Rumelhart, Hinton and Williams (1986) introduced back-propagation, which adjusts connection weights to reduce the gap between actual and desired outputs and lets hidden units learn useful internal features. Kingma and Ba (2015) proposed Adam, a first-order optimiser that adapts step sizes from running estimates of the gradient's first and second moments, with hyperparameters that rarely need much tuning. Our MLP uses all three ideas: a feedforward network, back-propagation and Adam (learning rate 1e-3).

## 2.2 Neural networks versus trees on tabular data

This is the most relevant strand for our comparison. Grinsztajn, Oyallon and Varoquaux (2022) found tree-based models still ahead of deep models on medium-sized tabular data, and explained it by three properties of such data (robustness to uninformative features, preserved feature orientation, and irregular target functions). Shwartz-Ziv and Armon (2022) reported that XGBoost beat several recent deep models on the datasets from those models' own papers, needed less tuning, and that an ensemble of deep models with XGBoost did better than XGBoost alone. Borisov et al. (2022) surveyed deep methods for mixed-type tabular data and reported that gradient-boosted trees still outperformed the deep models in their experiments. Gorishniy et al. (2021) argued that deep tabular models are compared under inconsistent protocols and that strong simple baselines are missing; they proposed a ResNet-like baseline and a Transformer adaptation, and found that neither deep models nor gradient-boosted trees are universally better. McElfresh et al. (2023) compared 19 algorithms on 176 datasets and concluded that the gap between neural networks and boosted trees is often negligible and that light tuning of a tree model can matter more than the choice of family; they also found that boosted trees handle skewed, heavy-tailed and irregular features better.

Taken together, these studies set a clear expectation for our project: a plain MLP should land close to a tuned XGBoost, not far above it. Our results fit that picture. On the station test split the MLP's RMSE (11.108) is within 0.05 of XGBoost's (11.058), and the paired bootstrap interval for the difference (−0.053 to +0.147) includes zero. Our target also has a feature that is awkward for networks: 43.7% of rows are exactly 100, so the target is heavily piled at one value. We keep the claim modest: with 14 inputs and one dataset, we cannot say that either family is better in general.

## 2.3 Regularisation and stopping

Srivastava et al. (2014) introduced dropout: units are randomly removed during training, which discourages units from depending on each other and approximates averaging many thinned networks. Prechelt's work on early stopping (Prechelt, 1998) studies stopping rules that watch the validation error and halt training when it stops improving, and keeps the weights from the best point. Weight decay is the standard small penalty on large weights. Our MLP uses dropout 0.2, weight decay 1e-4 and early stopping with patience 10 (best epoch 19, stopped at 29). An honest note from our own tuning, consistent with the cautious tone of the tabular literature: tested one at a time, dropout and weight decay changed the validation score very little, so they act as cheap safeguards rather than the reason the model works.

## 2.4 Combining model families

Bates and Granger (1969) showed that a weighted combination of two forecasts can beat either forecast alone when their errors are not perfectly correlated, and the idea of choosing the weights from past errors goes back to that paper. Shwartz-Ziv and Armon (2022) observed the same effect for deep models plus XGBoost on tabular data. Our optional hybrid follows this idea in its simplest form: a weighted average of the saved XGBoost and MLP predictions, with the weight (55% XGBoost, 45% MLP) chosen on the validation set only and the result clipped to the 0–100 range. We also tried the more elaborate version in which a second model learns the first model's errors, using out-of-fold predictions; the out-of-fold correlation with the true residual was only 0.088 on the station split and −0.001 on the year split, so we rejected it. The weighted blend is a small but consistent gain on the station test (RMSE 11.010 against 11.058 for XGBoost; paired interval −0.098 to −0.002).

## 2.5 Deep learning for water quality and hydrology

Sit et al. (2020) reviewed deep learning in hydrology and water resources, covering forecasting, data improvement and classification across monitoring, management and governance, and discuss challenges and ethical concerns. Zhi et al. (2021) trained an LSTM on sparse dissolved-oxygen measurements and daily weather data from 236 minimally disturbed US watersheds; the model captured the expected fall in dissolved oxygen as temperature rises and showed promise for basins with no measurements, but missed some peaks and troughs and did not do better where more data were available. Rahman, Chung and Ng (2026) applied a transformer, few-shot learning and GAN framework to five-class CCME classification on the Canadian subset of the same dataset we use, reporting high accuracy with the eight index parameters as inputs. Two cautions follow. First, the task is different (time-series dissolved oxygen, or classification), so numbers are not comparable with ours. Second, a model that is given the parameters the index is built from can recompute the formula, which our notebook 09 confirms: with the five hidden chemicals added to our inputs, validation R² rises from about 0.35 to 0.99 for both XGBoost and the MLP.

## 2.6 Evaluating fairly

Many applied papers report a single split and a single number. Efron's bootstrap (1979) gives a way to attach an uncertainty range to a test score by re-drawing the data. We re-draw whole stations, not rows, because rows of one station are alike, and we use the same re-drawn stations for both models so that the difference between two models can be tested directly (2,000 draws, seed 42, test set scored once). The MLP is chosen on the validation set only, from 13 settings (our default plus 12 random ones); the default won, and 12 of 13 settings landed within a narrow band (validation RMSE 11.23–11.41).

## 2.7 Research gap and contribution

The gap this project addresses is a fair, like-for-like comparison of a neural network and gradient boosting on the CCME index with the formula inputs hidden, tested on unseen stations and later years, with uncertainty on the difference between models. Our contribution is the comparison itself, an honest account of where the two families differ (the MLP is slightly better on later years, XGBoost slightly better on unseen stations, neither significantly), and the leakage experiment that shows why published scores built on the index parameters are not comparable.

# 3. Literature matrix

Summary table of the sources used in Phase 2.

| # | Citation | Theme | Main contribution | Limitation / caution | Relevance to project |
|---|---|---|---|---|---|
| 1 | Hornik, Stinchcombe & White (1989) — *Multilayer feedforward networks are universal approximators* | Neural network theory | Shows that multilayer feedforward networks can approximate a very wide class of functions. | An existence result; says nothing about training or generalisation. | Why an MLP is a reasonable model family. |
| 2 | Rumelhart, Hinton & Williams (1986) — *Learning representations by back-propagating errors* | Training | Introduces back-propagation for adjusting network weights. | Original method; modern practice adds better optimisers and regularisation. | How our MLP is trained. |
| 3 | Kingma & Ba (2015) — *Adam: A Method for Stochastic Optimization* | Optimiser | Adaptive first-order optimiser using running gradient moments. | Convergence guarantees are for the convex case; other optimisers may do as well. | Optimiser for our MLP (lr 1e-3). |
| 4 | Srivastava et al. (2014) — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* | Regularisation | Randomly drops units during training to reduce overfitting. | Gains depend on network size and data. | Dropout 0.2 in our MLP. |
| 5 | Prechelt (1998) — *Early Stopping — But When?* | Regularisation | Stopping rules based on validation error. | Needs a held-out validation set. | Early stopping, patience 10. |
| 6 | Grinsztajn, Oyallon & Varoquaux (2022) — *Why do tree-based models still outperform deep learning on typical tabular data?* | Tabular comparison | Trees stay ahead on medium-sized tabular data; explains why. | Our data (~260,000 rows) is larger than their medium setting. | Sets the expectation that the MLP will not clearly beat XGBoost. |
| 7 | Shwartz-Ziv & Armon (2022) — *Tabular data: Deep learning is not all you need* | Tabular comparison | XGBoost beat recent deep models, with less tuning; ensembles did best. | Uses datasets from the deep models' own papers. | Supports the hybrid idea and a fair comparison. |
| 8 | Borisov et al. (2022) — *Deep Neural Networks and Tabular Data: A Survey* | Survey | Organises deep tabular methods; trees still ahead in their tests. | Survey; arXiv version cited. | Background on deep tabular methods. |
| 9 | Gorishniy et al. (2021) — *Revisiting Deep Learning Models for Tabular Data* | Tabular comparison | Strong ResNet-like baseline and Transformer; neither family universally better. | Arxiv preprint cited; benchmarks differ from ours. | Justifies a simple MLP as a baseline. |
| 10 | McElfresh et al. (2023) — *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* | Tabular comparison | 19 algorithms on 176 datasets; gap often negligible. | arXiv version cited; mostly smaller datasets. | Supports reporting a modest, uncertain difference. |
| 11 | Bates & Granger (1969) — *The Combination of Forecasts* | Ensembles | Weighted combinations of forecasts can beat each forecast alone. | Gains depend on the errors not being perfectly correlated. | Basis of the optional weighted hybrid. |
| 12 | Sit et al. (2020) — *A Comprehensive Review of Deep Learning Applications in Hydrology and Water Resources* | Applied review | Surveys deep learning in water monitoring, management and forecasting. | Review; not a regression benchmark. | Context for DL in water science. |
| 13 | Zhi et al. (2021) — *From Hydrometeorology to River Water Quality: Can a Deep Learning Model Predict Dissolved Oxygen at the Continental Scale?* | Applied DL | LSTM for dissolved oxygen on 236 US watersheds. | Time-series task on minimally disturbed basins; not comparable with ours. | Shows promise and limits of DL for water quality. |
| 14 | Rahman, Chung & Ng (2026) — *A transformer-FSL-GAN framework for multiclass water quality classification under limited data conditions* | Applied DL, same dataset | Deep few-shot framework for CCME classification on the Canadian subset. | Classification; inputs are the index parameters; one country. | Why high scores there are not comparable with ours. |
| 15 | Efron (1979) — *Bootstrap Methods: Another Look at the Jackknife* | Evaluation | Introduces the bootstrap for uncertainty estimates. | Assumes the resampling unit is exchangeable; we resample whole stations. | Basis of our confidence intervals. |

## Details by source

## 1. Hornik, Stinchcombe & White (1989) — Multilayer feedforward networks are universal approximators

**Theme:** Neural network theory

**Main contribution:** Shows that multilayer feedforward networks are universal approximators.

**Key findings:** A network with enough hidden units can approximate a very broad class of functions. This is an existence result, not a recipe for training.

**Limitation / caution:** Says nothing about how many units are needed, whether training finds good weights, or whether the network generalises.

**Relevance to project:** Why an MLP is a reasonable model for a nonlinear target.

**Reference:** Hornik, K., Stinchcombe, M. & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks, 2(5), 359–366.

## 2. Rumelhart, Hinton & Williams (1986) — Learning representations by back-propagating errors

**Theme:** Training

**Main contribution:** Introduces back-propagation.

**Key findings:** The weights of a network are adjusted repeatedly to reduce the difference between the actual and desired outputs, which lets hidden units learn useful internal features.

**Limitation / caution:** Original method; modern training adds adaptive optimisers and regularisation.

**Relevance to project:** The way our MLP is trained.

**Reference:** Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536.

## 3. Kingma & Ba (2015) — Adam: A Method for Stochastic Optimization

**Theme:** Optimiser

**Main contribution:** A first-order optimiser that adapts learning rates from running estimates of gradient moments.

**Key findings:** Cheap in computation and memory; invariant to diagonal rescaling of gradients; suited to noisy or sparse gradients; hyperparameters are intuitive and rarely need much tuning; a regret bound of O(√T) in the convex online setting.

**Limitation / caution:** The theory is for convex problems; networks are not convex.

**Relevance to project:** The optimiser for the MLP (learning rate 1e-3, batch size 512).

**Reference:** Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015. arXiv:1412.6980.

## 4. Srivastava et al. (2014) — Dropout

**Theme:** Regularisation

**Main contribution:** Dropout, a way to reduce overfitting in large networks.

**Key findings:** Randomly removing units during training stops units from relying too heavily on each other and effectively trains a large set of thinned networks; at test time one full network with scaled weights approximates averaging them. The authors report that this reduces overfitting and improves results on many benchmarks in vision, speech, document classification and computational biology.

**Limitation / caution:** Benchmarks are mostly perceptual tasks, not small tabular problems; our own tests found dropout made little difference.

**Relevance to project:** Dropout 0.2 in the MLP.

**Reference:** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15, 1929–1958.

## 5. Prechelt (1998) — Early Stopping — But When?

**Theme:** Regularisation

**Main contribution:** Practical stopping rules based on the validation error.

**Key findings:** Training is stopped when the validation error stops improving, and the weights from the best point are kept. (Bibliographic details were verified; the abstract was not re-read for this review.)

**Limitation / caution:** Needs held-out validation data, which must not be the test set.

**Relevance to project:** Early stopping with patience 10 (best epoch 19, stopped at 29).

**Reference:** Prechelt, L. (1998). Early Stopping — But When? In Neural Networks: Tricks of the Trade, LNCS 1524, Springer, pp. 55–69.

## 6. Grinsztajn, Oyallon & Varoquaux (2022)

**Theme:** Tabular comparison

**Main contribution:** Investigates why trees remain competitive on typical tabular data.

**Key findings:** On a benchmark of 45 datasets, tree-based models remain state of the art on medium-sized data (about 10,000 samples). The authors point to robustness to uninformative features, preserved feature orientation and learning irregular functions.

**Limitation / caution:** Our 260,000 rows are larger than their setting, so this supports our expectation but does not prove it.

**Relevance to project:** Sets the expectation for the MLP versus XGBoost comparison.

**Reference:** Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 35. arXiv:2207.08815.

## 7. Shwartz-Ziv & Armon (2022)

**Theme:** Tabular comparison

**Main contribution:** Compares recent deep tabular models with XGBoost.

**Key findings:** XGBoost outperformed the deep models on the datasets from the papers that proposed them, needed much less tuning, and an ensemble of deep models with XGBoost did better than XGBoost alone.

**Limitation / caution:** No single family wins everywhere.

**Relevance to project:** Supports a fair comparison and the optional hybrid.

**Reference:** Shwartz-Ziv, R. & Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion, 81, 84–90.

## 8. Borisov et al. (2022) — Deep Neural Networks and Tabular Data: A Survey

**Theme:** Survey

**Main contribution:** Reviews deep methods for mixed-type tabular data.

**Key findings:** The survey groups methods into data transformations, specialised architectures and regularisation models, and compares traditional machine learning with deep learning on real datasets of different sizes and objectives; in those experiments gradient-boosted trees still outperformed the deep models.

**Limitation / caution:** The arXiv record lists no journal reference; the survey has since appeared in IEEE Transactions on Neural Networks and Learning Systems, but check the final citation before submission.

**Relevance to project:** Background on how deep models are used on tables.

**Reference:** Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M. & Kasneci, G. (2022). Deep Neural Networks and Tabular Data: A Survey. arXiv:2110.01889.

## 9. Gorishniy et al. (2021) — Revisiting Deep Learning Models for Tabular Data

**Theme:** Tabular comparison

**Main contribution:** A simple ResNet-like baseline and a Transformer adaptation for tabular data.

**Key findings:** Existing deep models are compared under inconsistent protocols and strong easy-to-use baselines were missing. The ResNet-like baseline is strong, the Transformer adaptation beats other deep solutions on most tasks, and neither deep models nor gradient-boosted trees are universally better.

**Limitation / caution:** The page we checked lists it as an arXiv preprint; check the NeurIPS 2021 proceedings entry before submission.

**Relevance to project:** Justifies using a simple MLP as the deep baseline.

**Reference:** Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021. arXiv:2106.11959.

## 10. McElfresh et al. (2023) — When Do Neural Nets Outperform Boosted Trees on Tabular Data?

**Theme:** Tabular comparison

**Main contribution:** The largest tabular comparison so far (19 algorithms, 176 datasets).

**Key findings:** The neural net versus boosted tree debate receives too much attention, since the performance gap is often negligible or light tuning of a tree model matters more. TabPFN is the main exception on small data. Boosted trees handle skewed, heavy-tailed and irregular features better.

**Limitation / caution:** Cited as an arXiv preprint (v4); it appeared at NeurIPS 2023 (Datasets and Benchmarks).

**Relevance to project:** Supports reporting a small, uncertain difference between the families.

**Reference:** McElfresh, D., Khandagale, S., Valverde, J., Prasad C, V., Feuer, B., Hegde, C., Ramakrishnan, G., Goldblum, M. & White, C. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? arXiv:2305.02997.

## 11. Bates & Granger (1969) — The Combination of Forecasts

**Theme:** Ensembles

**Main contribution:** Weighted combinations of forecasts.

**Key findings:** A combination of two forecasts can have a smaller error than either one alone when their errors are not perfectly correlated; weights can be chosen from past errors.

**Limitation / caution:** Gains are small when the two forecasts make the same mistakes, which is our case (both models over-predict polluted water by about 29 points).

**Relevance to project:** Basis for the optional weighted hybrid.

**Reference:** Bates, J. M. & Granger, C. W. J. (1969). The Combination of Forecasts. Operational Research Quarterly, 20(4), 451–468. DOI 10.1057/jors.1969.103 (volume and pages to be confirmed on the publisher page).

## 12. Sit et al. (2020) — A Comprehensive Review of Deep Learning Applications in Hydrology and Water Resources

**Theme:** Applied review

**Main contribution:** A systematic review of deep learning in the water sector.

**Key findings:** Growing volumes of water data make deep learning useful for forecasting, data improvement and classification across monitoring, management, governance and communication; the paper also discusses challenges and ethical concerns.

**Limitation / caution:** A review, not a benchmark for index regression.

**Relevance to project:** Context for deep learning in water science.

**Reference:** Sit, M., Demiray, B. Z., Xiang, Z., Ewing, G. J., Sermet, Y. & Demir, I. (2020). A Comprehensive Review of Deep Learning Applications in Hydrology and Water Resources. arXiv:2007.12269 (published in Water Science and Technology; confirm volume and pages).

## 13. Zhi et al. (2021) — From Hydrometeorology to River Water Quality

**Theme:** Applied DL

**Main contribution:** An LSTM for dissolved oxygen at continental scale.

**Key findings:** Trained on sparse dissolved-oxygen data and daily weather from 236 minimally disturbed US watersheds, the model captured the decline of dissolved oxygen as temperature rises and showed potential in basins with no measurements, but missed some peaks and troughs, and did not do better where more data existed. The authors argue for more monitoring at extremes and in sparsely monitored areas.

**Limitation / caution:** A time-series task on minimally disturbed basins; not comparable with index regression.

**Relevance to project:** Shows both the promise and the limits of deep learning for water quality, and that sparse extremes are hard, as in our polluted-water results.

**Reference:** Zhi, W., Feng, D., Tsai, W.-P., Sterle, G., Harpold, A., Shen, C. & Li, L. (2021). From Hydrometeorology to River Water Quality: Can a Deep Learning Model Predict Dissolved Oxygen at the Continental Scale? Environmental Science & Technology, 55(4), 2357–2368. DOI 10.1021/acs.est.0c06783.

## 14. Rahman, Chung & Ng (2026) — transformer-FSL-GAN framework

**Theme:** Applied DL on the same dataset

**Main contribution:** A deep few-shot framework for CCME classification on the Canadian subset.

**Key findings:** Reports 97.33% accuracy for five CCME classes, using the eight index parameters as inputs.

**Limitation / caution:** Classification, not regression; inputs are the parameters the index is made from; one country.

**Relevance to project:** Shows how the same dataset is used elsewhere. Our notebook 09 shows that with the five main chemicals added, even our simple models reach R² of about 0.99, so such scores say little about prediction from cheap sensors.

**Reference:** Rahman, A., Chung, G. C. & Ng, Y. H. (2026). A transformer-FSL-GAN framework for multiclass water quality classification under limited data conditions. Frontiers in Water, 8. DOI 10.3389/frwa.2026.1902772.

## 15. Efron (1979) — Bootstrap Methods: Another Look at the Jackknife

**Theme:** Evaluation

**Main contribution:** The bootstrap, a way to estimate the uncertainty of a statistic by re-drawing the data.

**Key findings:** Repeatedly re-drawing samples from the data approximates the sampling distribution of a statistic without a formula.

**Limitation / caution:** The resampling unit must be chosen with care; we re-draw whole stations because rows of one station are not independent.

**Relevance to project:** Our 2,000-draw station-cluster bootstrap.

**Reference:** Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 7(1), 1–26.

# 4. How the literature connects to our results

| Question | What the literature says | What we found |
|---|---|---|
| Does a neural network beat boosted trees on tabular data? | Usually not clearly; the gap is often negligible (Grinsztajn 2022; Borisov 2022; McElfresh 2023). | Station test RMSE: MLP 11.108 vs XGBoost 11.058; paired difference +0.049, interval −0.053 to +0.147, so not distinguishable. |
| Does it help to combine the two families? | Often yes (Bates & Granger 1969; Shwartz-Ziv & Armon 2022). | Weighted hybrid RMSE 11.010; paired difference −0.049, interval −0.098 to −0.002, small but consistent. |
| Do regularisation tools matter much? | They help large networks (Srivastava 2014; Prechelt 1998). | Dropout and weight decay changed validation score very little; early stopping kept the epoch-19 weights. |
| Are published water-quality scores comparable? | High scores often use index parameters as inputs (Rahman 2026). | With the five hidden chemicals added, validation R² ≈ 0.99 for both models. |

# 5. Verification note

Titles, authors, venues and key findings for Kingma & Ba, Srivastava et al., Gorishniy et al., Borisov et al., McElfresh et al., Sit et al. and Zhi et al. were checked against the paper's abstract page or the publisher/index record while preparing this review. For Prechelt (1998) and Bates & Granger (1969) only the bibliographic details were checked; the description of their findings is standard summary content, and the volume and page numbers for Bates & Granger should be confirmed on the publisher page. Hornik et al. (1989), Rumelhart et al. (1986) and Efron (1979) are classic references cited with their standard details. Phase 1 sources reused here (Grinsztajn, Shwartz-Ziv & Armon, Rahman) were verified in the Phase 1 review.

# 6. References

1. Hornik, K., Stinchcombe, M. & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks, 2(5), 359–366.
2. Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536.
3. Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015. arXiv:1412.6980.
4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15, 1929–1958.
5. Prechelt, L. (1998). Early Stopping — But When? In Neural Networks: Tricks of the Trade, LNCS 1524, Springer, 55–69.
6. Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 35. arXiv:2207.08815.
7. Shwartz-Ziv, R. & Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion, 81, 84–90.
8. Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M. & Kasneci, G. (2022). Deep Neural Networks and Tabular Data: A Survey. arXiv:2110.01889.
9. Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. arXiv:2106.11959.
10. McElfresh, D. et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? arXiv:2305.02997.
11. Bates, J. M. & Granger, C. W. J. (1969). The Combination of Forecasts. Operational Research Quarterly, 20(4), 451–468.
12. Sit, M., Demiray, B. Z., Xiang, Z., Ewing, G. J., Sermet, Y. & Demir, I. (2020). A Comprehensive Review of Deep Learning Applications in Hydrology and Water Resources. arXiv:2007.12269.
13. Zhi, W., Feng, D., Tsai, W.-P., Sterle, G., Harpold, A., Shen, C. & Li, L. (2021). From Hydrometeorology to River Water Quality: Can a Deep Learning Model Predict Dissolved Oxygen at the Continental Scale? Environmental Science & Technology, 55(4), 2357–2368.
14. Rahman, A., Chung, G. C. & Ng, Y. H. (2026). A transformer-FSL-GAN framework for multiclass water quality classification under limited data conditions. Frontiers in Water, 8.
15. Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 7(1), 1–26.
