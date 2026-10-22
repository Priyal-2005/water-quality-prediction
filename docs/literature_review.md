# Literature Review

**Water Quality Prediction via Gradient Boosting and Deep Neural Networks**

*Phase 1 \| Prepared from project repository and the assigned literature brief*

# 1. Introduction

Water-quality indices compress multiple physical and chemical measurements into a score that can be communicated more easily than a list of individual parameters. Machine-learning regression can estimate such a score from a restricted set of observations, but a high score is meaningful only if the target definition, input features and evaluation split match the intended prediction task. The project predicts target_ccme_value, a continuous CCME Water Quality Index (WQI) score on a 0–100 scale.

The project uses a cleaned subset of 259,774 rows from 9,940 sites across England, Ireland, the USA and Canada for 2000–2023. Phase 1 compares a mean baseline, Ridge Regression, Random Forest and XGBoost. The main design choices are excluding five target-driving parameters, evaluating unseen monitoring sites, and adding a separate chronological test.

# 2. Related work by theme

## 2.1 Water-quality indices and standards

The CCME WQI framework summarises water-quality guideline exceedances through scope, frequency and amplitude. It provides an interpretable index, but its construction creates a direct risk for predictive modelling: if the model receives the parameters used to calculate the index, it can learn to reconstruct the target formula. The project therefore separates the index-defining measurements from the permitted predictors and explicitly compares leakage-prone and restricted feature sets.

## 2.2 Machine learning for water-quality prediction

Water-quality studies cover several different tasks: regression of a WQI score, classification into quality categories, prediction of individual physicochemical parameters, and forecasting over time. Their results are not directly comparable unless the target, geography, sample size, feature set, and split protocol are similar. Karim et al. (2025) describe a broad surface-water dataset spanning 1940–2023. The project uses a cleaned subset, removes non-surface-water records and repeated placeholder values, and limits the study to four countries because the remaining Chinese records do not support a site-held-out split.

A recurring methodological concern is that random row splits can place observations from the same station in training and test sets. Because a station may have stable characteristics and repeated samples, this can inflate apparent generalisation. The project’s station-grouped split assigns all observations from a site to a single partition; a separate year split evaluates later years. This design is a project-specific safeguard and should not be represented as proof that all previous work uses random splits.

## 2.3 Tree ensembles on tabular data

Random Forests aggregate many decision trees trained with bootstrap samples and random feature subsets, which helps reduce variance and model nonlinear interactions. Gradient boosting builds an additive model sequentially; XGBoost adds regularisation and computational optimisations. These properties make tree ensembles strong candidates for heterogeneous environmental tables. The project’s validation results support this choice: Random Forest and XGBoost outperform Ridge Regression and the mean baseline, while their difference is small.

## 2.4 Neural networks on tabular data

Multilayer perceptrons can learn nonlinear relationships, but performance depends on feature preprocessing, architecture, regularisation, data volume and tuning. Grinsztajn et al. (2022) and Shwartz-Ziv and Armon (2022) are useful for explaining why deep learning does not automatically outperform tree-based models on typical tabular tasks. The Phase 2 MLP should therefore be compared using the same features, splits and metrics rather than assumed to be better in advance.

## 2.5 Interpretability

SHAP attributes a prediction to input features using a game-theoretic framework. In this project, SHAP plots can show which permitted features most influence XGBoost predictions and whether those patterns are plausible. Feature attribution is not causal evidence, and correlated variables can make individual attributions harder to interpret.

## 2.6 What published applied studies report

Applied studies report high scores, but they use different tasks, inputs and metrics. Ooko et al. (2023) report that Random Forest gave the best result with 79% accuracy in a water-quality classification study that used open data (from the abstract). Rahman et al. (2026) report 97.33% accuracy for five-class CCME classification on the Canadian subset of the dataset used here, with the eight index parameters as inputs. Because the CCME index is calculated from those parameters, such inputs make the index easy to recover (our leakage test gives R² 0.99 with all eight), so these figures are not comparable with the 0.293 R² we obtain with five parameters held out.

Index-based targets also have known weaknesses. Chidiac et al. (2023) describe eclipsing (a single very poor parameter masked by good ones), loss of information about interactions, subjectivity in parameter choice and weighting, and sensitivity to how the index is formulated. These support our decision to treat the CCME value as a modelling target with care, and to state its limits in the report.

## 2.7 Research gap and contribution

The relevant gap for this project is not simply that water-quality prediction has not been studied; rather, published studies use different targets, feature sets and evaluation protocols, and scores can be misleading when target-defining parameters leak into predictors or stations overlap between partitions. This project addresses that risk by excluding ammonia, BOD, orthophosphate, nitrogen and nitrate from the model inputs, comparing a mean baseline and three classical models, testing unseen stations, and conducting a separate future-period evaluation. It reports MAE, RMSE and R² with bootstrap intervals, then plans a like-for-like MLP comparison in Phase 2.

# 3. Literature matrix

Summary table of all twelve sources, followed by the key findings and references for each one.

| # | Citation | Theme | Main contribution | Limitation / caution | Relevance to project |
|---|---|---|---|---|---|
| 1 | CCME (2001) — *CCME WQI 1.0 Technical Report* | WQI standard | Defines the index calculation around the scope, frequency and amplitude of guideline exceedances. | Technical framework, not a predictive ML benchmark. | Defines the target and motivates leakage control. |
| 2 | Breiman (2001) — *Random Forests* | Ensemble method | Introduces randomised tree ensembles using bootstrap samples and feature subsampling. | General method paper, not specific to water quality. | Foundation for the Random Forest model. |
| 3 | Friedman (2001) — *Greedy Function Approximation: A Gradient Boosting Machine* | Gradient boosting | Develops stage-wise additive modelling via gradient boosting. | Foundational method; not a benchmark on this dataset. | Explains the boosting family. |
| 4 | Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System* | Boosted trees | Presents a scalable, regularised tree-boosting system. | Results from other benchmarks do not guarantee this project's performance. | Supports testing XGBoost. |
| 5 | Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* | Explainability | Introduces SHAP as a unified feature-attribution framework. | Attributions are not causal; correlated features complicate interpretation. | Supports model explanation. |
| 6 | Grinsztajn et al. (2022) — *Why do tree-based models still outperform deep learning on typical tabular data?* | Tabular ML comparison | Investigates why tree-based models remain competitive on typical tabular benchmarks. | Findings depend on benchmark design and data characteristics. | Motivates strong tree baselines before the Phase 2 MLP. |
| 7 | Shwartz-Ziv & Armon (2022) — *Tabular data: Deep learning is not all you need* | Tabular ML comparison | Discusses deep learning approaches for tabular data and their trade-offs. | No single model family dominates every dataset. | Supports a fair empirical Phase 2 comparison. |
| 8 | Karim et al. (2025) — *A Comprehensive Dataset of Surface Water Quality Spanning 1940–2023 for Empirical and ML Adopted Research* | Water-quality dataset | Describes a harmonised dataset with 2.82 million records, five countries, eight parameters and a 1940–2023 span. | The project uses a cleaned subset; coverage and data quality vary by country. | Source of our data; compare the original and the subset honestly. |
| 9 | Syeed et al. (2023) — *Surface water quality profiling using the water quality index, pollution index and statistical methods: A critical review* | Water-quality review | Reviews WQI and pollution-index approaches for surface-water profiling. | A review, not a benchmark for restricted-feature regression. | Context for index-based assessment. |
| 10 | Ooko et al. (2023) — *Use of Machine Learning for Realtime Water Quality Prediction* | Applied water-quality ML | Applies supervised learning in a real-time water-quality prediction setting. | Different task and data; its results are not numerically comparable with ours. | Example of applied water-quality ML. |
| 11 | Chidiac et al. (2023) — *A comprehensive review of water quality indices (WQIs): history, models, attempts and perspectives* | Water-quality index review | Reviews the history and models of water-quality indices (NSF, Dinius, CCME, Oregon, weighted arithmetic and others). | A review, not a predictive benchmark. | Why an index is an imperfect target; use for limitations. |
| 12 | Rahman, Chung & Ng (2026) — *A transformer-FSL-GAN framework for multiclass water quality classification under limited data conditions* | Applied ML on the same dataset | Applies a deep few-shot framework to CCME classification on the Canadian subset of the same dataset. | Classification, not regression; inputs are the index parameters; single country. | Shows how the same dataset is used elsewhere and why high scores there are not comparable with ours. |

## Details by source


## 1. CCME (2001) — CCME WQI 1.0 Technical Report

**Theme:** WQI standard

**Main contribution:** Defines index calculation around scope, frequency and amplitude of guideline exceedances.

**Key findings:** The index is WQI = 100 − √(F1² + F2² + F3²) / 1.732, where F1 (scope) is the share of variables that fail their guideline, F2 (frequency) is the share of tests that fail, and F3 (amplitude) is how far failing values exceed the guideline. Because the score is a direct function of the measured parameters, a model given those parameters can recompute it, which is the leakage this project tests for.
  
**Limitation / caution:** Technical framework, not a predictive ML benchmark.

**Relevance to project:** Defines the target and motivates leakage control.

**Reference:** CCME (2001), CCME Water Quality Index 1.0 Technical Report (formula checked against a provincial technical appendix that reproduces it).

## 2. Breiman (2001) — Random Forests

**Theme:** Ensemble method

**Main contribution:** Introduces randomised tree ensembles using bootstrap samples and feature subsampling.

**Key findings:** Breiman shows that the generalisation error of a forest converges as trees are added, so adding trees does not cause overfitting, and that the error depends on the strength of the individual trees and the correlation between them. This is why random feature selection at each split is used.
  
**Limitation / caution:** General method paper, not specific to water quality.

**Relevance to project:** Foundation for Random Forest baseline.

**Reference:** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

## 3. Friedman (2001) — Greedy Function Approximation: A Gradient Boosting Machine

**Theme:** Gradient boosting

**Main contribution:** Develops stage-wise additive modelling via gradient boosting.

**Key findings:** Friedman casts boosting as gradient descent in function space: each new tree is fitted to the negative gradient of the loss, and shrinking each tree's contribution (a small learning rate) improves generalisation. This is the idea behind XGBoost's learning-rate setting.
  
**Limitation / caution:** Foundational method; not a benchmark on this dataset.

**Relevance to project:** Explains the boosting family.

**Reference:** Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.

## 4. Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System

**Theme:** Boosted trees

**Main contribution:** Presents a scalable, regularised tree-boosting system.

**Key findings:** The paper contributes a sparsity-aware algorithm for tree learning, a weighted quantile sketch for approximate split finding, and system-level techniques for cache access, compression and sharding. It claims XGBoost scales to billions of examples with far fewer resources than existing systems. The abstract gives no benchmark numbers.
  
**Limitation / caution:** Results from other benchmarks do not guarantee this project’s performance.

**Relevance to project:** Supports testing XGBoost.

**Reference:** Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. arXiv:1603.02754; DOI 10.1145/2939672.2939785.

## 5. Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions

**Theme:** Explainability

**Main contribution:** Introduces SHAP as a unified feature-attribution framework.

**Key findings:** The paper unifies six existing attribution methods into one class of additive feature-importance measures and shows that one unique solution in that class has a set of desirable properties; this solution is SHAP. The authors report that it agrees better with human intuition than earlier methods, and that their methods are computationally more efficient.
  
**Limitation / caution:** Attributions are not causal; correlated features complicate interpretation.

**Relevance to project:** Supports model explanation.

**Reference:** Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. arXiv:1705.07874.

## 6. Grinsztajn et al. (2022) — Why do tree-based models still outperform deep learning on typical tabular data?

**Theme:** Tabular ML comparison

**Main contribution:** Investigates why tree-based models remain competitive on typical tabular benchmarks.

**Key findings:** On a benchmark of 45 tabular datasets, tree-based models (XGBoost and Random Forest) remain state of the art on medium-sized data (around 10,000 samples), even before counting their speed advantage. The authors attribute this to three properties that neural networks need: robustness to uninformative features, preserving the orientation of the data (trees split on single features), and learning irregular functions. Our data (about 260,000 rows) is larger than the 'medium-sized' setting, so this is supporting evidence, not proof for us.
  
**Limitation / caution:** Findings depend on benchmark design and data characteristics.

**Relevance to project:** Motivates strong tree baselines before MLP.

**Reference:** Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). arXiv:2207.08815.

## 7. Shwartz-Ziv & Armon (2022) — Tabular data: Deep learning is not all you need

**Theme:** Tabular ML comparison

**Main contribution:** Discusses deep learning approaches for tabular data and their trade-offs.

**Key findings:** The authors compared several recent deep models for tabular data with XGBoost, using the datasets from the papers that proposed those models. XGBoost outperformed the deep models, needed much less hyperparameter tuning, and an ensemble of deep models with XGBoost did better than XGBoost alone.
  
**Limitation / caution:** No single model family dominates every dataset.

**Relevance to project:** Supports fair empirical Phase 2 comparison.

**Reference:** Shwartz-Ziv, R. & Armon, A. (2022). arXiv:2106.03253 (check the published journal version before citing).

## 8. Karim et al. (2025) — A Comprehensive Dataset of Surface Water Quality Spanning 1940–2023 for Empirical and ML Adopted Research

**Theme:** Water-quality dataset

**Main contribution:** Describes a harmonised dataset with 2.82 million records, five countries, eight parameters and a 1940–2023 span.

**Key findings:** The dataset combines records from regional and global databases, water-management agencies and research projects in the USA, Canada, Ireland, England and China: about 2.82 million measurements of eight water-quality parameters from 1940 to 2023, intended for empirical analysis, prediction and forecasting. Our own inspection (not a claim of the paper) found that large shares of several parameters are one repeated value per country and water type, which we set to missing during cleaning (docs/cleaning_log.md).
  
**Limitation / caution:** The project uses a cleaned subset; coverage and data quality vary by country.

**Relevance to project:** Primary dataset provenance; compare original and subset honestly.

**Reference:** Karim, M. R., Syeed, M. M. M., Rahman, A., Rabbani, K. A., Fatema, K., Khan, R. H., Hossain, M. S. & Uddin, M. F. (2025). Scientific Data, 12. DOI 10.1038/s41597-025-04715-4.

## 9. Syeed et al. (2023) — Surface water quality profiling using the water quality index, pollution index and statistical methods: A critical review

**Theme:** Water-quality review

**Main contribution:** Reviews WQI/pollution-index approaches for surface-water profiling.

**Key findings:** A systematic review of surface-water quality profiling. It finds 13 distinct freshwater sources used mainly across five sectors, about 23 water-quality-index models and about 10 pollution-index models, and builds a taxonomy of 69 parameters in three categories. It lists the limitations of manual measurement and proposes seven requirements for technology-driven profiling. (Syeed is also a co-author of the dataset paper.)
  
**Limitation / caution:** Review rather than direct benchmark for restricted-feature regression.

**Relevance to project:** Context for index-based assessment.

**Reference:** Syeed et al. (2023). Environmental and Sustainability Indicators, June 2023. Author list and DOI still to be confirmed from the publisher page.

## 10. Ooko et al. (2023) — Use of Machine Learning for Realtime Water Quality Prediction

**Theme:** Applied water-quality ML

**Main contribution:** Applies supervised learning in a real-time water-quality prediction setting.

**Key findings:** The abstract says the authors tested supervised models for predicting water quality from physico-chemical properties, using open data, and that Random Forest gave the best result with 79% accuracy. The abstract does not name the dataset or the other models, and it uses accuracy, so the result is for classification and cannot be compared with our MAE, RMSE or R². Only the abstract was read.
  
**Limitation / caution:** Different task and data; not numerically comparable without checking original details.

**Relevance to project: Example of applied water-quality ML.**

**Reference:** Ooko, S. O., Pamela, E. K. & Kwagalakwe, G. (2023). IEEE Xplore, document 10293701 (venue not stated on the source page).

## 11. Chidiac et al. (2023) — A comprehensive review of water quality indices (WQIs): history, models, attempts and perspectives

**Theme: Water-quality index review**

**Main contribution:** Provides context for why an index is an imperfect target: it can hide a single very poor parameter and loses information about interactions.

**Key findings:** A review of water-quality indices (NSF, Dinius, CCME, Oregon, weighted arithmetic and others). Its main limitations of indices are: eclipsing (one very poor parameter can be masked by good ones), loss of information and of interactions between variables, subjectivity in choosing and weighting parameters, sensitivity to how the index is formulated, and limited portability between regions. It notes that the CCME index can perform poorly when few variables are used or variables covary strongly. No single index is applicable across all countries and uses.
  
**Limitation / caution: A review, not a predictive benchmark.**

**Relevance to project: Use as an additional source for the limitations of index-based targets.**

**Reference:** Chidiac, S., El Najjar, P., Ouaini, N., El Rayess, Y. & El Azzi, D. (2023). Reviews in Environmental Science and Bio/Technology, 22(2), 349–395. DOI 10.1007/s11157-023-09650-7.

## 12. Rahman, Chung & Ng (2026) — A transformer-FSL-GAN framework for multiclass water quality classification under limited data conditions

**Theme: Applied ML on the same dataset**

**Main contribution:** Applies a deep few-shot framework to CCME classification on the Canadian subset of the same dataset.

**Key findings:** The study uses the Canadian subset of the same dataset (3,949 observations from 2,354 stations) for five-class CCME classification, with a Transformer + few-shot + GAN framework, and reports 97.33% accuracy (10-shot). According to the paper, the inputs are the eight physico-chemical parameters, which are the ingredients of the index; our leakage experiment shows such inputs make the index easy to recover, so scores like this are not comparable with our restricted-input results. Its own stated limitations: one country only, only 19 'Poor' samples, and fixed random episodes.
  
**Limitation / caution: Classification, not regression; inputs are the index parameters; single country.**

**Relevance to project: Shows how the same dataset is used elsewhere and why high scores there are not comparable with ours.**

**Reference:** Rahman, A., Chung, G. C. & Ng, Y. H. (2026). A transformer-FSL-GAN framework for multiclass water quality classification under limited data conditions. Frontiers in Water, 8. DOI 10.3389/frwa.2026.1902772.

# 4. Project results to connect with the literature

On the station-split validation set, XGBoost is the best model (MAE 7.729, RMSE 11.205, R² 0.348), narrowly ahead of Random Forest (MAE 7.773, RMSE 11.247, R² 0.343). On the held-out station test set, XGBoost achieves MAE 7.543, RMSE 11.058, R² 0.293; Random Forest achieves MAE 7.598, RMSE 11.179, R² 0.278. The drop in R² from validation to test (0.348 to 0.293 for XGBoost) has two likely causes that this project has not separated: the hyperparameters were chosen on the validation stations, which makes validation scores slightly optimistic, and the test stations are different stations that may be harder. It is therefore a reason to report the test score, with its 95% bootstrap interval (R² 0.246 to 0.340 for XGBoost), rather than the best validation score. A paired bootstrap over the same resampled stations shows that XGBoost’s RMSE is lower than Random Forest’s by 0.12 (95% interval −0.19 to −0.06), so its advantage is consistent but small; both clearly outperform Ridge Regression and the mean baseline.

The chronological test gives XGBoost MAE 5.449, RMSE 8.047, R² 0.243, with Random Forest close behind. The lower MAE/RMSE but lower R² relative to station test reflects a different target distribution and split; metrics across these splits should not be treated as interchangeable. The target has a ceiling effect: 43.7% of labels are exactly 100, so MAE/RMSE/R² should be interpreted with per-country and residual plots.

### Station split: held-out test (`results/final_results.csv`)

| Model | MAE | RMSE | R² | 95% bootstrap CI for R² |
|---|---:|---:|---:|---|
| Mean baseline | 9.570 | 13.159 | -0.001 | [-0.009, -0.000] |
| Ridge | 8.329 | 11.733 | 0.205 | [0.174, 0.233] |
| Random Forest | 7.598 | 11.179 | 0.278 | [0.230, 0.324] |
| XGBoost | 7.543 | 11.058 | 0.293 | [0.246, 0.340] |

### Chronological test, 2018 onward (`results/year_split_results.csv`)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 9.101 | 10.693 | -0.338 |
| Ridge | 6.738 | 8.786 | 0.097 |
| Random Forest | 5.596 | 8.310 | 0.192 |
| XGBoost | 5.449 | 8.047 | 0.243 |

The two tables use different test populations (new stations vs. later years), so compare models within one table, not across the two.
