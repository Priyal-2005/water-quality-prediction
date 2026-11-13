# Phase 2 Ablation Studies: Design Choices & Sensitivity Analysis

> **Notice:** Prepared by Sarthak as part of his Ablation Studies role. To be merged into the IEEE report by Vaishnavi.

---

## 1. Purpose & Motivation

In tabular deep learning, standard empirical benchmarks often mask whether performance gains arise from increased parameter capacity, regularization mechanisms, specific feature domains, or ensembling. The objective of this ablation study is to systematically isolate and quantify the marginal impact of each key design choice in our Phase 2 Multilayer Perceptron (MLP) pipeline. 

By varying **exactly one setting at a time** from the tuned baseline, we answer five critical research questions:
1. **Architectural Depth:** Does deeper non-linear hierarchy (3 hidden layers vs. 2) improve representation or exacerbate overfitting?
2. **Layer Width:** Does expanding channel capacity (128 vs. 64 neurons) capture more complex interactions among water quality indicators?
3. **Stochastic Regularisation (Dropout):** Does randomly muting 20% of activations prevent co-adaptation, or does it impair convergence on tabular inputs?
4. **Weight Decay ($L_2$ Regularisation):** Does constraining the Frobenius norm of layer weights improve station generalisation?
5. **Feature Context:** How much predictive signal originates from historical sensor trends (`station_history`) versus regional baselines (`country`), compared to raw physicochemical sensors alone?
6. **Model Formulation:** Does an MLP alone capture different residual dynamics than XGBoost, and does a hybrid ensemble add verifiable predictive value?

---

## 2. Experimental Setup & Fairness Controls

To guarantee complete scientific integrity and direct comparability with Phase 1 results, the ablation framework enforces strict fairness controls:

- **Partition Integrity (Zero Test Leakage):** All ablation trials are evaluated exclusively on the validation partition (`split == 'val'`). The 20% holdout test partition is strictly locked and reserved for Anusha’s final one-time model evaluation (`src/evaluate_dl.py`). Sarthak never touches or scores the test set.
- **Strict Station Splitting:** We reuse the identical station-based split (`split`) stored in `data/processed/final_dataset.csv`. Stations are partitioned 70% train, 10% validation, and 20% test with **zero station overlap** ($\text{Train} \cap \text{Val} = \emptyset$). This directly evaluates out-of-station spatial generalisation.
- **Hidden Chemical Parameters:** The five primary driving ions and biochemical measures (`ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`) remain strictly excluded from model inputs to prevent trivial re-learning of the CCME index formula.
- **Preprocessing Isolation:** Imputation medians, standard scaling ($\mu, \sigma$), and one-hot categorical encodings are fitted strictly on the training partition ($X_{\text{train}}$) and applied downstream.
- **Station-Level Bootstrap Confidence Intervals:** For all comparative metrics, standard errors and 95% confidence intervals are computed by resampling entire monitoring stations (`site_id`) rather than individual sample rows, accounting for strong serial and spatial autocorrelation.
- **Out-of-Fold Hybrid Protocol:** Any hybrid model evaluation strictly requires out-of-fold-by-station validation predictions (`results/hybrid_val_preds.csv`) from Anusha and Priyal.
- **Reproducibility:** Seed 42 is fixed across all data splitting, mini-batch shuffling, and PyTorch weight initialization routines.

---

## 3. Consolidated Ablation Results Table

The table below presents the consolidated evaluation protocol across all planned ablation axes. All rows reflect append-only records logged in `results/ablations.csv` with 95% station-bootstrap confidence intervals ($[\text{lo}, \text{hi}]$ from 2,000 resamples of whole monitoring stations).

| Setting / Axis | Evaluated Value | Status | Val MAE [95% CI] | Val RMSE [95% CI] | Val $R^2$ [95% CI] | Notes / Experimental Condition |
|---|---|---|---|---|---|---|
| **Tuned Baseline** | Reference | `PLACEHOLDER` | *NA* | *NA* | *NA* | Pending Priyal's committed tuned config (`configs/tuned_mlp.yaml`). Scaffolding default: [64, 64], lr $10^{-3}$, bs 32. |
| **Depth** | 2 hidden layers | `PLACEHOLDER` | *NA* | *NA* | *NA* | Preserves baseline width; tests 2-layer representation. |
| **Depth** | 3 hidden layers | `PLACEHOLDER` | *NA* | *NA* | *NA* | Adds 3rd hidden layer; tests higher non-linear abstraction. |
| **Width** | 64 neurons | `PLACEHOLDER` | *NA* | *NA* | *NA* | Compressed width across hidden layers. |
| **Width** | 128 neurons | `PLACEHOLDER` | *NA* | *NA* | *NA* | Expanded width across hidden layers. |
| **Dropout** | 0.0 (Disabled) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Eliminates stochastic feature dropping. |
| **Dropout** | 0.2 (Enabled) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Standard regularisation rate. |
| **Weight Decay** | 0.0 (Disabled) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Unpenalized Adam optimisation ($L_2 = 0$). |
| **Weight Decay** | $10^{-4}$ (Enabled) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Standard weight shrinkage. |
| **Station History** | `true` (Included) | `PLACEHOLDER` | *NA* | *NA* | *NA* | All 5 temporal past-mean sensor features included. |
| **Station History** | `false` (Excluded) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Removes 5 historical features; relies purely on concurrent readings. |
| **Country** | `true` (Included) | `PLACEHOLDER` | *NA* | *NA* | *NA* | One-hot geographic region indicator included. |
| **Country** | `false` (Excluded) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Removes regional indicators; enforces cross-national invariance. |
| **Model** | `mlp` (Standalone) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Standalone tuned neural architecture reference. |
| **Model** | `hybrid` (Ensemble) | `PLACEHOLDER` | *NA* | *NA* | *NA* | Out-of-fold XGBoost + MLP blend; pending Anusha's validation predictions. |

*Note: In accordance with academic honesty and our team charter, all rows are designated `status=PLACEHOLDER` with `NA` metrics until Priyal provides `configs/tuned_mlp.yaml` and Anusha provides `results/hybrid_val_preds.csv`. No numbers are fabricated.*

---

## 4. Empirical Findings

*(To be updated after real runs. No empirical superiority claims are made prior to execution on Priyal's tuned baseline.)*

Planned analysis points once genuine `RUN` executions complete:
- **Depth & Width Capacity:** Comparison of whether expanding hidden dimensions from 64 to 128 or adding a 3rd hidden layer provides statistically meaningful RMSE reductions on unseen stations, or if the 14-input tabular manifold is already saturated by 2 layers.
- **Regularisation Impact:** Evaluation of validation loss curves with and without dropout and weight decay. In tabular settings, dropout can sometimes hinder training efficiency if feature co-adaptation is already constrained by low dimensionality.
- **Feature Attribution:** Isolation of validation performance drops when removing `station_history` versus `country`. In Phase 1 XGBoost ablations (`results/ablation_validation.csv`), removing station history dropped $R^2$ from 0.348 to 0.271, and removing country dropped $R^2$ to 0.274. We will verify whether the MLP exhibits identical feature dependencies.
- **Hybrid Value-Add:** Determining whether blending tree-based recursive partitioning (XGBoost) with smooth gradient-based linear-ReLU hyperplanes (MLP) reduces residual bias on extreme water quality bands.

---

## 5. Limitations & Threats to Validity

1. **Discrete Grid Resolution:** Testing binary variations (e.g. depth 2 vs 3, dropout 0.0 vs 0.2) explores isolated orthogonal axes rather than continuous hyperparameter surfaces.
2. **Validation Partition Variance:** Although the 10% station validation set contains 25,977 rows from 994 distinct stations, localized sensor calibration quirks or regional sampling densities (predominantly UK stations) introduce sampling noise.
3. **Tabular Inductive Biases:** Prior literature (Grinsztajn et al., 2022) indicates that gradient-boosted decision trees inherently handle unnormalized marginal distributions, coordinate-aligned boundaries, and missing indicators more robustly than smooth neural activations. The MLP is not guaranteed to outperform XGBoost.
4. **Hybrid Validity:** A hybrid model is only valid if trained on strictly out-of-fold predictions. In-sample predictions would induce severe leakage and yield overly optimistic validation metrics.

