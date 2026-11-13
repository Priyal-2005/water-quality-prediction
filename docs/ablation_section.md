# Phase 2 Ablation Studies: Design Choices & Sensitivity Analysis

## 1. Purpose & Motivation

In tabular deep learning, standard benchmarks frequently obscure whether performance changes stem from network capacity, regularisation, feature subsets, or ensembling. This ablation study systematically isolates and quantifies the marginal impact of each key design choice in our Phase 2 Multilayer Perceptron (MLP) architecture.

Adhering strictly to a One-Factor-At-A-Time (OFAT) protocol, we mutate exactly one setting at a time from the tuned baseline across six targeted axes:
1. **Hierarchical Depth:** 1, 2, or 3 hidden layers via geometric pyramid.
2. **Channel Capacity:** Half, tuned, and double layer widths.
3. **Stochastic Regularisation:** Dropout on (0.2) versus off (0.0).
4. **Weight Decay:** $L_2$ penalty enabled ($10^{-4}$) versus disabled (0.0).
5. **Domain Context Attribution:** Historical sensor persistence (`station_history`) and macro-regional identity (`country`).
6. **Model Formulation:** Standalone MLP versus tree-neural hybrid ensembling.

---

## 2. Experimental Setup & Protocol

The ablation pipeline enforces four core controls to guarantee out-of-station validity and continuity with Phase 1:
- **Validation Partition Isolation:** Every ablation trial is evaluated strictly on the 10% station validation set (`split == 'val'`), comprising 26,054 rows from 1,136 distinct monitoring stations. The 20% test partition is strictly locked for the one-time final evaluation; no ablation trial touches test data.
- **Station Independence:** Splitting is partitioned by `site_id` with zero station overlap across splits ($\text{Train} \cap \text{Val} = \emptyset$).
- **Hidden Chemical Exclusion:** The five driving ions (`ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`) remain strictly excluded from model inputs.
- **Station-Bootstrap Confidence Intervals:** Standard errors and 95% confidence intervals resample entire physical monitoring stations (`site_id`) with replacement over 2,000 draws to capture spatial autocorrelation.

All runs fix seed 42, logging results to an append-only ledger (`results/ablations.csv`).

---

## 3. Consolidated Ablation Results Table

The table below presents the consolidated validation performance across all ablation axes. All metrics report the point estimate accompanied by two-sided 95% station-bootstrap confidence intervals ($[\text{lo}, \text{hi}]$) derived from 2,000 cluster draws. Every row was trained with the same code, seed and data split as the tuned baseline, changing exactly one setting.

| Setting | Value | Val MAE [95% CI] | Val RMSE [95% CI] | Val R² [95% CI] |
|---|---|---|---|---|
| **Tuned Baseline** | Reference (128-64, drop 0.2, wd 1e-4) | 7.767 [7.118, 8.503] | 11.231 [10.374, 12.216] | 0.345 [0.280, 0.396] |
| **Depth** | 1 layer (128) | 7.824 [7.174, 8.586] | 11.369 [10.470, 12.422] | 0.329 [0.270, 0.376] |
| **Depth** | 2 layers (128, 64) [Tuned] | 7.767 [7.118, 8.503] | 11.231 [10.374, 12.216] | 0.345 [0.280, 0.396] |
| **Depth** | 3 layers (128, 64, 32) | 7.897 [7.263, 8.623] | 11.279 [10.423, 12.281] | 0.339 [0.274, 0.390] |
| **Width** | Half (64, 32) | 7.914 [7.295, 8.624] | 11.284 [10.440, 12.254] | 0.339 [0.273, 0.390] |
| **Width** | Double (256, 128) | 7.890 [7.266, 8.615] | 11.273 [10.425, 12.278] | 0.340 [0.275, 0.389] |
| **Dropout** | 0.0 (Disabled) | 7.665 [7.016, 8.406] | 11.245 [10.366, 12.230] | 0.343 [0.280, 0.396] |
| **Dropout** | 0.2 (Enabled) [Tuned] | 7.767 [7.118, 8.503] | 11.231 [10.374, 12.216] | 0.345 [0.280, 0.396] |
| **Weight Decay** | 0.0 (Disabled) | 7.803 [7.156, 8.528] | 11.245 [10.398, 12.213] | 0.343 [0.278, 0.395] |
| **Weight Decay** | 1e-4 (Enabled) [Tuned] | 7.767 [7.118, 8.503] | 11.231 [10.374, 12.216] | 0.345 [0.280, 0.396] |
| **Station History** | true (Included) [Tuned] | 7.767 [7.118, 8.503] | 11.231 [10.374, 12.216] | 0.345 [0.280, 0.396] |
| **Station History** | false (Excluded) | 8.115 [7.418, 8.989] | 11.878 [10.829, 13.124] | 0.267 [0.218, 0.306] |
| **Country** | true (Included) [Tuned] | 7.767 [7.118, 8.503] | 11.231 [10.374, 12.216] | 0.345 [0.280, 0.396] |
| **Country** | false (Excluded) | 8.241 [7.619, 9.008] | 11.733 [10.868, 12.761] | 0.285 [0.218, 0.337] |
| **Model** | hybrid (PENDING: awaiting results/hybrid_val_preds.csv) | NA | NA | NA |

*Note: The hybrid model row remains PENDING with NA metrics, awaiting out-of-fold validation predictions (`results/hybrid_val_preds.csv`). No hybrid numbers are fabricated.*

---

## 4. Empirical Findings

Differences below are measured against the tuned baseline (RMSE 11.231, MAE 7.767, R² 0.345). The confidence intervals in the table are computed separately for each model, so they overlap whenever two models are close. We therefore read overlapping intervals as "no clear difference", not as proof of equality.

- **Architectural depth (1, 2 or 3 layers):** The tuned 2-layer network (128, 64) has the lowest validation RMSE (11.231). One layer gives 11.369 (+0.138) and three layers give 11.279 (+0.048). The intervals overlap heavily, so there is no clear evidence that more depth helps on this 14-input table; a single hidden layer is only slightly worse.
- **Layer width (half, tuned, double):** Half width (64, 32) gives RMSE 11.284 (+0.053) and double width (256, 128) gives 11.273 (+0.042). Both are marginally worse than the tuned size and well within each other's intervals, so extra capacity brings no benefit and the tuned size is a reasonable choice.
- **Dropout (0.0 vs 0.2):** Removing dropout gives RMSE 11.245 (+0.014) and MAE 7.665 (-0.102). The RMSE is not better and the MAE is slightly lower, but both are inside the intervals. Dropout is therefore low-impact here; early stopping already limits overfitting.
- **Weight decay (0.0 vs 1e-4):** Without weight decay the RMSE is 11.245 (+0.014) and the MAE is 7.803 (+0.035). The difference is negligible, so L2 shrinkage has little effect when early stopping is used.
- **Station history (included vs excluded):** This is the largest effect in the study. Without it RMSE rises from 11.231 to 11.878 (+0.647), MAE from 7.767 to 8.115, and R² falls from 0.345 to 0.267 (-0.078). This matches Phase 1, where removing station history dropped the XGBoost R² from 0.348 to 0.271. The two intervals still overlap because they are computed separately; since both models are scored on the same stations, a paired bootstrap of the difference would be tighter and is the right test to confirm it.
- **Country indicator (included vs excluded):** Without country, RMSE rises from 11.231 to 11.733 (+0.502), MAE from 7.767 to 8.241, and R² falls from 0.345 to 0.285 (-0.060). Phase 1 XGBoost showed a similar drop (R² 0.348 to 0.274). The same note about overlapping separate intervals applies.
- **Model formulation (MLP vs hybrid):** The standalone tuned MLP gives R² 0.345 and RMSE 11.231. The comparison with the hybrid is pending the out-of-fold predictions (`results/hybrid_val_preds.csv`).

**Summary:** the inputs matter far more than the network design. Station history and country each cost clearly more than any change to depth, width, dropout or weight decay, and none of those four settings moved the result by more than about 0.14 RMSE.

---

## 5. Limitations & Threats to Validity

1. **Discrete Grid Resolution:** Evaluating isolated settings explores orthogonal axes rather than continuous response surfaces.
2. **Validation Partition Variance:** While the validation split contains 26,054 rows from 1,136 distinct monitoring stations, localized sensor calibration quirks introduce sampling variance.
3. **Tabular Inductive Biases:** Prior literature demonstrates that tree ensembles partition tabular spaces naturally. Model accuracy is primarily bounded by domain feature representation rather than neural capacity.
4. **Hybrid Protocol Dependency:** Hybrid benchmarking requires genuine out-of-fold predictions to prevent optimistic leakage; until verified out-of-fold predictions are available, ensembling claims remain unverified.
