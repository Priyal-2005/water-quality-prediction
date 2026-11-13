# Phase 2 Ablation Studies: Design Choices & Sensitivity Analysis

## 1. Purpose & Motivation

In tabular deep learning, standard benchmarks frequently obscure whether performance changes stem from network capacity, regularisation, feature subsets, or ensembling. This ablation study systematically isolates and quantifies the marginal impact of each key design choice in our Phase 2 Multilayer Perceptron (MLP) architecture.

Adhering strictly to a One-Factor-At-A-Time (OFAT) protocol, we mutate exactly one setting at a time from Priyal's tuned baseline across six targeted axes:
1. **Hierarchical Depth:** 1, 2, or 3 hidden layers via geometric pyramid.
2. **Channel Capacity:** Half, tuned, and double layer widths.
3. **Stochastic Regularisation:** Dropout on (0.2) versus off (0.0).
4. **Weight Decay:** $L_2$ penalty enabled ($10^{-4}$) versus disabled (0.0).
5. **Domain Context Attribution:** Historical sensor persistence (`station_history`) and macro-regional identity (`country`).
6. **Model Formulation:** Standalone MLP versus tree-neural hybrid ensembling.

---

## 2. Experimental Setup & Protocol

The ablation pipeline enforces four core controls to guarantee out-of-station validity and continuity with Phase 1:
- **Validation Partition Isolation:** Every ablation trial is evaluated strictly on the 10% station validation set (`split == 'val'`), comprising 26,054 rows from 1,136 distinct monitoring stations. The 20% test partition is strictly locked for Anusha's final evaluation; Sarthak never evaluates on test data.
- **Station Independence:** Splitting is partitioned by `site_id` with zero station overlap across splits ($\text{Train} \cap \text{Val} = \emptyset$).
- **Hidden Chemical Exclusion:** The five driving ions (`ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`) remain strictly excluded from model inputs.
- **Station-Bootstrap Confidence Intervals:** Standard errors and 95% confidence intervals resample entire physical monitoring stations (`site_id`) with replacement over 2,000 draws to capture spatial autocorrelation.

All runs fix seed 42, logging results to an append-only ledger (`results/ablations.csv`).

---

## 3. Consolidated Ablation Results Table

The table below presents the consolidated validation performance across all ablation axes. All metrics report the point estimate accompanied by two-sided 95% station-bootstrap confidence intervals ($[\text{lo}, \text{hi}]$) derived from 2,000 cluster draws.

| Setting | Value | Val MAE [95% CI] | Val RMSE [95% CI] | Val R² [95% CI] |
|---|---|---|---|---|
| **Tuned Baseline** | Reference (128-64, drop 0.2, wd 1e-4) | 7.793 [7.146, 8.529] | 11.246 [10.383, 12.248] | 0.343 [0.278, 0.393] |
| **Depth** | 1 layer (128) | 7.823 [7.168, 8.587] | 11.369 [10.468, 12.422] | 0.329 [0.270, 0.376] |
| **Depth** | 2 layers (128, 64) [Tuned] | 7.793 [7.146, 8.529] | 11.246 [10.383, 12.248] | 0.343 [0.278, 0.393] |
| **Depth** | 3 layers (128, 64, 32) | 7.891 [7.258, 8.603] | 11.269 [10.414, 12.258] | 0.340 [0.274, 0.391] |
| **Width** | Half (64, 32) | 7.924 [7.302, 8.633] | 11.295 [10.456, 12.261] | 0.337 [0.272, 0.389] |
| **Width** | Double (256, 128) | 7.835 [7.224, 8.505] | 11.223 [10.444, 12.112] | 0.346 [0.267, 0.406] |
| **Dropout** | 0.0 (Disabled) | 7.672 [7.028, 8.410] | 11.241 [10.362, 12.214] | 0.344 [0.279, 0.397] |
| **Dropout** | 0.2 (Enabled) [Tuned] | 7.793 [7.146, 8.529] | 11.246 [10.383, 12.248] | 0.343 [0.278, 0.393] |
| **Weight Decay** | 0.0 (Disabled) | 7.785 [7.143, 8.505] | 11.243 [10.391, 12.235] | 0.343 [0.277, 0.393] |
| **Weight Decay** | 1e-4 (Enabled) [Tuned] | 7.793 [7.146, 8.529] | 11.246 [10.383, 12.248] | 0.343 [0.278, 0.393] |
| **Station History** | true (Included) [Tuned] | 7.793 [7.146, 8.529] | 11.246 [10.383, 12.248] | 0.343 [0.278, 0.393] |
| **Station History** | false (Excluded) | 8.181 [7.486, 9.039] | 11.889 [10.877, 13.098] | 0.266 [0.213, 0.306] |
| **Country** | true (Included) [Tuned] | 7.793 [7.146, 8.529] | 11.246 [10.383, 12.248] | 0.343 [0.278, 0.393] |
| **Country** | false (Excluded) | 8.237 [7.616, 8.998] | 11.723 [10.862, 12.736] | 0.286 [0.217, 0.338] |
| **Model** | hybrid (PENDING: awaiting results/hybrid_val_preds.csv from Anusha/Priyal) | NA | NA | NA |

*Note: The hybrid model row remains PENDING with NA metrics, awaiting out-of-fold validation predictions from Anusha and Priyal (`results/hybrid_val_preds.csv`). No hybrid numbers are fabricated.*

---

## 4. Empirical Findings

Systematic evaluation across each isolated axis reveals clear structural and domain dynamics:

- **Architectural Depth (1 vs. 2 vs. 3 Layers):** The 2-layer pyramid (128, 64) achieves validation RMSE 11.246 [10.383, 12.248], compared to 11.369 [10.468, 12.422] for 1 layer ($\Delta = -0.123$) and 11.269 [10.414, 12.258] for 3 layers ($\Delta = -0.023$). Although depth 2 achieves the lowest point RMSE, 95% bootstrap confidence intervals overlap substantially across all three depths. Consequently, depths 1, 2, and 3 are **statistically indistinguishable under station-bootstrap CIs**, indicating that depth beyond 2 layers offers no abstraction advantage on this 14-input tabular manifold.
- **Layer Width (Half vs. Tuned vs. Double):** Compressing width to half (64, 32) yields RMSE 11.295 [10.456, 12.261], whereas doubling width to (256, 128) achieves 11.223 [10.444, 12.112] ($\Delta = -0.023$ relative to tuned 11.246). Due to broad CI overlap, width scaling is statistically indistinguishable under station-bootstrap CIs. The tuned baseline (128, 64) occupies the **sweet spot**, maximising parameter efficiency without unnecessary computational overhead.
- **Stochastic Regularisation (Dropout 0.0 vs. 0.2):** Disabling dropout yields RMSE 11.241 [10.362, 12.214] and MAE 7.672 [7.028, 8.410], versus 11.246 and 7.793 with 20% dropout ($\Delta \text{MAE} = -0.121$, $\Delta \text{RMSE} = -0.005$). While point errors are slightly lower without dropout, both settings are statistically indistinguishable under station-bootstrap CIs. In low-dimensional tabular regimes, early stopping regularises adequately, rendering dropout unhelpful.
- **Weight Decay ($0.0$ vs. $10^{-4}$):** Unregularized Adam ($L_2 = 0$) achieves RMSE 11.243 [10.391, 12.235] and MAE 7.785 [7.143, 8.505], essentially identical to $10^{-4}$ weight decay ($\Delta \text{RMSE} = +0.003$, $\Delta \text{MAE} = +0.008$). The difference is statistically indistinguishable under station-bootstrap CIs, showing $L_2$ shrinkage has minimal effect given early stopping.
- **Station History Attribution (True vs. False):** Removing historical past-mean sensor features causes severe degradation: validation RMSE deteriorates from 11.246 to 11.889 [10.877, 13.098] ($\Delta = +0.643$), and $R^2$ collapses from 0.343 to 0.266 [0.213, 0.306] ($\Delta = -0.077$). This directly mirrors Phase 1 XGBoost ablations where removing station history dropped $R^2$ from 0.348 to 0.271, proving that local temporal persistence is a foundational predictor across both tree and neural model families.
- **Country Indicator Attribution (True vs. False):** Omitting regional country indicators increases RMSE from 11.246 to 11.723 [10.862, 12.736] ($\Delta = +0.478$) and degrades $R^2$ from 0.343 to 0.286 [0.217, 0.338] ($\Delta = -0.057$). Macro-geographical indicators provide vital baseline calibration across jurisdictions.
- **Model Formulation (MLP vs. Hybrid):** The standalone tuned MLP achieves $R^2 = 0.343$ [0.278, 0.393] and RMSE 11.246. Comparative evaluation against the hybrid ensemble remains strictly **pending Anusha's out-of-fold predictions** (`results/hybrid_val_preds.csv`).

---

## 5. Limitations & Threats to Validity

1. **Discrete Grid Resolution:** Evaluating isolated settings explores orthogonal axes rather than continuous response surfaces.
2. **Validation Partition Variance:** While the validation split contains 26,054 rows from 1,136 distinct monitoring stations, localized sensor calibration quirks introduce sampling variance.
3. **Tabular Inductive Biases:** Prior literature demonstrates that tree ensembles partition tabular spaces naturally. Model accuracy is primarily bounded by domain feature representation rather than neural capacity.
4. **Hybrid Protocol Dependency:** Hybrid benchmarking requires genuine out-of-fold predictions to prevent optimistic leakage; until Anusha delivers verified predictions, ensembling claims remain unverified.
