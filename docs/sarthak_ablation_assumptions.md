# Sarthak's Ablation Studies: Interface Assumptions & Repository Status Log

**Author:** Sarthak Mathapati (Role: Ablation Studies, Phase 2 Deep Learning)  
**Date:** November 2026  
**Context:** Team T109, Project 18 — Water Quality Prediction (AML + DL)  
**Notice:** This document logs all interface assumptions, identified deliverables, confirmed function signatures, and missing teammate dependencies. No teammate deliverables are fabricated.

---

## 1. Executive Summary of Repository State (Phase 2 Inspection)

During repository inspection (Step 0), the existing codebase was examined to identify data pipelines, splitting schemes, input features, target variables, model implementations, and hyperparameters:

| Item | Status in Repository | Source / Reference File | Notes & Assumptions |
|---|---|---|---|
| **Data Split Scheme** | Available & Verified | `data/processed/final_dataset.csv`, `src/features.py`, `src/dl_data.py` | `split` (by station: 70% train, 10% val, 20% test; strictly zero station overlap; validation partition contains 26,054 rows from 1,136 distinct monitoring stations) and `split_time` (by year). Station ID column is `site_id`. |
| **Input Features (14 inputs)** | Available & Verified | `src/features.py` (`FEATURE_COLS`) | 3 sensor features (`dissolved_oxygen_mg_l`, `ph`, `temperature_c`), 4 engineered (`month_sin`, `month_cos`, `do_sat_pct`, `do_x_temp`), 5 station-history (`*_site_prev_mean`, `site_prev_n`, `days_since_prev`), and 2 categorical (`waterbody_type`, `country`). Preprocessed to 30 neural inputs. Explicitly asserted `waterbody_type in df.columns`. |
| **Hidden Chemicals (5 parameters)** | Available & Verified | `src/features.py` (`HELD_OUT_COLS`) | `ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`. Strictly kept out of model inputs. |
| **Target Variable** | Available & Verified | `src/features.py` (`TARGET_COL`) | `target_ccme_value` (continuous regression, 0–100 scale). |
| **Station ID Column** | Available & Verified | `src/features.py`, `src/dl_data.py` | `site_id` (e.g. `Canada|LI110048`, `Ireland|IE_...`). Used for GroupKFold and station-level bootstrap. |
| **Random Seed** | Standardized | `src/reproducibility.py` | `SEED = 42`. Used for initialization, loaders, and bootstrapping. |
| **Priyal's Tuned Config (`configs/tuned_mlp.yaml`)** | **Missing** | `configs/` directory was absent | `results/dl/mlp_config.json` exists from Priyal's initial exploration in `train_dl.py`, but no formal YAML config was checked in. |
| **Anusha / Priyal Hybrid Predictions (`results/hybrid_val_preds.csv`)** | **Missing** | `results/` | `src/hybrid.py` is a 5-line stub; no out-of-fold-by-station validation predictions exist yet. |
| **XGBoost Val Metrics** | Found in `results/validation_metrics.csv` | `results/validation_metrics.csv` | Read dynamically by `src/make_ablation_figures.py` (Dummy: 9.762, Ridge: 8.688, RF: 7.773, XGBoost: 7.729). Standalone `results/xgboost_val_metrics.csv` was not created. |

---

## 2. Documented Assumptions & Verified Implementations

To adhere strictly to project rules (no fabricated results, no modification of teammate files, and append-only timestamped logging), the following protocols are enforced:

### Assumption 1: Handling Missing `configs/tuned_mlp.yaml`
- **Rule:** Never treat `configs/ablation_baseline.yaml` as Priyal's tuned configuration. Never train it and report it as the tuned baseline.
- **Action:** Created `configs/ablation_baseline.yaml` marked prominently with status `PLACEHOLDER` and a notice: *"Replace with Priyal’s tuned config when available. Do not report as tuned baseline."*
- **Ablation CLI:** `src/ablations.py` defaults to searching for `configs/tuned_mlp.yaml`. If missing, it accepts `--config configs/ablation_baseline.yaml` strictly for dry-run validation (`--dry-run`) or scaffold verification, producing rows tagged `status=PLACEHOLDER` or `status=DRY_RUN`.

### Assumption 2: Confirmed Signatures of Reproducibility & Model Loaders
- Verified signature in `src/reproducibility.py`:
  ```python
  def make_train_loader(data, batch_size=256, seed=SEED):
  def set_seed(seed=SEED):
  ```
  `make_train_loader` accepts a dictionary containing `X_train` and `y_train` tensors/arrays, batch size, and seed.
- In `src/ablations.py`, imports are placed at module top inside a robust `try/except` block with informative error diagnostics (`TORCH_IMPORT_ERROR`), allowing `--help`, `--dry-run`, data checks, and placeholder generation to operate cleanly even if PyTorch is not yet installed in a lightweight audit environment.

### Assumption 3: Station-Level Bootstrap Implementation (`station_bootstrap_ci`)
- Fully implemented in both `src/ablations.py` and `src/make_ablation_figures.py`:
  ```python
  def station_bootstrap_ci(df, group_col="site_id", y_true_col="y_true", y_pred_col="y_pred", metric="mae", n_boot=2000, seed=42, alpha=0.05):
  ```
- **Algorithm:** Identifies all unique monitoring stations (`site_id`), pre-groups observation arrays by station, and resamples all 1,136 distinct monitoring stations across the 26,054 rows in the validation set with replacement over 2,000 bootstrap draws. Computes two-sided 95% percentile confidence intervals (`alpha=0.05`).
- Accounting for clustering by station prevents artificially deflated confidence intervals caused by treating serially and spatially correlated observations from the same river site as independent.

### Assumption 4: Hybrid Model Comparison Protocol & 95% CIs
- Hybrid rows must ONLY use genuine out-of-fold-by-station validation predictions produced by Anusha or Priyal (`results/hybrid_val_preds.csv`).
- The predictions must align on `row_id`, `site_id`, `target_ccme_value`, and the validation split.
- `evaluate_hybrid_validation()` verifies alignment, merges station IDs (`site_id`), and computes 95% station bootstrap CIs for MAE, RMSE, and $R^2$.
- Because `results/hybrid_val_preds.csv` is currently missing, `src/ablations.py` writes a `PLACEHOLDER` row with `NA` metrics and note: *"Placeholder: out-of-fold hybrid validation predictions not yet generated by Anusha/Priyal."*
- Centralized ledger (`results/ablations.csv`) records confidence intervals via columns:
  `run_id,timestamp,status,setting,value,val_MAE,val_MAE_lo,val_MAE_hi,val_RMSE,val_RMSE_lo,val_RMSE_hi,val_R2,val_R2_lo,val_R2_hi,epochs_run,seed,config_hash,notes`.

### Assumption 5: Dynamic Phase 1 Metric Ingestion
- `src/make_ablation_figures.py` dynamically ingests Phase 1 baseline metrics from `results/validation_metrics.csv` instead of hardcoding values.
- If required Phase 1 models (`DummyRegressor`, `Ridge`, `RandomForestRegressor`, `XGBRegressor`) are absent, it raises a clear error and falls back to a watermarked placeholder figure.
- If `results/validation_predictions.csv` is present, it computes genuine station-level bootstrap CIs on Phase 1 validation predictions.

### Assumption 6: Architectural Parameter Mapping (Pyramid & Multiplier Scheme)
- In `src/train_dl.py`, architecture is configured via `widths: (w1, w2, ...)`.
- From Priyal's committed tuned config (`configs/tuned_mlp.yaml`), `hidden_layers: [128, 64]` is loaded, defining base layer width `base = 128`.
- For depth ablations: geometric pyramid derived from base width:
  `trial_cfg["widths"] = tuple(max(8, base // (2 ** i)) for i in range(depth_val))`
  - Depth 1: `(128,)`
  - Depth 2: `(128, 64)` [Tuned baseline]
  - Depth 3: `(128, 64, 32)`
- For width ablations: multiplier derived from tuned widths `[128, 64]`:
  `trial_cfg["widths"] = tuple(max(8, round(w * factor)) for w in tuned_widths)`
  - Width half: `(64, 32)`
  - Width double: `(256, 128)`

### Assumption 7: Feature Ablation Isolation
- When ablating **station history** (`use_station_history: false`), only the 5 station history features are excluded:
  `dissolved_oxygen_mg_l_site_prev_mean`, `ph_site_prev_mean`, `temperature_c_site_prev_mean`, `site_prev_n`, and `days_since_prev`.
- When ablating **country** (`use_country: false`), only the categorical column `country` is excluded.
- Preprocessing (median imputer + missing-indicator columns + standard scaler + one-hot encoder) and the station-based split (`split`) remain strictly identical to preserve experimental isolation.

### Assumption 8: Handoff Protocol
- Sarthak’s owned deliverable is the **Ablation Studies** module.
- The finalized markdown documentation in `docs/ablation_section.md` is formatted in Sarthak's voice and is designated to be handed off to **Vaishnavi** for inclusion in the consolidated IEEE conference report.
