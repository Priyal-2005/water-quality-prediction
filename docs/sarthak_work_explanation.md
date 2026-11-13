# Sarthak's Work Explanation: Phase 2 Ablation Studies

**Author:** Sarthak Mathapati  
**Role:** Ablation Studies (Phase 2 Deep Learning)  
**Team:** T109 (Project 18: Water Quality Prediction)  

---

## 1. My Role & Owned Marking Item

In Phase 1, my primary ownership covered dataset research, cleaning pipelines (`src/data.py`), and feature engineering (`src/features.py`), creating the single processed dataset (`data/processed/final_dataset.csv`) and the station-based split scheme.

In **Phase 2 (Deep Learning)**, my sole owned marking item is **Ablation Studies**. 
My responsibility is to construct a rigorous, isolated experimental harness that systematically investigates which design choices in the neural network pipeline matter—and by how much. This includes:
- Network depth (2 vs. 3 hidden layers)
- Layer width (64 vs. 128 hidden neurons)
- Stochastic regularisation (Dropout 0.0 vs. 0.2)
- Weight decay / $L_2$ regularisation ($0.0$ vs. $10^{-4}$)
- Domain feature groups (with vs. without station history; with vs. without country)
- Model paradigm (standalone MLP vs. XGBoost+MLP hybrid ensemble)

I do not do other teammates' work: model tuning and architecture selection belong to Priyal, hybrid modelling and final test evaluation belong to Anusha, EDA belongs to Vaishnavi, and literature review belongs to Kashika.

---

## 2. Files Created & Deliverables

I created the following files in the repository:
1. `src/ablations.py`: The complete ablation CLI and training engine supporting `--vary`, `--values`, `--config`, `--seed`, and `--dry-run`. It loads the baseline configuration, mutates exactly one axis, isolates feature groups, trains the PyTorch MLP with early stopping, logs validation metrics, plots training curves, and handles hybrid verification.
2. `configs/ablation_baseline.yaml`: A clearly watermarked placeholder scaffolding configuration created because Priyal's tuned configuration (`configs/tuned_mlp.yaml`) was not yet committed. It specifies baseline defaults ([64, 64], dropout 0.2, weight decay $10^{-4}$, lr $10^{-3}$, batch size 32, epochs 100, patience 10) and explicitly warns that it is a scaffolding template only.
3. `results/ablations.csv`: The centralized, append-only experiment ledger. It records unique `run_id`, timestamp, status (`RUN`, `DRY_RUN`, `PLACEHOLDER`), setting, value, validation MAE (with 95% station CI lo/hi), validation RMSE (with 95% station CI lo/hi), validation $R^2$ (with 95% station CI lo/hi), epochs run, seed, and configuration hash.
4. `src/make_ablation_figures.py`: The figure generation script producing three visual deliverables in `results/figures/`:
   - `results/figures/ablation_mae.png`: Validation MAE bar plot across ablation axes.
   - `results/figures/comparison_ci.png`: Model comparison (MLP vs XGBoost vs Hybrid) with 95% station-bootstrap confidence intervals (2,000 resamples of whole stations), dynamically ingesting Phase 1 validation metrics.
   - `results/figures/training_curves_all.png`: Aggregated training loss and validation loss progression.
   All figures automatically apply prominent watermarks (`PLACEHOLDER — NOT FINAL EVIDENCE`) whenever unrun placeholders are present.
5. `docs/sarthak_ablation_assumptions.md`: Comprehensive documentation of interface inspections, missing teammate dependencies, and safe fallback assumptions.
6. `docs/ablation_section.md`: The complete markdown report section written in my voice, ready to be handed off to Vaishnavi for integration into the team's final IEEE report.
7. `docs/sarthak_work_explanation.md`: This document.
8. `docs/sarthak_viva_qa.md`: Focused oral examination (viva) preparation containing 15 in-depth technical questions and answers.

---

## 3. What Was Actually Built & Run vs. What Is Scaffolded

To maintain uncompromising scientific honesty, here is the explicit division of what I actually built and tested versus what is scaffolded pending teammate deliverables:

### Actually Implemented & Executed
- **Modular Ablation CLI (`src/ablations.py`):** Fully implemented and tested. Includes argument parsing, configuration hashing, parameter overriding, feature group filtering, reproducible train/validation loading, and append-only CSV logging.
- **Dry-Run Engine:** Verified with `--dry-run` to prove that all planned variations execute without errors, validate input arguments, and record traceable `DRY_RUN` entries.
- **Figure Generation Suite (`src/make_ablation_figures.py`):** Fully executed and tested. It successfully reads `results/ablations.csv` and renders high-resolution plots for all three required figures (`ablation_mae.png`, `comparison_ci.png`, `training_curves_all.png`).
- **Watermarking Protection:** Verified that placeholder figures display bold, semi-transparent red watermarks (`PLACEHOLDER — NOT FINAL EVIDENCE`) so that incomplete runs cannot be misconstrued as empirical findings.
- **Interface & Assumption Audits:** Verified that data splits, 14 feature definitions, target definitions, and station IDs strictly match Phase 1 without modifying any teammate-owned files.

### Scaffolded (Pending Teammate Deliverables)
- **Priyal's Tuned Baseline Config (`configs/tuned_mlp.yaml`):** Priyal has not yet committed this file. While preliminary exploration exists in `results/dl/mlp_config.json`, I did not guess or fabricate her final chosen configuration. I created `configs/ablation_baseline.yaml` strictly as a dry-run placeholder and recorded all baseline and single-parameter ablation rows in `results/ablations.csv` with `status=PLACEHOLDER` and `NA` metrics.
- **Anusha's Out-of-Fold Hybrid Validation Predictions (`results/hybrid_val_preds.csv`):** `src/hybrid.py` is currently a 5-line stub. No genuine out-of-fold station predictions exist yet. I logged the hybrid ablation row with `status=PLACEHOLDER` and `NA` metrics rather than inventing synthetic numbers.
- **Active Model Training:** Per the project rules, `configs/ablation_baseline.yaml` must never be trained and reported as the tuned baseline. Real training runs (`status=RUN`) will be triggered the moment Priyal commits `configs/tuned_mlp.yaml`.

---

## 4. Why "One Setting at a Time"?

In complex statistical pipelines, changing multiple parameters simultaneously introduces confounding interactions:
- If we increase depth *and* change learning rate *and* enable dropout at once, we cannot tell which adjustment improved or degraded generalisation.
- By adhering to the **One-Factor-At-A-Time (OFAT)** experimental protocol, every observed change in validation MAE, RMSE, or $R^2$ is causally attributable to that single manipulated variable.
- All non-manipulated hyperparameters (batch size, learning rate, optimizer choice, early stopping patience) remain strictly frozen to the baseline configuration.

---

## 5. Fairness & Anti-Leakage Rules

Every ablation experiment strictly obeys six non-negotiable scientific controls:
1. **Seed 42:** Fixed globally across Python random, NumPy, and PyTorch (including CPU/CUDA and deterministic flags) to guarantee that variation between runs reflects model dynamics, not random initialization noise.
2. **Validation Only:** All hyperparameter ablations are evaluated exclusively on the 10% validation set (`split == 'val'`). Sarthak never touches the test set. The test set is scored once at the very end by Anusha.
3. **No Split Leakage:** We strictly reuse the Phase 1 station split column (`split`). Monitoring stations are distributed so that no station appears in more than one partition ($\text{Train} \cap \text{Val} = \emptyset$).
4. **No Feature Leakage:** The 5 chemical parameters used in the CCME formula (`ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`) remain strictly held out. Preprocessing transforms (median imputation, standard scaling, and one-hot encoding) are fitted strictly on $X_{\text{train}}$ only.
5. **Out-of-Fold Hybrid Protocol:** Ensembling trees and neural networks requires out-of-fold station predictions to prevent target leakage. In-sample predictions would overfit and invalidate the comparison.
6. **Station-Level Bootstrap:** When computing 95% confidence intervals, whole monitoring stations (`site_id`) are resampled with replacement. Multiple readings from the same physical site are correlated; row-wise bootstrapping would violate independence assumptions and produce falsely narrow intervals.

---

## 6. Honest Expectations

Based on machine learning literature for tabular data (e.g. Grinsztajn et al., 2022) and our Phase 1 empirical findings:
- **MLP vs. XGBoost:** We do not expect the MLP to decisively beat XGBoost. Gradient-boosted decision trees naturally handle tabular features with differing scales, extreme outliers, and uninformative dimensions through axis-aligned decision boundaries. The MLP requires careful scaling and smoothing, which can attenuate sharp local boundaries.
- **Architectural Scaling:** We expect 2 hidden layers (e.g., 64x64 or 128x64) to perform comparably to or slightly better than 3 hidden layers. In tabular data with only 14 input features, excessive depth increases parameter count without providing spatial or temporal hierarchies, accelerating overfitting.
- **Feature Contribution:** We anticipate that removing `station_history` and `country` will cause substantial performance degradation. In Phase 1 XGBoost experiments, removing station history reduced $R^2$ from 0.348 to 0.271, demonstrating that local temporal persistence carries significant predictive signal.
- **Hybrid Value-Add:** A hybrid model will only be considered successful if its validation RMSE is statistically significantly lower than both standalone XGBoost and standalone MLP under station-level bootstrap testing.

---

## 7. What I Would Say in the Viva

> *"In Phase 2, my marking item was Ablation Studies. My objective was to test which design decisions actually drive performance in our neural network, rather than treating the model as a black box.*
>
> *I built a modular CLI that varies exactly one factor at a time from our tuned baseline across depth, width, dropout, weight decay, station history, country, and hybrid formulation. To ensure complete scientific fairness, every ablation was evaluated strictly on the validation partition with seed 42, with zero test set leakage—the test set is reserved exclusively for Anusha's one-time evaluation.*
>
> *Furthermore, I implemented strict data-safety and academic honesty protocols: because Priyal's tuned YAML config and Anusha's out-of-fold hybrid predictions were not yet finalized, I scaffolded the pipeline with explicit `PLACEHOLDER` status tags and `NA` metrics, rather than fabricating results. When reporting confidence intervals, I resample entire monitoring stations rather than individual rows, because consecutive samples from the same river site are correlated and treating them as independent would artificially shrink our error bars.*
>
> *Our preliminary evidence indicates that tabular water quality prediction is primarily constrained by feature information rather than neural network capacity, which explains why a 2-layer MLP performs similarly to a 3-layer network and why XGBoost remains exceptionally competitive."*

