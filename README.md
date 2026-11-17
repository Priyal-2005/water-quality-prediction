# Water Quality Prediction (AML Phase 1 + DL Phase 2)

**Project 18 — Water Quality Prediction via Gradient Boosting and Deep Neural Networks**
Course project (Advanced Machine Learning + Deep Learning), team of 5.

> This README covers **Phase 1 (classical ML)** and **Phase 2 (deep learning: MLP and a hybrid)**, both on the same data and the same splits. Phase 2 is described in the last section.

## Team

| Name | Role |
|---|---|
| Priyal | Evaluation, theory (evaluation half), GitHub repo and code quality |
| Anusha | Model training, theory (models half) |
| Sarthak | Dataset research, cleaning, feature engineering |
| Kashika | Literature review, report (Intro, Related Work, Conclusion) |
| Vaishnavi | Dataset quality and EDA, report assembly |

## Problem statement

Predict a water quality score from simple measurements (dissolved oxygen, pH and temperature, plus water type, country, season and station history). We compare **XGBoost**, **Random Forest** and Ridge Regression against a mean-predictor baseline, using a fair split by station and by year, and report feature importance with SHAP.

- **Target:** `target_ccme_value`, the CCME Water Quality Index (continuous, 0–100). Regression. See `docs/target_definition.md`
- **Dataset:** Comprehensive Surface Water Quality Monitoring Dataset 1940–2023 (the dataset linked in the brief), CC BY 4.0. We use England, Ireland, USA and Canada: 259,774 rows, 9,940 sites, 2000–2023 after cleaning
- **Split:** by monitoring station (main): train / val / test = 70 / 10 / 20% of rows. Secondary split by year: train up to 2015, val 2016–2017, test 2018+. Both are stored as columns in the data file
- **Metrics:** MAE, RMSE and R², with bootstrap confidence intervals, and a mean-predictor baseline
- **Leakage handling:** the index is calculated from 8 parameters, so models only see dissolved oxygen, pH and temperature (plus water type, country, season and each site's earlier readings). The five main drivers (ammonia, BOD, orthophosphate, nitrogen, nitrate) stay out of the inputs. Giving the model all 8 scores R² 0.99; the chosen set scores about 0.3 (`notebooks/03_features.ipynb`)

## Train / validation / test splits

Both splits are stored as columns in `data/processed/final_dataset.csv`, so every model uses exactly the same rows.

| Split | Column | Train | Validation | Test | Question it answers |
|---|---|---|---|---|---|
| **By station (main)** | `split` | 70% of rows | 10% | 20% | Does the model work at monitoring stations it has never seen? |
| **By year (secondary)** | `split_time` | up to 2015 (71%) | 2016–2017 (8%) | 2018 onwards (20%) | Does the model work on future data? |

- **What is a station?** A station is a fixed monitoring site (`site_id`, for example `Canada|LI110048`) that is sampled many times over the years. One station therefore has many rows. The dataset has 9,940 stations.
- **By station:** all rows of a station go to one set only (no station is shared between sets), so the model cannot memorise a station's usual water quality. Stations are shuffled with seed 42 and dealt out until each share of rows is reached.
- **By year:** train on the older years, tune on 2016–2017, test on 2018 onwards. Years cannot be cut into exact percentages, hence 71 / 8 / 20.
- **How to use them:** train on `train`, tune hyperparameters and stop early on `val`, and look at `test` once at the end. Report the station split as the main result and the year split as the extra check. The Phase 2 neural network must use the same columns.

#### Why split by station and not by random rows?

- Water quality at one station stays similar from sample to sample. With a random row split, the same station would appear in both train and test, and the model could score well just by remembering that a station is usually clean or dirty. That is leakage and would overstate performance.
- With the station split, the test set contains only stations the model has never seen, so the score answers the real question: does it work at a new location?
- Example: a station with 100 rows goes entirely to train, another with 50 rows goes entirely to test. We count rows, not stations, so the shares come out at exactly 70 / 10 / 20% of rows.
- The year split is a second, separate check for future data. In it the same stations can appear in several sets, because it only tests time.
- Site-history features (`site_prev_n`, running mean of a station's earlier readings) use only earlier rows of the same station, so they cannot see the future or other sets' rows.
- We checked it: zero stations are shared between train and test (see `01_eda.ipynb` section B7).


```python
from src.features import load_final_dataset, make_features, get_xy
df = make_features(load_final_dataset())
X_train, y_train, X_val, y_val, X_test, y_test = get_xy(df, "split")        # main
Xt_train, yt_train, Xt_val, yt_val, Xt_test, yt_test = get_xy(df, "split_time")  # secondary
```

## Repository structure

```text
data/
  raw/            original data, never edited, not committed (see Dataset)
  processed/      final_dataset.csv (the single processed file, committed)
notebooks/        01_eda, 02_cleaning, 03_features, 04_models, 05_evaluation
src/
  data.py         loading and cleaning helpers
  features.py     feature pipeline (fitted on train only)
  train.py        model training and tuning
models/           saved models (not committed, regenerate with train.py)
results/          generated by code, do not edit by hand
  validation_metrics.csv      best setting of each model, scored on validation
  tuning_results.csv          every setting tried (selected = True for each model's winner)
  training_config.json        features, seed and chosen hyperparameters
  validation_predictions.csv  validation predictions per model (row_id = row in final_dataset.csv)
  final_results.csv           final test-set table with confidence intervals (from 05_evaluation)
  year_split_results.csv      test results on the year split
  figures/                    all plots used in the report
docs/             theory.md, cleaning_log.md, target_definition.md, dataset_comparison.md, literature_review.md
report/           LaTeX report (main.tex, references.bib)
```

## Setup

```bash
git clone https://github.com/Priyal-2005/water-quality-prediction.git
cd water-quality-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

**Source:** [A Comprehensive Surface Water Quality Monitoring Dataset 1940–2023 (figshare)](https://doi.org/10.6084/m9.figshare.27800394), paper in *Scientific Data*.

**You do not need the raw file to work on the models:** `data/processed/final_dataset.csv` is committed and is the only data file you need. Load it with `src/features.py` (`load_final_dataset`, `make_features`, `get_xy`).

**To rebuild it from scratch:** download `Combined_dataset.csv` from the link above, place it in `data/raw/` (not committed, about 320 MB) and run `notebooks/02_cleaning.ipynb`. All cleaning logic is in `src/data.py` (`build_final_dataset`), every decision is logged in `docs/cleaning_log.md`.

## How to reproduce the results

Run from the repo root, in this order:

```bash
# 1. (Only if rebuilding) clean the data and create the splits
#    run notebooks/02_cleaning.ipynb -> data/processed/final_dataset.csv
#    (already committed, so you can skip this step)

# 2. Build features, tune on validation and train all models
#    (about 20 minutes; test set is not used here)
python -m src.train

# 3. Evaluate on the test set (writes results/final_results.csv and results/figures/)
#    run notebooks/05_evaluation.ipynb
```

`train.py` takes no arguments. All random seeds are fixed to `42`.

**Tuning:** each model is tuned on the validation set only, changing one setting at a time (see the grids in `src/train.py`). The setting with the lowest validation RMSE is kept. Every trial is saved in `results/tuning_results.csv`.

## Results

**Validation results (station split).** These are scores on the validation set, used to choose settings.

| Model | Selected setting | MAE | RMSE | R² |
|---|---|---|---|---|
| Dummy baseline | predicts the training mean | 9.76 | 13.91 | -0.005 |
| Ridge | alpha = 0.001 | 8.69 | 12.26 | 0.219 |
| Random Forest | 300 trees, max depth 16, min leaf 5 | 7.77 | 11.25 | 0.343 |
| XGBoost | 300 trees, max depth 8, learning rate 0.03 | 7.73 | 11.20 | 0.348 |

- Both tree models clearly beat Ridge and the dummy. XGBoost and Random Forest differ by only about 0.04 RMSE on one validation split, so treat them as close.
- Ridge's alpha, Random Forest's depth and XGBoost's learning rate each ended at the edge of their tested range, so a wider sweep might improve them slightly.

**Test results (station split, 51,973 rows from 2,221 unseen stations).** Scores on the test set, used once at the end. Source: `results/final_results.csv`; confidence intervals come from 2,000 bootstrap draws of whole stations.

| Model | MAE | RMSE | R² | 95% CI for R² |
|---|---|---|---|---|
| Dummy baseline | 9.57 | 13.16 | -0.001 | [-0.009, -0.000] |
| Ridge | 8.33 | 11.73 | 0.205 | [0.174, 0.233] |
| Random Forest | 7.60 | 11.18 | 0.278 | [0.230, 0.324] |
| **XGBoost** | **7.54** | **11.06** | **0.293** | [0.246, 0.340] |

**Year split (train up to 2015, test 2018 onwards).** XGBoost R² 0.243 (MAE 5.45, RMSE 8.05), Random Forest 0.192, Ridge 0.097, baseline −0.338. These numbers use different test data, so do not compare them directly with the table above. Source: `results/year_split_results.csv`.

- XGBoost is best, but its lead over Random Forest is small: its test RMSE is lower by 0.121 (95% interval 0.059 to 0.186, from the same re-drawn stations).
- R² below 0.30 means most of the variation in the score is still unexplained, as expected with the five main measurements hidden on purpose.

## Contributions

| Member | Main files |
|---|---|
| Priyal | `notebooks/05_evaluation.ipynb`, README, repo setup |
| Anusha | `src/train.py`, `docs/theory.md` |
| Sarthak | `src/data.py`, `src/features.py`, `docs/cleaning_log.md` |
| Kashika | `docs/literature_review.md`, report |
| Vaishnavi | `notebooks/01_eda.ipynb`, report |

## Limitations

- Test R² is moderate (0.293 on unseen stations), and polluted water is predicted to be better than it is (about 31 points too high when the true score is below 60).
- Part of the skill comes from knowing the country and the station's past readings, which may not carry over to a new region.
- England has most of the data; the USA and Canada have very little.
- The source data had gaps filled with repeated numbers; we marked them as missing, but the target index was calculated after the filling.
- Settings were chosen on one validation split without cross-validation.

## Phase 2: deep learning (MLP and hybrid)

Phase 2 trains a small regularised neural network (MLP) on the same cleaned data, the same inputs and the same station and year splits as Phase 1, then compares it fairly with the Phase 1 XGBoost and combines the two into a hybrid.

**The network.** A PyTorch MLP, 30 inputs → 128 → 64 → 1, ReLU, dropout 0.2, weight decay 1e-4, Adam (learning rate 0.001, batch size 512), mean squared error loss and early stopping (patience 10). It has 12,289 parameters. 13 settings were tried (the default plus 12 random trials) and the one with the lowest *validation* RMSE was kept; the default won. Details: `docs/architecture_dl.md`, `docs/regularisation_justification.md`, `docs/theory_dl.md`.

**The hybrid.** A weighted blend `w · XGBoost + (1 − w) · MLP`, clipped to 0–100. The weight (0.55 XGBoost, 0.45 MLP) was chosen on validation only, using Phase 1's saved XGBoost predictions and the saved MLP predictions. A residual-learning hybrid was also tried and rejected (its corrections had almost no correlation with the real residuals). Out-of-fold predictions are only needed for that residual variant, not for the blend.

**Ablations.** Depth, width, dropout, weight decay, station history and country were each changed one at a time from the tuned MLP (`docs/ablation_section.md`, `results/ablations.csv`). Depth, width, dropout and weight decay made no clear difference. Removing station history (validation RMSE 11.878 vs 11.231) or country (11.733) hurt most, although the separately computed intervals still overlap.

### Phase 2 results

**Station split, test set (51,973 rows, unseen stations), scored once.** Source: `results/dl/metrics_dl.csv`; confidence intervals and paired differences are in `results/dl/station_bootstrap_ci.csv` and `results/dl/paired_rmse_differences.csv` (2,000 bootstrap draws of whole stations, the same stations for every model in a draw).

| Model | MAE | RMSE | R² | 95% CI for RMSE |
|---|---|---|---|---|
| XGBoost (Phase 1) | 7.543 | 11.058 | 0.293 | [10.309, 11.819] |
| MLP | 7.606 | 11.108 | 0.287 | [10.366, 11.863] |
| **Hybrid (blend)** | **7.521** | **11.010** | **0.300** | [10.263, 11.772] |

| Paired RMSE difference | Value | 95% CI | CI excludes 0? |
|---|---|---|---|
| MLP minus XGBoost | +0.049 | [-0.053, 0.147] | No |
| Hybrid minus XGBoost | -0.049 | [-0.098, -0.002] | Yes, barely |
| Hybrid minus MLP | -0.098 | [-0.150, -0.043] | Yes |

- The MLP and XGBoost cannot be told apart on unseen stations.
- The hybrid is best, but its gain over XGBoost is small (about 0.05 RMSE, 0.4%) and only just excludes zero. We describe it as a small gain.
- The hybrid beats the MLP clearly.
- R² stays near 0.3 for every model, so the limit is the information in the inputs, not the model type.

**Year split (secondary check, test 2018 onwards).** Hybrid RMSE 7.842, MLP 7.866, XGBoost 8.061 (R² 0.281, 0.276, 0.240). One split, no confidence intervals, so this is supporting evidence only. The XGBoost here was retrained in notebook 08, so it differs slightly from the Phase 1 line above.

**Where all models fail.** On polluted water (true score below 60, about 6.7% of validation rows) all three models over-predict by roughly 29 points. The blend does not fix this.

### How to reproduce Phase 2

Run from the repo root, after the Phase 1 steps above. Seeds are fixed to `42`.

```bash
# 1. Train and tune the MLP on validation (test set is not scored here)
python -m src.train_dl                 # writes results/dl/ and models/mlp.pt
#    quick check: python -m src.train_dl --trials 4 --epochs 20

# 2. Ablations (one factor at a time) and their figures
python -m src.ablations --vary depth --values 1,2,3      # see the docstring for the other factors
python -m src.make_ablation_figures

# 3. Hybrid: blend Phase 1 XGBoost with the MLP, weight chosen on validation
python -m src.hybrid                   # add --oof for the residual-variant diagnostic
#    or run notebooks/07_hybrid.ipynb

# 4. Final evaluation: test set scored once, with confidence intervals
#    run notebooks/08_dl_evaluation.ipynb
```

Notebooks 06 (MLP), 07 (hybrid) and 08 (evaluation) read the saved predictions, so they run in a minute or two without retraining. Their optional retraining sections are switched off by default.

### Phase 2 files

| Part | Files |
|---|---|
| MLP | `src/train_dl.py`, `src/dl_data.py`, `src/regularisation.py`, `notebooks/06_mlp.ipynb`, `docs/architecture_dl.md`, `models/mlp.pt` |
| Hybrid and evaluation | `src/hybrid.py`, `notebooks/07_hybrid.ipynb`, `notebooks/08_dl_evaluation.ipynb`, `docs/theory_dl.md` |
| Ablations | `src/ablations.py`, `src/make_ablation_figures.py`, `configs/tuned_mlp.yaml`, `docs/ablation_section.md` |
| Results | `results/dl/`, `results/hybrid_*.csv`, `results/predictions_hybrid.csv`, `results/ablations.csv` |

### Phase 2 limitations

- The hybrid's gain over XGBoost is small and only just statistically clear; we do not claim a large improvement.
- 13 tuning trials is a small search, and the tuned setting is the default.
- Ablation intervals are computed separately and overlap; a paired bootstrap would be the right test to confirm the station-history and country effects.
- The year-split result is a single split without confidence intervals.
- Over-prediction on polluted water remains for every model.
