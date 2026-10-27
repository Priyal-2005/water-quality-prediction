# Water Quality Prediction (AML Phase 1)

**Project 18 — Water Quality Prediction via Gradient Boosting and Deep Neural Networks**
Course project (Advanced Machine Learning + Deep Learning), team of 5.

> This README covers **Phase 1 (classical ML)**. Phase 2 (deep learning, MLP) will be added later on the same data and split.

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

#### Viva answer: why split by station and not by random rows?

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

## Phase 2 (planned)

A small regularised MLP will be trained on the same cleaned data and the same saved splits (the `split` and `split_time` columns in `data/processed/final_dataset.csv`) and compared fairly with the Phase 1 baselines.
