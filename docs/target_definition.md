# Target definition and leakage handling

- **Dataset:** Comprehensive Surface Water Quality Monitoring Dataset 1940–2023 (the dataset linked in the project brief), file `Combined_dataset.csv`
- **Processed file:** `data/processed/final_dataset.csv` (259,774 rows, 9,940 sites, 2000–2023; England, Ireland, USA, Canada)
- **Task:** regression
- **Target column:** `target_ccme_value` (CCME Water Quality Index, 0–100, higher = better water)
- **Metrics:** MAE, RMSE and R² (as the brief asks for a continuous target)
- **Main split:** by monitoring station, `split` column: train / val / test = 70% / 10% / 20% of rows (stations shuffled with seed 42 and dealt out until each share is reached; no station is in two sets). Validation is for tuning and early stopping; the test set is used once
- **Secondary split:** by year, `split_time` column: train up to 2015 (71%), validation 2016–2017 (8%), test 2018 onwards (20%); years cannot be cut to exact percentages

## 1. The target

The dataset authors computed the Canadian Council of Ministers of the Environment (CCME) index from 8 parameters: ammonia, BOD, dissolved oxygen, orthophosphate, pH, temperature, nitrogen and nitrate. We predict the numeric index (`target_ccme_value`). The class label (`target_ccme_class`) is made from the same number, so it is never used as an input, and we do not classify (the brief asks for macro F1 and calibration for classes, and the bad classes are rare).

| Statistic | Value |
|---|---|
| Rows | 259,774 |
| Mean / std | 90.2 / 12.9 |
| Median | 93.1 |
| Share exactly 100 | 43.7% (England 29%, Ireland 82%, USA 46%, Canada 35%) |
| Classes | Excellent 43.7%, Good 39.9%, Fair 9.6%, Marginal 6.1%, Poor 0.8% |

Because 44% of the target sits at the ceiling of 100, R² and RMSE are reported next to MAE and per country.

## 2. Why the models cannot see all 8 parameters (leakage)

The brief says: *do not derive a target index from the same components without acknowledging leakage.* Our test (`notebooks/03_features.ipynb`, section 2; trained on train rows, scored on the validation rows of the station split, so the test set stays untouched):

| Inputs given to the model | R² |
|---|---|
| All 8 parameters | 0.991 |
| Original proposal (pH, temperature, ammonia, orthophosphate, nitrogen + type) | 0.982 |
| Dissolved oxygen, pH, temperature only | 0.127 |
| + context (type, country, month) | 0.268 |
| + engineered features and site history **(chosen)** | 0.343 |
| Mean-predictor baseline | -0.005 |

An R² of 0.98 means the model re-learned the formula, not that it predicts well. Each parameter alone: orthophosphate 0.935, ammonia 0.367, nitrogen 0.285, BOD 0.179, nitrate 0.175, but dissolved oxygen 0.040, temperature 0.039 and pH 0.013.

## 3. Column roles

| Columns | Role |
|---|---|
| `dissolved_oxygen_mg_l`, `ph`, `temperature_c` | **Model inputs** (the sensors the brief names; they explain very little of the index alone) |
| `ammonia_mg_l`, `bod_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l` | **Held out of the models**; kept in the file for EDA only |
| `waterbody_type`, `country`, `month` | Context inputs |
| Engineered: month sin/cos, DO saturation %, DO × temperature, running mean of the site's earlier sensor readings, number of earlier samples, days since the previous sample | Inputs (built in `src/features.py`) |
| `site_id`, `date` | Grouping and ordering only, never inputs |
| `target_ccme_value`, `target_ccme_class`, `n_genuine_params`, `split`, `split_time` | Never inputs |

## 4. Statement for the report

> To avoid direct leakage from the CCME index formula, the five parameters that drive it most (ammonia, BOD, orthophosphate, total nitrogen and nitrate) are excluded from the model inputs. Models predict the index from dissolved oxygen, pH and temperature, the sensor readings named in the project brief, plus water type, country, season and each site's earlier readings. Dissolved oxygen, pH and temperature are still ingredients of the index, so some dependence remains; each explains very little of it on its own (R² ≤ 0.05), and the reduction in score from 0.99 to about 0.3 shows the formula is no longer recoverable.

## 5. Other things to state honestly

- **Filled-in source values.** The dataset authors filled gaps with one repeated value per country and water type (for example dissolved oxygen 10.2 in England). We set these to NaN (details in `docs/cleaning_log.md`). The authors computed the index after filling, so part of the target's variation comes from those fillers.
- **Modest scores.** About 0.3 R² is the honest ceiling with three sensors.
- **Time split.** The three sensors alone do not carry over to later years (validation R² about −0.04); the chosen set does (about 0.34). We report both splits.
