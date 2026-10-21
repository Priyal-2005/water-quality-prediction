
# Model Training and Theoretical Background

## 1. Problem Formulation

This project treats water-quality prediction as a supervised regression problem.

- **Target:** `target_ccme_value`
- **Target meaning:** CCME Water Quality Index (WQI), ranging from 0 to 100, with higher values indicating better water quality.
- **Primary split:** Station-based training, validation, and test sets.
- **Evaluation metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and coefficient of determination (R²).

The primary objective is to predict the continuous CCME WQI value from sensor readings, contextual variables, engineered features, and historical sensor information.

## 2. Feature Selection and Leakage Prevention

The CCME WQI target is calculated using multiple water-quality parameters. Including those same constituent parameters as predictors could allow a model to reconstruct the target formula instead of learning a useful predictive relationship.

The model therefore uses three sensor readings:

- Dissolved oxygen (`dissolved_oxygen_mg_l`)
- pH (`ph`)
- Temperature (`temperature_c`)

Additional features include:

- Seasonal features: `month_sin` and `month_cos`
- Dissolved oxygen saturation percentage: `do_sat_pct`
- Dissolved oxygen–temperature interaction: `do_x_temp`
- Historical sensor means for the station
- Number of earlier sampling dates: `site_prev_n`
- Days since the previous sampling date: `days_since_prev`
- Context: `waterbody_type` and `country`

The held-out constituent parameters are ammonia, BOD, orthophosphate, nitrogen, and nitrate. The target, target class, station identifier, date, split labels, and `n_genuine_params` are not model inputs.

### Historical Features

Historical features are calculated from strictly earlier dates at the same station. All observations sharing a station and date receive identical history features. This prevents one observation from the current date from becoming historical input for another observation from that same date.

## 3. Preprocessing

The preprocessing pipeline is fitted as part of each model pipeline, using training data only.

For numerical features:

1. Missing values are imputed using the training-set median.
2. Missing-value indicators are added.
3. Standardisation is applied to Ridge Regression, but not required for the tree-based models.

For categorical features, one-hot encoding is applied with unknown categories ignored during transformation.

Keeping preprocessing inside the pipeline ensures that validation information does not determine the fitted imputation values, scaling parameters, or category vocabulary.

## 4. Baseline Models

### 4.1 Dummy Regressor

The Dummy Regressor predicts the mean target value observed in the training data for every input.

It establishes a simple reference point. A useful predictive model should improve upon this baseline on validation data.

### 4.2 Ridge Regression

Ridge Regression is a linear regression model with L2 regularisation. Its objective is to minimise the squared prediction error while penalising large coefficients.

The regularisation strength is controlled by `alpha`. Larger values penalise coefficients more strongly.

The tested values were `0.1`, `1.0`, `10.0`, and `100.0`. The selected value was `0.1`, based on the lowest validation RMSE.

## 5. Random Forest Regression

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

Each tree is trained using a bootstrap sample of the training observations. At each split, the algorithm considers a random subset of features. Averaging the predictions reduces the variance associated with an individual decision tree.

Random Forest can model nonlinear relationships and feature interactions without requiring the input features to be linearly related to the target.

The following configurations were evaluated:

| Number of trees | Maximum depth | Minimum samples per leaf |
|---:|---:|---:|
| 200 | 12 | 2 |
| 300 | 20 | 2 |
| 300 | Unlimited | 2 |

The selected configuration contained 200 trees, a maximum depth of 12, and a minimum leaf size of 2. It achieved a validation RMSE of approximately 11.285.

## 6. XGBoost Regression

XGBoost is a gradient-boosting method that builds decision trees sequentially. Each new tree improves the current ensemble by optimising an objective based on prediction errors and regularisation.

Important hyperparameters include:

- `n_estimators`: number of boosting rounds
- `max_depth`: maximum tree depth
- `learning_rate`: contribution of each new tree
- `subsample`: fraction of training observations sampled per boosting round
- `colsample_bytree`: fraction of features sampled for each tree
- `reg_lambda`: L2 regularisation strength

The experiment used squared-error regression, a learning-rate-dependent boosting process, row and feature subsampling, and L2 regularisation.

The following configurations were compared:

| Trees | Maximum depth | Learning rate |
|---:|---:|---:|
| 300 | 4 | 0.05 |
| 500 | 5 | 0.03 |
| 300 | 7 | 0.05 |

The best validation result came from 300 trees, maximum depth 7, and learning rate 0.05. The remaining settings were `subsample=0.8`, `colsample_bytree=0.8`, and `reg_lambda=1.0`.

## 7. Experimental Methodology

The main station-based split was used for model development.

1. Fit preprocessing and models on the training set.
2. Compare candidate configurations on the validation set.
3. Select configurations using validation RMSE.
4. Save the selected models, validation metrics, validation predictions, and training configuration.
5. Reserve the test set for separate final evaluation.

The test set was not used to choose the hyperparameters reported below.

The fixed random seed was `42` for reproducibility where supported.

## 8. Validation Results

Results obtained from the initial training run:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 9.7617 | 13.9120 | -0.0054 |
| Ridge Regression | 8.6875 | 12.2626 | 0.2188 |
| Random Forest | 7.7912 | 11.2855 | 0.3384 |
| XGBoost | 7.7489 | 11.2319 | 0.3446 |

XGBoost achieved the lowest validation RMSE among the tested configurations. Its validation RMSE was approximately 19.3% lower than the Dummy Regressor's RMSE.

Random Forest performed similarly, while both tree-based methods outperformed the two baselines on the validation set.

The XGBoost R² of approximately 0.345 indicates that predictive performance is moderate. The result supports further evaluation but does not establish generalisation performance on the held-out test set.

## 9. Saved Outputs

The training script saves the following artifacts:

- `models/dummy_regressor.joblib`
- `models/ridge_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost_regressor.joblib`
- `results/validation_metrics.csv`
- `results/validation_predictions.csv`
- `results/training_config.json`

Model binaries are excluded from Git because they can be regenerated using `src/train.py`. The validation results and configuration are retained as reproducible experiment outputs.

## 10. Limitations and Next Steps

- The initial hyperparameter search is small and does not guarantee globally optimal configurations.
- Validation R² indicates that substantial target variation remains unexplained.
- The target is bounded between 0 and 100, while standard regression estimators do not inherently enforce those bounds.
- Final test metrics must be reported separately after model selection is complete.
- Feature importance and SHAP analysis can help investigate which inputs influence predictions, but importance does not establish causation.
