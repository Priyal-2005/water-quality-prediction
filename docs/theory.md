# Model Training and Theoretical Background

## 1. Problem Formulation

This project treats water-quality prediction as a supervised regression problem.

- **Target variable:** `target_ccme_value`
- **Target meaning:** CCME Water Quality Index (WQI), ranging from 0 to 100, with higher values indicating better water quality.
- **Primary split:** Station-based training, validation and test sets.
- **Evaluation metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and coefficient of determination (R²).

The objective is to predict the continuous CCME WQI value using sensor measurements, contextual information, engineered features and historical sensor information.

The dataset contains 259,774 observations, divided into:

| Dataset split | Number of observations |
|---|---:|
| Training | 181,747 |
| Validation | 26,054 |
| Test | 51,973 |
| **Total** | **259,774** |

The test set is reserved for final evaluation and is not used for hyperparameter selection.

## 2. Feature Selection and Leakage Prevention

The CCME WQI target is calculated using multiple water-quality parameters. Including the same constituent parameters used to calculate the target could allow a model to reconstruct the target formula instead of learning a useful predictive relationship from the available sensor readings.

The model therefore uses three primary sensor measurements:

- Dissolved oxygen: `dissolved_oxygen_mg_l`
- pH: `ph`
- Temperature: `temperature_c`

Additional features include the following.

### 2.1 Seasonal features

- `month_sin`
- `month_cos`

These encode the month cyclically, allowing the model to represent the relationship between adjacent months, including December and January.

### 2.2 Derived sensor features

- `do_sat_pct`: dissolved oxygen saturation percentage.
- `do_x_temp`: interaction between dissolved oxygen and temperature.

These features represent additional relationships that may be relevant to water quality.

### 2.3 Historical features

The historical features include:

- `dissolved_oxygen_mg_l_site_prev_mean`
- `ph_site_prev_mean`
- `temperature_c_site_prev_mean`
- `site_prev_n`
- `days_since_prev`

Historical means use measurements from strictly earlier dates at the same station. All observations sharing the same station and date receive identical historical features. This prevents observations from the same date from leaking information into one another through the history calculation.

`site_prev_n` represents the number of earlier sampling dates, while `days_since_prev` represents the number of days since the preceding sampling date.

### 2.4 Contextual features

- `waterbody_type`
- `country`

The current training run uses 14 features in total.

### 2.5 Excluded variables

The following constituent parameters are excluded from the model inputs because they contribute to target calculation:

- Ammonia: `ammonia_mg_l`
- Biochemical oxygen demand: `bod_mg_l`
- Orthophosphate: `orthophosphate_mg_l`
- Nitrogen: `nitrogen_mg_l`
- Nitrate: `nitrate_mg_l`

The target itself, `target_ccme_class`, station identifiers, date-related split metadata and `n_genuine_params` are also excluded from the feature set.

This feature-selection strategy aims to reduce target leakage and make predictions more representative of a setting where only the selected sensor measurements, context and permitted history are available.

## 3. Preprocessing

Preprocessing is implemented within the model pipeline so that fitted transformations use training data rather than information learned from the validation or test sets.

### 3.1 Numerical features

Numerical preprocessing includes:

1. Missing-value imputation using training-set medians.
2. Missing-value indicators.
3. Standardisation for Ridge Regression.

Tree-based models do not require standardisation in the same way as linear models, so scaling is disabled for Random Forest and XGBoost.

### 3.2 Categorical features

Categorical variables are one-hot encoded. Unknown categories are ignored during transformation, allowing the fitted pipeline to process categories not encountered during training.

Keeping preprocessing within the pipeline helps prevent information leakage during model fitting and evaluation.

## 4. Baseline Models

Two baseline approaches are used to establish reference performance before assessing the more flexible tree-based models.

### 4.1 Dummy Regressor

The Dummy Regressor predicts the mean target value observed in the training set for every input.

It does not learn relationships between the input features and the target. Its purpose is to provide a simple benchmark against which the other models can be compared.

If a predictive model cannot outperform this baseline, its additional complexity may not be justified.

### 4.2 Ridge Regression

Ridge Regression is a linear regression model with L2 regularisation. It minimises squared prediction error while penalising large coefficients.

The regularisation strength is controlled by `alpha`. Larger values apply a stronger penalty to the coefficients.

The current training pipeline evaluates the following values:

`0.001`, `0.01`, `0.1`, `1`, `10`, `100` and `1000`.

The selected configuration is `alpha=0.001`, which achieves the lowest validation RMSE among the tested Ridge configurations.

Ridge Regression provides a useful linear benchmark. Comparing it with tree-based models helps assess whether nonlinear relationships and interactions improve predictive performance.

## 5. Random Forest Regression

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

Each tree is trained using a bootstrap sample of the training observations, and random feature selection at each split helps diversify the trees. Averaging predictions across trees generally reduces the variance associated with an individual decision tree.

Random Forest can model nonlinear relationships and feature interactions without requiring a linear relationship between the predictors and target.

### 5.1 Hyperparameter search

The current pipeline evaluates eight configurations. The number of trees is held at 300 while maximum depth and minimum leaf size are varied.

**Maximum-depth sweep**

| Number of trees | Maximum depth | Minimum samples per leaf |
|---:|---:|---:|
| 300 | 6 | 5 |
| 300 | 8 | 5 |
| 300 | 10 | 5 |
| 300 | 12 | 5 |
| 300 | 16 | 5 |

**Minimum-leaf-size sweep**

| Number of trees | Maximum depth | Minimum samples per leaf |
|---:|---:|---:|
| 300 | 10 | 2 |
| 300 | 10 | 10 |
| 300 | 10 | 20 |

Configurations are compared using validation RMSE, with lower values indicating better performance under this metric.

### 5.2 Selected configuration

The selected Random Forest configuration is:

- `n_estimators=300`
- `max_depth=16`
- `min_samples_leaf=5`
- `random_state=42`

The model achieves the following validation results:

- MAE: 7.7728
- RMSE: 11.2470
- R²: 0.3429

This is the best Random Forest configuration among the candidates evaluated in the current run. The results do not establish that it is the globally optimal configuration.

## 6. XGBoost Regression

XGBoost is a gradient-boosting method that builds decision trees sequentially. Each new tree contributes to improving the current ensemble by optimising an objective based on prediction error and regularisation.

Unlike Random Forest, which builds many trees whose predictions are averaged, gradient boosting builds trees in sequence, with subsequent trees improving the existing ensemble.

### 6.1 Regularised objective

A simplified representation of the XGBoost objective is:

\[
\mathcal{L} =
\sum_{i=1}^{n} l(y_i,\hat{y}_i)
+
\sum_{k=1}^{K}\Omega(f_k)
\]

where:

- \(y_i\) is the actual target value.
- \(\hat{y}_i\) is the predicted target value.
- \(l\) is the prediction-loss function.
- \(f_k\) represents an individual tree.
- \(\Omega(f_k)\) penalises model complexity.

For this regression task, the pipeline uses squared-error regression loss. The regularisation term discourages unnecessarily complex trees and can help reduce overfitting.

### 6.2 Important hyperparameters

- `n_estimators`: number of boosting rounds.
- `max_depth`: maximum depth of each tree.
- `learning_rate`: contribution of each new tree.
- `subsample`: fraction of observations sampled for each boosting round.
- `colsample_bytree`: fraction of features sampled for each tree.
- `reg_lambda`: L2 regularisation strength.

The current pipeline uses `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` and `random_state=42`, alongside the candidate values for tree count, depth and learning rate.

### 6.3 Hyperparameter search

The current search evaluates seven configurations.

**Maximum-depth sweep**

| Number of trees | Maximum depth | Learning rate |
|---:|---:|---:|
| 300 | 4 | 0.05 |
| 300 | 6 | 0.05 |
| 300 | 8 | 0.05 |
| 300 | 10 | 0.05 |

**Learning-rate sweep**

| Number of trees | Maximum depth | Learning rate |
|---:|---:|---:|
| 300 | 8 | 0.03 |
| 300 | 8 | 0.10 |

**Tree-count experiment**

| Number of trees | Maximum depth | Learning rate |
|---:|---:|---:|
| 600 | 8 | 0.05 |

The configurations are compared using validation RMSE.

### 6.4 Selected configuration

The selected XGBoost configuration is:

- `n_estimators=300`
- `max_depth=8`
- `learning_rate=0.03`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_lambda=1.0`
- `random_state=42`

It achieves:

- MAE: 7.7014
- RMSE: 11.1942
- R²: 0.3490

This is the best XGBoost configuration among the candidates evaluated in the current run.

## 7. Experimental Methodology

The main station-based split is used for model development. Model selection is performed using training and validation data, while the test set remains separate.

The current experimental procedure is:

1. Load the processed dataset and engineer the permitted features.
2. Separate the training, validation and test partitions.
3. Fit preprocessing and models on the training data.
4. Evaluate the candidate Ridge, Random Forest and XGBoost configurations on the validation data.
5. Select the configuration with the lowest validation RMSE for each model.
6. Record the model configurations and validation metrics.
7. Save the fitted models and experiment outputs.
8. Reserve the test set for final evaluation after model selection is complete.

The training script uses a fixed random seed of `42` where supported to improve reproducibility.

### 7.1 Cross-validation status

The results presented in this document come from a single validation split and the manually defined hyperparameter searches described above.

The required five-fold `GroupKFold` tuning grouped by station must be implemented and run separately if it is not already present in the current training script. It should operate on training data only, keeping observations from the same station within the same fold.

The validation results in this document must not be described as cross-validation results.

### 7.2 Test-set evaluation

The test set contains 51,973 observations and is reserved for final evaluation. Its labels must not be used to select hyperparameters or otherwise influence model selection.

Once the model configurations are finalised, the selected models should be evaluated on the held-out test set. Final metrics and the prediction files required by the evaluation workflow should then be generated separately.

## 8. Validation Results

The following table reports the results from the latest training run. For each model family, the reported row corresponds to the selected configuration under the current search procedure.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 9.7617 | 13.9120 | -0.0054 |
| Ridge Regression | 8.6875 | 12.2626 | 0.2188 |
| Random Forest | 7.7728 | 11.2470 | 0.3429 |
| XGBoost | 7.7014 | 11.1942 | 0.3490 |

### 8.1 Interpretation

**XGBoost achieves the best validation performance** among the four selected models, with the lowest RMSE of approximately 11.1942.

Compared with the Dummy Regressor, XGBoost reduces validation RMSE by approximately 19.5%. Random Forest performs similarly, with a validation RMSE of approximately 11.2470.

Both tree-based models outperform Ridge Regression and the Dummy Regressor on the current validation split. This suggests that nonlinear relationships and interactions may help predict the target beyond what the tested linear model captures.

The XGBoost R² of approximately 0.349 indicates that the model explains about 34.9% of the variation in the validation targets relative to predicting their mean.

This is moderate predictive performance, and substantial variation remains unexplained. These findings are limited to the current validation split; final generalisation performance must be assessed on the held-out test set.

## 9. Saved Outputs

The training pipeline saves fitted models and experiment results under the project directory.

### 9.1 Model artifacts

- `models/dummy_regressor.joblib`
- `models/ridge_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost_regressor.joblib`

These files contain fitted model artifacts and can be regenerated by running:

```bash
python -m src.train
```

The model binaries may be excluded from Git if the repository intentionally stores only reproducible code and experiment outputs.

### 9.2 Experiment outputs

- `results/validation_metrics.csv`: validation metrics for the selected models.
- `results/tuning_results.csv`: individual candidate configurations, validation metrics and selection markers, when generated by the current script.
- `results/training_config.json`: target, split information, random seed and selected hyperparameters.
- `results/validation_predictions.csv`: validation predictions, if generated by the current script.

The presence of a file should be verified against the current training script and output directory.

Final test predictions are a separate deliverable. Where required by the evaluation workflow, generate one CSV per model with the columns `row_id`, `y_true` and `y_pred`, using the original dataset row identifiers and the held-out test observations. These files must not be confused with validation predictions.

## 10. Limitations and Next Steps

The current experiments establish a baseline comparison and identify promising candidate configurations, but several limitations remain.

1. **Limited hyperparameter search:** The manually selected configurations do not guarantee globally optimal parameters.
2. **Cross-validation:** The required five-fold station-grouped `GroupKFold` tuning must be completed and its results recorded before claiming that cross-validation has been performed.
3. **Final test evaluation:** Final test metrics and test prediction CSVs must be generated after model selection is complete.
4. **Moderate predictive performance:** The validation R² values indicate that a substantial proportion of target variation remains unexplained.
5. **Target bounds:** The target lies between 0 and 100, but the regression estimators do not inherently enforce these bounds on their predictions.
6. **Interpretability:** Feature importance and SHAP analysis can help explain model behaviour, but feature importance does not establish causation.
7. **Reproducibility:** The training configuration, metrics and prediction artifacts should be retained so that the experiments can be reproduced and compared.

The next priorities are to complete the required station-grouped cross-validation, finalise model selection without using test data, generate test predictions for the evaluation workflow, and coordinate the final evaluation and interpretability results with the rest of the project team.