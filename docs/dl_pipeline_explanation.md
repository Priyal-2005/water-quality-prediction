
# 5.4 DL Data Pipeline — Explanation

## 1. Why is scaling essential for Neural Networks but not strictly required for XGBoost?

Neural Networks use gradient-based optimisation. Features with very different scales can affect gradient updates and make training less stable or slower. StandardScaler brings numerical features to a comparable scale.

XGBoost uses decision-tree splits, which are generally insensitive to feature scaling. Therefore, scaling is not strictly required for XGBoost.

## 2. Why are the imputer and scaler fitted only on training data?

The imputer and scaler must learn their parameters only from training data.

- The imputer learns median values from training data.
- The scaler learns the mean and standard deviation from training data.
- Validation and test data are only transformed using these learned parameters.

This prevents data leakage and keeps evaluation fair.

## 3. Why are missing-value indicators included?

Median imputation replaces missing numerical values with median values. However, this replacement hides the information that a value was originally missing.

Missing-value indicators preserve this information by adding an indicator feature for missing values.

## 4. Explain Dropout, Weight Decay and Early Stopping.

- **Dropout:** Randomly disables some neurons during training to reduce overfitting.
- **Weight Decay:** Penalises large weights during optimisation to control model complexity.
- **Early Stopping:** Stops training when validation performance stops improving to prevent overfitting.

## 5. Why must the same train/validation/test splits be reused?

Reusing the same splits ensures a fair comparison between models and prevents differences in evaluation caused by different data partitions.

The project reuses the existing `split` and `split_time` columns instead of creating new splits.
