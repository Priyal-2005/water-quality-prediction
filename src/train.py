
"""
Train baseline and tree-based regression models for water-quality prediction.

Target: target_ccme_value
Primary split: station-based train / validation / test
Test data is reserved for final evaluation and is not used for tuning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.features import (
    FEATURE_COLS,
    TARGET_COL,
    build_preprocessor,
    get_xy,
    load_final_dataset,
    make_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

RANDOM_STATE = 42


def regression_metrics(y_true, y_pred):
    """Return standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_pipeline(model, scale=True):
    """Fit preprocessing and a model together to prevent preprocessing leakage."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale=scale)),
            ("model", model),
        ]
    )


def evaluate_validation(pipeline, X_train, y_train, X_val, y_val):
    """Fit on training data and evaluate on validation data."""
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)
    return regression_metrics(y_val, predictions)


def main():
    start_time = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and engineering features...")
    df = make_features(load_final_dataset(str(DATA_PATH)))

    X_train, y_train, X_val, y_val, X_test, y_test = get_xy(
        df, split_col="split"
    )

    print(f"Target: {TARGET_COL}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Train: {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test: {len(X_test):,}")
    print("\nThe test set will not be used for model selection.")

    # Keep all model selection on train/validation only.
    experiments = []

    # 1. Dummy baseline: predicts the training-target mean.
    print("\n[1/4] Training DummyRegressor...")
    dummy = make_pipeline(
        DummyRegressor(strategy="mean"),
        scale=False,
    )
    dummy_metrics = evaluate_validation(
        dummy, X_train, y_train, X_val, y_val
    )
    experiments.append(
        {
            "model": "DummyRegressor",
            "parameters": {"strategy": "mean"},
            **dummy_metrics,
        }
    )
    joblib.dump(dummy, MODEL_DIR / "dummy_regressor.joblib")
    print("Validation:", dummy_metrics)

    # 2. Ridge baseline.
    print("\n[2/4] Training Ridge baselines...")
    ridge_candidates = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = make_pipeline(Ridge(alpha=alpha), scale=True)
        metrics = evaluate_validation(
            model, X_train, y_train, X_val, y_val
        )
        ridge_candidates.append((metrics["rmse"], alpha, model, metrics))
        print(f"Ridge alpha={alpha}: {metrics}")

    _, best_ridge_alpha, best_ridge, best_ridge_metrics = min(
        ridge_candidates, key=lambda item: item[0]
    )
    experiments.append(
        {
            "model": "Ridge",
            "parameters": {"alpha": best_ridge_alpha},
            **best_ridge_metrics,
        }
    )
    joblib.dump(best_ridge, MODEL_DIR / "ridge_regression.joblib")

    # 3. Random Forest: select parameters using validation RMSE.
    print("\n[3/4] Tuning Random Forest...")
    rf_candidates = []
    rf_settings = [
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
    ]

    for params in rf_settings:
        print("Trying Random Forest:", params)
        model = make_pipeline(
            RandomForestRegressor(
                **params,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            scale=False,
        )
        metrics = evaluate_validation(
            model, X_train, y_train, X_val, y_val
        )
        rf_candidates.append((metrics["rmse"], params, model, metrics))
        print("Validation:", metrics)

    _, best_rf_params, best_rf, best_rf_metrics = min(
        rf_candidates, key=lambda item: item[0]
    )
    experiments.append(
        {
            "model": "RandomForestRegressor",
            "parameters": best_rf_params,
            **best_rf_metrics,
        }
    )
    joblib.dump(best_rf, MODEL_DIR / "random_forest.joblib")

    # 4. XGBoost: select parameters using validation RMSE.
    print("\n[4/4] Tuning XGBoost...")
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install it with: pip install xgboost"
        ) from exc

    xgb_settings = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05},
    ]
    xgb_candidates = []

    for params in xgb_settings:
        print("Trying XGBoost:", params)
        model = make_pipeline(
            XGBRegressor(
                **params,
                objective="reg:squarederror",
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            ),
            scale=False,
        )
        metrics = evaluate_validation(
            model, X_train, y_train, X_val, y_val
        )
        xgb_candidates.append((metrics["rmse"], params, model, metrics))
        print("Validation:", metrics)

    _, best_xgb_params, best_xgb, best_xgb_metrics = min(
        xgb_candidates, key=lambda item: item[0]
    )
    experiments.append(
        {
            "model": "XGBRegressor",
            "parameters": best_xgb_params,
            **best_xgb_metrics,
        }
    )
    joblib.dump(best_xgb, MODEL_DIR / "xgboost_regressor.joblib")

    # Save validation results and selected settings.
    results = pd.DataFrame(experiments).sort_values("rmse")
    results.to_csv(RESULT_DIR / "validation_metrics.csv", index=False)

    with open(RESULT_DIR / "training_config.json", "w") as f:
        json.dump(
            {
                "target": TARGET_COL,
                "split": "split (station-based)",
                "random_state": RANDOM_STATE,
                "feature_columns": FEATURE_COLS,
                "selected_parameters": {
                    "Ridge": {"alpha": best_ridge_alpha},
                    "RandomForestRegressor": best_rf_params,
                    "XGBRegressor": best_xgb_params,
                },
                "selection_metric": "validation RMSE",
                "test_used_for_tuning": False,
            },
            f,
            indent=2,
        )

    # Save validation predictions for later evaluation and diagnostics.
    prediction_rows = []
    for name, model in [
        ("DummyRegressor", dummy),
        ("Ridge", best_ridge),
        ("RandomForestRegressor", best_rf),
        ("XGBRegressor", best_xgb),
    ]:
        pred = model.predict(X_val)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "row_id": X_val.index,
                    "model": name,
                    "y_true": y_val.to_numpy(),
                    "y_pred": pred,
                    "split": "val",
                }
            )
        )

    pd.concat(prediction_rows, ignore_index=True).to_csv(
        RESULT_DIR / "validation_predictions.csv", index=False
    )

    print("\nValidation leaderboard (lower RMSE is better):")
    print(results.to_string(index=False))
    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULT_DIR}")
    print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")
    print("\nTraining complete. Test-set evaluation remains separate.")


if __name__ == "__main__":
    main()
