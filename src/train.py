"""
Train baseline and tree-based regression models for water-quality prediction.

Target: target_ccme_value
Primary split: station-based train / validation / test
Test data is reserved for final evaluation and is not used for tuning.

Run from the project root:  python -m src.train
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


def log_trial(trials, name, params, metrics):
    """Record one tried setting and return its index in the trials list."""
    trials.append(
        {
            "model": name,
            "parameters": json.dumps(params, sort_keys=True),
            **metrics,
            "selected": False,
        }
    )
    return len(trials) - 1


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
    # Every setting tried is logged in `trials`; the winner of each model is
    # the one with the lowest validation RMSE, and only that model is kept.
    trials = []
    winners = {}  # model name -> index of the winning trial in `trials`

    # 1. Dummy baseline: predicts the training-target mean (needs no preprocessing).
    print("\n[1/4] Training DummyRegressor...")
    dummy = DummyRegressor(strategy="mean")
    metrics = evaluate_validation(dummy, X_train, y_train, X_val, y_val)
    winners["DummyRegressor"] = log_trial(
        trials, "DummyRegressor", {"strategy": "mean"}, metrics
    )
    joblib.dump(dummy, MODEL_DIR / "dummy_regressor.joblib")
    print("Validation:", metrics)

    # Check results/tuning_results.csv: a winner at the edge of its sweep means the sweep could be extended.

    # 2. Ridge baseline.
    print("\n[2/4] Training Ridge baselines...")
    ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    best_ridge = None
    best_ridge_rmse = float("inf")
    for alpha in ridge_alphas:
        model = make_pipeline(Ridge(alpha=alpha), scale=True)
        metrics = evaluate_validation(model, X_train, y_train, X_val, y_val)
        index = log_trial(trials, "Ridge", {"alpha": alpha}, metrics)
        print(f"Ridge alpha={alpha}: {metrics}")
        if metrics["rmse"] < best_ridge_rmse:
            best_ridge, best_ridge_rmse = model, metrics["rmse"]
            winners["Ridge"] = index
    joblib.dump(best_ridge, MODEL_DIR / "ridge_regression.joblib")

    # 3. Random Forest: one setting changes at a time.
    print("\n[3/4] Tuning Random Forest...")
    rf_settings = (
        # depth sweep
        [{"n_estimators": 300, "max_depth": d, "min_samples_leaf": 5} for d in [6, 8, 10, 12, 16]]
        # leaf sweep
        + [{"n_estimators": 300, "max_depth": 10, "min_samples_leaf": m} for m in [2, 10, 20]]
    )

    best_rf = None
    best_rf_rmse = float("inf")
    for params in rf_settings:
        print("Trying Random Forest:", params)
        model = make_pipeline(
            RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE),
            scale=False,
        )
        metrics = evaluate_validation(model, X_train, y_train, X_val, y_val)
        index = log_trial(trials, "RandomForestRegressor", params, metrics)
        print("Validation:", metrics)
        if metrics["rmse"] < best_rf_rmse:
            best_rf, best_rf_rmse = model, metrics["rmse"]
            winners["RandomForestRegressor"] = index
    joblib.dump(best_rf, MODEL_DIR / "random_forest.joblib")

    # 4. XGBoost: one setting changes at a time.
    print("\n[4/4] Tuning XGBoost...")
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install it with: pip install xgboost"
        ) from exc

    xgb_settings = (
        # depth sweep
        [{"n_estimators": 300, "max_depth": d, "learning_rate": 0.05} for d in [4, 6, 8, 10]]
        # learning-rate sweep
        + [{"n_estimators": 300, "max_depth": 8, "learning_rate": lr} for lr in [0.03, 0.1]]
        # more trees
        + [{"n_estimators": 600, "max_depth": 8, "learning_rate": 0.05}]
    )

    best_xgb = None
    best_xgb_rmse = float("inf")
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
        metrics = evaluate_validation(model, X_train, y_train, X_val, y_val)
        index = log_trial(trials, "XGBRegressor", params, metrics)
        print("Validation:", metrics)
        if metrics["rmse"] < best_xgb_rmse:
            best_xgb, best_xgb_rmse = model, metrics["rmse"]
            winners["XGBRegressor"] = index
    joblib.dump(best_xgb, MODEL_DIR / "xgboost_regressor.joblib")

    # Save every trial (winner marked by its index) and the validation leaderboard.
    for index in winners.values():
        trials[index]["selected"] = True
    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(RESULT_DIR / "tuning_results.csv", index=False)

    results = (
        trials_df[trials_df["selected"]]
        .drop(columns="selected")
        .sort_values("rmse")
    )
    results.to_csv(RESULT_DIR / "validation_metrics.csv", index=False)

    selected_parameters = {
        name: json.loads(trials[index]["parameters"])
        for name, index in winners.items()
        if name != "DummyRegressor"
    }
    with open(RESULT_DIR / "training_config.json", "w") as f:
        json.dump(
            {
                "target": TARGET_COL,
                "split": "split (station-based)",
                "random_state": RANDOM_STATE,
                "feature_columns": FEATURE_COLS,
                "selected_parameters": selected_parameters,
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
                    "row_id": df.loc[X_val.index, "row_id"].to_numpy(),
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
