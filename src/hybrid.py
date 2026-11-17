
"""
Reusable XGBoost + MLP hybrid pipeline.

Selected approach: weighted prediction blending.
Residual learning was investigated separately in notebook 07
and was not selected because its validation corrections had
very weak predictive signal.

Run from the project root:
    python -m src.hybrid
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.features import (
    build_preprocessor,
    get_xy,
    load_final_dataset,
    make_features,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
RESULT_DIR = ROOT / "results"
DL_RESULT_DIR = RESULT_DIR / "dl"

SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------
# 1. XGBoost model
# ---------------------------------------------------------

def make_xgb():
    """Create XGBoost using the selected Phase 1 parameters."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def make_xgb_pipeline():
    """Create a preprocessing + XGBoost pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor(scale=False)),
        ("model", make_xgb()),
    ])


# ---------------------------------------------------------
# 2. Out-of-fold predictions
# ---------------------------------------------------------

def oof_predictions(X, y, groups, n_splits=5, seed=42):
    """
    Generate out-of-fold predictions using GroupKFold.

    Each validation fold contains stations not used to train
    that fold's model. Preprocessing is fitted separately
    inside each fold to prevent preprocessing leakage.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y and groups must have equal lengths.")

    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique groups for the requested folds.")

    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        model = make_xgb_pipeline()

        model.fit(
            X.iloc[train_idx],
            y[train_idx],
        )

        oof[valid_idx] = model.predict(X.iloc[valid_idx])

        fold_rmse = np.sqrt(
            mean_squared_error(y[valid_idx], oof[valid_idx])
        )
        print(f"OOF fold {fold}/{n_splits}: RMSE = {fold_rmse:.4f}")

    if np.isnan(oof).any():
        raise RuntimeError("Some training rows did not receive OOF predictions.")

    return oof


# ---------------------------------------------------------
# 3. Blending utilities
# ---------------------------------------------------------

def blend_predictions(xgb_pred, mlp_pred, xgb_weight=0.55):
    """Combine XGBoost and MLP predictions."""
    xgb_pred = np.asarray(xgb_pred, dtype=float)
    mlp_pred = np.asarray(mlp_pred, dtype=float)

    if xgb_pred.shape != mlp_pred.shape:
        raise ValueError("Prediction arrays must have identical shapes.")

    if not 0 <= xgb_weight <= 1:
        raise ValueError("xgb_weight must be between 0 and 1.")

    return (
        xgb_weight * xgb_pred
        + (1 - xgb_weight) * mlp_pred
    )


def clip_predictions(predictions, lower=0, upper=100):
    """Keep predicted CCME values within the valid 0–100 range."""
    return np.clip(np.asarray(predictions), lower, upper)


def evaluate_predictions(y_true, predictions):
    """Calculate regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def select_blend_weight(
    y_true,
    xgb_pred,
    mlp_pred,
    weights=None,
    clip=True,
):
    """
    Select the XGBoost weight using validation RMSE only.
    """
    if weights is None:
        weights = np.arange(0, 1.001, 0.05)

    rows = []

    for weight in weights:
        prediction = blend_predictions(
            xgb_pred,
            mlp_pred,
            xgb_weight=float(weight),
        )

        if clip:
            prediction = clip_predictions(prediction)

        metrics = evaluate_predictions(y_true, prediction)

        rows.append({
            "xgb_weight": float(weight),
            "mlp_weight": float(1 - weight),
            **metrics,
        })

    results = pd.DataFrame(rows)
    best = results.loc[results["rmse"].idxmin()]

    return float(best["xgb_weight"]), results


# ---------------------------------------------------------
# 4. Prediction alignment
# ---------------------------------------------------------

def load_mlp_predictions(path, expected_row_ids):
    """Load committed MLP predictions and align by original row ID."""
    predictions = pd.read_csv(path)

    required = {"row_id", "y_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{path} must contain {required}.")

    if predictions["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row IDs found in {path}.")

    aligned = pd.DataFrame({"row_id": expected_row_ids}).merge(
        predictions[["row_id", "y_pred"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    if aligned["y_pred"].isna().any():
        raise ValueError(f"Missing MLP predictions while aligning {path}.")

    return aligned["y_pred"].to_numpy(dtype=float)


# ---------------------------------------------------------
# 5. Main station-split hybrid workflow
# ---------------------------------------------------------

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and engineering dataset...")
    df = make_features(load_final_dataset(str(DATA_PATH)))

    X_train, y_train, X_val, y_val, X_test, y_test = get_xy(
        df, split_col="split"
    )

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    train_rows = df.loc[X_train.index]
    val_rows = df.loc[X_val.index]
    test_rows = df.loc[X_test.index]

    train_ids = train_rows["row_id"].to_numpy()
    val_ids = val_rows["row_id"].to_numpy()
    test_ids = test_rows["row_id"].to_numpy()

    groups_train = train_rows["site_id"].to_numpy()

    print(
        f"Train: {len(y_train):,} | "
        f"Validation: {len(y_val):,} | "
        f"Test: {len(y_test):,}"
    )

    # Reuse the saved Phase 1 XGBoost predictions so the
    # hybrid and standalone XGBoost comparisons use the same model.
    print("\nLoading saved Phase 1 XGBoost predictions...")

    xgb_val_all = pd.read_csv(RESULT_DIR / "validation_predictions.csv")
    xgb_test_all = pd.read_csv(RESULT_DIR / "test_predictions.csv")

    xgb_val_frame = xgb_val_all.loc[
        xgb_val_all["model"] == "XGBRegressor",
        ["row_id", "y_true", "y_pred"],
    ].copy()

    xgb_test_frame = xgb_test_all.loc[
        xgb_test_all["model"] == "XGBoost",
        ["row_id", "y_true", "y_pred"],
    ].copy()

    def align_xgb(frame, expected_ids, expected_y, name):
        if frame["row_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate row IDs")

        aligned = pd.DataFrame({"row_id": expected_ids}).merge(
            frame,
            on="row_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if aligned.isna().any().any():
            raise ValueError(f"{name}: missing or unaligned predictions")

        if not np.allclose(aligned["y_true"], expected_y):
            raise ValueError(f"{name}: target values do not match dataset")

        return aligned["y_pred"].to_numpy(dtype=float)

    xgb_val = align_xgb(
        xgb_val_frame, val_ids, y_val, "Phase 1 XGBoost validation"
    )
    xgb_test = align_xgb(
        xgb_test_frame, test_ids, y_test, "Phase 1 XGBoost test"
    )

    print("Loaded and verified saved Phase 1 XGBoost predictions.")

    # Reuse the committed MLP predictions from Phase 2.
    print("\nLoading committed MLP predictions...")
    mlp_val = load_mlp_predictions(
        DL_RESULT_DIR / "predictions_mlp_val.csv",
        val_ids,
    )

    mlp_test = load_mlp_predictions(
        DL_RESULT_DIR / "predictions_mlp.csv",
        test_ids,
    )

    # Select blend weight using validation data only.
    print("\nSelecting blend weight using validation RMSE...")
    best_weight, weight_results = select_blend_weight(
        y_val,
        xgb_val,
        mlp_val,
        clip=True,
    )

    mlp_weight = 1 - best_weight

    print(
        f"Selected weights: XGBoost={best_weight:.2f}, "
        f"MLP={mlp_weight:.2f}"
    )

    weight_results.to_csv(
        RESULT_DIR / "hybrid_blend_weights.csv",
        index=False,
    )

    # Validation predictions.
    hybrid_val = clip_predictions(
        blend_predictions(xgb_val, mlp_val, best_weight)
    )

    val_output = pd.DataFrame({
        "row_id": val_ids,
        "y_true": y_val,
        "y_pred": hybrid_val,
        "xgb_pred": xgb_val,
        "mlp_pred": mlp_val,
        "blend_pred": hybrid_val,
    })

    val_output.to_csv(
        RESULT_DIR / "hybrid_val_preds.csv",
        index=False,
    )

    print("\nValidation hybrid metrics:")
    print(evaluate_predictions(y_val, hybrid_val))

    # Test predictions are generated only after validation selection.
    # No test metrics are calculated in this script.
    hybrid_test = clip_predictions(
        blend_predictions(xgb_test, mlp_test, best_weight)
    )

    test_output = pd.DataFrame({
    "row_id": test_ids,
    "y_pred": hybrid_test,
})

    test_output.to_csv(
        RESULT_DIR / "predictions_hybrid.csv",
        index=False,
    )

    print("\nSaved:")
    print(RESULT_DIR / "hybrid_val_preds.csv")
    print(RESULT_DIR / "predictions_hybrid.csv")
    print("\nTest predictions saved without scoring the test set.")


if __name__ == "__main__":
    main()
