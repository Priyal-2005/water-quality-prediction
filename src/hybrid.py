"""
XGBoost + MLP hybrid pipeline (weighted prediction blending).

Selected approach: a weighted blend of the saved Phase 1 XGBoost predictions
and the saved Phase 2 MLP predictions, with the blend weight chosen on the
validation set only. Both inputs are predictions from models trained on the
training rows only, so no out-of-fold step is needed for the blend.

Out-of-fold (OOF) XGBoost predictions are needed only for the residual-learning
variant (an MLP trained on y - XGBoost prediction). That variant was tested in
notebook 07 and not selected (its corrections had almost no predictive signal).
`oof_predictions` is kept for it; run it with `--oof` as a diagnostic.

Run from the project root:
    python -m src.hybrid            # build the blend from saved predictions
    python -m src.hybrid --oof      # also run the station-grouped OOF diagnostic

No test metrics are calculated here. The test set is scored once, in
`notebooks/08_dl_evaluation.ipynb`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
RESULT_DIR = ROOT / "results"
DL_RESULT_DIR = RESULT_DIR / "dl"

SEED = 42
N_SPLITS = 5

# Phase 1 saved XGBoost predictions (the same predictions the Phase 1 results use).
XGB_VAL_PATH = RESULT_DIR / "validation_predictions.csv"
XGB_TEST_PATH = RESULT_DIR / "test_predictions.csv"
XGB_VAL_NAME = "XGBRegressor"   # model name in validation_predictions.csv
XGB_TEST_NAME = "XGBoost"       # model name in test_predictions.csv


# ---------------------------------------------------------
# 1. XGBoost model (only needed for the OOF / residual variant)
# ---------------------------------------------------------

def make_xgb():
    """Create XGBoost using the selected Phase 1 parameters."""
    from xgboost import XGBRegressor

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
    from sklearn.pipeline import Pipeline
    from src.features import build_preprocessor

    return Pipeline([
        ("preprocessor", build_preprocessor(scale=False)),
        ("model", make_xgb()),
    ])


# ---------------------------------------------------------
# 2. Out-of-fold predictions (needed for the residual variant only)
# ---------------------------------------------------------

def oof_predictions(X, y, groups, n_splits=5, seed=42):
    """
    Generate out-of-fold predictions using GroupKFold.

    Each validation fold contains stations not used to train
    that fold's model. Preprocessing is fitted separately
    inside each fold to prevent preprocessing leakage.

    These predictions are required when XGBoost predictions on the TRAINING
    rows are used as inputs or residual targets for a second model. The
    weighted blend does not need them.
    """
    from sklearn.model_selection import GroupKFold

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
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = model.predict(X.iloc[valid_idx])

        fold_rmse = np.sqrt(mean_squared_error(y[valid_idx], oof[valid_idx]))
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

    return xgb_weight * xgb_pred + (1 - xgb_weight) * mlp_pred


def clip_predictions(predictions, lower=0, upper=100):
    """Keep predicted CCME values within the valid 0-100 range."""
    return np.clip(np.asarray(predictions), lower, upper)


def evaluate_predictions(y_true, predictions):
    """Calculate regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def select_blend_weight(y_true, xgb_pred, mlp_pred, weights=None, clip=True):
    """Select the XGBoost weight using validation RMSE only."""
    if weights is None:
        weights = np.arange(0, 1.001, 0.05)

    rows = []
    for weight in weights:
        prediction = blend_predictions(xgb_pred, mlp_pred, xgb_weight=float(weight))
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
# 4. Loading saved predictions, aligned by original row ID
# ---------------------------------------------------------

def load_saved_predictions(path, expected_row_ids, model=None):
    """
    Load saved predictions and align them to `expected_row_ids` by row ID.
    If `model` is given, keep only that model's rows (Phase 1 files hold
    several models).
    """
    predictions = pd.read_csv(path)

    required = {"row_id", "y_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"{path} must contain {required}.")

    if model is not None:
        predictions = predictions[predictions["model"] == model]
        if predictions.empty:
            raise ValueError(f"No rows for model '{model}' in {path}.")

    if predictions["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row IDs found in {path}.")

    aligned = pd.DataFrame({"row_id": expected_row_ids}).merge(
        predictions[["row_id", "y_pred"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    if aligned["y_pred"].isna().any():
        raise ValueError(f"Missing predictions while aligning {path}.")

    return aligned["y_pred"].to_numpy(dtype=float)


# ---------------------------------------------------------
# 5. Main station-split hybrid workflow
# ---------------------------------------------------------

def main(run_oof: bool = False):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Row IDs and validation targets come from the saved MLP prediction files,
    # so the hybrid is aligned to exactly the rows the MLP was scored on.
    mlp_val_file = pd.read_csv(DL_RESULT_DIR / "predictions_mlp_val.csv")
    mlp_test_file = pd.read_csv(DL_RESULT_DIR / "predictions_mlp.csv")

    val_ids = mlp_val_file["row_id"].to_numpy()
    test_ids = mlp_test_file["row_id"].to_numpy()
    y_val = mlp_val_file["y_true"].to_numpy(dtype=float)

    print(f"Validation rows: {len(val_ids):,} | Test rows: {len(test_ids):,}")

    # Saved predictions from models trained on the training rows only.
    xgb_val = load_saved_predictions(XGB_VAL_PATH, val_ids, model=XGB_VAL_NAME)
    xgb_test = load_saved_predictions(XGB_TEST_PATH, test_ids, model=XGB_TEST_NAME)
    mlp_val = load_saved_predictions(DL_RESULT_DIR / "predictions_mlp_val.csv", val_ids)
    mlp_test = load_saved_predictions(DL_RESULT_DIR / "predictions_mlp.csv", test_ids)

    print("Validation XGBoost (Phase 1 saved):", evaluate_predictions(y_val, xgb_val))
    print("Validation MLP (saved):", evaluate_predictions(y_val, mlp_val))

    # Select blend weight using validation data only.
    best_weight, weight_results = select_blend_weight(
        y_val, xgb_val, mlp_val, clip=True
    )
    print(f"\nSelected weights: XGBoost={best_weight:.2f}, MLP={1 - best_weight:.2f}")

    weight_results.to_csv(RESULT_DIR / "hybrid_blend_weights.csv", index=False)

    hybrid_val = clip_predictions(blend_predictions(xgb_val, mlp_val, best_weight))

    pd.DataFrame({
        "row_id": val_ids,
        "y_true": y_val,
        "y_pred": hybrid_val,
        "xgb_pred": xgb_val,
        "mlp_pred": mlp_val,
        "blend_pred": hybrid_val,
    }).to_csv(RESULT_DIR / "hybrid_val_preds.csv", index=False)

    print("Validation hybrid metrics:", evaluate_predictions(y_val, hybrid_val))

    # Test predictions use the validation-selected weight. They are saved but
    # NOT scored here.
    hybrid_test = clip_predictions(blend_predictions(xgb_test, mlp_test, best_weight))
    pd.DataFrame({"row_id": test_ids, "y_pred": hybrid_test}).to_csv(
        RESULT_DIR / "predictions_hybrid.csv", index=False
    )

    print("\nSaved:")
    print(RESULT_DIR / "hybrid_blend_weights.csv")
    print(RESULT_DIR / "hybrid_val_preds.csv")
    print(RESULT_DIR / "predictions_hybrid.csv")
    print("Test predictions saved without scoring the test set.")

    if run_oof:
        # Diagnostic needed only for the residual-learning variant.
        from src.features import get_xy, load_final_dataset, make_features

        print("\nRunning station-grouped OOF diagnostic (residual variant only)...")
        df = make_features(load_final_dataset(str(DATA_PATH)))
        X_train, y_train, *_ = get_xy(df, split_col="split")
        groups_train = df.loc[X_train.index, "site_id"].to_numpy()
        oof = oof_predictions(
            X_train, y_train.to_numpy(), groups_train,
            n_splits=N_SPLITS, seed=SEED,
        )
        print("OOF metrics:", evaluate_predictions(y_train.to_numpy(), oof))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--oof", action="store_true",
        help="Also run the station-grouped OOF XGBoost diagnostic (needs xgboost).",
    )
    main(run_oof=parser.parse_args().oof)
