"""
src/hybrid.py
Hybrid of XGBoost and the MLP (blend or residual). XGBoost predictions used as inputs or targets MUST be out-of-fold by station (GroupKFold).
Run from the project root:  python -m src.hybrid
"""
"""
Reusable utilities for the XGBoost + MLP hybrid model.

The selected hybrid approach is weighted prediction blending.
Residual learning was investigated separately in notebook 07.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def blend_predictions(xgb_pred, mlp_pred, xgb_weight=0.55):
    """
    Combine XGBoost and MLP predictions using a weighted average.

    Parameters
    ----------
    xgb_pred : array-like
        XGBoost predictions.
    mlp_pred : array-like
        MLP predictions.
    xgb_weight : float
        Weight assigned to XGBoost. MLP receives the remaining weight.

    Returns
    -------
    numpy.ndarray
        Blended predictions.
    """
    if not 0 <= xgb_weight <= 1:
        raise ValueError("xgb_weight must be between 0 and 1.")

    xgb_pred = np.asarray(xgb_pred, dtype=float).ravel()
    mlp_pred = np.asarray(mlp_pred, dtype=float).ravel()

    if xgb_pred.shape != mlp_pred.shape:
        raise ValueError("XGBoost and MLP predictions must have equal shapes.")

    return (
        xgb_weight * xgb_pred
        + (1 - xgb_weight) * mlp_pred
    )


def clip_predictions(predictions, lower=0, upper=100):
    """Restrict predictions to the valid CCME score range."""
    if lower >= upper:
        raise ValueError("lower must be smaller than upper.")

    return np.clip(np.asarray(predictions, dtype=float), lower, upper)


def evaluate_predictions(y_true, predictions):
    """Calculate regression evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    predictions = np.asarray(predictions, dtype=float).ravel()

    if y_true.shape != predictions.shape:
        raise ValueError("Actual and predicted values must have equal shapes.")

    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2": float(r2_score(y_true, predictions)),
    }


def select_blend_weight(
    y_true,
    xgb_pred,
    mlp_pred,
    weights=None,
    clip=True,
):
    """
    Select the XGBoost blending weight using validation RMSE only.

    The MLP weight is calculated as 1 - xgb_weight.
    """
    if weights is None:
        weights = np.arange(0, 1.001, 0.05)

    results = []

    for weight in weights:
        predictions = blend_predictions(
            xgb_pred,
            mlp_pred,
            xgb_weight=float(weight),
        )

        if clip:
            predictions = clip_predictions(predictions)

        metrics = evaluate_predictions(y_true, predictions)

        results.append({
            "xgb_weight": float(weight),
            "mlp_weight": float(1 - weight),
            **metrics,
        })

    results_df = pd.DataFrame(results).sort_values(
        "RMSE"
    ).reset_index(drop=True)

    best = results_df.iloc[0].to_dict()

    return best, results_df
