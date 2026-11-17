"""Recalculate station-split validation results using the Phase 1 XGBoost baseline."""

import numpy as np
import pandas as pd

from src.dl_data import load_dl_data
from src.evaluate_station_test import (
    metrics,
    station_bootstrap,
    paired_rmse_bootstrap,
)

RESULTS = "results"
DL = "results/dl"


def align_predictions(frame, row_ids, name):
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate row IDs")

    aligned = pd.DataFrame({"row_id": row_ids}).merge(
        frame,
        on="row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if aligned.isna().any().any():
        raise ValueError(f"{name}: missing or unaligned values")

    return aligned


def main():
    print("Loading station-split validation metadata...")
    data = load_dl_data(split_col="split")

    row_ids = data["row_id_val"]
    y = data["y_val"].astype(float)
    groups = data["groups_val"]

    print("Validation rows:", len(row_ids))
    print("Unique validation stations:", len(np.unique(groups)))

    xgb_all = pd.read_csv(f"{RESULTS}/validation_predictions.csv")
    xgb = xgb_all.loc[
        xgb_all["model"] == "XGBRegressor",
        ["row_id", "y_true", "y_pred"],
    ].copy()

    mlp = pd.read_csv(f"{DL}/predictions_mlp_val.csv")
    hybrid = pd.read_csv(f"{RESULTS}/hybrid_val_preds.csv")

    xgb = align_predictions(xgb, row_ids, "XGBoost")
    mlp = align_predictions(mlp, row_ids, "MLP")
    hybrid = align_predictions(hybrid, row_ids, "Hybrid")

    for name, frame in [
        ("XGBoost", xgb),
        ("MLP", mlp),
        ("Hybrid", hybrid),
    ]:
        if not np.allclose(frame["y_true"], y):
            raise ValueError(f"{name}: target values do not match validation targets")

    # Verify that the hybrid is the fixed 55:45 blend.
    expected = np.clip(
        0.55 * xgb["y_pred"].to_numpy()
        + 0.45 * mlp["y_pred"].to_numpy(),
        0,
        100,
    )

    if not np.allclose(expected, hybrid["y_pred"], atol=1e-5):
        raise ValueError("Hybrid predictions do not match the fixed 55:45 blend.")

    predictions = {
        "XGBoost": xgb["y_pred"].to_numpy(),
        "MLP": mlp["y_pred"].to_numpy(),
        "Hybrid": hybrid["y_pred"].to_numpy(),
    }

    print("\nRecalculating validation metrics...")
    summary = pd.DataFrame([
        {"Model": name, **metrics(y, pred)}
        for name, pred in predictions.items()
    ])
    print(summary.to_string(index=False))

    print("\nCalculating station-bootstrap confidence intervals...")
    ci, _ = station_bootstrap(y, predictions, groups)

    ci_rows = []
    for name, values in ci.groupby("model"):
        record = {"Model": name}
        for _, row in values.iterrows():
            metric_name = row["metric"]
            record[metric_name] = row["estimate"]
            record[f"{metric_name}_CI_Lower"] = row["ci_lower"]
            record[f"{metric_name}_CI_Upper"] = row["ci_upper"]
        ci_rows.append(record)

    ci_output = pd.DataFrame(ci_rows)[[
        "Model",
        "MAE", "MAE_CI_Lower", "MAE_CI_Upper",
        "RMSE", "RMSE_CI_Lower", "RMSE_CI_Upper",
    ]]
    print(ci_output.to_string(index=False))

    print("\nCalculating paired RMSE differences against XGBoost...")
    paired = paired_rmse_bootstrap(
        y, predictions, groups, baseline="XGBoost"
    )

    paired_output = paired.rename(columns={
        "comparison": "Model",
        "rmse_difference": "Mean_RMSE_Difference",
        "ci_lower": "CI_Lower",
        "ci_upper": "CI_Upper",
    })
    paired_output["Model"] = paired_output["Model"].str.replace(
        " - XGBoost", "", regex=False
    )
    paired_output = paired_output[[
        "Model",
        "Mean_RMSE_Difference",
        "CI_Lower",
        "CI_Upper",
    ]]
    print(paired_output.to_string(index=False))

    # Save the corrected validation outputs.
    summary.to_csv(f"{DL}/validation_metrics.csv", index=False)
    ci_output.to_csv(f"{DL}/station_bootstrap_ci.csv", index=False)
    paired_output.to_csv(f"{DL}/paired_rmse_differences.csv", index=False)

    # Replace only station-split validation rows; preserve all other evaluations.
    consolidated = pd.read_csv(f"{DL}/metrics_dl.csv")
    is_validation = (
        consolidated["Evaluation"] == "Station-split validation"
    )
    consolidated = consolidated.loc[~is_validation].copy()

    new_rows = pd.DataFrame({
        "Model": summary["Model"],
        "MAE": summary["MAE"],
        "RMSE": summary["RMSE"],
        "R2": summary["R2"],
        "Evaluation": "Station-split validation",
        "model": np.nan,
        "split": np.nan,
    })

    consolidated = pd.concat(
        [new_rows, consolidated],
        ignore_index=True,
    )
    consolidated.to_csv(f"{DL}/metrics_dl.csv", index=False)

    print("\nUpdated validation files successfully.")
    print("Year-split and station-test rows preserved.")


if __name__ == "__main__":
    main()
