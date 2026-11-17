"""One-time evaluation on the unseen station-split test set."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dl_data import load_dl_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DL_RESULTS = RESULTS / "dl"
SEED = 42
N_BOOTSTRAPS = 2000

XGB_WEIGHT = 0.55
MLP_WEIGHT = 0.45


def metrics(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
    }


def station_bootstrap(y, predictions, groups):
    """Bootstrap complete stations and calculate metric confidence intervals."""
    rng = np.random.default_rng(SEED)
    unique_stations = np.unique(groups)
    station_rows = {
        station: np.flatnonzero(groups == station)
        for station in unique_stations
    }

    samples = {name: [] for name in predictions}

    for _ in range(N_BOOTSTRAPS):
        sampled = rng.choice(
            unique_stations,
            size=len(unique_stations),
            replace=True,
        )
        indices = np.concatenate([station_rows[s] for s in sampled])

        for name, pred in predictions.items():
            result = metrics(y[indices], pred[indices])
            samples[name].append(result)

    rows = []
    for name, values in samples.items():
        frame = pd.DataFrame(values)
        for metric in ("MAE", "RMSE", "R2"):
            rows.append({
                "model": name,
                "metric": metric,
                "estimate": metrics(y, predictions[name])[metric],
                "ci_lower": frame[metric].quantile(0.025),
                "ci_upper": frame[metric].quantile(0.975),
            })

    return pd.DataFrame(rows), samples


def paired_rmse_bootstrap(y, predictions, groups, baseline):
    """Paired station bootstrap of RMSE differences against baseline."""
    rng = np.random.default_rng(SEED + 1)
    unique_stations = np.unique(groups)
    station_rows = {
        station: np.flatnonzero(groups == station)
        for station in unique_stations
    }

    rows = []
    baseline_pred = predictions[baseline]

    for name, pred in predictions.items():
        if name == baseline:
            continue

        differences = []
        for _ in range(N_BOOTSTRAPS):
            sampled = rng.choice(
                unique_stations,
                size=len(unique_stations),
                replace=True,
            )
            indices = np.concatenate([station_rows[s] for s in sampled])

            rmse_model = mean_squared_error(
                y[indices], pred[indices]
            ) ** 0.5
            rmse_baseline = mean_squared_error(
                y[indices], baseline_pred[indices]
            ) ** 0.5

            differences.append(rmse_model - rmse_baseline)

        observed = (
            mean_squared_error(y, pred) ** 0.5
            - mean_squared_error(y, baseline_pred) ** 0.5
        )

        rows.append({
            "comparison": f"{name} - {baseline}",
            "rmse_difference": observed,
            "ci_lower": np.quantile(differences, 0.025),
            "ci_upper": np.quantile(differences, 0.975),
            "bootstrap_probability_below_zero": np.mean(
                np.asarray(differences) < 0
            ),
        })

    return pd.DataFrame(rows)


def main():
    print("Loading station split metadata...")
    data = load_dl_data(split_col="split")

    row_ids = data["row_id_test"]
    y = data["y_test"].astype(float)
    groups = data["groups_test"]

    print("Test rows:", len(row_ids))
    print("Unique test stations:", len(np.unique(groups)))
    print("Station overlap:", data["station_overlap"])

    xgb_all = pd.read_csv(RESULTS / "test_predictions.csv")
    xgb = xgb_all.loc[
        xgb_all["model"] == "XGBoost",
        ["row_id", "y_true", "y_pred"],
    ].copy()

    mlp = pd.read_csv(DL_RESULTS / "predictions_mlp.csv")
    hybrid = pd.read_csv(RESULTS / "predictions_hybrid.csv")

    def align(frame, name):
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
            raise ValueError(f"{name}: missing or unaligned predictions")
        return aligned

    xgb = align(xgb, "XGBoost")
    mlp = align(mlp, "MLP")
    hybrid = align(hybrid, "Hybrid")

    # XGBoost and MLP files contain ground-truth values.
    # The hybrid test file intentionally contains predictions only.
    for name, frame in [
        ("XGBoost", xgb),
        ("MLP", mlp),
    ]:
        if not np.allclose(frame["y_true"], y):
            raise ValueError(f"{name}: target values do not match test targets")

    expected_hybrid = np.clip(
        XGB_WEIGHT * xgb["y_pred"].to_numpy()
        + MLP_WEIGHT * mlp["y_pred"].to_numpy(),
        0,
        100,
    )

    if not np.allclose(expected_hybrid, hybrid["y_pred"], atol=1e-5):
        raise ValueError(
            "Hybrid predictions do not match the frozen 55:45 Phase 1 XGB + MLP blend. "
            "Regenerate results/predictions_hybrid.csv before scoring."
        )

    predictions = {
        "XGBoost": xgb["y_pred"].to_numpy(),
        "MLP": mlp["y_pred"].to_numpy(),
        "Hybrid": hybrid["y_pred"].to_numpy(),
    }

    print("\nScoring station-split test set...")
    summary = pd.DataFrame([
        {"model": name, **metrics(y, pred)}
        for name, pred in predictions.items()
    ])

    print("\nTest metrics:")
    print(summary.to_string(index=False))

    print("\nCalculating station-bootstrap confidence intervals...")
    ci, _ = station_bootstrap(y, predictions, groups)
    print(ci.to_string(index=False))

    print("\nCalculating paired RMSE differences against XGBoost...")
    paired = paired_rmse_bootstrap(
        y, predictions, groups, baseline="XGBoost"
    )
    print(paired.to_string(index=False))

    summary.to_csv(DL_RESULTS / "station_test_metrics.csv", index=False)
    ci.to_csv(DL_RESULTS / "station_test_bootstrap_ci.csv", index=False)
    paired.to_csv(DL_RESULTS / "station_test_paired_rmse.csv", index=False)

    consolidated = pd.read_csv(DL_RESULTS / "metrics_dl.csv")
    new_rows = summary.assign(split="station_test")
    consolidated = pd.concat(
        [consolidated, new_rows],
        ignore_index=True,
    )
    consolidated.to_csv(DL_RESULTS / "metrics_dl.csv", index=False)

    print("\nSaved:")
    print(DL_RESULTS / "station_test_metrics.csv")
    print(DL_RESULTS / "station_test_bootstrap_ci.csv")
    print(DL_RESULTS / "station_test_paired_rmse.csv")
    print(DL_RESULTS / "metrics_dl.csv")


if __name__ == "__main__":
    main()
