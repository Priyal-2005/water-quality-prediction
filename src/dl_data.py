
"""
src/dl_data.py

Prepare network-ready data for the Phase 2 deep learning model.
Reuses Phase 1 feature engineering and existing dataset splits.
"""

import numpy as np

from src.features import (
    load_final_dataset,
    make_features,
    get_xy,
    build_preprocessor,
)


def load_dl_data(
    path="data/processed/final_dataset.csv",
    split_col="split",
):
    """Return preprocessed NumPy arrays, feature names, station ids and row ids.

    Extra keys (not model inputs):
      groups_train / groups_val / groups_test  station id (site_id) of each row, for GroupKFold
      row_id_train / row_id_val / row_id_test  row number in final_dataset.csv, for joining predictions
      station_overlap  dict with the computed number of stations shared between splits
    """

    if split_col not in ("split", "split_time"):
        raise ValueError("split_col must be 'split' or 'split_time'")

    # 1. Load dataset and reuse Phase 1 feature engineering
    df = load_final_dataset(path)
    df = make_features(df)

    # 2. Reuse existing train, validation and test splits
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy(
        df, split_col=split_col
    )

    # 3. Station ids and row ids of every split (same order as X and y)
    parts = {k: df[df[split_col] == k] for k in ("train", "val", "test")}
    sites = {k: set(v["site_id"]) for k, v in parts.items()}
    station_overlap = {
        "train_test": len(sites["train"] & sites["test"]),
        "train_val": len(sites["train"] & sites["val"]),
        "val_test": len(sites["val"] & sites["test"]),
    }

    # The main split is by station, so no station may appear in two sets.
    # (The year split reuses stations across years on purpose, so it is only reported.)
    if split_col == "split":
        assert sum(station_overlap.values()) == 0, (
            f"Station overlap detected: {station_overlap}"
        )

    # 4. Build preprocessor and fit ONLY on training rows
    preprocessor = build_preprocessor(scale=True)
    X_train_processed = preprocessor.fit_transform(X_train)

    # 5. Transform validation and test using the fitted preprocessor
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # 6. Convert features and targets to NumPy arrays
    result = {
        "X_train": np.asarray(X_train_processed, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.float32),
        "X_val": np.asarray(X_val_processed, dtype=np.float32),
        "y_val": np.asarray(y_val, dtype=np.float32),
        "X_test": np.asarray(X_test_processed, dtype=np.float32),
        "y_test": np.asarray(y_test, dtype=np.float32),
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "groups_train": parts["train"]["site_id"].to_numpy(),
        "groups_val": parts["val"]["site_id"].to_numpy(),
        "groups_test": parts["test"]["site_id"].to_numpy(),
        "row_id_train": parts["train"]["row_id"].to_numpy(),
        "row_id_val": parts["val"]["row_id"].to_numpy(),
        "row_id_test": parts["test"]["row_id"].to_numpy(),
        "station_overlap": station_overlap,
    }

    return result


if __name__ == "__main__":
    data = load_dl_data()

    print("Network-ready data shapes:")

    for name, value in data.items():
        if name == "feature_names":
            print(f"{name}: {len(value)} features")
        elif name == "station_overlap":
            continue
        else:
            print(f"{name}: {value.shape}")

    print("\nStation overlap (split):", data["station_overlap"])
    print("Station overlap (split_time, reported only):",
          load_dl_data(split_col="split_time")["station_overlap"])
    print("Data pipeline test completed.")
