
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
    """Return preprocessed NumPy arrays and feature names."""

    if split_col not in ("split", "split_time"):
        raise ValueError("split_col must be 'split' or 'split_time'")

    # 1. Load dataset and reuse Phase 1 feature engineering
    df = load_final_dataset(path)
    df = make_features(df)

    # 2. Reuse existing train, validation and test splits
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy(
        df, split_col=split_col
    )

    # 3. Verify station separation for the main split
    if split_col == "split":
        train_sites = set(df.loc[df[split_col] == "train", "site_id"])
        test_sites = set(df.loc[df[split_col] == "test", "site_id"])

        overlap = train_sites.intersection(test_sites)
        assert len(overlap) == 0, (
            f"Train-test station overlap detected: {len(overlap)}"
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
    }

    return result


if __name__ == "__main__":
    data = load_dl_data()

    print("Network-ready data shapes:")

    for name, value in data.items():
        if name == "feature_names":
            print(f"{name}: {len(value)} features")
        else:
            print(f"{name}: {value.shape}")

    print("\nTrain-test station overlap: 0")
    print("Data pipeline test completed.")
