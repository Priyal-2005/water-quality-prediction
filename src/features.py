"""
src/features.py
Feature engineering for the water-quality models.

The target (CCME index) is calculated from 8 parameters, so giving a model all 8
would let it re-learn the formula (R^2 ~ 0.99, see notebooks/03_features.ipynb).
Models therefore only see the three sensor readings the brief lists (dissolved
oxygen, pH, temperature) plus context and history built from them.

Usage
-----
    from src.features import load_final_dataset, make_features, get_xy, build_preprocessor
    df = make_features(load_final_dataset())
    X_train, y_train, X_val, y_val, X_test, y_test = get_xy(df, split_col="split")
    prep = build_preprocessor()          # fit on X_train ONLY
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "target_ccme_value"

# Parameters held OUT of the model inputs (the CCME formula uses them).
HELD_OUT_COLS = [
    "ammonia_mg_l", "bod_mg_l", "orthophosphate_mg_l", "nitrogen_mg_l", "nitrate_mg_l",
]
# Columns that must never be model inputs.
FORBIDDEN_COLS = HELD_OUT_COLS + [
    "target_ccme_value", "target_ccme_class", "n_genuine_params",
    "site_id", "date", "split", "split_time",
]

SENSOR_COLS = ["dissolved_oxygen_mg_l", "ph", "temperature_c"]
ENGINEERED_COLS = ["month_sin", "month_cos", "do_sat_pct", "do_x_temp"]
HISTORY_COLS = [f"{c}_site_prev_mean" for c in SENSOR_COLS] + ["site_prev_n", "days_since_prev"]
NUMERIC_COLS = SENSOR_COLS + ENGINEERED_COLS + HISTORY_COLS
CATEGORICAL_COLS = ["waterbody_type", "country"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def load_final_dataset(path: str = "data/processed/final_dataset.csv") -> pd.DataFrame:
    """Load the single processed file, sorted by site and date (needed for history features)."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month as a circle (December is next to January), so the model sees seasons."""
    out = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_chemistry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    do_sat_pct: dissolved oxygen as % of the most the water could hold at that
        temperature (freshwater, sea level; Standard Methods polynomial). Cold water
        holds more oxygen, so 8 mg/L means different things at 5 C and 25 C.
    do_x_temp: simple interaction of oxygen and temperature.
    """
    out = df.copy()
    t = out["temperature_c"]
    do_sat = 14.652 - 0.41022 * t + 0.007991 * t**2 - 0.000077774 * t**3
    out["do_sat_pct"] = 100 * out["dissolved_oxygen_mg_l"] / do_sat.where(do_sat > 0)
    out["do_x_temp"] = out["dissolved_oxygen_mg_l"] * t
    return out


def add_site_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses ONLY earlier rows of the same site (never the row itself or later rows) and
    ONLY the three sensor columns (never the target or held-out parameters):
      <sensor>_site_prev_mean : running mean of the site's previous readings
      site_prev_n             : how many earlier samples this site has
      days_since_prev         : days since the site's previous sample
    Requires df sorted by site_id then date (load_final_dataset does this).
    """
    out = df.copy()
    grp = out.groupby("site_id", sort=False)
    for c in SENSOR_COLS:
        out[f"{c}_site_prev_mean"] = grp[c].transform(lambda s: s.shift(1).expanding().mean())
    out["site_prev_n"] = grp.cumcount()
    out["days_since_prev"] = grp["date"].diff().dt.days
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered column. Row-wise except the site-history block (past rows only)."""
    return add_site_history_features(add_chemistry_features(add_time_features(df)))


def get_xy(df: pd.DataFrame, split_col: str = "split"):
    """
    Return X_train, y_train, X_val, y_val, X_test, y_test using only the allowed columns.
    split_col="split" is the main (by station) split, "split_time" the secondary (by year).
    Use val for tuning / early stopping, and look at test once at the end.
    """
    assert split_col in ("split", "split_time"), "split_col must be 'split' or 'split_time'"
    leaked = set(FEATURE_COLS) & set(FORBIDDEN_COLS)
    assert not leaked, f"Forbidden columns in feature list: {leaked}"
    parts = {k: df[df[split_col] == k] for k in ("train", "val", "test")}
    return (parts["train"][FEATURE_COLS], parts["train"][TARGET_COL],
            parts["val"][FEATURE_COLS], parts["val"][TARGET_COL],
            parts["test"][FEATURE_COLS], parts["test"][TARGET_COL])


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """
    Median imputation (+ missing-indicator columns), optional scaling, one-hot categories.
    Fit on the training rows only (call .fit(X_train)); then transform train and test.
    Trees do not need scaling but the Phase 2 neural network and Ridge do.
    """
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(steps), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
