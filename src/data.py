"""
src/data.py
Reusable data loading, cleaning, domain validation, and station-based splitting.

Features:
- Safe raw-data loading and column-name standardization
- Complete alias mapping for Ireland (Plan A) and Cork Harbour (Plan B) schemas
- Robust date and numeric parsing
- Left-censored detection-limit handling (<X -> 0.5 * X + binary BDL flag)
- Duplicate auditing with reproducible logs
- Physical-domain validation without deleting complete rows
- Leakage-resistant station-based train/test splitting (GroupShuffleSplit)
- Explicit validation of missing identifiers and invalid inputs
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {
    # Location / Station Identifiers
    "country": "country",
    "area": "site_id",
    "waterbodyname": "site_id",
    "waterbody_name": "site_id",
    "site": "site_id",
    "site_id": "site_id",
    "station_id": "site_id",
    "station": "site_id",
    "monitoring_location_identifier": "site_id",

    # Waterbody & Temporal
    "waterbody_type": "waterbody_type",
    "water_body_type": "waterbody_type",
    "date": "date",
    "sampledate": "date",
    "sample_date": "date",
    "years": "years",
    "year": "years",

    # Physicochemical Parameters (Normalized Ireland & Cork headers)
    "ammonia_mg_l": "ammonia_mg_l",
    "ammonia_total_as_n": "ammonia_mg_l",

    "bod_mg_l": "bod_mg_l",
    "biochemical_oxygen_demand_mg_l": "bod_mg_l",
    "bod_5_days_total": "bod_mg_l",

    "dissolved_oxygen_mg_l": "dissolved_oxygen_mg_l",
    "dissolved_oxygen": "dissolved_oxygen_mg_l",

    "orthophosphate_mg_l": "orthophosphate_mg_l",
    "ortho_phosphate_as_p_unspecified": "orthophosphate_mg_l",

    "ph": "ph",
    "ph_ph_units": "ph",

    "temperature_c": "temperature_c",
    "temperature_cel": "temperature_c",
    "temperature": "temperature_c",

    "nitrogen_mg_l": "nitrogen_mg_l",
    "nitrate_mg_l": "nitrate_mg_l",

    # Additional Cork Harbour Parameters (if Plan B is tested)
    "alkalinity_total_as_caco3": "alkalinity_mg_l",
    "chloride": "chloride_mg_l",
    "conductivity_25_c": "conductivity_us_cm",
    "total_hardness_as_caco3": "hardness_mg_l",
    "true_colour": "true_colour_hazen",

    # Target Columns
    "ccme_values": "target_ccme_value",
    "ccme_value": "target_ccme_value",
    "target_ccme_value": "target_ccme_value",
    "wqi_value": "target_ccme_value",

    "ccme_wqi": "target_ccme_class",
    "target_ccme_class": "target_ccme_class",
    "label": "target_ccme_class",
}


def _normalize_header(name: Any) -> str:
    """Convert a column header into a lowercase snake_case identifier."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _log_entry(
    issue: str,
    count: int | str,
    action: str,
    reason: str,
) -> dict[str, Any]:
    """Create a consistently structured cleaning-log entry."""
    return {
        "Issue found": issue,
        "How many rows/values": count,
        "Action": action,
        "Why": reason,
    }


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file without modifying the source file."""
    if not isinstance(filepath, str) or not filepath.strip():
        raise ValueError("filepath must be a non-empty string.")

    filepath = os.path.abspath(os.path.expanduser(filepath))

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)

    if df.empty:
        raise ValueError(f"The input dataset is empty: {filepath}")

    if len(df.columns) == 0:
        raise ValueError(f"The input file contains no columns: {filepath}")

    return df


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize known water-quality headers and normalize other
    headers to snake_case.

    Raises an error if normalization creates duplicate column names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df = df.copy()

    normalized_names = [_normalize_header(col) for col in df.columns]
    renamed_names = [
        COLUMN_ALIASES.get(name, name)
        for name in normalized_names
    ]

    duplicates = pd.Index(renamed_names)[
        pd.Index(renamed_names).duplicated()
    ].unique().tolist()

    if duplicates:
        raise ValueError(
            "Column-name normalization would create duplicate columns: "
            f"{duplicates}. Resolve these columns explicitly before cleaning."
        )

    df.columns = renamed_names
    return df


# ============================================================
# 4. PARSE DATES, NUMERIC VALUES, AND DETECTION LIMITS
# ============================================================

def parse_types_and_detection_limits(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Parse dates and numeric measurement columns.

    Left-censored values such as '<0.02' are replaced with half
    the stated detection limit, and a binary BDL flag is generated.

    Right-censored values such as '>100' are not interpreted as
    exact measurements; they become NaN and are logged.

    Returns:
        (cleaned_dataframe, cleaning_log)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if isinstance(numeric_cols, str):
        raise TypeError("numeric_cols must be a sequence of column names.")

    df = df.copy()
    log_entries: list[dict[str, Any]] = []

    # Parse dates using common formats, preferring day-first dates.
    if "date" in df.columns:
        original = df["date"]
        text_dates = original.astype("string").str.strip()

        parsed = pd.to_datetime(
            text_dates,
            format="%d-%m-%Y",
            errors="coerce",
        )

        remaining = parsed.isna() & text_dates.notna() & text_dates.ne("")

        # Support ISO dates and other common day-first formats.
        if remaining.any():
            try:
                fallback = pd.to_datetime(
                    text_dates.loc[remaining],
                    format="mixed",
                    dayfirst=True,
                    errors="coerce",
                )
            except (TypeError, ValueError):
                fallback = pd.to_datetime(
                    text_dates.loc[remaining],
                    dayfirst=True,
                    errors="coerce",
                )

            parsed.loc[remaining] = fallback

        # Preserve genuinely missing values as missing.
        df["date"] = parsed

        invalid_dates = int(
            (original.notna() & text_dates.ne("") & df["date"].isna()).sum()
        )

        if invalid_dates:
            log_entries.append(
                _log_entry(
                    "Unparseable date values",
                    invalid_dates,
                    "Set invalid dates to NaT",
                    "Unparseable dates must be investigated before temporal analysis.",
                )
            )
        else:
            log_entries.append(
                _log_entry(
                    "Date format standardization",
                    int(df["date"].notna().sum()),
                    "Converted dd-mm-yyyy strings to datetime64[ns]",
                    "Enables chronological sorting and seasonal feature extraction.",
                )
            )

    # Parse each requested measurement column.
    for col in numeric_cols:
        if col not in df.columns:
            log_entries.append(
                _log_entry(
                    f"Measurement column not found: {col}",
                    0,
                    "Skipped",
                    "The requested column is absent from the dataset.",
                )
            )
            continue

        original = df[col]

        if pd.api.types.is_numeric_dtype(original):
            df[col] = pd.to_numeric(original, errors="coerce")
            continue

        values = original.astype("string").str.strip()

        # Recognize conventional missing-value markers.
        missing_markers = {
            "", "na", "n/a", "nan", "null", "none",
            "nd", "n.d.", "not detected", "not available",
        }

        missing_mask = (
            values.isna()
            | values.str.lower().isin(missing_markers)
        )

        # Recognize left-censored values: <0.02, < 0.02, etc.
        left_censored = values.str.match(
            r"^\s*<\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][+-]?\d+)?\s*$",
            na=False,
        )

        # Right-censored measurements require a dedicated treatment.
        right_censored = values.str.match(
            r"^\s*>\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][+-]?\d+)?\s*$",
            na=False,
        )

        left_count = int(left_censored.sum())
        right_count = int(right_censored.sum())

        # Extract the numeric detection limit after the '<' symbol.
        numeric_text = values.str.replace(
            r"^\s*<\s*", "", regex=True
        )

        extracted = pd.to_numeric(
            numeric_text.where(left_censored),
            errors="coerce",
        )

        # Ensure malformed limits do not silently become zero.
        valid_limits = extracted.notna() & (extracted >= 0)
        valid_left = left_censored & valid_limits

        # Parse standard numeric entries.
        parsed_numeric = pd.to_numeric(
            values.where(~left_censored & ~right_censored),
            errors="coerce",
        )

        # Preserve any existing BDL flag rather than overwriting it.
        flag_col = f"{col}_bdl_flag"

        if left_count and flag_col in df.columns:
            raise ValueError(
                f"Cannot create '{flag_col}': the column already exists."
            )

        if left_count:
            df[flag_col] = valid_left.astype("int8")
            parsed_numeric.loc[valid_left] = extracted.loc[valid_left] * 0.5

            log_entries.append(
                _log_entry(
                    f"Values below detection limit in {col}",
                    left_count,
                    "Imputed half the detection limit and created a BDL flag",
                    "Half-limit substitution is a documented approximation for left-censored measurements.",
                )
            )

            invalid_limits = int((left_censored & ~valid_limits).sum())

            if invalid_limits:
                log_entries.append(
                    _log_entry(
                        f"Invalid detection limits in {col}",
                        invalid_limits,
                        "Set to NaN",
                        "The stated detection limit could not be parsed as a non-negative number.",
                    )
                )

        if right_count:
            parsed_numeric.loc[right_censored] = np.nan

            log_entries.append(
                _log_entry(
                    f"Right-censored values in {col}",
                    right_count,
                    "Set to NaN for separate review",
                    "Values greater than a threshold are not exact measurements; use a suitable censored-data method when needed.",
                )
            )

        # Count non-missing values that could not be parsed.
        unparsed_mask = (
            ~missing_mask
            & ~left_censored
            & ~right_censored
            & parsed_numeric.isna()
        )

        unparsed_count = int(unparsed_mask.sum())

        if unparsed_count:
            log_entries.append(
                _log_entry(
                    f"Unparseable numeric values in {col}",
                    unparsed_count,
                    "Set to NaN",
                    "Non-numeric values must be reviewed rather than treated as valid measurements.",
                )
            )

        # Avoid turning recognized missing markers into parsing errors.
        parsed_numeric.loc[missing_mask] = np.nan
        df[col] = parsed_numeric

    return df, log_entries


# ============================================================
# 5. AUDIT AND REMOVE EXACT DUPLICATES
# ============================================================

def audit_and_remove_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Remove exact duplicate rows and reset the index.

    This checks exact row duplicates only. Records from the same
    station and date are not automatically considered duplicates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df = df.copy()
    log_entries: list[dict[str, Any]] = []

    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        df = df.loc[~duplicate_mask].reset_index(drop=True)

        log_entries.append(
            _log_entry(
                "Exact duplicate rows across all columns",
                duplicate_count,
                "Removed duplicate rows and reset the index",
                "Exact duplicates can over-weight repeated observations and complicate validation.",
            )
        )
    else:
        df = df.reset_index(drop=True)

        log_entries.append(
            _log_entry(
                "Exact duplicate rows check",
                0,
                "No rows removed",
                "No exact duplicate rows were found.",
            )
        )

    return df, log_entries


# ============================================================
# 6. VALIDATE PHYSICAL DOMAIN CONSTRAINTS
# ============================================================

def filter_physically_impossible_values(
    df: pd.DataFrame,
    rules: Mapping[str, tuple[float | None, float | None, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Set measurements outside configured bounds to NaN without
    deleting the complete row.

    rules format:
        {
            "column": (minimum, maximum, scientific_reason)
        }

    Missing values are not counted as out-of-range measurements.
    Bounds are inclusive: values equal to a bound are retained.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df = df.copy()
    log_entries: list[dict[str, Any]] = []

    for col, rule in rules.items():
        if len(rule) != 3:
            raise ValueError(
                f"Rule for '{col}' must contain (minimum, maximum, reason)."
            )

        low, high, reason = rule

        if low is not None and high is not None and low > high:
            raise ValueError(
                f"Invalid bounds for '{col}': minimum exceeds maximum."
            )

        if col not in df.columns:
            log_entries.append(
                _log_entry(
                    f"Domain validation skipped: {col}",
                    0,
                    "Skipped",
                    "The configured measurement column does not exist.",
                )
            )
            continue

        # Do not silently compare strings with numeric bounds.
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Column '{col}' must be numeric before domain validation. "
                "Run parse_types_and_detection_limits first."
            )

        numeric_values = df[col]
        invalid_mask = pd.Series(False, index=df.index)

        if low is not None:
            invalid_mask |= numeric_values.notna() & (numeric_values < low)

        if high is not None:
            invalid_mask |= numeric_values.notna() & (numeric_values > high)

        invalid_count = int(invalid_mask.sum())

        if invalid_count:
            df.loc[invalid_mask, col] = np.nan

        log_entries.append(
            _log_entry(
                f"{col} physical bounds check [{low}, {high}]",
                invalid_count,
                (
                    "Set out-of-range values to NaN; retained rows"
                    if invalid_count
                    else "No out-of-range values found"
                ),
                str(reason),
            )
        )

    return df, log_entries


# ============================================================
# 7. STATION-BASED TRAIN / TEST SPLIT
# ============================================================

def generate_station_split(
    df: pd.DataFrame,
    group_col: str = "site_id",
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split rows into train and test sets by group, usually monitoring
    station, ensuring a group is never shared across the two sets.

    Returns positional indices compatible with df.iloc.

    Missing station identifiers are rejected to avoid silently
    treating unknown locations as a valid station.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    if group_col not in df.columns:
        raise ValueError(
            f"Grouping column '{group_col}' is missing from the dataframe."
        )

    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
        raise TypeError("test_size must be a number between 0 and 1.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")

    if df[group_col].isna().any():
        missing_count = int(df[group_col].isna().sum())
        raise ValueError(
            f"Column '{group_col}' contains {missing_count:,} missing group IDs. "
            "Resolve missing station IDs before splitting."
        )

    unique_groups = df[group_col].nunique()

    if unique_groups < 2:
        raise ValueError(
            "At least two distinct groups are required for a group split."
        )

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(test_size),
        random_state=int(seed),
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df[group_col])
    )

    train_groups = set(df.iloc[train_idx][group_col].unique())
    test_groups = set(df.iloc[test_idx][group_col].unique())

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Group leakage detected: {len(overlap)} groups overlap."
        )

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("The generated split contains an empty partition.")

    return train_idx.tolist(), test_idx.tolist()
