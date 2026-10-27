"""
src/data.py
Reusable data loading, cleaning, domain validation and station-based splitting.

Pipeline (see section 8): data/raw/Combined_dataset.csv  ->  build_final_dataset()
 -> data/processed/final_dataset.csv, the single processed file used by everyone.

Features:
- Chunked loading of the large combined file, filtered to chosen countries
- Surface-water filter (drops effluent / sewage / drainage)
- Detection of placeholder (filled-in) values per country x water type
- Physical-domain validation without deleting whole rows
- Exact-duplicate removal with a reproducible log
- Whole-site sampling to keep the output GitHub-sized
- Leakage-resistant station split (GroupShuffleSplit) plus a time split
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


def generate_station_split_3way(
    df: pd.DataFrame,
    group_col: str = "site_id",
    fractions: tuple[float, float, float] = (0.70, 0.10, 0.20),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split rows into train / val / test by group (station) so that the shares of
    ROWS are as close as possible to `fractions` and no station is in two sets.

    Stations are shuffled (seed) and dealt out in order until the train share of rows
    is reached, then the val share, and the rest is test. Returns positional indices
    (for df.iloc).
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1.")
    if df[group_col].isna().any():
        raise ValueError(f"Column '{group_col}' contains missing group IDs.")
    sizes = df.groupby(group_col).size()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    cum = sizes.loc[order].cumsum().to_numpy() / len(df)
    train_end = fractions[0]
    val_end = fractions[0] + fractions[1]
    which = np.where(cum <= train_end, 0, np.where(cum <= val_end, 1, 2))
    assignment = dict(zip(order, which))
    codes = df[group_col].map(assignment).to_numpy()
    parts = [np.flatnonzero(codes == k).tolist() for k in range(3)]
    if any(len(p) == 0 for p in parts):
        raise RuntimeError("A partition is empty.")
    sets = [set(df.iloc[p][group_col]) for p in parts]
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        raise RuntimeError("Group leakage detected between partitions.")
    return parts[0], parts[1], parts[2]


# ============================================================
# 8. FINAL DATASET PIPELINE (Combined_dataset.csv -> final_dataset.csv)
# ============================================================
# Everything below turns the raw combined file into the single
# data/processed/final_dataset.csv that the whole team uses.
# Findings that drive these steps (see docs/cleaning_log.md):
#   * The source file has one row per sample but many values are NOT real
#     measurements: for each (country, water type) the authors filled
#     missing parameters with one repeated value (e.g. BOD = 2.7 for 44% of
#     England rows). We detect these placeholders and set them to NaN.
#   * ~30% (29.9%) of England rows are effluent / sewage / drainage water, not
#     surface water.
#   * The target (CCME index) is computed from the same 8 parameters, so the
#     model must NOT be given all 8 (see docs/target_definition.md).

PARAM_COLS = [
    "ammonia_mg_l",
    "bod_mg_l",
    "dissolved_oxygen_mg_l",
    "orthophosphate_mg_l",
    "ph",
    "temperature_c",
    "nitrogen_mg_l",
    "nitrate_mg_l",
]

# The only measured inputs the models are allowed to use (leakage control).
SENSOR_COLS = ["dissolved_oxygen_mg_l", "ph", "temperature_c"]

DEFAULT_COUNTRIES = ("England", "USA", "Ireland", "Canada")

# Rows per country kept in final_dataset.csv (None = keep all that pass the
# quality filters). Whole monitoring sites are sampled, never single rows.
DEFAULT_COUNTRY_CAPS = {
    "England": 180_000,
    "Ireland": 70_000,
    "USA": None,
    "Canada": None,
}

NON_SURFACE_TYPES = ("Effluent", "Sewage", "Drainage")

WATERBODY_GROUPS = {
    "River": "River",
    "Lake": "Lake",
    "Canal": "Canal",
    "Estuarine": "Estuarine/Transitional",
    "Transitional": "Estuarine/Transitional",
    "Sea Water": "Marine/Coastal",
    "Marine": "Marine/Coastal",
    "Coastal": "Marine/Coastal",
}

MIN_YEAR = 2000

PHYSICAL_RULES = {
    "ph": (0.0, 14.0, "Outside the physical pH scale; instrument error"),
    "temperature_c": (-5.0, 40.0, "Surface water outside -5..40 C is a unit error (e.g. Fahrenheit) or a fault"),
    "dissolved_oxygen_mg_l": (0.0, 25.0, "Above ~25 mg/L is not physically possible in surface water"),
    "bod_mg_l": (0.0, 150.0, "Above 150 mg/L looks like effluent, not surface water"),
    "ammonia_mg_l": (0.0, 100.0, "Negative or extreme concentrations are errors"),
    "orthophosphate_mg_l": (0.0, 50.0, "Negative or extreme concentrations are errors"),
    "nitrogen_mg_l": (0.0, 200.0, "Negative or extreme concentrations are errors"),
    "nitrate_mg_l": (0.0, 200.0, "Negative or extreme concentrations are errors"),
}


def load_combined_for_countries(
    path: str,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read the large combined CSV in chunks, keeping only the chosen countries."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Combined dataset not found at: {path}")
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        parts.append(chunk[chunk["Country"].isin(countries)])
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        raise ValueError(f"No rows found for countries {list(countries)}.")
    return df


def detect_placeholder_values(
    df: pd.DataFrame,
    cols: Sequence[str] = PARAM_COLS,
    group_cols: Sequence[str] = ("country", "waterbody_type"),
    min_share: float = 0.05,
    dominance: float = 2.5,
    max_passes: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find values that are repeated placeholders rather than measurements.

    Within each group (country x water type) and parameter, the most common
    value is flagged when it makes up at least `min_share` of the group's
    remaining values AND is at least `dominance` times more common than the
    second most common value. The check is repeated (up to `max_passes`) on the
    values that are left, because some groups contain more than one filler.
    Natural rounding peaks (e.g. pH 7.8 at 9% vs 8% for 7.9) are not flagged.

    Returns (mask, audit): a boolean frame (True = placeholder) and a table
    listing every flagged (group, parameter, pass, value, share).
    """
    if not df.index.is_unique:
        raise ValueError("df index must be unique; call reset_index(drop=True) first.")

    mask = pd.DataFrame(False, index=df.index, columns=list(cols))
    audit_rows: list[dict[str, Any]] = []

    for keys, grp in df.groupby(list(group_cols), observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for col in cols:
            values = grp[col].round(6)
            for pass_no in range(1, max_passes + 1):
                shares = values.value_counts(normalize=True)
                if shares.empty:
                    break
                top_value, top_share = shares.index[0], float(shares.iloc[0])
                second_share = float(shares.iloc[1]) if len(shares) > 1 else 0.0
                if not (top_share >= min_share and top_share >= dominance * second_share):
                    break
                hit = values == top_value
                mask.loc[hit[hit].index, col] = True
                audit_rows.append(
                    dict(
                        zip(group_cols, keys),
                        parameter=col,
                        pass_no=pass_no,
                        placeholder_value=float(top_value),
                        share_of_remaining_pct=round(100 * top_share, 1),
                        rows_flagged=int(hit.sum()),
                    )
                )
                values = values.where(~hit)

    return mask, pd.DataFrame(audit_rows)


def sample_sites_with_caps(
    df: pd.DataFrame,
    caps: Mapping[str, int | None],
    site_col: str = "site_id",
    country_col: str = "country",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Keep whole monitoring sites, in random order, until each country's row cap
    is reached. Sampling sites (not rows) keeps every kept station's full history.
    """
    rng = np.random.default_rng(seed)
    kept = []
    for country, grp in df.groupby(country_col):
        cap = caps.get(country)
        if cap is None or len(grp) <= cap:
            kept.append(grp)
            continue
        sizes = grp.groupby(site_col).size()
        order = rng.permutation(sizes.index.to_numpy())
        cum = sizes.loc[order].cumsum()
        n_sites = int((cum < cap).sum()) + 1
        chosen = set(order[:n_sites])
        kept.append(grp[grp[site_col].isin(chosen)])
    return pd.concat(kept).sort_index()


def build_final_dataset(
    raw_path: str,
    out_path: str,
    countries: Sequence[str] = DEFAULT_COUNTRIES,
    caps: Mapping[str, int | None] = DEFAULT_COUNTRY_CAPS,
    min_genuine_params: int = 5,
    min_genuine_sensors: int = 2,
    test_size: float = 0.20,
    val_size: float = 0.10,
    time_cutoff_year: int = 2017,
    time_val_start_year: int = 2016,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """
    Raw combined file -> one cleaned, split, model-ready CSV.

    Returns (final_df, cleaning_log, placeholder_audit).
    """
    log: list[dict[str, Any]] = []

    # 1. Load only the chosen countries
    df = clean_column_names(load_combined_for_countries(raw_path, countries))
    per_country = df["country"].value_counts().to_dict()
    log.append(_log_entry(
        "Rows loaded for the chosen countries",
        f"{len(df):,} " + str({k: int(v) for k, v in per_country.items()}),
        f"Kept countries {list(countries)} from Combined_dataset.csv",
        "Other countries are not used (China has a single site, so no site split is possible).",
    ))

    # 2. Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    log.append(_log_entry("Unparseable dates (dd-mm-yyyy)", bad_dates,
                          "Dropped rows" if bad_dates else "None found",
                          "Date is needed for the time split and seasonal features."))

    # 3. Surface water only
    non_surface = df["waterbody_type"].isin(NON_SURFACE_TYPES)
    by_type = df.loc[non_surface, "waterbody_type"].value_counts().to_dict()
    df = df.loc[~non_surface].reset_index(drop=True)
    log.append(_log_entry(
        "Non-surface water types (effluent, sewage, drainage)",
        f"{int(non_surface.sum()):,} {by_type}",
        "Dropped rows",
        "The task is surface-water quality; effluent/sewage has a different chemistry and was about 30% of England (29.9%).",
    ))

    # 4. Modern records only
    old = df["date"].dt.year < MIN_YEAR
    df = df.loc[~old].reset_index(drop=True)
    log.append(_log_entry(
        f"Records before {MIN_YEAR}", int(old.sum()), "Dropped rows",
        "England only has data from 2000; older USA/Ireland records used different methods and are sparse.",
    ))

    # 5. Duplicates
    df, dup_log = audit_and_remove_duplicates(df)
    log.extend(dup_log)

    # 6. Physical bounds (before placeholder detection)
    df, bound_log = filter_physically_impossible_values(df, PHYSICAL_RULES)
    log.extend(bound_log)

    # 7. Placeholder (filled-in) values -> NaN
    mask, audit = detect_placeholder_values(df)
    for col in PARAM_COLS:
        n = int(mask[col].sum())
        df.loc[mask[col], col] = np.nan
        log.append(_log_entry(
            f"{col}: repeated placeholder value (filled in by the dataset authors)",
            f"{n:,} ({100 * n / len(df):.1f}% of rows)",
            "Set to NaN (imputed later, train-only, with a missing-indicator)",
            "A value repeated for 5%+ of a country/water-type group, 2.5x more than any other (checked up to 3 times), is not a measurement.",
        ))

    df["n_genuine_params"] = df[PARAM_COLS].notna().sum(axis=1)
    df["n_genuine_sensors"] = df[SENSOR_COLS].notna().sum(axis=1)

    # 8. Row-quality filter
    weak = (df["n_genuine_params"] < min_genuine_params) | (df["n_genuine_sensors"] < min_genuine_sensors)
    weak_by_country = df.loc[weak, "country"].value_counts().to_dict()
    df = df.loc[~weak].reset_index(drop=True)
    log.append(_log_entry(
        f"Rows with fewer than {min_genuine_params} of 8 genuine parameters or fewer than {min_genuine_sensors} of 3 genuine sensor readings",
        f"{int(weak.sum()):,} {weak_by_country}",
        "Dropped rows",
        "Mostly filled-in rows: their CCME target is computed from constants, so they teach the model nothing real. This removes most USA rows.",
    ))

    # 9. Tidy columns and identifiers
    df["waterbody_type"] = df["waterbody_type"].map(WATERBODY_GROUPS)
    df["site_id"] = df["country"].astype(str) + "|" + df["site_id"].astype(str).str.strip()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    log.append(_log_entry("Site identifiers", df["site_id"].nunique(),
                          "Prefixed with country so ids are unique across countries",
                          "Needed so the site split cannot mix two stations that share a name."))

    # 10. Sample whole sites to keep the file GitHub-sized
    before = len(df)
    df = sample_sites_with_caps(df, caps, seed=seed).reset_index(drop=True)
    log.append(_log_entry(
        "File size (GitHub limit)", f"{before:,} -> {len(df):,} rows, {df['site_id'].nunique():,} sites",
        f"Random sample of whole sites per country, caps {dict(caps)}, seed {seed}",
        "A full file would be hundreds of MB; whole-site sampling keeps each station's history intact.",
    ))

    # 11. Splits: train / validation / test (70 / 10 / 20 of rows, by station)
    train_idx, val_idx, test_idx = generate_station_split_3way(df, "site_id", (1 - test_size - val_size, val_size, test_size), seed)
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    log.append(_log_entry(
        "Station split (main): train / val / test",
        f"{len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,} rows; "
        f"{df.loc[train_idx, 'site_id'].nunique():,} / {df.loc[val_idx, 'site_id'].nunique():,} / {df.loc[test_idx, 'site_id'].nunique():,} sites",
        "Stations shuffled (seed 42) and dealt out until 70% / 10% / 20% of rows are reached, stored in the `split` column",
        "No station is in more than one set. Validation is for tuning and early stopping; the test set is touched once.",
    ))
    # time split: train up to cutoff-1 ... validation = years in between, test = after cutoff
    df["split_time"] = np.where(
        df["year"] > time_cutoff_year, "test",
        np.where(df["year"] >= time_val_start_year, "val", "train"),
    )
    tc = df["split_time"].value_counts()
    log.append(_log_entry(
        "Time split (secondary): train / val / test",
        f"{tc.get('train', 0):,} / {tc.get('val', 0):,} / {tc.get('test', 0):,} rows",
        f"Train up to {time_val_start_year - 1}, validation {time_val_start_year}-{time_cutoff_year}, test {time_cutoff_year + 1}+, stored in `split_time`",
        "Checks that the model still works on later years.",
    ))

    # 12. Final column order and save
    final_cols = (
        ["country", "site_id", "waterbody_type", "date", "year", "month"]
        + PARAM_COLS
        + ["n_genuine_params", "target_ccme_value", "target_ccme_class", "split", "split_time"]
    )
    final = df[final_cols].copy()
    final["date"] = final["date"].dt.strftime("%Y-%m-%d")
    final[PARAM_COLS + ["target_ccme_value"]] = final[PARAM_COLS + ["target_ccme_value"]].round(4)

    out_path = os.path.abspath(os.path.expanduser(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final.to_csv(out_path, index=False)
    log.append(_log_entry(
        "Output", f"{len(final):,} rows x {final.shape[1]} columns, {os.path.getsize(out_path) / 1e6:.1f} MB",
        f"Saved {os.path.basename(out_path)}", "The single processed dataset used by every later step.",
    ))
    return final, log, audit


def write_cleaning_log_md(log: Sequence[Mapping[str, Any]], path: str, header: str = "", footer: str = "") -> None:
    """Write the cleaning log as a Markdown table (docs/cleaning_log.md)."""
    lines = [header.rstrip(), "", "| Issue found | How many rows/values | Action | Why |", "| :--- | :--- | :--- | :--- |"]
    for e in log:
        cells = [str(e[k]).replace("|", "/").replace("\n", " ") for k in ("Issue found", "How many rows/values", "Action", "Why")]
        lines.append("| " + " | ".join(cells) + " |")
    if footer:
        lines += ["", footer.rstrip()]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
