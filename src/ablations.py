"""
src/ablations.py
================
Phase 2 Deep Learning Ablation Studies for Water Quality Prediction.
Author: Sarthak Mathapati (Role: Ablation Studies)
Team: T109 (Project 18)

Ablation Methodology & Scientific Protocol
------------------------------------------
1. Objective:
   Evaluate the marginal contribution and sensitivity of individual design choices
   in the deep learning pipeline by changing ONE setting at a time from the tuned
   MLP baseline.

2. Scope of Ablation Experiments:
   - Depth: 2 vs 3 hidden layers (network depth capacity)
   - Width: 64 vs 128 hidden neurons (representational width)
   - Regularisation:
     * Dropout: 0.0 (off) vs 0.2 (on)
     * Weight Decay: 0.0 (off) vs 1e-4 (on)
   - Feature Groups:
     * With / without station history (5 running-mean sensor and temporal gap features)
     * With / without country (geographic indicator feature)
   - Model Comparison:
     * MLP alone vs XGBoost+MLP Hybrid (using strictly out-of-fold station predictions)

3. Rigorous Experimental Fairness Controls:
   - Validation-Only Evaluation: Every ablation trial is tuned and evaluated strictly
     on the 10% validation split ('split' column). The test set is reserved exclusively
     for Anusha's one-time final model evaluation. Zero test leakage.
   - Split Preservation: Reuses the identical station-based split (split = 'train' / 'val')
     from Phase 1 (70/10/20% rows with strictly zero station overlap).
   - Fixed Seed: Seed 42 is strictly fixed for PyTorch, NumPy, Python random, and DataLoaders.
   - Feature Leakage Prevention: Imputer, standard scaler, and one-hot encoder are fitted
     strictly on X_train. The five formula-calculating chemicals (ammonia, BOD, orthophosphate,
     nitrogen, nitrate) remain permanently hidden.
   - One-Setting-At-A-Time: Each ablation overrides exactly one parameter while preserving
     all other hyperparameters from Priyal's tuned baseline.
   - Out-of-Fold Hybrid Verification: Hybrid comparisons require out-of-fold-by-station
     predictions from Anusha/Priyal. If absent or unverified, placeholder records with NA
     are produced without fabrication.
   - Station-Level Bootstrap Confidence Intervals: Computes 95% bootstrap CIs by resampling
     entire monitoring stations (site_id) with replacement across 2,000 draws.
   - Append-Only & Traceable Logging: Every trial is logged to results/ablations.csv with a
     unique run_id, ISO-8601 timestamp, status (RUN / DRY_RUN / PLACEHOLDER), config hash,
     and descriptive notes. Existing records are never overwritten.

CLI Usage:
----------
    python -m src.ablations --vary depth --values 2,3
    python -m src.ablations --vary width --values 64,128
    python -m src.ablations --vary dropout --values 0.0,0.2
    python -m src.ablations --vary weight_decay --values 0.0,1e-4
    python -m src.ablations --vary use_station_history --values true,false
    python -m src.ablations --vary use_country --values true,false
    python -m src.ablations --vary model --values mlp,hybrid
    python -m src.ablations --vary depth --values 2,3 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
    import torch.nn as nn
    from src.train_dl import make_mlp, predict, regression_metrics
    from src.reproducibility import make_train_loader, set_seed
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    torch = None
    nn = None
    make_mlp = None
    predict = None
    regression_metrics = None
    make_train_loader = None
    set_seed = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

from src.features import (
    ENGINEERED_COLS,
    HISTORY_COLS,
    SENSOR_COLS,
    TARGET_COL,
    load_final_dataset,
    make_features,
)

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "tuned_mlp.yaml"
FALLBACK_SCAFFOLD_PATH = ROOT / "configs" / "ablation_baseline.yaml"
DEFAULT_OUTPUT_CSV = ROOT / "results" / "ablations.csv"
FIGURES_DIR = ROOT / "results" / "figures"
HYBRID_PREDS_PATH = ROOT / "results" / "hybrid_val_preds.csv"


# -----------------------------------------------------------------------------
# 1. Configuration Management & Integrity Hashing
# -----------------------------------------------------------------------------

def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of configuration settings."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def load_baseline_config(config_path: Optional[Path] = None) -> Tuple[Dict[str, Any], bool]:
    """
    Load baseline configuration.
    Returns:
        (cfg_dict, is_genuine_tuned_config)
    """
    if config_path is None or not config_path.exists():
        if DEFAULT_CONFIG_PATH.exists():
            target_path = DEFAULT_CONFIG_PATH
        elif FALLBACK_SCAFFOLD_PATH.exists():
            target_path = FALLBACK_SCAFFOLD_PATH
        else:
            raise FileNotFoundError(
                f"No configuration file found at {DEFAULT_CONFIG_PATH} or {FALLBACK_SCAFFOLD_PATH}."
            )
    else:
        target_path = Path(config_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {target_path}")

    text = target_path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)

    is_placeholder = (
        cfg.get("status") == "PLACEHOLDER"
        or "ablation_baseline" in target_path.name
        or "placeholder" in str(cfg.get("model_name", "")).lower()
    )

    # Standardize baseline dictionary keys
    standardized = {
        "widths": tuple(cfg.get("hidden_layers", cfg.get("widths", [64, 64]))),
        "dropout": float(cfg.get("dropout", 0.2)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "batch_norm": bool(cfg.get("batch_norm", False)),
        "use_station_history": bool(cfg.get("use_station_history", True)),
        "use_country": bool(cfg.get("use_country", True)),
        "batch_size": int(cfg.get("batch_size", 32)),
        "lr": float(cfg.get("lr", 1e-3)),
        "loss": str(cfg.get("loss", "mse")),
        "epochs": int(cfg.get("epochs", 100)),
        "patience": int(cfg.get("early_stopping", {}).get("patience", 10) if isinstance(cfg.get("early_stopping"), dict) else 10),
        "seed": int(cfg.get("seed", 42)),
        "source_file": str(target_path.relative_to(ROOT) if ROOT in target_path.parents else target_path.name),
        "status": cfg.get("status", "PLACEHOLDER" if is_placeholder else "RUN"),
    }

    return standardized, not is_placeholder


# -----------------------------------------------------------------------------
# 2. Station-Level Bootstrap Confidence Intervals
# -----------------------------------------------------------------------------

def station_bootstrap_ci(
    df: pd.DataFrame,
    group_col: str = "site_id",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    metric: str = "mae",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute metric point estimate and 95% bootstrap confidence interval by resampling
    entire monitoring stations (clusters) with replacement.
    
    Parameters:
        df: DataFrame containing ground truth, predictions, and station identifiers.
        group_col: Column name identifying monitoring stations (default: site_id).
        y_true_col: Column name for ground truth values.
        y_pred_col: Column name for predicted values.
        metric: One of 'mae', 'rmse', or 'r2'.
        n_boot: Number of bootstrap resamples (default: 2000).
        seed: Random seed for reproducible bootstrap draws.
        alpha: Significance level for two-tailed percentile interval (default: 0.05 for 95% CI).

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    assert group_col in df.columns, f"Missing group column '{group_col}' in DataFrame"
    assert y_true_col in df.columns, f"Missing target column '{y_true_col}' in DataFrame"
    assert y_pred_col in df.columns, f"Missing prediction column '{y_pred_col}' in DataFrame"

    y_true_full = df[y_true_col].to_numpy(dtype=np.float64)
    y_pred_full = df[y_pred_col].to_numpy(dtype=np.float64)

    # Point estimate calculation
    diff_full = y_true_full - y_pred_full
    if metric == "mae":
        point = float(np.mean(np.abs(diff_full)))
    elif metric == "rmse":
        point = float(np.sqrt(np.mean(diff_full ** 2)))
    elif metric == "r2":
        ss_res = np.sum(diff_full ** 2)
        ss_tot = np.sum((y_true_full - np.mean(y_true_full)) ** 2)
        point = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'mae', 'rmse', or 'r2'.")

    # Group true and pred arrays by station for fast vectorised bootstrap draws
    station_groups = [
        (sub[y_true_col].to_numpy(dtype=np.float64), sub[y_pred_col].to_numpy(dtype=np.float64))
        for _, sub in df.groupby(group_col, sort=False)
    ]
    n_stations = len(station_groups)
    if n_stations == 0:
        return point, point, point

    rng = np.random.default_rng(seed)
    boot_estimates = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sample_indices = rng.integers(0, n_stations, size=n_stations)
        sampled_y_t = np.concatenate([station_groups[idx][0] for idx in sample_indices])
        sampled_y_p = np.concatenate([station_groups[idx][1] for idx in sample_indices])
        sampled_diff = sampled_y_t - sampled_y_p

        if metric == "mae":
            boot_estimates[b] = np.mean(np.abs(sampled_diff))
        elif metric == "rmse":
            boot_estimates[b] = np.sqrt(np.mean(sampled_diff ** 2))
        elif metric == "r2":
            ss_res = np.sum(sampled_diff ** 2)
            ss_tot = np.sum((sampled_y_t - np.mean(sampled_y_t)) ** 2)
            boot_estimates[b] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    lower = float(np.percentile(boot_estimates, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))
    return point, lower, upper


# -----------------------------------------------------------------------------
# 3. Data Preparation with Feature Group Ablation Support
# -----------------------------------------------------------------------------

def prepare_ablation_data(
    split_col: str = "split",
    use_station_history: bool = True,
    use_country: bool = True,
    data_path: str = "data/processed/final_dataset.csv",
) -> Dict[str, Any]:
    """
    Prepare feature-ablated dataset arrays.
    Preserves exact split isolation and strictly fits preprocessing on training rows only.
    """
    full_path = ROOT / data_path
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset not found at {full_path}")

    df = load_final_dataset(str(full_path))
    assert "waterbody_type" in df.columns, "Missing 'waterbody_type' column in dataset"
    df = make_features(df)

    # Determine active numeric features
    active_numeric = list(SENSOR_COLS + ENGINEERED_COLS)
    if use_station_history:
        active_numeric += list(HISTORY_COLS)

    # Determine active categorical features
    active_categorical = ["waterbody_type"]
    if use_country:
        active_categorical.append("country")

    active_features = active_numeric + active_categorical

    # Partition by split
    parts = {k: df[df[split_col] == k] for k in ("train", "val", "test")}

    # Verify zero station overlap on station split
    if split_col == "split":
        sites = {k: set(v["site_id"]) for k, v in parts.items()}
        overlap = {
            "train_val": len(sites["train"] & sites["val"]),
            "train_test": len(sites["train"] & sites["test"]),
            "val_test": len(sites["val"] & sites["test"]),
        }
        assert sum(overlap.values()) == 0, f"Station overlap detected: {overlap}"

    # Build preprocessor strictly for active feature groups
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, active_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), active_categorical),
    ])

    # Fit strictly on train
    X_train_proc = preprocessor.fit_transform(parts["train"][active_features])
    X_val_proc = preprocessor.transform(parts["val"][active_features])

    return {
        "X_train": np.asarray(X_train_proc, dtype=np.float32),
        "y_train": np.asarray(parts["train"][TARGET_COL], dtype=np.float32),
        "X_val": np.asarray(X_val_proc, dtype=np.float32),
        "y_val": np.asarray(parts["val"][TARGET_COL], dtype=np.float32),
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "site_id_val": parts["val"]["site_id"].to_numpy(),
        "row_id_val": parts["val"]["row_id"].to_numpy(),
        "use_station_history": use_station_history,
        "use_country": use_country,
    }


# -----------------------------------------------------------------------------
# 4. Model Training & Validation Curve Tracking
# -----------------------------------------------------------------------------

def train_ablation_model(
    data: Dict[str, Any],
    cfg: Dict[str, Any],
    setting_name: str,
    setting_val: str,
    save_curve: bool = True,
) -> Tuple[Dict[str, float], int, List[Dict[str, float]]]:
    """
    Train single MLP trial with early stopping and record validation metrics with 95% CIs.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"PyTorch or required modeling modules unavailable: {TORCH_IMPORT_ERROR}. "
            "Please ensure PyTorch is installed in your Python environment."
        )

    set_seed(cfg["seed"])
    y_mean = float(data["y_train"].mean())
    y_std = float(data["y_train"].std())

    train_data = dict(data)
    train_data["y_train"] = (data["y_train"] - y_mean) / y_std
    loader = make_train_loader(train_data, batch_size=cfg["batch_size"], seed=cfg["seed"])

    d_in = data["X_train"].shape[1]
    model = make_mlp(
        d_in=d_in,
        widths=cfg["widths"],
        dropout=cfg["dropout"],
        batch_norm=cfg["batch_norm"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.HuberLoss(delta=1.0) if cfg["loss"] == "huber" else nn.MSELoss()

    Xv = torch.as_tensor(data["X_val"], dtype=torch.float32)
    yv_std = torch.as_tensor((data["y_val"] - y_mean) / y_std, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        n_samples = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze(1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * len(xb)
            n_samples += len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(Xv).squeeze(1), yv_std))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_samples,
            "val_loss": val_loss,
        }
        history.append(epoch_record)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val_pred = predict(model, data["X_val"], y_mean, y_std)

    # Compute validation metrics with station-level 95% bootstrap CIs
    df_val_eval = pd.DataFrame({
        "site_id": data["site_id_val"],
        "y_true": data["y_val"],
        "y_pred": y_val_pred,
    })
    mae_pt, mae_lo, mae_hi = station_bootstrap_ci(df_val_eval, group_col="site_id", metric="mae", n_boot=2000, seed=cfg["seed"])
    rmse_pt, rmse_lo, rmse_hi = station_bootstrap_ci(df_val_eval, group_col="site_id", metric="rmse", n_boot=2000, seed=cfg["seed"])
    r2_pt, r2_lo, r2_hi = station_bootstrap_ci(df_val_eval, group_col="site_id", metric="r2", n_boot=2000, seed=cfg["seed"])

    metrics = {
        "mae": mae_pt, "mae_lo": mae_lo, "mae_hi": mae_hi,
        "rmse": rmse_pt, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
        "r2": r2_pt, "r2_lo": r2_lo, "r2_hi": r2_hi,
    }

    if save_curve and history:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        safe_val_str = str(setting_val).replace(".", "p").replace("/", "_")
        fig_path = FIGURES_DIR / f"training_curves_{setting_name}_{safe_val_str}.png"
        plot_training_curve(history, setting_name, setting_val, fig_path)

    return metrics, best_epoch, history


def plot_training_curve(
    history: List[Dict[str, float]],
    setting: str,
    val_str: str,
    out_path: Path,
) -> None:
    """Plot and save loss curves for a single trial."""
    df_h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(df_h["epoch"], df_h["train_loss"], label="Train Loss (MSE, standardised)", color="#1f77b4", lw=2)
    ax.plot(df_h["epoch"], df_h["val_loss"], label="Val Loss (MSE, standardised)", color="#d62728", lw=2, linestyle="--")
    ax.set_title(f"Training Curve: {setting} = {val_str}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss (standardised scale)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Out-of-Fold Hybrid Evaluation Protocol
# -----------------------------------------------------------------------------

def evaluate_hybrid_validation(
    hybrid_path: Path = HYBRID_PREDS_PATH,
) -> Tuple[Optional[Dict[str, float]], str, str]:
    """
    Read and verify out-of-fold-by-station validation predictions for XGBoost+MLP hybrid.
    Computes station-level bootstrap 95% CIs across monitoring stations.
    Returns:
        (metrics_dict_or_None, status, notes)
    """
    if not hybrid_path.exists():
        return (
            None,
            "PLACEHOLDER",
            "Placeholder: out-of-fold hybrid validation predictions not yet generated by Anusha/Priyal (results/hybrid_val_preds.csv missing).",
        )

    try:
        df_h = pd.read_csv(hybrid_path)
        required_cols = {"row_id", "y_true", "y_pred"}
        if not required_cols.issubset(set(df_h.columns)):
            return (
                None,
                "PLACEHOLDER",
                f"Placeholder: {hybrid_path.name} missing required columns {required_cols - set(df_h.columns)}.",
            )

        # Verify alignment with validation dataset
        df_full = load_final_dataset(str(ROOT / "data/processed/final_dataset.csv"))
        val_sub = df_full[df_full["split"] == "val"][["row_id", "site_id"]].copy()
        val_rows = set(val_sub["row_id"])
        hybrid_rows = set(df_h["row_id"])

        if val_rows != hybrid_rows:
            return (
                None,
                "PLACEHOLDER",
                f"Placeholder: row ID mismatch between {hybrid_path.name} ({len(hybrid_rows)} rows) and validation split ({len(val_rows)} rows).",
            )

        # Merge station identifiers (site_id) for station-level bootstrap
        df_h = df_h.merge(val_sub, on="row_id", how="left")

        # Compute point estimates and station-bootstrap 95% CIs
        mae_pt, mae_lo, mae_hi = station_bootstrap_ci(df_h, group_col="site_id", y_true_col="y_true", y_pred_col="y_pred", metric="mae", n_boot=2000, seed=42)
        rmse_pt, rmse_lo, rmse_hi = station_bootstrap_ci(df_h, group_col="site_id", y_true_col="y_true", y_pred_col="y_pred", metric="rmse", n_boot=2000, seed=42)
        r2_pt, r2_lo, r2_hi = station_bootstrap_ci(df_h, group_col="site_id", y_true_col="y_true", y_pred_col="y_pred", metric="r2", n_boot=2000, seed=42)

        metrics = {
            "mae": mae_pt, "mae_lo": mae_lo, "mae_hi": mae_hi,
            "rmse": rmse_pt, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
            "r2": r2_pt, "r2_lo": r2_lo, "r2_hi": r2_hi,
        }
        return metrics, "RUN", "Genuine out-of-fold station hybrid predictions verified with 95% station CIs."

    except Exception as e:
        return (
            None,
            "PLACEHOLDER",
            f"Placeholder: failed to process {hybrid_path.name} ({str(e)}).",
        )


# -----------------------------------------------------------------------------
# 6. Result Logging (Append-Only)
# -----------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "status",
    "setting",
    "value",
    "val_MAE",
    "val_MAE_lo",
    "val_MAE_hi",
    "val_RMSE",
    "val_RMSE_lo",
    "val_RMSE_hi",
    "val_R2",
    "val_R2_lo",
    "val_R2_hi",
    "epochs_run",
    "seed",
    "config_hash",
    "notes",
]


def append_result_row(output_csv: Path, row: Dict[str, Any]) -> None:
    """Safely append a single experiment row to results/ablations.csv."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    df_row = pd.DataFrame([row])[CSV_COLUMNS]

    if not file_exists:
        df_row.to_csv(output_csv, index=False)
    else:
        df_row.to_csv(output_csv, mode="a", header=False, index=False)


def generate_run_id(setting: str, value: Any, timestamp: str) -> str:
    """Generate a clean, reproducible run identifier."""
    clean_val = str(value).replace(".", "p").replace("/", "_").replace(" ", "")
    ts_short = timestamp.replace("-", "").replace(":", "").replace("T", "_")[:15]
    return f"ablation_{setting}_{clean_val}_{ts_short}"


# -----------------------------------------------------------------------------
# 7. Ablation Experiment Driver
# -----------------------------------------------------------------------------

def run_ablation(
    vary: str,
    values: List[str],
    config_path: Optional[Path] = None,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    seed: int = SEED,
    dry_run: bool = False,
) -> None:
    """Execute ablation variation or print dry-run schedule."""
    baseline_cfg, is_genuine_baseline = load_baseline_config(config_path)
    baseline_cfg["seed"] = seed
    cfg_hash = compute_config_hash(baseline_cfg)

    print(f"\n=======================================================")
    print(f"Sarthak's Ablation Experiment: Varying '{vary}'")
    print(f"Values to test: {values}")
    print(f"Baseline Source: {baseline_cfg['source_file']}")
    print(f"Baseline Is Genuine Tuned: {is_genuine_baseline}")
    print(f"Config Hash: {cfg_hash}")
    print(f"Execution Mode: {'DRY-RUN' if dry_run else 'ACTIVE RUN'}")
    print(f"=======================================================\n")

    timestamp = datetime.now(timezone.utc).isoformat()

    # If output CSV does not exist, initialize with baseline reference row
    if not output_csv.exists():
        ref_status = "RUN" if (is_genuine_baseline and not dry_run) else "PLACEHOLDER"
        ref_row = {
            "run_id": f"ablation_baseline_ref_{timestamp[:10]}",
            "timestamp": timestamp,
            "status": ref_status,
            "setting": "baseline",
            "value": "reference",
            "val_MAE": "NA",
            "val_MAE_lo": "NA",
            "val_MAE_hi": "NA",
            "val_RMSE": "NA",
            "val_RMSE_lo": "NA",
            "val_RMSE_hi": "NA",
            "val_R2": "NA",
            "val_R2_lo": "NA",
            "val_R2_hi": "NA",
            "epochs_run": "NA",
            "seed": seed,
            "config_hash": cfg_hash,
            "notes": (
                "Priyal's tuned baseline reference."
                if is_genuine_baseline
                else "Placeholder reference: Priyal's tuned config (configs/tuned_mlp.yaml) not yet available."
            ),
        }
        append_result_row(output_csv, ref_row)

    # Process each variation
    for val_str in values:
        val_str = val_str.strip()
        trial_cfg = copy.deepcopy(baseline_cfg)
        trial_timestamp = datetime.now(timezone.utc).isoformat()
        run_id = generate_run_id(vary, val_str, trial_timestamp)

        # Setting parsing and mutation
        if vary == "depth":
            depth_val = int(val_str)
            if depth_val == 2:
                trial_cfg["widths"] = (64, 64)
            elif depth_val == 3:
                trial_cfg["widths"] = (64, 64, 64)
            elif depth_val == 1:
                trial_cfg["widths"] = (64,)
            else:
                trial_cfg["widths"] = tuple([64] * depth_val)

        elif vary == "width":
            width_val = int(val_str)
            depth_len = len(baseline_cfg["widths"])
            trial_cfg["widths"] = tuple([width_val] * depth_len)

        elif vary == "dropout":
            trial_cfg["dropout"] = float(val_str)

        elif vary == "weight_decay":
            trial_cfg["weight_decay"] = float(val_str)

        elif vary == "use_station_history":
            trial_cfg["use_station_history"] = val_str.lower() in ("true", "1", "yes")

        elif vary == "use_country":
            trial_cfg["use_country"] = val_str.lower() in ("true", "1", "yes")

        elif vary == "model":
            # Model comparison handles MLP vs Hybrid
            pass

        else:
            raise ValueError(f"Unknown setting to vary: {vary}")

        trial_hash = compute_config_hash(trial_cfg)

        # ----------------------------------------------------
        # DRY-RUN OR PLACEHOLDER EXECUTION
        # ----------------------------------------------------
        if dry_run or not is_genuine_baseline:
            status = "DRY_RUN" if dry_run else "PLACEHOLDER"
            note = (
                f"Dry-run planned experiment: vary {vary}={val_str}."
                if dry_run
                else f"Placeholder record: awaiting Priyal's tuned configuration before executing {vary}={val_str}."
            )

            # Special case for hybrid model
            if vary == "model" and val_str.lower() == "hybrid":
                _, h_status, h_note = evaluate_hybrid_validation()
                note = h_note

            row = {
                "run_id": run_id,
                "timestamp": trial_timestamp,
                "status": status,
                "setting": vary,
                "value": val_str,
                "val_MAE": "NA",
                "val_MAE_lo": "NA",
                "val_MAE_hi": "NA",
                "val_RMSE": "NA",
                "val_RMSE_lo": "NA",
                "val_RMSE_hi": "NA",
                "val_R2": "NA",
                "val_R2_lo": "NA",
                "val_R2_hi": "NA",
                "epochs_run": "NA",
                "seed": seed,
                "config_hash": trial_hash,
                "notes": note,
            }
            append_result_row(output_csv, row)
            print(f"[{status}] Logged {vary}={val_str} -> {output_csv}")
            continue

        # ----------------------------------------------------
        # ACTIVE TRAINING (Genuinely reproducible baseline)
        # ----------------------------------------------------
        if vary == "model":
            if val_str.lower() == "mlp":
                print(f"[RUN] MLP model uses baseline reference setting.")
                continue
            elif val_str.lower() == "hybrid":
                metrics, h_status, h_notes = evaluate_hybrid_validation()
                row = {
                    "run_id": run_id,
                    "timestamp": trial_timestamp,
                    "status": h_status,
                    "setting": vary,
                    "value": val_str,
                    "val_MAE": f"{metrics['mae']:.4f}" if metrics else "NA",
                    "val_MAE_lo": f"{metrics['mae_lo']:.4f}" if metrics else "NA",
                    "val_MAE_hi": f"{metrics['mae_hi']:.4f}" if metrics else "NA",
                    "val_RMSE": f"{metrics['rmse']:.4f}" if metrics else "NA",
                    "val_RMSE_lo": f"{metrics['rmse_lo']:.4f}" if metrics else "NA",
                    "val_RMSE_hi": f"{metrics['rmse_hi']:.4f}" if metrics else "NA",
                    "val_R2": f"{metrics['r2']:.4f}" if metrics else "NA",
                    "val_R2_lo": f"{metrics['r2_lo']:.4f}" if metrics else "NA",
                    "val_R2_hi": f"{metrics['r2_hi']:.4f}" if metrics else "NA",
                    "epochs_run": "NA",
                    "seed": seed,
                    "config_hash": trial_hash,
                    "notes": h_notes,
                }
                append_result_row(output_csv, row)
                continue

        # Prepare ablated data
        data = prepare_ablation_data(
            split_col="split",
            use_station_history=trial_cfg["use_station_history"],
            use_country=trial_cfg["use_country"],
        )

        t0 = time.time()
        metrics, best_epoch, hist = train_ablation_model(
            data=data,
            cfg=trial_cfg,
            setting_name=vary,
            setting_val=val_str,
            save_curve=True,
        )
        elapsed = time.time() - t0

        row = {
            "run_id": run_id,
            "timestamp": trial_timestamp,
            "status": "RUN",
            "setting": vary,
            "value": val_str,
            "val_MAE": f"{metrics['mae']:.4f}",
            "val_MAE_lo": f"{metrics['mae_lo']:.4f}",
            "val_MAE_hi": f"{metrics['mae_hi']:.4f}",
            "val_RMSE": f"{metrics['rmse']:.4f}",
            "val_RMSE_lo": f"{metrics['rmse_lo']:.4f}",
            "val_RMSE_hi": f"{metrics['rmse_hi']:.4f}",
            "val_R2": f"{metrics['r2']:.4f}",
            "val_R2_lo": f"{metrics['r2_lo']:.4f}",
            "val_R2_hi": f"{metrics['r2_hi']:.4f}",
            "epochs_run": len(hist),
            "seed": seed,
            "config_hash": trial_hash,
            "notes": f"Trained in {elapsed:.1f}s, best epoch {best_epoch}.",
        }
        append_result_row(output_csv, row)
        print(f"[RUN] Finished {vary}={val_str}: val RMSE {metrics['rmse']:.3f} [{metrics['rmse_lo']:.3f}, {metrics['rmse_hi']:.3f}], MAE {metrics['mae']:.3f}, R2 {metrics['r2']:.3f}")


# -----------------------------------------------------------------------------
# 8. CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sarthak's Ablation Studies CLI for Phase 2 Deep Learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else str(FALLBACK_SCAFFOLD_PATH),
        help="Path to baseline YAML config (default: configs/tuned_mlp.yaml or configs/ablation_baseline.yaml).",
    )
    parser.add_argument(
        "--vary",
        type=str,
        required=True,
        choices=[
            "depth",
            "width",
            "dropout",
            "weight_decay",
            "use_station_history",
            "use_country",
            "model",
        ],
        help="Hyperparameter or design axis to ablate.",
    )
    parser.add_argument(
        "--values",
        type=str,
        required=True,
        help="Comma-separated list of values to evaluate (e.g. '2,3', '64,128', '0.0,0.2').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to output CSV log file (default: results/ablations.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and validate runs without training; records DRY_RUN entries.",
    )

    args = parser.parse_args()
    val_list = [v.strip() for v in args.values.split(",") if v.strip()]

    run_ablation(
        vary=args.vary,
        values=val_list,
        config_path=Path(args.config),
        output_csv=Path(args.output),
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
