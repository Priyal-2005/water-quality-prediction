"""
src/train_dl.py
Train and tune the Phase 2 MLP (PyTorch) on the same data, splits and inputs as Phase 1.

Rules (same as Phase 1):
  * every setting is chosen on VALIDATION RMSE only;
  * the test set is never scored here. Test predictions are only saved to disk so that
    the one-time evaluation (src/evaluate_dl.py) can score them once.

Regularisation switches (dropout, weight decay, early stopping, batch norm) are read from
src/regularisation.py, so ablations can change them one at a time.

Run from the project root:
    python -m src.train_dl                 # full tuning (default 12 random trials + the default setting)
    python -m src.train_dl --trials 4 --epochs 20   # quick smoke test

Outputs (results/dl/):
    experiments_dl.csv         every trial: settings, validation MAE / RMSE / R2, best epoch, time
    mlp_config.json            chosen settings and data description
    mlp_history.csv            train / validation loss per epoch of the chosen model
    predictions_mlp.csv        test predictions: row_id, y_true, y_pred (scored once, later)
    predictions_mlp_val.csv    validation predictions (for choosing a hybrid blend weight)
models/mlp.pt                  chosen weights (not committed, regenerate with this script)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import regularisation as reg
from src.dl_data import load_dl_data
from src.reproducibility import SEED, make_train_loader, set_seed

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "dl"
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------- model
def make_mlp(d_in, widths=(128, 64), dropout=0.2, batch_norm=False):
    """Small MLP for a table: [Linear -> (BatchNorm) -> ReLU -> Dropout] x depth -> Linear(1).

    The last layer has one neuron and no activation, because the score is a plain number.
    """
    layers, d = [], d_in
    for w in widths:
        layers.append(nn.Linear(d, w))
        if batch_norm:
            layers.append(nn.BatchNorm1d(w))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = w
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


@torch.no_grad()
def predict(model, X, y_mean, y_std, batch=8192):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32)
        out.append(model(xb).squeeze(1).numpy())
    return np.concatenate(out) * y_std + y_mean


# ---------------------------------------------------------------- one training run
def train_one(data, cfg, max_epochs=60, verbose=False):
    """Train one MLP. Early stopping and model choice use validation loss only."""
    set_seed(SEED)
    y_mean, y_std = float(data["y_train"].mean()), float(data["y_train"].std())

    # The target is standardised for stable training; predictions are converted back.
    train = dict(data)
    train["y_train"] = (data["y_train"] - y_mean) / y_std
    loader = make_train_loader(train, batch_size=cfg["batch_size"], seed=SEED)

    model = make_mlp(data["X_train"].shape[1], cfg["widths"], cfg["dropout"], cfg["batch_norm"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.HuberLoss(delta=1.0) if cfg["loss"] == "huber" else nn.MSELoss()

    Xv = torch.as_tensor(data["X_val"], dtype=torch.float32)
    yv_std = torch.as_tensor((data["y_val"] - y_mean) / y_std, dtype=torch.float32)

    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(Xv).squeeze(1), yv_std))
        history.append({"epoch": epoch, "train_loss": total / n, "val_mse_std": val_loss})
        if verbose:
            print(f"  epoch {epoch:3d} train {total / n:.4f} val {val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if reg.USE_EARLY_STOPPING and bad >= reg.EARLY_STOPPING_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, y_mean, y_std, best_epoch, pd.DataFrame(history)


def default_cfg():
    """Starting setting: regularisation values come from src/regularisation.py."""
    return {
        "widths": (128, 64),
        "dropout": reg.DROPOUT_RATE if reg.USE_DROPOUT else 0.0,
        "weight_decay": reg.WEIGHT_DECAY if reg.USE_WEIGHT_DECAY else 0.0,
        "batch_norm": reg.USE_BATCH_NORM,
        "lr": 1e-3,
        "loss": "mse",
        "batch_size": 512,
    }


def random_cfgs(n, seed=SEED):
    """Random search around the starting setting (fixed seed, so reproducible)."""
    rng = random.Random(seed)
    space = {
        "widths": [(64, 32), (128, 64), (256, 128), (128, 64, 32), (256, 128, 64)],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
        "lr": [3e-4, 1e-3, 3e-3],
        "loss": ["mse", "huber"],
        "batch_size": [256, 512, 1024],
    }
    cfgs, seen = [], {json.dumps(default_cfg(), default=list, sort_keys=True)}
    while len(cfgs) < n:
        c = {k: rng.choice(v) for k, v in space.items()}
        c["batch_norm"] = reg.USE_BATCH_NORM
        key = json.dumps(c, default=list, sort_keys=True)
        if key not in seen:
            seen.add(key)
            cfgs.append(c)
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12, help="random trials (the default setting is always run first)")
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs per trial")
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(4)

    data = load_dl_data(split_col="split")
    print("Train / val / test rows:", len(data["y_train"]), len(data["y_val"]), len(data["y_test"]))

    rows, best = [], None
    cfgs = [default_cfg()] + random_cfgs(args.trials)
    for i, cfg in enumerate(cfgs):
        t0 = time.time()
        model, y_mean, y_std, best_epoch, hist = train_one(data, cfg, max_epochs=args.epochs)
        m = regression_metrics(data["y_val"], predict(model, data["X_val"], y_mean, y_std))
        row = {"trial": i, "is_default": i == 0, **{k: (str(v) if k == "widths" else v) for k, v in cfg.items()},
               "best_epoch": best_epoch, "epochs_run": len(hist), **{f"val_{k}": v for k, v in m.items()},
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"trial {i:2d} val RMSE {m['rmse']:.3f} MAE {m['mae']:.3f} R2 {m['r2']:.3f} "
              f"epoch {best_epoch}/{len(hist)} {row['seconds']}s  {cfg}")
        if best is None or m["rmse"] < best["rmse"]:
            best = {"rmse": m["rmse"], "cfg": cfg, "model": model, "y_mean": y_mean, "y_std": y_std,
                    "hist": hist, "trial": i, "metrics": m}
        pd.DataFrame(rows).to_csv(RESULT_DIR / "experiments_dl.csv", index=False)

    # Mark the winner and save everything for the later steps.
    exp = pd.DataFrame(rows)
    exp["selected"] = exp["trial"] == best["trial"]
    exp.to_csv(RESULT_DIR / "experiments_dl.csv", index=False)

    model = best["model"]
    torch.save({"state_dict": model.state_dict(), "cfg": best["cfg"], "y_mean": best["y_mean"],
                "y_std": best["y_std"], "d_in": data["X_train"].shape[1]}, MODEL_DIR / "mlp.pt")
    best["hist"].to_csv(RESULT_DIR / "mlp_history.csv", index=False)

    pd.DataFrame({"row_id": data["row_id_val"], "y_true": data["y_val"],
                  "y_pred": predict(model, data["X_val"], best["y_mean"], best["y_std"])}
                 ).to_csv(RESULT_DIR / "predictions_mlp_val.csv", index=False)
    # Test predictions are saved but NOT scored here (the test set is used once, in evaluate_dl.py).
    pd.DataFrame({"row_id": data["row_id_test"], "y_true": data["y_test"],
                  "y_pred": predict(model, data["X_test"], best["y_mean"], best["y_std"])}
                 ).to_csv(RESULT_DIR / "predictions_mlp.csv", index=False)

    config = {"seed": SEED, "split": "split (by station)", "selected_trial": best["trial"],
              "selection_rule": "lowest validation RMSE", "n_trials": len(cfgs),
              "max_epochs": args.epochs, "early_stopping_patience": reg.EARLY_STOPPING_PATIENCE,
              "mlp": {k: (list(v) if k == "widths" else v) for k, v in best["cfg"].items()},
              "validation_metrics": best["metrics"], "n_features": len(data["feature_names"]),
              "feature_names": data["feature_names"]}
    (RESULT_DIR / "mlp_config.json").write_text(json.dumps(config, indent=2))
    print("\nChosen (validation RMSE %.3f, trial %d):" % (best["rmse"], best["trial"]), best["cfg"])


if __name__ == "__main__":
    main()
