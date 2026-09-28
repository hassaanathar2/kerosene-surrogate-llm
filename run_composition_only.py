#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Composition-only regression baseline for kerosene surrogate fuels.

The regression head sees only the mole fractions of the surrogate components:

    X = [x_1, x_2, ..., x_N]

No molecular embeddings are used. This is the ablation that any
representation-learning claim needs: if an embedding-based representation
(ChemBERTa, MolFormer, TinyLlama -- see run_chemberta.py / run_molformer.py /
run_tinyllama.py) cannot clearly beat this baseline, that is expected rather
than a sign of a broken pipeline whenever every sample is built from the same
fixed set of molecules -- in that case, a per-component embedding is a fixed
function of composition and can match, but not exceed, composition alone. It
becomes informative on its own once the set of molecules varies between
samples (e.g. combined Jet A / SAF systems).

To make the comparison with the embedding-based scripts meaningful, this
script uses the identical protocol:

  * the same train/test split (--split_seed, --test_size)
  * the same validation split carved out of the TRAINING data only (15%),
    used for early stopping -- the test set is touched exactly once
  * the identical regression-head architecture and optimiser settings as
    run_chemberta.py, run_molformer.py and run_tinyllama.py (128-64 MLP,
    dropout 0.10, AdamW lr=1e-3, weight_decay=1e-4, batch=64, patience=80)
  * the same five seeds, reported as mean +/- std

Auto-detects composition columns for RP-3 (4-component), Jet A / JETA5
(5-component) or a 5-component RP-3 variant with 1,3,5-trimethylbenzene --
whichever are present in your Excel file. No internet and no GPU needed --
only numpy/pandas/scikit-learn/torch (CPU is plenty fast here; this script
trains in seconds to a couple of minutes, unlike the embedding-based ones).

Requires:
    pip install torch scikit-learn pandas numpy matplotlib openpyxl

Run (use the SAME --excel, --split_seed, --test_size, --seeds as your
run_chemberta.py / run_molformer.py / run_tinyllama.py runs, so the four
result rows are directly comparable):
    python run_composition_only.py --excel "surrogate_compositions_6000.xlsx"
"""
from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Component names recognised in the input Excel file, mapped to the canonical
# label used in the manuscript's tables. Matching is case/whitespace-
# insensitive (see `norm`). The same set of names is recognised identically
# by run_chemberta.py, run_molformer.py and run_tinyllama.py, so the same
# Excel file is auto-detected the same way by all four scripts.
LIBRARY_LABELS = {
    "tetradecane": "Tetradecane", "iso-dodecane": "Iso-dodecane", "isododecane": "Iso-dodecane",
    "n-pch": "n-PCH", "toluene": "Toluene", "n-dodecane": "n-Dodecane",
    "iso-cetane": "Iso-cetane", "isocetane": "Iso-cetane",
    "iso-octane": "Iso-octane", "isooctane": "Iso-octane",
    "decalin": "Decalin", "135tmb": "135TMB", "1,3,5-trimethylbenzene": "135TMB",
}

TARGET_ALIASES = {
    "molecular weight": "Molecular Weight",
    "lower heating value": "Lower Heating Value",
    "cetane number": "Cetane Number",
    "density": "Density",
    "viscosity": "Viscosity",
    "threshold sooting index": "Threshold Sooting Index",
    "h/c": "H/C",
}


def norm(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s).strip().lower())


def detect_columns(df: pd.DataFrame):
    col_by_norm = {norm(c): c for c in df.columns}
    input_cols, seen_labels = [], set()
    for key, label in LIBRARY_LABELS.items():
        nk = norm(key)
        if nk in col_by_norm and label not in seen_labels:
            input_cols.append(col_by_norm[nk])
            seen_labels.add(label)
    target_cols = []
    for key, canon in TARGET_ALIASES.items():
        nk = norm(key)
        if nk in col_by_norm:
            target_cols.append(col_by_norm[nk])
    if not input_cols:
        raise ValueError(
            "No recognised composition columns found. Expected some subset of: "
            + ", ".join(sorted(set(LIBRARY_LABELS.values())))
            + f"\nActual columns: {list(df.columns)}"
        )
    return input_cols, target_cols


def composition_to_fraction(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    if np.nanmedian(s) > 2.0:
        x = x / 100.0
        s = x.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
    return x / s


class Head(nn.Module):
    """128-64 MLP with dropout and a linear output layer. Identical
    architecture to run_chemberta.py / run_molformer.py / run_tinyllama.py --
    only the input dimension differs, since there are no embeddings here."""
    def __init__(self, input_dim, output_dim, hidden1=128, hidden2=64, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_head(X_tr, Y_tr, X_val, Y_val, seed, device, epochs=1000, batch_size=64,
               lr=1e-3, weight_decay=1e-4, patience=80):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Head(X_tr.shape[1], Y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                       torch.tensor(Y_tr, dtype=torch.float32)),
                         batch_size=batch_size, shuffle=True, generator=g)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best, best_state, bad = float("inf"), None, 0
    hist = {"epoch": [], "train_loss": [], "val_loss": []}
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        hist["epoch"].append(epoch)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(vl)
        if epoch == 1 or epoch % 50 == 0:
            print(f"    epoch {epoch:4d} | train_loss={hist['train_loss'][-1]:.6f} | val_loss={vl:.6f}")
        if vl < best - 1e-8:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    Early stopping at epoch {epoch}. Best val_loss={best:.6f}")
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["stopped_epoch"] = epoch
    return model, hist


def calc_metrics(y_true, y_pred, target_cols):
    rows = []
    for i, t in enumerate(target_cols):
        yt, yp = y_true[:, i], y_pred[:, i]
        mse = mean_squared_error(yt, yp)
        rows.append({
            "Property": t, "MSE": mse, "RMSE": math.sqrt(mse),
            "MAE": mean_absolute_error(yt, yp),
            "MRE_percent": float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-12)) * 100.0),
            "R2": r2_score(yt, yp),
        })
    df = pd.DataFrame(rows)
    avg = df.drop(columns="Property").mean().to_dict()
    avg["Property"] = "Average"
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--output", default="composition_only_outputs")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_fraction", type=float, default=0.15, help="from the TRAIN split only")
    ap.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--split_seed", type=int, default=42,
                     help="must match the split_seed used for the chemberta/molformer/tinyllama runs")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_excel(args.excel, sheet_name=args.sheet)
    df.columns = [str(c).strip() for c in df.columns]
    input_cols, target_cols = detect_columns(df)
    print(f"Detected composition columns: {input_cols}")
    print(f"Detected target columns:      {target_cols}")
    if not target_cols:
        raise SystemExit("No target property columns found -- cannot train/evaluate a regression head.")

    df = df.dropna(subset=input_cols + target_cols).reset_index(drop=True)
    x_frac = composition_to_fraction(df[input_cols].to_numpy(float))
    y = df[target_cols].to_numpy(float)

    # X = mole fractions only -- no embeddings, no PCA
    X_full = x_frac.copy()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    X_train_full, X_test = X_full[train_idx], X_full[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    fit_idx, val_idx = train_test_split(np.arange(len(X_train_full)), test_size=args.val_fraction,
                                         random_state=0, shuffle=True)

    xs = StandardScaler().fit(X_train_full[fit_idx])
    ys = StandardScaler().fit(y_train_full[fit_idx])
    A_fit, A_val, A_te = xs.transform(X_train_full[fit_idx]), xs.transform(X_train_full[val_idx]), xs.transform(X_test)
    B_fit, B_val = ys.transform(y_train_full[fit_idx]), ys.transform(y_train_full[val_idx])

    print(f"\nSamples: fit={len(fit_idx)}  val={len(val_idx)}  test={len(X_test)}")
    print(f"Regression-head input dimension: {A_fit.shape[1]}  (composition only, no embeddings)")

    all_seeds = []
    for si, seed in enumerate(args.seeds):
        print(f"\n--- seed {seed} ---")
        t0 = time.time()
        model_head, hist = train_head(A_fit, B_fit, A_val, B_val, seed, device, epochs=args.epochs)
        print(f"  trained in {time.time()-t0:.1f}s")
        with torch.no_grad():
            y_pred = ys.inverse_transform(model_head(torch.tensor(A_te, dtype=torch.float32, device=device)).cpu().numpy())
        m = calc_metrics(y_test, y_pred, target_cols)
        m.insert(0, "seed", seed)
        all_seeds.append(m)
        avg = m[m.Property == "Average"].iloc[0]
        print(f"  R2={avg.R2:.6f}  RMSE={avg.RMSE:.5f}  MRE={avg.MRE_percent:.4f}%")

        if si == 0:
            pred_df = pd.DataFrame(df.iloc[test_idx][input_cols].values, columns=input_cols)
            for j, t in enumerate(target_cols):
                pred_df[f"Actual_{t}"] = y_test[:, j]
                pred_df[f"Predicted_{t}"] = y_pred[:, j]
                pred_df[f"Residual_{t}"] = y_pred[:, j] - y_test[:, j]
            pred_df.to_csv(out / f"predictions_seed{seed}.csv", index=False)
            pd.DataFrame(hist).to_csv(out / f"loss_history_seed{seed}.csv", index=False)
            torch.save({"state_dict": model_head.state_dict()}, out / f"head_seed{seed}.pt")

            plt.figure(figsize=(7, 5))
            plt.plot(hist["epoch"], hist["train_loss"], label="Training loss")
            plt.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
            plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE (standardised targets)")
            plt.legend(); plt.title("Composition-only head training"); plt.tight_layout()
            plt.savefig(out / "loss_curve.png", dpi=300); plt.close()

            for j, t in enumerate(target_cols):
                lo = min(y_test[:, j].min(), y_pred[:, j].min())
                hi = max(y_test[:, j].max(), y_pred[:, j].max())
                plt.figure(figsize=(5.5, 5.5))
                plt.scatter(y_test[:, j], y_pred[:, j], s=14, alpha=0.6)
                plt.plot([lo, hi], [lo, hi], "--", color="k")
                plt.xlabel(f"Actual {t}"); plt.ylabel(f"Predicted {t}")
                plt.title(f"Composition-only: {t}"); plt.tight_layout()
                plt.savefig(out / f"parity_{re.sub(r'[^A-Za-z0-9]+','_',t)}.png", dpi=300); plt.close()

    all_df = pd.concat(all_seeds, ignore_index=True)
    all_df.to_csv(out / "metrics_per_seed.csv", index=False)
    cols = ["MSE", "RMSE", "MAE", "MRE_percent", "R2"]
    summary = all_df.groupby("Property", sort=False)[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "metrics_mean_std.csv", index=False)

    print("\n=== FINAL (mean +/- std across seeds) ===")
    print(summary.to_string(index=False))
    print(f"\nAll outputs saved in: {out.resolve()}")
    print("\nNOTE: compare this row-for-row against your run_chemberta.py, run_molformer.py "
          "and run_tinyllama.py outputs -- make sure --split_seed and --test_size matched "
          "across all four runs, or the test sets differ and the comparison is not apples-to-apples.")


if __name__ == "__main__":
    main()
