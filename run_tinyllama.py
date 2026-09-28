#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TinyLlama-based regression pipeline for kerosene/RP-3/Jet-A surrogate datasets.

Unlike run_chemberta.py / run_molformer.py, which embed each component's
SMILES string once, this script builds one structured text prompt per
*sample* -- composition, component identities, SMILES, and a task
description, never the target property values -- and encodes each prompt
once with a frozen TinyLlama (masked mean pooling over the last hidden
state). Because there is one embedding per sample rather than per component,
its PCA is fitted on the TRAINING prompt embeddings only, and test-row
embeddings are projected with that fitted PCA -- fitting on all samples
before the split would leak test information into the projection. The
regression input is

    X = [x_1 ... x_N, pc_1 ... pc_m]

The default of 20 PCA components (--pca_components) is a sample-level
budget, distinct from run_chemberta.py / run_molformer.py's component-level
PCA (capped at N_components - 1, since those are fitted on only a handful of
pure-component embeddings) -- the two are not meant to use the same number of
components, only the same regression head, which they do: the identical
128-64 MLP used by run_composition_only.py and run_molformer.py. Early
stopping uses a validation split carved out of the TRAINING split only; the
test set is touched exactly once.

Auto-detects composition columns for RP-3 (4-component), Jet A / JETA5
(5-component) or a 5-component RP-3 variant with 1,3,5-trimethylbenzene --
whichever are present in your Excel file.

Requires (on YOUR machine, with internet -- TinyLlama is a ~1.1B parameter
model, so a GPU is strongly recommended; on CPU use a small
--embedding_batch_size, e.g. 1 or 2):
    pip install torch transformers scikit-learn pandas numpy matplotlib openpyxl

Run:
    python run_tinyllama.py --excel "surrogate_compositions_6000.xlsx"
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
from transformers import AutoModel, AutoTokenizer

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LIBRARY = {
    "tetradecane":   {"label": "Tetradecane",   "smiles": "CCCCCCCCCCCCCC"},
    "iso-dodecane":  {"label": "Iso-dodecane",  "smiles": "CC(C)(C)CC(C)CC(C)(C)C"},
    "isododecane":   {"label": "Iso-dodecane",  "smiles": "CC(C)(C)CC(C)CC(C)(C)C"},
    "n-pch":         {"label": "n-PCH",         "smiles": "CCCC1CCCCC1"},
    "toluene":       {"label": "Toluene",       "smiles": "Cc1ccccc1"},
    "n-dodecane":    {"label": "n-Dodecane",    "smiles": "CCCCCCCCCCCC"},
    "iso-cetane":    {"label": "Iso-cetane",    "smiles": "CC(C)(C)CC(C)(C)CC(C)CC(C)(C)C"},
    "isocetane":     {"label": "Iso-cetane",    "smiles": "CC(C)(C)CC(C)(C)CC(C)CC(C)(C)C"},
    "iso-octane":    {"label": "Iso-octane",    "smiles": "CC(C)CC(C)(C)C"},
    "isooctane":     {"label": "Iso-octane",    "smiles": "CC(C)CC(C)(C)C"},
    "decalin":       {"label": "Decalin",       "smiles": "C1CCC2CCCCC2C1"},
    "135tmb":        {"label": "135TMB",        "smiles": "Cc1cc(C)cc(C)c1"},
    "1,3,5-trimethylbenzene": {"label": "135TMB", "smiles": "Cc1cc(C)cc(C)c1"},
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
    input_cols, smiles_dict = [], {}
    for key, info in LIBRARY.items():
        nk = norm(key)
        if nk in col_by_norm:
            col = col_by_norm[nk]
            if info["label"] not in smiles_dict:
                input_cols.append(col)
                smiles_dict[col] = info["smiles"]
    target_cols = []
    for key, canon in TARGET_ALIASES.items():
        nk = norm(key)
        if nk in col_by_norm:
            target_cols.append(col_by_norm[nk])
    if not input_cols:
        raise ValueError(
            "No recognised composition columns found. Expected some subset of: "
            + ", ".join(sorted({v['label'] for v in LIBRARY.values()}))
            + f"\nActual columns: {list(df.columns)}"
        )
    return input_cols, smiles_dict, target_cols


def composition_to_fraction(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    if np.nanmedian(s) > 2.0:
        x = x / 100.0
        s = x.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
    return x / s


def make_prompt(row_percent: dict, input_cols: list, smiles_dict: dict) -> str:
    """Composition + molecular identity + task description. NEVER includes target values."""
    comp_lines = "\n".join(f"{c} = {row_percent[c]:.6f} mole percent." for c in input_cols)
    smiles_lines = "\n".join(f"{c}: {smiles_dict[c]}." for c in input_cols)
    return (
        "Aviation kerosene surrogate mixture for numerical property regression.\n\n"
        f"Composition:\n{comp_lines}\n\n"
        f"Molecular structures in SMILES:\n{smiles_lines}\n\n"
        "Task representation:\nCreate a latent representation for predicting continuous physicochemical "
        "properties of this surrogate fuel mixture. Important factors include composition, hydrocarbon chain "
        "length, branching, cyclic structure, aromaticity, and hydrocarbon type."
    )


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed_prompts(prompts, tokenizer, model, device, max_length=256, batch_size=2):
    out = []
    n = len(prompts)
    for i in range(0, n, batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            res = model(**enc)
        hidden = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
        pooled = mean_pool(hidden, enc["attention_mask"])
        out.append(pooled.float().cpu().numpy())
        if (i // batch_size) % 25 == 0:
            print(f"    encoded {min(i+batch_size, n)}/{n}")
    return np.vstack(out)


class Head(nn.Module):
    """128-64 MLP with dropout and a linear output layer. Identical
    architecture to run_chemberta.py, run_molformer.py and
    run_composition_only.py."""
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
    ap.add_argument("--output", default="tinyllama_outputs")
    ap.add_argument("--pca_components", type=int, default=20)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_fraction", type=float, default=0.15, help="from the TRAIN split only")
    ap.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--embedding_batch_size", type=int, default=2, help="use 1-2 on CPU")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--force_reembed", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_excel(args.excel, sheet_name=args.sheet)
    df.columns = [str(c).strip() for c in df.columns]
    input_cols, smiles_dict, target_cols = detect_columns(df)
    print(f"Detected composition columns: {input_cols}")
    print(f"Detected target columns:      {target_cols}")
    if not target_cols:
        raise SystemExit("No target property columns found -- cannot train/evaluate a regression head.")

    df = df.dropna(subset=input_cols + target_cols).reset_index(drop=True)
    x_frac = composition_to_fraction(df[input_cols].to_numpy(float))
    y = df[target_cols].to_numpy(float)
    pct = 100.0 * x_frac
    prompts = [make_prompt(dict(zip(input_cols, pct[i])), input_cols, smiles_dict) for i in range(len(df))]
    pd.DataFrame({"prompt": prompts}).to_csv(out / "prompts.csv", index=False)
    print("\nExample prompt:\n" + "-" * 70 + f"\n{prompts[0]}\n" + "-" * 70)

    print(f"\nLoading TinyLlama: {MODEL_NAME}  (this can take a while, and is slow on CPU)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    cache_path = out / "prompt_embeddings.npy"
    if cache_path.exists() and not args.force_reembed:
        print(f"Loading cached prompt embeddings: {cache_path}")
        E_full = np.load(cache_path)
    else:
        E_full = embed_prompts(prompts, tokenizer, model, device, args.max_length, args.embedding_batch_size)
        np.save(cache_path, E_full)
    print(f"Prompt embedding dim: {E_full.shape[1]}")

    # split BEFORE PCA and BEFORE scaling
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.split_seed, shuffle=True)
    x_train, x_test = x_frac[train_idx], x_frac[test_idx]
    E_train, E_test = E_full[train_idx], E_full[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    # PCA fitted on TRAINING prompt embeddings only
    n_pca = min(args.pca_components, E_train.shape[0], E_train.shape[1])
    pca = PCA(n_components=n_pca, random_state=0).fit(E_train)
    Z_train, Z_test = pca.transform(E_train), pca.transform(E_test)
    print(f"Prompt PCA: {n_pca} components (fit on TRAIN prompts only), "
          f"explained variance = {pca.explained_variance_ratio_.sum()*100:.2f}%")

    X_train_full = np.hstack([x_train, Z_train])
    X_test = np.hstack([x_test, Z_test])

    fit_idx, val_idx = train_test_split(np.arange(len(X_train_full)), test_size=args.val_fraction,
                                         random_state=0, shuffle=True)

    xs = StandardScaler().fit(X_train_full[fit_idx])
    ys = StandardScaler().fit(y_train_full[fit_idx])
    A_fit, A_val, A_te = xs.transform(X_train_full[fit_idx]), xs.transform(X_train_full[val_idx]), xs.transform(X_test)
    B_fit, B_val = ys.transform(y_train_full[fit_idx]), ys.transform(y_train_full[val_idx])

    print(f"\nSamples: fit={len(fit_idx)}  val={len(val_idx)}  test={len(X_test)}")
    print(f"Regression-head input dimension: {A_fit.shape[1]}")

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
            plt.legend(); plt.title("TinyLlama head training"); plt.tight_layout()
            plt.savefig(out / "loss_curve.png", dpi=300); plt.close()

            for j, t in enumerate(target_cols):
                lo = min(y_test[:, j].min(), y_pred[:, j].min())
                hi = max(y_test[:, j].max(), y_pred[:, j].max())
                plt.figure(figsize=(5.5, 5.5))
                plt.scatter(y_test[:, j], y_pred[:, j], s=14, alpha=0.6)
                plt.plot([lo, hi], [lo, hi], "--", color="k")
                plt.xlabel(f"Actual {t}"); plt.ylabel(f"Predicted {t}")
                plt.title(f"TinyLlama: {t}"); plt.tight_layout()
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


if __name__ == "__main__":
    main()
