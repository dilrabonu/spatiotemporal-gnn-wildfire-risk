"""
Phase 5A — GNN Figure Generation

Run:
    python scripts/phase5a_make_figures.py

Creates:
- training curves
- prediction vs truth plots
- binned MAE plots
- comparison against Phase 4 and Phase 4B baselines
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, filename: str, dpi: int = 160):
    out = FIG_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"✓ Saved: {out}")


def find_phase5a_models():
    files = sorted(TABLE_DIR.glob("phase5a_*_metrics.csv"))
    models = []

    for f in files:
        name = f.name.replace("phase5a_", "").replace("_metrics.csv", "")
        models.append(name)

    return models


def plot_training_history(model_name: str):
    path = TABLE_DIR / f"phase5a_{model_name}_history.csv"

    if not path.exists():
        print(f"Missing: {path}")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], label="Validation loss")
    ax.set_title(f"Phase 5A Training History — {model_name.upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss in transformed target space")
    ax.legend()
    ax.grid(alpha=0.25)

    save_fig(fig, f"p5a_{model_name}_training_history.png")


def plot_predictions_vs_truth(model_name: str):
    path = PRED_DIR / f"phase5a_{model_name}_test_predictions.npz"

    if not path.exists():
        print(f"Missing: {path}")
        return

    data = np.load(path)
    y_true = data["y_test_raw"]
    y_pred = data["pred_test_raw"]

    rng = np.random.default_rng(42)
    n = min(20_000, len(y_true))
    idx = rng.choice(len(y_true), size=n, replace=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.35)

    min_v = min(y_true[idx].min(), y_pred[idx].min())
    max_v = max(y_true[idx].max(), y_pred[idx].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)

    ax.set_title(f"Phase 5A {model_name.upper()} — Predicted vs True BP")
    ax.set_xlabel("True Burn Probability")
    ax.set_ylabel("Predicted Burn Probability")

    save_fig(fig, f"p5a_{model_name}_pred_vs_truth.png")


def plot_binned_mae(model_name: str):
    path = TABLE_DIR / f"phase5a_{model_name}_binned_metrics.csv"

    if not path.exists():
        print(f"Missing: {path}")
        return

    df = pd.read_csv(path).sort_values("bin")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["bin"], df["mae"], marker="o")
    ax.set_title(f"Phase 5A {model_name.upper()} — MAE by Risk Bin")
    ax.set_xlabel("Risk bin: 1 = low BP, 5 = high BP")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)

    save_fig(fig, f"p5a_{model_name}_binned_mae.png")


def load_combined_metrics():
    frames = []

    phase4_path = TABLE_DIR / "phase4_baseline_metrics.csv"
    if phase4_path.exists():
        frames.append(pd.read_csv(phase4_path))

    cnn_path = TABLE_DIR / "phase4b_cnn_metrics.csv"
    if cnn_path.exists():
        frames.append(pd.read_csv(cnn_path))

    for path in sorted(TABLE_DIR.glob("phase5a_*_metrics.csv")):
        frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_combined_comparison():
    df = load_combined_metrics()

    if df.empty:
        print("No metrics found for combined comparison.")
        return

    for metric in ["r2", "mae", "spearman", "brier", "ece"]:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(df["model"], df[metric])
        ax.set_title(f"Phase 4 + 4B + 5A Comparison — {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=25)

        for i, v in enumerate(df[metric]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        save_fig(fig, f"p5a_combined_metric_{metric}.png")


def main():
    print("=" * 65)
    print("Phase 5A — GNN Figure Generation")
    print("=" * 65)

    models = find_phase5a_models()

    if not models:
        print("No Phase 5A model outputs found.")
        print("Run: python scripts/phase5a_run_gnn.py --model gcn --epochs 5 --device cpu")
        return

    for model_name in models:
        plot_training_history(model_name)
        plot_predictions_vs_truth(model_name)
        plot_binned_mae(model_name)

    plot_combined_comparison()

    print("\n✓ Phase 5A figures complete.")


if __name__ == "__main__":
    main()