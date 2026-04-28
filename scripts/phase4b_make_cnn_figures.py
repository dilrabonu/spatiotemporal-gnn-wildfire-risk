"""
Phase 4B — CNN Figure Generation

Run:
    python scripts/phase4b_make_cnn_figures.py
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


def save_fig(fig, filename):
    out = FIG_DIR / filename
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"✓ Saved: {out}")


def plot_training_history():
    path = TABLE_DIR / "phase4b_cnn_history.csv"
    if not path.exists():
        print(f"Missing: {path}")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["train_loss"], label="Train loss")
    ax.plot(df["val_loss"], label="Validation loss")
    ax.set_title("CNN Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(alpha=0.25)

    save_fig(fig, "p4b_cnn_training_history.png")


def plot_predictions_vs_truth():
    path = PRED_DIR / "phase4b_cnn_test_predictions.npz"
    if not path.exists():
        print(f"Missing: {path}")
        return

    data = np.load(path)
    y_true = data["y_test_raw"]
    y_pred = data["pred_cnn_raw"]

    rng = np.random.default_rng(42)
    n = min(20000, len(y_true))
    idx = rng.choice(len(y_true), size=n, replace=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.35)

    min_v = min(y_true[idx].min(), y_pred[idx].min())
    max_v = max(y_true[idx].max(), y_pred[idx].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)

    ax.set_title("CNN Predicted vs True Burn Probability")
    ax.set_xlabel("True BP")
    ax.set_ylabel("Predicted BP")

    save_fig(fig, "p4b_cnn_pred_vs_truth.png")


def plot_binned_mae():
    path = TABLE_DIR / "phase4b_cnn_binned_metrics.csv"
    if not path.exists():
        print(f"Missing: {path}")
        return

    df = pd.read_csv(path).sort_values("bin")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["bin"], df["mae"], marker="o")
    ax.set_title("CNN MAE by Burn Probability Risk Bin")
    ax.set_xlabel("Risk bin: 1 = low BP, 5 = high BP")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)

    save_fig(fig, "p4b_cnn_binned_mae.png")


def plot_compare_with_phase4():
    baseline_path = TABLE_DIR / "phase4_baseline_metrics.csv"
    cnn_path = TABLE_DIR / "phase4b_cnn_metrics.csv"

    if not baseline_path.exists() or not cnn_path.exists():
        print("Missing Phase 4 or Phase 4B metrics. Skipping combined comparison.")
        return

    base = pd.read_csv(baseline_path)
    cnn = pd.read_csv(cnn_path)

    combined = pd.concat([base, cnn], ignore_index=True)

    for metric in ["r2", "mae", "spearman", "brier", "ece"]:
        if metric not in combined.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(combined["model"], combined[metric])
        ax.set_title(f"Phase 4 + 4B Comparison — {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)

        for i, v in enumerate(combined[metric]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        save_fig(fig, f"p4b_combined_metric_{metric}.png")


def main():
    print("=" * 60)
    print("Phase 4B — CNN Figure Generation")
    print("=" * 60)

    plot_training_history()
    plot_predictions_vs_truth()
    plot_binned_mae()
    plot_compare_with_phase4()

    print("\n✓ Phase 4B figures complete.")


if __name__ == "__main__":
    main()