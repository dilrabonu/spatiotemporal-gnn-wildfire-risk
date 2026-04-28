"""
Phase 4 — Baseline Figure Generation

Inputs:
- reports/tables/phase4_baseline_metrics.csv
- reports/tables/phase4_binned_metrics.csv
- reports/tables/phase4_feature_importances.csv
- reports/predictions/phase4_test_predictions.npz

Outputs:
- reports/figures/p4_model_metrics_comparison.png
- reports/figures/p4_binned_mae.png
- reports/figures/p4_predictions_vs_truth.png
- reports/figures/p4_feature_importances.png
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
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PRED_DIR = PROJECT_ROOT / "reports" / "predictions"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, filename: str, dpi: int = 160):
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"✓ Saved: {out_path}")


def plot_metric_comparison(metrics_df: pd.DataFrame):
    print("\n[1/4] Model metric comparison")

    metrics = ["r2", "mae", "spearman", "brier", "ece"]

    for metric in metrics:
        if metric not in metrics_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(metrics_df["model"], metrics_df[metric])
        ax.set_title(f"Phase 4 Baseline Comparison — {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)

        for i, v in enumerate(metrics_df[metric]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        save_fig(fig, f"p4_metric_{metric}.png")


def plot_binned_mae(binned_df: pd.DataFrame):
    print("\n[2/4] Binned MAE by burn probability risk bin")

    if binned_df.empty:
        print("No binned metrics found.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for model in binned_df["model"].unique():
        df_m = binned_df[binned_df["model"] == model].sort_values("bin")
        ax.plot(df_m["bin"], df_m["mae"], marker="o", label=model)

    ax.set_title("MAE by True Burn Probability Quantile Bin")
    ax.set_xlabel("Risk bin: 1 = lowest BP, 5 = highest BP")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "p4_binned_mae.png")


def plot_predictions_vs_truth():
    print("\n[3/4] Predictions vs truth")

    pred_path = PRED_DIR / "phase4_test_predictions.npz"
    if not pred_path.exists():
        print(f"Missing: {pred_path}")
        return

    data = np.load(pred_path)
    y_true = data["y_test_raw"]

    pred_keys = [k for k in data.files if k not in {"y_test_raw", "test_idx"}]

    rng = np.random.default_rng(42)
    n = min(20_000, len(y_true))
    idx = rng.choice(len(y_true), size=n, replace=False)

    for key in pred_keys:
        y_pred = data[key]

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.35)

        min_v = min(y_true[idx].min(), y_pred[idx].min())
        max_v = max(y_true[idx].max(), y_pred[idx].max())
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)

        ax.set_title(f"Predicted vs True Burn Probability\n{key}")
        ax.set_xlabel("True BP")
        ax.set_ylabel("Predicted BP")

        fig.tight_layout()
        save_fig(fig, f"p4_pred_vs_truth_{key}.png")


def plot_feature_importances(importance_df: pd.DataFrame):
    print("\n[4/4] Feature importances")

    if importance_df.empty:
        print("No feature importances found.")
        return

    for model in importance_df["model"].unique():
        df_m = importance_df[importance_df["model"] == model].sort_values("rank").head(20)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(df_m["feature"], df_m["importance"])
        ax.set_title(f"Top 20 Feature Importances — {model}")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()

        fig.tight_layout()
        safe_name = model.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_fig(fig, f"p4_feature_importances_{safe_name}.png")


def main():
    print("=" * 60)
    print("Phase 4 — Safe Figure Generation")
    print("=" * 60)

    metrics_path = TABLE_DIR / "phase4_baseline_metrics.csv"
    binned_path = TABLE_DIR / "phase4_binned_metrics.csv"
    importance_path = TABLE_DIR / "phase4_feature_importances.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run python scripts/phase4_run_baselines.py first."
        )

    metrics_df = pd.read_csv(metrics_path)
    binned_df = pd.read_csv(binned_path) if binned_path.exists() else pd.DataFrame()
    importance_df = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()

    plot_metric_comparison(metrics_df)
    plot_binned_mae(binned_df)
    plot_predictions_vs_truth()
    plot_feature_importances(importance_df)

    print("\n✓ Phase 4 figures complete.")


if __name__ == "__main__":
    main()