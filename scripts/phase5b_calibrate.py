"""
Phase 5B — Uncertainty Calibration (Temperature Scaling)

WHAT THIS SCRIPT DOES
----------------------
1. Loads Phase 5A predictions (from phase5a_save_predictions.py)
2. Learns a temperature T on the VALIDATION set (no retraining)
3. Evaluates calibration BEFORE and AFTER temperature scaling
4. Generates reliability diagrams and calibration figures
5. Saves all calibration results

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/phase5b_calibrate.py
    python scripts/phase5b_calibrate.py --arch GAT      (default)
    python scripts/phase5b_calibrate.py --arch GCN
    python scripts/phase5b_calibrate.py --arch GraphSAGE
    python scripts/phase5b_calibrate.py --all            (all arches)

PRE-CONDITIONS
--------------
    reports/predictions/phase5a_gat_preds.npz       (from phase5a_save_predictions.py)
    reports/predictions/phase5a_gcn_preds.npz
    reports/predictions/phase5a_graphsage_preds.npz
    data/features/target_transformer.pkl
    data/processed/graph_data_enriched.pt            (for validation split)

OUTPUTS
-------
    reports/tables/phase5b_{arch}_calibration.csv
    reports/tables/phase5b_{arch}_temperature.csv
    reports/figures/p5b_{arch}_reliability.png
    reports/figures/p5b_{arch}_uncertainty_hist.png
    reports/figures/p5b_{arch}_interval_coverage.png
    reports/figures/p5b_all_reliability.png
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.models.calibration import (
    TemperatureScaling,
    compute_all_calibration_metrics,
    reliability_curve,
    compute_picp,
    compute_mpiw,
)
from wildfire_gnn.models.gnn import build_model
from wildfire_gnn.evaluation.metrics import (
    r2_score, mae_score, spearman_rho,
    brier_score, expected_calibration_error,
)

config   = load_yaml_config(PROJECT_ROOT / "configs" / "gnn_config.yaml")
p        = config["paths"]

GRAPH_PATH  = PROJECT_ROOT / p["graph_data"]
TRANS_PATH  = PROJECT_ROOT / p["target_transformer"]
CKPT_DIR    = PROJECT_ROOT / "checkpoints"
ARCHIVE_DIR = CKPT_DIR / "archive"
PRED_DIR    = PROJECT_ROOT / "reports" / "predictions"
TBL_DIR     = PROJECT_ROOT / "reports" / "tables"
FIG_DIR     = PROJECT_ROOT / "reports" / "figures"
TBL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5B — Calibration")
    p.add_argument("--arch", default="GAT",
                   choices=["GAT", "GCN", "GraphSAGE"])
    p.add_argument("--all", action="store_true",
                   help="Calibrate all architectures")
    return p.parse_args()


def load_transformer():
    with open(TRANS_PATH, "rb") as f:
        return pickle.load(f)


def load_preds(arch: str) -> dict | None:
    path = PRED_DIR / f"phase5a_{arch.lower()}_preds.npz"
    if not path.exists():
        print(f"  ✗  {path.name} not found.")
        print(f"     Run: python scripts/phase5a_save_predictions.py")
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_val_predictions(arch: str, transformer) -> dict | None:
    """
    Run model on validation split to get predictions for temperature fitting.
    Loads archive checkpoint and runs 30 MC passes on val nodes.
    """
    ckpt_name = f"phase5a_{arch.lower()}_best.pt"
    ckpt_path = ARCHIVE_DIR / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / f"gnn_{arch.lower()}_best.pt"
    if not ckpt_path.exists():
        print(f"  ✗  No checkpoint for {arch}")
        return None

    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m     = ckpt.get("config", config)["model"]

    model = build_model(
        architecture = arch,
        in_channels  = m["in_channels"],
        hidden       = m["hidden_channels"],
        num_layers   = m.get("num_layers", 4),
        heads        = m.get("heads", 8),
        dropout      = m.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.train()  # dropout ON

    print(f"  Getting validation predictions for temperature fitting...")
    sample_means = []
    sample_lvs   = []
    with torch.no_grad():
        for _ in range(30):
            mean, lv = model(graph.x, graph.edge_index)
            sample_means.append(mean[graph.val_mask].numpy())
            sample_lvs.append(lv[graph.val_mask].numpy())

    samples     = np.stack(sample_means)   # (30, N_val)
    lv_stack    = np.stack(sample_lvs)

    mean_pred_t = samples.mean(axis=0)
    std_pred_t  = samples.std(axis=0)
    y_true_t    = graph.y[graph.val_mask].numpy().ravel()

    # Also get original scale for reporting
    y_pred_bp   = transformer.inverse_transform(
        mean_pred_t.reshape(-1,1)).ravel()
    y_true_bp   = graph.y_raw[graph.val_mask].numpy().ravel()

    aleatoric   = np.sqrt(np.exp(lv_stack.mean(axis=0)))
    total_std_t = np.sqrt(std_pred_t**2 + aleatoric**2)

    del graph, model
    gc.collect()

    return {
        "mean_pred_t":  mean_pred_t,
        "std_pred_t":   std_pred_t,
        "total_std_t":  total_std_t,
        "y_true_t":     y_true_t,
        "y_pred_bp":    y_pred_bp,
        "y_true_bp":    y_true_bp,
        "aleatoric":    aleatoric,
    }


def savefig(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"  ✓  Figure: {path.name}")


def plot_reliability(
    arch: str,
    expected_before: np.ndarray,
    actual_before:   np.ndarray,
    expected_after:  np.ndarray,
    actual_after:    np.ndarray,
    T: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(expected_before, actual_before, "o-",
            color="#E74C3C", lw=2, ms=5, label=f"Before scaling (T=1.0)")
    ax.plot(expected_after, actual_after, "s-",
            color="#2ECC71", lw=2, ms=5, label=f"After scaling (T={T:.3f})")

    ax.fill_between([0,1], [0,1], [0,1], alpha=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage", fontsize=12)
    ax.set_ylabel("Actual coverage (PICP)", fontsize=12)
    ax.set_title(
        f"Phase 5B — {arch} Reliability Diagram\n"
        f"Perfect calibration = diagonal line",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, f"p5b_{arch.lower()}_reliability.png")


def plot_uncertainty_histogram(
    arch: str,
    std_before: np.ndarray,
    std_after:  np.ndarray,
    T: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(std_before, bins=60, color="#E74C3C", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Total Uncertainty (std)")
    ax.set_ylabel("Node count")
    ax.set_title(f"{arch} — Before Scaling\nmean={std_before.mean():.4f}")

    ax2 = axes[1]
    ax2.hist(std_after, bins=60, color="#2ECC71", alpha=0.8, edgecolor="none")
    ax2.set_xlabel("Calibrated Total Uncertainty (std)")
    ax2.set_ylabel("Node count")
    ax2.set_title(f"{arch} — After Scaling (T={T:.3f})\nmean={std_after.mean():.4f}")

    fig.suptitle(
        f"Phase 5B — {arch} Uncertainty Distribution\n"
        "T>1: intervals widened (was overconfident)",
        fontsize=12
    )
    plt.tight_layout()
    savefig(fig, f"p5b_{arch.lower()}_uncertainty_hist.png")


def plot_interval_coverage(
    arch: str,
    y_true_bp:  np.ndarray,
    mean_bp:    np.ndarray,
    std_before: np.ndarray,
    std_after:  np.ndarray,
) -> None:
    """Show 90% prediction intervals before and after calibration."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true_bp), min(3000, len(y_true_bp)), replace=False)
    sort_idx = idx[np.argsort(y_true_bp[idx])]

    z   = 1.645  # 90% interval
    lo_b = mean_bp[sort_idx] - z * std_before[sort_idx]
    hi_b = mean_bp[sort_idx] + z * std_before[sort_idx]
    lo_a = mean_bp[sort_idx] - z * std_after[sort_idx]
    hi_a = mean_bp[sort_idx] + z * std_after[sort_idx]
    yt   = y_true_bp[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lo, hi, label, color in zip(
        axes,
        [lo_b, lo_a], [hi_b, hi_a],
        ["Before scaling", f"After scaling (T={TemperatureScaling().T:.3f})"],
        ["#E74C3C", "#2ECC71"]
    ):
        picp = np.mean((yt >= lo) & (yt <= hi))
        x    = np.arange(len(yt))
        ax.fill_between(x, lo, hi, alpha=0.3, color=color,
                        label=f"90% PI (PICP={picp:.3f})")
        ax.plot(x, yt, "k.", ms=1.5, alpha=0.4, label="True BP")
        ax.plot(x, mean_bp[sort_idx], color=color, lw=1,
                alpha=0.8, label="Predicted mean")
        ax.set_xlabel("Test nodes (sorted by true BP)")
        ax.set_ylabel("Burn Probability")
        ax.set_title(f"{arch} — 90% Prediction Intervals\n{label}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig(fig, f"p5b_{arch.lower()}_interval_coverage.png")


def calibrate_arch(arch: str) -> dict | None:
    print(f"\n{'─'*65}")
    print(f"  Architecture: {arch}")
    print(f"{'─'*65}")

    # Load test predictions from Phase 5A
    preds = load_preds(arch)
    if preds is None:
        return None

    transformer = load_transformer()

    y_true_bp   = preds["y_true_bp"]
    y_pred_bp   = preds["y_pred_bp"]
    mean_pred_t = preds["mean_pred_t"]    # transformed scale
    std_pred    = preds["std_pred"]       # epistemic (transformed)
    aleatoric   = preds["aleatoric"]      # aleatoric (transformed)
    total_unc_t = np.sqrt(std_pred**2 + aleatoric**2)  # total, transformed

    # Convert total uncertainty to original BP scale (approximate)
    # We scale by the mean derivative of the inverse transform
    eps       = 0.01
    mean_up   = transformer.inverse_transform(
        (mean_pred_t + eps).reshape(-1,1)).ravel()
    mean_dn   = transformer.inverse_transform(
        (mean_pred_t - eps).reshape(-1,1)).ravel()
    deriv     = (mean_up - mean_dn) / (2 * eps)
    total_unc_bp = total_unc_t * np.abs(deriv)  # chain rule

    # ── Step 1: Get validation predictions for temperature fitting ─────
    print("\n  [1/4] Getting validation predictions for temperature fitting...")
    val_preds = get_val_predictions(arch, transformer)
    if val_preds is None:
        return None

    # ── Step 2: Fit temperature on validation set ─────────────────────
    print("\n  [2/4] Fitting temperature (validation set)...")
    ts = TemperatureScaling()
    ts.fit(
        mean_pred = val_preds["mean_pred_t"],
        std_pred  = val_preds["total_std_t"],
        y_true    = val_preds["y_true_t"],
    )

    # ── Step 3: Apply to test set ─────────────────────────────────────
    print("\n  [3/4] Applying temperature scaling to test predictions...")
    std_after_t  = ts.scale(total_unc_t)
    std_after_bp = std_after_t * np.abs(deriv)

    # ── Step 4: Calibration metrics before and after ──────────────────
    print("\n  [4/4] Computing calibration metrics...")

    print("\n  BEFORE temperature scaling:")
    metrics_before = compute_all_calibration_metrics(
        y_true_bp, y_pred_bp, total_unc_bp,
        label=f"{arch} (before)", verbose=True
    )

    print("\n  AFTER temperature scaling:")
    metrics_after = compute_all_calibration_metrics(
        y_true_bp, y_pred_bp, std_after_bp,
        label=f"{arch} (after)", verbose=True
    )

    # ── Save tables ───────────────────────────────────────────────────
    cal_df = pd.DataFrame([
        {
            "arch": arch, "stage": "before",
            "T": 1.0,
            "picp_50": metrics_before["picp_50"],
            "picp_90": metrics_before["picp_90"],
            "picp_95": metrics_before["picp_95"],
            "mpiw_90": metrics_before["mpiw_90"],
            "ace":     metrics_before["ace"],
            "ence":    metrics_before["ence"],
        },
        {
            "arch": arch, "stage": "after",
            "T": ts.T,
            "picp_50": metrics_after["picp_50"],
            "picp_90": metrics_after["picp_90"],
            "picp_95": metrics_after["picp_95"],
            "mpiw_90": metrics_after["mpiw_90"],
            "ace":     metrics_after["ace"],
            "ence":    metrics_after["ence"],
        },
    ])
    cal_path = TBL_DIR / f"phase5b_{arch.lower()}_calibration.csv"
    cal_df.to_csv(cal_path, index=False)
    print(f"\n  ✓  Saved: {cal_path.name}")

    # Temperature record
    temp_df = pd.DataFrame([{
        "arch": arch, "T": ts.T,
        "interpretation": (
            "overconfident" if ts.T > 1.05 else
            "underconfident" if ts.T < 0.95 else
            "well-calibrated"
        )
    }])
    temp_path = TBL_DIR / f"phase5b_{arch.lower()}_temperature.csv"
    temp_df.to_csv(temp_path, index=False)

    # ── Generate figures ──────────────────────────────────────────────
    exp_b, act_b = metrics_before["expected"], metrics_before["actual"]
    exp_a, act_a = metrics_after["expected"],  metrics_after["actual"]

    plot_reliability(arch, exp_b, act_b, exp_a, act_a, ts.T)
    plot_uncertainty_histogram(arch, total_unc_bp, std_after_bp, ts.T)
    plot_interval_coverage(
        arch, y_true_bp, y_pred_bp, total_unc_bp, std_after_bp
    )

    return {
        "arch":          arch,
        "T":             ts.T,
        "picp_90_before": metrics_before["picp_90"],
        "picp_90_after":  metrics_after["picp_90"],
        "ace_before":     metrics_before["ace"],
        "ace_after":      metrics_after["ace"],
        "ence_before":    metrics_before["ence"],
        "ence_after":     metrics_after["ence"],
        "exp_before":     exp_b,
        "act_before":     act_b,
        "exp_after":      exp_a,
        "act_after":      act_a,
    }


def plot_all_reliability(all_results: list[dict]) -> None:
    """Combined reliability diagram for all architectures."""
    colors_before = ["#E74C3C", "#E67E22", "#9B59B6"]
    colors_after  = ["#2ECC71", "#3498DB", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration")

    for r, cb, ca in zip(all_results, colors_before, colors_after):
        arch = r["arch"]
        ax.plot(r["exp_before"], r["act_before"], "o:",
                color=cb, lw=1.5, ms=4, alpha=0.7,
                label=f"{arch} before (T=1.0)")
        ax.plot(r["exp_after"], r["act_after"], "s-",
                color=ca, lw=2, ms=5,
                label=f"{arch} after (T={r['T']:.3f})")

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Expected coverage", fontsize=12)
    ax.set_ylabel("Actual coverage (PICP)", fontsize=12)
    ax.set_title(
        "Phase 5B — All Architectures Reliability Diagram\n"
        "Dashed=before scaling  |  Solid=after scaling",
        fontsize=12
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "p5b_all_reliability.png")


def main():
    args = parse_args()
    arches = ["GAT", "GCN", "GraphSAGE"] if args.all else [args.arch]

    print("\n" + "="*65)
    print("  Phase 5B — Uncertainty Calibration")
    print("="*65)

    all_results = []
    for arch in arches:
        result = calibrate_arch(arch)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        plot_all_reliability(all_results)

    # Final summary
    print(f"\n{'='*65}")
    print("  PHASE 5B CALIBRATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Arch':<12} {'T':>7} {'PICP-90 before':>16} "
          f"{'PICP-90 after':>15} {'ACE after':>11}")
    print(f"  {'-'*65}")
    for r in all_results:
        ideal = "✓" if abs(r["picp_90_after"] - 0.90) < 0.05 else "✗"
        print(f"  {r['arch']:<12} {r['T']:>7.4f} "
              f"{r['picp_90_before']:>16.4f} "
              f"{r['picp_90_after']:>15.4f} {ideal} "
              f"{r['ace_after']:>+11.4f}")
    print(f"{'='*65}")
    print()
    print("  All calibration tables and figures saved.")
    print("  Proceed to Phase 5D — Intervention Analysis")
    print()


if __name__ == "__main__":
    main()