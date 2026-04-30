"""
Phase 5A — Evaluate ALL architectures from archive checkpoints.

Fixes the broken metrics CSVs and generates all figures for:
  - GAT        (from checkpoints/archive/phase5a_gat_best.pt)
  - GCN        (from checkpoints/archive/phase5a_gcn_best.pt)
  - GraphSAGE  (from checkpoints/archive/phase5a_graphsage_best.pt)

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/phase5a_evaluate_all_arches.py

OUTPUTS
-------
    reports/tables/phase5a_gat_metrics.csv        (fixed)
    reports/tables/phase5a_gcn_metrics.csv        (fixed)
    reports/tables/phase5a_graphsage_metrics.csv  (fixed)
    reports/tables/phase5a_all_models_comparison.csv (updated)
    reports/figures/p5a_gat_scatter.png
    reports/figures/p5a_gcn_scatter.png
    reports/figures/p5a_graphsage_scatter.png
    reports/figures/p5a_all_comparison.png
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.models.gnn import build_model, count_parameters
from wildfire_gnn.evaluation.metrics import (
    r2_score, mae_score, spearman_rho,
    brier_score, expected_calibration_error, binned_metrics,
)

config   = load_yaml_config(PROJECT_ROOT / "configs" / "gnn_config.yaml")
p        = config["paths"]

GRAPH_PATH  = PROJECT_ROOT / p["graph_data"]
TRANS_PATH  = PROJECT_ROOT / p["target_transformer"]
CKPT_DIR    = PROJECT_ROOT / "checkpoints"
ARCHIVE_DIR = CKPT_DIR / "archive"
TBL_DIR     = PROJECT_ROOT / "reports" / "tables"
FIG_DIR     = PROJECT_ROOT / "reports" / "figures"
TBL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_MC = 30   # MC Dropout passes


def load_graph():
    print(f"  Loading graph: {GRAPH_PATH.name}")
    g = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    print(f"  Nodes={g.num_nodes:,}  Features={g.num_node_features}"
          f"  Test={int(g.test_mask.sum()):,}")

    # Guard: y must be transformed (mean ≈ 0)
    assert abs(float(g.y.mean())) < 0.5, \
        "y looks un-transformed — check graph file"
    assert (g.train_mask & g.test_mask).sum() == 0, \
        "Train/Test overlap detected!"
    print("  ✓ Graph assertions passed")
    return g


def load_model_from_archive(arch: str) -> torch.nn.Module:
    """
    Load checkpoint from archive directory.
    Falls back to main checkpoints/ if archive copy not found.
    """
    ckpt_name = f"phase5a_{arch.lower()}_best.pt"
    ckpt_path = ARCHIVE_DIR / ckpt_name

    if not ckpt_path.exists():
        # Try main checkpoints directory
        ckpt_path = CKPT_DIR / f"gnn_{arch.lower()}_best.pt"

    if not ckpt_path.exists():
        print(f"  ✗ No checkpoint found for {arch}")
        print(f"    Tried: {ARCHIVE_DIR / ckpt_name}")
        print(f"    Tried: {CKPT_DIR / f'gnn_{arch.lower()}_best.pt'}")
        return None

    print(f"  Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Read model config from checkpoint (saved during training)
    saved_cfg = ckpt.get("config", config)
    m = saved_cfg["model"]

    model = build_model(
        architecture = arch,
        in_channels  = m["in_channels"],
        hidden       = m["hidden_channels"],
        num_layers   = m.get("num_layers", 4),
        heads        = m.get("heads", 8),
        dropout      = m.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    print(f"  ✓ {arch} loaded — {count_parameters(model):,} parameters")
    return model


def run_mc_dropout(model, graph, n_samples: int = N_MC) -> dict:
    """
    Full-graph MC Dropout: run model n_samples times with dropout ON.
    Uses graph.test_mask to select test nodes.
    """
    model.train()   # dropout ON — critical for MC Dropout

    sample_means   = []
    sample_logvars = []

    print(f"  Running {n_samples} MC Dropout passes (full graph)...")
    with torch.no_grad():
        for i in range(n_samples):
            mean, lv = model(graph.x, graph.edge_index)
            sample_means.append(mean[graph.test_mask].numpy())
            sample_logvars.append(lv[graph.test_mask].numpy())
            if (i + 1) % 10 == 0:
                print(f"    Pass {i+1}/{n_samples}")

    samples        = np.stack(sample_means)    # (N_MC, N_test)
    logvar_stack   = np.stack(sample_logvars)  # (N_MC, N_test)

    mean_pred  = samples.mean(axis=0)
    std_pred   = samples.std(axis=0)           # epistemic uncertainty
    aleatoric  = np.sqrt(np.exp(logvar_stack.mean(axis=0)))  # aleatoric
    total_unc  = np.sqrt(aleatoric**2 + std_pred**2)

    return {
        "mean_pred": mean_pred,
        "std_pred":  std_pred,
        "aleatoric": aleatoric,
        "total_unc": total_unc,
    }


def compute_metrics(graph, mc: dict, transformer) -> dict:
    """Inverse-transform predictions and compute all metrics."""
    y_pred_bp = transformer.inverse_transform(
        mc["mean_pred"].reshape(-1, 1)
    ).ravel()
    y_true_bp = graph.y_raw[graph.test_mask].numpy().ravel()

    return {
        "y_true_bp": y_true_bp,
        "y_pred_bp": y_pred_bp,
        "r2":        r2_score(y_true_bp, y_pred_bp),
        "mae":       mae_score(y_true_bp, y_pred_bp),
        "spearman":  spearman_rho(y_true_bp, y_pred_bp),
        "brier":     brier_score(y_true_bp, y_pred_bp),
        "ece":       expected_calibration_error(y_true_bp, y_pred_bp),
        "n_test":    len(y_true_bp),
        "total_unc": mc["total_unc"],
        "std_pred":  mc["std_pred"],
        "aleatoric": mc["aleatoric"],
        "binned":    binned_metrics(y_true_bp, y_pred_bp),
    }


def print_results(arch: str, m: dict, baselines: dict) -> None:
    print(f"\n  ── {arch} Results (test split, original BP scale) ──")
    print(f"  R²       = {m['r2']:.4f}")
    print(f"  MAE      = {m['mae']:.5f}")
    print(f"  Spearman = {m['spearman']:.4f}")
    print(f"  Brier    = {m['brier']:.5f}")
    print(f"  ECE      = {m['ece']:.5f}")
    print(f"  n_test   = {m['n_test']:,}")

    print(f"\n  Comparison vs baselines:")
    for name, row in baselines.items():
        diff = m["r2"] - row.get("r2", 0)
        sym  = "✓" if diff > 0 else "✗"
        print(f"  {sym} {name:<22} R²={row.get('r2',0):.4f}  diff={diff:+.4f}")

    # Binned
    print(f"\n  Binned evaluation:")
    print(f"  {'Bin':<5} {'BP range':<22} {'n':>8} {'R²':>8} {'MAE':>10} {'Spearman':>10}")
    print(f"  {'-'*65}")
    for b in m["binned"]:
        flag = " ← HIGH RISK" if b["bin"] == len(m["binned"]) else ""
        print(f"  {b['bin']:<5} [{b['bin_low']:.4f}, {b['bin_high']:.4f}]  "
              f"{b['n']:>8,} {b['r2']:>8.3f} {b['mae']:>10.5f} "
              f"{b['spearman']:>10.3f}{flag}")


def fix_metrics_csv(arch: str, m: dict) -> None:
    """Overwrite the metrics CSV with correct values from this evaluation."""
    path = TBL_DIR / f"phase5a_{arch.lower()}_metrics.csv"
    df = pd.DataFrame([{
        "model":    arch,
        "r2":       m["r2"],
        "mae":      m["mae"],
        "spearman": m["spearman"],
        "brier":    m["brier"],
        "ece":      m["ece"],
        "n_test":   m["n_test"],
    }])
    df.to_csv(path, index=False)
    print(f"  ✓ Fixed: {path.name}  (R²={m['r2']:.4f})")

    # Also fix binned
    binned_path = TBL_DIR / f"phase5a_{arch.lower()}_binned.csv"
    bd = pd.DataFrame(m["binned"])
    bd["model"] = arch
    bd.to_csv(binned_path, index=False)
    print(f"  ✓ Fixed: {binned_path.name}")


def generate_scatter_figure(arch: str, m: dict) -> None:
    """Prediction vs truth + uncertainty vs BP scatter."""
    y_true     = m["y_true_bp"]
    y_pred     = m["y_pred_bp"]
    total_unc  = m["total_unc"]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(20_000, len(y_true)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: prediction vs truth
    ax = axes[0]
    ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.3, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect")
    ax.set_xlabel("True Burn Probability", fontsize=11)
    ax.set_ylabel("Predicted Burn Probability", fontsize=11)
    ax.set_title(
        f"{arch} — Predicted vs True\n"
        f"R²={m['r2']:.4f}  MAE={m['mae']:.4f}"
    )
    ax.legend()

    # Right: uncertainty vs true BP
    ax2 = axes[1]
    ax2.scatter(y_true[idx], total_unc[idx], s=2, alpha=0.3,
                color="orange", rasterized=True)
    ax2.set_xlabel("True Burn Probability", fontsize=11)
    ax2.set_ylabel("Total Uncertainty (epistemic + aleatoric)", fontsize=11)
    ax2.set_title(
        f"{arch} — Uncertainty vs True BP\n"
        "High uncertainty on high-risk cells = Gap 1+2 addressed"
    )

    plt.tight_layout()
    out = FIG_DIR / f"p5a_{arch.lower()}_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure: {out.name}")


def generate_comparison_figure(all_results: list[dict], baselines: dict) -> None:
    """All models R² bar chart."""
    rows = []
    for name, row in baselines.items():
        rows.append({"model": name, "r2": row.get("r2", 0)})
    for r in all_results:
        rows.append({"model": r["arch"], "r2": r["r2"]})

    df = pd.DataFrame(rows).sort_values("r2", ascending=True)
    gnn_names = {r["arch"] for r in all_results}
    colors = ["#2980B9" if m in gnn_names else "#95A5A6"
              for m in df["model"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["model"], df["r2"], color=colors, height=0.6)

    ax.axvline(0,      color="black",       lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.7187, color="darkorange",  lw=1.5, ls=":",
               label="CNN baseline (R²=0.7187)")
    ax.axvline(0.6761, color="gray",        lw=1.2, ls=":",
               label="XGBoost (R²=0.6761)")

    for bar, v in zip(bars, df["r2"]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)

    ax.set_xlabel("R² (test split, original BP scale, geographic split)")
    ax.set_title(
        "Phase 5A — All Models Final Comparison\n"
        "Blue = GNN  |  Grey = Phase 4 baselines"
    )
    ax.legend(loc="lower right")
    ax.set_xlim(df["r2"].min() - 0.15, df["r2"].max() + 0.07)
    plt.tight_layout()

    out = FIG_DIR / "p5a_all_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Comparison figure: {out.name}")


def update_all_comparison_csv(all_results: list[dict],
                               baselines: dict) -> None:
    """Rewrite phase5a_all_models_comparison.csv with correct values."""
    rows = []
    for name, row in baselines.items():
        rows.append({
            "model":    name,
            "r2":       row.get("r2", 0),
            "mae":      row.get("mae", 0),
            "spearman": row.get("spearman", 0),
            "brier":    row.get("brier", 0),
            "ece":      row.get("ece", 0),
        })
    for r in all_results:
        rows.append({
            "model":    r["arch"],
            "r2":       r["r2"],
            "mae":      r["mae"],
            "spearman": r["spearman"],
            "brier":    r["brier"],
            "ece":      r["ece"],
        })

    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    path = TBL_DIR / "phase5a_all_models_comparison.csv"
    df.to_csv(path, index=False)
    print(f"\n  ✓ Updated: {path.name}")
    print(df.to_string(index=False))


def main():
    print("\n" + "="*65)
    print("  Phase 5A — Evaluate All Architectures from Archive")
    print("="*65 + "\n")

    # Load graph and transformer
    graph = load_graph()
    with open(TRANS_PATH, "rb") as f:
        transformer = pickle.load(f)

    # Load baselines for comparison
    baselines = {}
    for fname in ["phase4_baseline_metrics.csv", "phase4b_cnn_metrics.csv"]:
        csv = TBL_DIR / fname
        if csv.exists():
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                baselines[row["model"]] = row.to_dict()
    print(f"\n  Baselines loaded: {list(baselines.keys())}")

    # Architectures to evaluate
    arches = ["GAT", "GCN", "GraphSAGE"]
    all_results = []

    for arch in arches:
        print(f"\n{'─'*65}")
        print(f"  Architecture: {arch}")
        print(f"{'─'*65}")

        model = load_model_from_archive(arch)
        if model is None:
            print(f"  ⚠  Skipping {arch} — no checkpoint found")
            continue

        # MC Dropout inference
        mc = run_mc_dropout(model, graph, n_samples=N_MC)

        # Compute all metrics
        m = compute_metrics(graph, mc, transformer)

        # Print results
        print_results(arch, m, baselines)

        # Fix CSV files
        print(f"\n  Fixing CSV files for {arch}...")
        fix_metrics_csv(arch, m)

        # Generate scatter figure
        generate_scatter_figure(arch, m)

        all_results.append({
            "arch":     arch,
            "r2":       m["r2"],
            "mae":      m["mae"],
            "spearman": m["spearman"],
            "brier":    m["brier"],
            "ece":      m["ece"],
        })

    # Update combined comparison table
    print(f"\n{'─'*65}")
    print("  Updating combined comparison table...")
    update_all_comparison_csv(all_results, baselines)

    # Generate comparison figure
    generate_comparison_figure(all_results, baselines)

    # Final summary
    print(f"\n{'='*65}")
    print("  PHASE 5A — FINAL CONFIRMED RESULTS")
    print(f"{'='*65}")
    print(f"  {'Model':<15} {'R²':>8} {'MAE':>10} {'Spearman':>10} "
          f"{'Brier':>10} {'ECE':>8}")
    print(f"  {'-'*60}")
    for r in sorted(all_results, key=lambda x: x["r2"], reverse=True):
        beats_cnn = "✓ beats CNN" if r["r2"] > 0.7187 else "✗"
        print(f"  {r['arch']:<15} {r['r2']:>8.4f} {r['mae']:>10.5f} "
              f"{r['spearman']:>10.4f} {r['brier']:>10.5f} "
              f"{r['ece']:>8.5f}  {beats_cnn}")
    print(f"{'='*65}")
    print()
    print("  All CSVs fixed. All figures generated.")
    print("  Proceed to Phase 5B — Calibration (Temperature Scaling)")
    print()


if __name__ == "__main__":
    main()