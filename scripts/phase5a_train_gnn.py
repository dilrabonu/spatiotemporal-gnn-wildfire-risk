"""
Phase 5A — GNN Training Orchestrator

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn

    # Train primary GAT model (default)
    python scripts/phase5a_train_gnn.py

    # Train all architectures for ablation
    python scripts/phase5a_train_gnn.py --arch GAT
    python scripts/phase5a_train_gnn.py --arch GCN
    python scripts/phase5a_train_gnn.py --arch GraphSAGE

    # Quick smoke test (5 epochs)
    python scripts/phase5a_train_gnn.py --epochs 5 --smoke-test

PRE-CONDITIONS
--------------
    data/processed/graph_data_enriched.pt     (Phase 3)
    data/features/target_transformer.pkl      (Phase 2)
    data/features/feature_names.json          (Phase 3)
    configs/gnn_config.yaml                   (in_channels=61)
    Phase 4 baselines complete (CNN R²=0.7187 is the target to beat)

OUTPUTS
-------
    checkpoints/gnn_{arch}_best.pt            model weights
    reports/tables/phase5a_{arch}_metrics.csv results
    reports/figures/p5a_{arch}_loss.png       training curve
    reports/predictions/phase5a_{arch}_preds.npz

TARGET (updated after Phase 4B CNN)
------------------------------------
    Must beat CNN R²=0.7187
    Minimum acceptable: R²  > 0.72, MAE < 0.012
    Strong result:      R²  > 0.75

WATCH FOR — previous project failure
--------------------------------------
    val_loss plateau ≈ 0.88 with train_loss 0.56 = generalization gap
    If this happens: DO NOT increase epochs. Instead:
      1. Check feature normalisation
      2. Reduce hidden_channels to 128
      3. Increase dropout to 0.4
      4. Add label smoothing to loss
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config        import load_yaml_config
from wildfire_gnn.utils.reproducibility import set_seed
from wildfire_gnn.models.gnn_pipeline  import GNNPipeline
from wildfire_gnn.evaluation.metrics   import (
    r2_score, mae_score, spearman_rho, brier_score,
    expected_calibration_error, binned_metrics, print_comparison_table,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5A — Train GNN")
    p.add_argument("--config",      default="configs/gnn_config.yaml")
    p.add_argument("--arch",        default="GAT",
                   choices=["GAT", "GCN", "GraphSAGE"])
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--hidden",      type=int,   default=None)
    p.add_argument("--dropout",     type=float, default=None)
    p.add_argument("--loss",        default=None,
                   choices=["gaussian_nll", "mse"])
    p.add_argument("--mc-samples",  type=int,   default=30)
    p.add_argument("--smoke-test",  action="store_true",
                   help="5-epoch smoke test")
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def load_phase4_baselines() -> list[dict]:
    """Load Phase 4 baseline results for comparison table."""
    csv_path = PROJECT_ROOT / "reports" / "tables" / "phase4_baseline_metrics.csv"
    cnn_path = PROJECT_ROOT / "reports" / "tables" / "phase4b_cnn_metrics.csv"
    baselines = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            baselines.append(row.to_dict())
    if cnn_path.exists():
        df2 = pd.read_csv(cnn_path)
        for _, row in df2.iterrows():
            baselines.append(row.to_dict())
    return baselines


def plot_training_curve(history: pd.DataFrame, arch: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss", lw=2)
    ax.plot(history["epoch"], history["val_loss"],   label="Val loss",   lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Gaussian NLL)")
    ax.set_title(f"Phase 5A — {arch} Training Curve\n"
                 f"Watch: val plateau ≈ 0.88 with train 0.56 = overfitting")
    ax.legend()
    ax.grid(alpha=0.3)

    # Mark best val epoch
    best_idx = history["val_loss"].idxmin()
    ax.axvline(history["epoch"][best_idx], color="red", ls="--", alpha=0.5,
               label=f"Best val (epoch {history['epoch'][best_idx]})")
    ax.legend()

    out = fig_dir / f"p5a_{arch.lower()}_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Loss curve saved: {out.name}")


def plot_pred_vs_truth(
    y_true: np.ndarray, y_pred: np.ndarray,
    arch: str, fig_dir: Path
) -> None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(20_000, len(y_true)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.3)
    lo = min(y_true[idx].min(), y_pred[idx].min())
    hi = max(y_true[idx].max(), y_pred[idx].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True Burn Probability")
    ax.set_ylabel("Predicted Burn Probability")
    ax.set_title(f"Phase 5A — {arch}\nPredicted vs True BP (test split)")

    r2  = r2_score(y_true, y_pred)
    mae = mae_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"R²={r2:.4f}  MAE={mae:.4f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8))

    out = fig_dir / f"p5a_{arch.lower()}_pred_vs_truth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Pred vs truth saved: {out.name}")


def main():
    args = parse_args()
    t0   = time.time()

    print("\n" + "="*65)
    print(f"  Phase 5A — GNN Training  [{args.arch}]")
    print("="*65 + "\n")

    # ── Config ─────────────────────────────────────────────────────────────
    config = load_yaml_config(PROJECT_ROOT / args.config)
    set_seed(config["training"]["seed"])

    # CLI overrides
    if args.epochs:  config["training"]["epochs"]           = args.epochs
    if args.lr:      config["training"]["lr"]               = args.lr
    if args.hidden:  config["model"]["hidden_channels"]     = args.hidden
    if args.dropout: config["model"]["dropout"]             = args.dropout
    if args.loss:    config["uncertainty"]["loss_function"] = args.loss
    if args.smoke_test:
        config["training"]["epochs"]  = 5
        config["training"]["patience"] = 5
    config["model"]["architecture"] = args.arch

    p = config["paths"]
    graph_path       = PROJECT_ROOT / p["graph_data"]
    transformer_path = PROJECT_ROOT / p["target_transformer"]
    feat_names_path  = PROJECT_ROOT / p["feature_names"]

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    tbl_dir  = PROJECT_ROOT / "reports" / "tables"
    fig_dir  = PROJECT_ROOT / "reports" / "figures"
    pred_dir = PROJECT_ROOT / "reports" / "predictions"
    for d in [ckpt_dir, tbl_dir, fig_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"gnn_{args.arch.lower()}_best.pt"
    if ckpt_path.exists() and not args.overwrite:
        print(f"  Checkpoint exists: {ckpt_path}")
        print("  Use --overwrite to retrain.\n")
        sys.exit(0)

    # ── Load graph ─────────────────────────────────────────────────────────
    print(f"  Loading graph: {graph_path.name}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    print(f"  Nodes     : {graph.num_nodes:,}")
    print(f"  Features  : {graph.num_node_features}")
    print(f"  Train     : {int(graph.train_mask.sum()):,}")
    print(f"  Val       : {int(graph.val_mask.sum()):,}")
    print(f"  Test      : {int(graph.test_mask.sum()):,}")

    assert graph.num_node_features == config["model"]["in_channels"], (
        f"Feature mismatch: graph has {graph.num_node_features}, "
        f"config expects {config['model']['in_channels']}. "
        f"Update gnn_config.yaml model.in_channels."
    )

    # ── Train ──────────────────────────────────────────────────────────────
    pipeline = GNNPipeline(config)
    pipeline.build_model()
    train_outputs = pipeline.train(graph)
    history = train_outputs["history"]

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics = pipeline.evaluate(
        data             = graph,
        transformer_path = str(transformer_path),
        n_mc_samples     = args.mc_samples,
        verbose          = True,
    )

    # ── Save model ─────────────────────────────────────────────────────────
    pipeline.save(str(ckpt_path))

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics_row = {k: v for k, v in metrics.items()
                   if k not in ("binned", "mc", "y_true_bp", "y_pred_bp")}
    metrics_df  = pd.DataFrame([metrics_row])
    metrics_csv = tbl_dir / f"phase5a_{args.arch.lower()}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  ✓  Metrics saved: {metrics_csv.name}")

    # Binned
    binned_df  = pd.DataFrame(metrics["binned"])
    binned_df["model"] = args.arch
    binned_csv = tbl_dir / f"phase5a_{args.arch.lower()}_binned.csv"
    binned_df.to_csv(binned_csv, index=False)

    # History
    history_csv = tbl_dir / f"phase5a_{args.arch.lower()}_history.csv"
    history.to_csv(history_csv, index=False)

    # Predictions NPZ
    mc   = metrics["mc"]
    pred_npz = pred_dir / f"phase5a_{args.arch.lower()}_preds.npz"
    np.savez_compressed(
        pred_npz,
        y_true_bp   = metrics["y_true_bp"],
        y_pred_bp   = metrics["y_pred_bp"],
        mean_pred   = mc["mean_pred"],
        std_pred    = mc["std_pred"],
        aleatoric   = mc["aleatoric"],
        total_unc   = mc["total_unc"],
    )
    print(f"  ✓  Predictions saved: {pred_npz.name}")

    # ── Figures ────────────────────────────────────────────────────────────
    plot_training_curve(history, args.arch, fig_dir)
    plot_pred_vs_truth(
        metrics["y_true_bp"], metrics["y_pred_bp"],
        args.arch, fig_dir
    )

    # ── Final comparison table ─────────────────────────────────────────────
    baselines   = load_phase4_baselines()
    gnn_result  = {k: v for k, v in metrics_row.items()}
    all_results = baselines + [gnn_result]
    print_comparison_table(all_results)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Phase 5A Complete [{args.arch}] — {elapsed/60:.1f} min")
    print(f"{'='*65}")
    print(f"  R²       = {metrics['r2']:.4f}")
    print(f"  MAE      = {metrics['mae']:.5f}")
    print(f"  Spearman = {metrics['spearman']:.4f}")
    print(f"  ECE      = {metrics['ece']:.5f}")
    print()
    print(f"  vs CNN baseline   (R²=0.7187): "
          f"{'✓ BEATS CNN' if metrics['r2']>0.7187 else '✗ below CNN'}")
    print(f"  vs XGB baseline   (R²=0.6761): "
          f"{'✓ BEATS XGB' if metrics['r2']>0.6761 else '✗ below XGB'}")
    print()
    if metrics['r2'] > 0.7187:
        print("  ✓  STRONG RESULT — Phase 5B (uncertainty calibration) ready")
    elif metrics['r2'] > 0.6761:
        print("  ⚠  Beats tabular but not CNN — review architecture/hyperparams")
    else:
        print("  ✗  Does not beat XGBoost — check features, split, config")


if __name__ == "__main__":
    main()