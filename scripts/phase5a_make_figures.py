"""
Phase 5A — Figure Generation (standalone script, no notebook required)

Run AFTER phase5a_train_gnn.py completes for at least one architecture.

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn

    # Generate figures for GAT (default)
    python scripts/phase5a_make_figures.py

    # Generate for specific arch
    python scripts/phase5a_make_figures.py --arch GAT
    python scripts/phase5a_make_figures.py --arch GCN
    python scripts/phase5a_make_figures.py --arch GraphSAGE

    # Generate all available arches at once
    python scripts/phase5a_make_figures.py --all

OUTPUTS (saved to reports/figures/)
-------------------------------------
    p5a_{arch}_loss_curve.png           Training/val loss curve
    p5a_{arch}_pred_vs_truth.png        Prediction vs true BP scatter
    p5a_{arch}_uncertainty_vs_bp.png    Uncertainty vs true BP
    p5a_{arch}_binned_mae.png           High-risk tail binned MAE
    p5a_{arch}_binned_spearman.png      High-risk tail Spearman
    p5a_{arch}_error_distribution.png  Prediction error histogram
    p5a_{arch}_uncertainty_map.png      Epistemic vs aleatoric 2D
    p5a_all_comparison.png             All models + baselines R² bar chart

PRE-CONDITIONS
--------------
    reports/tables/phase5a_{arch}_history.csv     (from training)
    reports/predictions/phase5a_{arch}_preds.npz  (from training)
    reports/tables/phase5a_{arch}_metrics.csv     (from training)
    reports/tables/phase4_baseline_metrics.csv    (from Phase 4)
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TBL_DIR  = PROJECT_ROOT / "reports" / "tables"
FIG_DIR  = PROJECT_ROOT / "reports" / "figures"
PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"  ✓  Saved: {path.name}")


def load_preds(arch: str) -> dict | None:
    path = PRED_DIR / f"phase5a_{arch.lower()}_preds.npz"
    if not path.exists():
        print(f"  ✗  Predictions not found: {path.name}")
        print(f"     Run: python scripts/phase5a_train_gnn.py --arch {arch}")
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_history(arch: str) -> pd.DataFrame | None:
    path = TBL_DIR / f"phase5a_{arch.lower()}_history.csv"
    if not path.exists():
        print(f"  ✗  History not found: {path.name}")
        return None
    return pd.read_csv(path)


def load_metrics(arch: str) -> dict | None:
    path = TBL_DIR / f"phase5a_{arch.lower()}_metrics.csv"
    if not path.exists():
        print(f"  ✗  Metrics not found: {path.name}")
        return None
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if len(df) > 0 else None


def load_baselines() -> pd.DataFrame:
    dfs = []
    for fname in [
        "phase4_baseline_metrics.csv",
        "phase4b_cnn_metrics.csv",
    ]:
        p = TBL_DIR / fname
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_binned(y_true: np.ndarray, y_pred: np.ndarray,
                   n_bins: int = 5) -> list[dict]:
    """Compute binned R², MAE, Spearman per quantile bin."""
    from scipy import stats as sp_stats
    thresholds = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    results = []
    for i in range(n_bins):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask   = (y_true >= lo) & (y_true <= hi if i == n_bins - 1 else y_true < hi)
        n = mask.sum()
        if n < 5:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae    = float(np.mean(np.abs(yt - yp)))
        rho, _ = sp_stats.spearmanr(yt, yp)
        results.append({
            "bin": i + 1, "bin_low": float(lo), "bin_high": float(hi),
            "n": int(n), "r2": float(r2), "mae": mae,
            "spearman": float(rho) if not np.isnan(rho) else 0.0,
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# Figure functions
# ════════════════════════════════════════════════════════════════════════════

def fig_loss_curve(arch: str) -> None:
    """Training and validation loss curve with early stopping marker."""
    history = load_history(arch)
    if history is None:
        return

    best_idx   = history["val_loss"].idxmin()
    best_epoch = int(history["epoch"][best_idx])
    best_val   = float(history["val_loss"].min())
    final_train = float(history["train_loss"].iloc[-1])
    final_val   = float(history["val_loss"].iloc[-1])
    gap         = final_val - final_train

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["train_loss"],
            lw=2, label="Train loss")
    ax.plot(history["epoch"], history["val_loss"],
            lw=2, label="Val loss")
    ax.axvline(best_epoch, color="red", ls="--", alpha=0.6,
               label=f"Best val — epoch {best_epoch} (loss={best_val:.4f})")

    # Gap warning
    gap_str = f"gap={gap:.4f}  {'⚠ Overfitting' if gap > 0.3 else '✓ OK'}"
    ax.set_title(
        f"Phase 5A — {arch} Training Curve\n"
        f"train={final_train:.4f}  val={final_val:.4f}  {gap_str}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Gaussian NLL)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, f"p5a_{arch.lower()}_loss_curve.png")


def fig_pred_vs_truth(arch: str, preds: dict) -> None:
    """Prediction vs true burn probability scatter."""
    y_true = preds["y_true_bp"]
    y_pred = preds["y_pred_bp"]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(25_000, len(y_true)), replace=False)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(y_true - y_pred)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true[idx], y_pred[idx], s=3, alpha=0.25, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True Burn Probability", fontsize=12)
    ax.set_ylabel("Predicted Burn Probability", fontsize=12)
    ax.set_title(
        f"Phase 5A — {arch}\nPredicted vs True BP (test split)\n"
        f"R²={r2:.4f}   MAE={mae:.5f}   n={len(y_true):,}"
    )
    ax.text(0.05, 0.93, f"R²={r2:.4f}\nMAE={mae:.5f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8))
    ax.legend()
    plt.tight_layout()
    savefig(fig, f"p5a_{arch.lower()}_pred_vs_truth.png")


def fig_uncertainty_vs_bp(arch: str, preds: dict) -> None:
    """Total uncertainty and epistemic vs aleatoric vs true BP."""
    y_true    = preds["y_true_bp"]
    total_unc = preds.get("total_unc", None)
    epistemic = preds.get("std_pred",  None)
    aleatoric = preds.get("aleatoric", None)

    if total_unc is None:
        print(f"  ⚠  No uncertainty data in {arch} predictions NPZ")
        return

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(20_000, len(y_true)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, unc, label, color in zip(
        axes,
        [epistemic, aleatoric, total_unc],
        ["Epistemic\n(MC Dropout std)", "Aleatoric\n(sqrt(exp(log_var)))",
         "Total Uncertainty\n(sqrt(ep²+al²))"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        if unc is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.scatter(y_true[idx], unc[idx], s=2, alpha=0.2,
                   color=color, rasterized=True)
        ax.set_xlabel("True Burn Probability", fontsize=10)
        ax.set_ylabel("Uncertainty", fontsize=10)
        ax.set_title(f"{label}\nmean={unc.mean():.4f}", fontsize=10)

    fig.suptitle(
        f"Phase 5A — {arch}  Uncertainty Decomposition\n"
        "Higher uncertainty on high-risk cells → Gap 1+2 addressed",
        fontsize=12
    )
    plt.tight_layout()
    savefig(fig, f"p5a_{arch.lower()}_uncertainty_vs_bp.png")


def fig_error_distribution(arch: str, preds: dict) -> None:
    """Distribution of prediction errors (y_pred - y_true)."""
    y_true = preds["y_true_bp"]
    y_pred = preds["y_pred_bp"]
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Error histogram
    ax = axes[0]
    ax.hist(errors, bins=80, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(0,            color="black", lw=1.5, ls="--", label="Zero error")
    ax.axvline(errors.mean(), color="red",   lw=1.5, ls=":",
               label=f"Mean error={errors.mean():.5f}")
    ax.set_xlabel("Prediction Error (pred − true)", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title(f"{arch} — Error Distribution\n"
                 f"mean={errors.mean():.5f}  std={errors.std():.5f}")
    ax.legend()

    # Absolute error vs true BP
    ax2 = axes[1]
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(y_true), min(20_000, len(y_true)), replace=False)
    ax2.scatter(y_true[idx], np.abs(errors[idx]),
                s=2, alpha=0.2, rasterized=True)
    ax2.set_xlabel("True Burn Probability", fontsize=11)
    ax2.set_ylabel("|Prediction Error|")
    ax2.set_title(f"{arch} — |Error| vs True BP\n"
                  "High-risk cells (right) have largest errors")

    fig.suptitle(f"Phase 5A — {arch} Prediction Error Analysis", fontsize=13)
    plt.tight_layout()
    savefig(fig, f"p5a_{arch.lower()}_error_distribution.png")


def fig_binned_evaluation(arch: str, preds: dict) -> None:
    """Binned MAE and Spearman across burn probability quantiles."""
    y_true = preds["y_true_bp"]
    y_pred = preds["y_pred_bp"]
    bins   = compute_binned(y_true, y_pred, n_bins=5)

    if not bins:
        print(f"  ⚠  Could not compute bins for {arch}")
        return

    bin_labels = [f"Bin {b['bin']}\n[{b['bin_low']:.3f},{b['bin_high']:.3f}]"
                  for b in bins]
    mae_vals  = [b["mae"]      for b in bins]
    spr_vals  = [b["spearman"] for b in bins]
    n_vals    = [b["n"]        for b in bins]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # MAE per bin
    ax = axes[0]
    colors = ["#2ecc71" if i < 4 else "#e74c3c" for i in range(len(bins))]
    bars = ax.bar(bin_labels, mae_vals, color=colors, alpha=0.85)
    for bar, n in zip(bars, n_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE (original BP scale)")
    ax.set_title(f"{arch} — MAE per Quantile Bin\nRed = highest risk bin")
    ax.set_xlabel("Burn Probability Quantile Bin (1=low, 5=high)")

    # Spearman per bin
    ax2 = axes[1]
    ax2.bar(bin_labels, spr_vals, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title(f"{arch} — Spearman ρ per Quantile Bin\n"
                  "Rank-ordering ability within each risk level")
    ax2.set_xlabel("Burn Probability Quantile Bin (1=low, 5=high)")
    ax2.set_ylim(-0.1, 1.05)

    fig.suptitle(f"Phase 5A — {arch} Binned Evaluation\n"
                 "High-risk tail (Bin 5) is the hardest and most important",
                 fontsize=12)
    plt.tight_layout()
    savefig(fig, f"p5a_{arch.lower()}_binned_eval.png")


def fig_uncertainty_map(arch: str, preds: dict) -> None:
    """Epistemic vs aleatoric uncertainty 2D scatter (uncertainty decomposition)."""
    epistemic = preds.get("std_pred",  None)
    aleatoric = preds.get("aleatoric", None)

    if epistemic is None or aleatoric is None:
        print(f"  ⚠  Missing uncertainty arrays for {arch}")
        return

    rng = np.random.default_rng(42)
    idx = rng.choice(len(epistemic), min(20_000, len(epistemic)), replace=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        aleatoric[idx], epistemic[idx],
        s=3, alpha=0.2,
        c=preds["y_true_bp"][idx],
        cmap="YlOrRd", rasterized=True
    )
    plt.colorbar(sc, ax=ax, label="True Burn Probability")
    ax.set_xlabel("Aleatoric Uncertainty (label noise from FSim)", fontsize=11)
    ax.set_ylabel("Epistemic Uncertainty (MC Dropout std)", fontsize=11)
    ax.set_title(
        f"Phase 5A — {arch}  Uncertainty Decomposition\n"
        "Colour = true BP. High-BP cells should have high uncertainty.",
        fontsize=11
    )
    plt.tight_layout()
    savefig(fig, f"p5a_{arch.lower()}_uncertainty_map.png")


def fig_all_comparison(trained_arches: list[str]) -> None:
    """
    Bar chart comparing all Phase 4 baselines + all trained GNN arches.
    This is Figure 4 of the paper (comparison table visualised).
    """
    baselines = load_baselines()
    rows = []

    # Baselines
    for _, row in baselines.iterrows():
        rows.append({
            "model": row.get("model", "?"),
            "r2":    row.get("r2",    np.nan),
            "mae":   row.get("mae",   np.nan),
            "spearman": row.get("spearman", np.nan),
        })

    # GNN results
    for arch in trained_arches:
        m = load_metrics(arch)
        if m:
            rows.append({
                "model":    arch,
                "r2":       m.get("r2",    np.nan),
                "mae":      m.get("mae",   np.nan),
                "spearman": m.get("spearman", np.nan),
            })

    if not rows:
        print("  ⚠  No data to plot comparison")
        return

    df = pd.DataFrame(rows).sort_values("r2", ascending=True)

    # Colour: baselines = grey, GNN models = blue family
    gnn_names = set(trained_arches)
    colors = [
        "#2980B9" if m in gnn_names else "#95A5A6"
        for m in df["model"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["model"], df["r2"], color=colors, height=0.6)

    # Reference lines
    ax.axvline(0,      color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.7187, color="darkorange", lw=1.5, ls=":",
               label="CNN baseline (R²=0.7187)")
    ax.axvline(0.6761, color="gray", lw=1.2, ls=":",
               label="XGBoost (R²=0.6761)")

    # Value labels
    for bar, v in zip(bars, df["r2"]):
        if not np.isnan(v):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)

    ax.set_xlabel("R² (test split, original BP scale)", fontsize=12)
    ax.set_title(
        "Phase 5A — All Models Comparison\n"
        "Blue = GNN architectures  |  Grey = Phase 4 baselines",
        fontsize=12
    )
    ax.legend(loc="lower right")
    ax.set_xlim(df["r2"].min() - 0.15, df["r2"].max() + 0.08)
    plt.tight_layout()
    savefig(fig, "p5a_all_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def generate_for_arch(arch: str) -> bool:
    """Generate all figures for one architecture. Returns True if successful."""
    print(f"\n  Generating figures for {arch}...")

    preds = load_preds(arch)
    if preds is None:
        return False

    print(f"  [1/6] Loss curve")
    fig_loss_curve(arch)

    print(f"  [2/6] Prediction vs truth")
    fig_pred_vs_truth(arch, preds)

    print(f"  [3/6] Uncertainty vs BP")
    fig_uncertainty_vs_bp(arch, preds)

    print(f"  [4/6] Error distribution")
    fig_error_distribution(arch, preds)

    print(f"  [5/6] Binned evaluation")
    fig_binned_evaluation(arch, preds)

    print(f"  [6/6] Uncertainty map")
    fig_uncertainty_map(arch, preds)

    return True


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5A — Figure Generation")
    p.add_argument("--arch",    default="GAT",
                   choices=["GAT", "GCN", "GraphSAGE"])
    p.add_argument("--all",     action="store_true",
                   help="Generate figures for all available architectures")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  Phase 5A — Figure Generation")
    print("="*60)

    all_arches    = ["GAT", "GCN", "GraphSAGE"]
    trained_arches = []

    if args.all:
        target_arches = all_arches
    else:
        target_arches = [args.arch]

    for arch in target_arches:
        ok = generate_for_arch(arch)
        if ok:
            trained_arches.append(arch)

    # Combined comparison chart
    print(f"\n  Generating combined comparison chart...")
    fig_all_comparison(trained_arches)

    print(f"\n{'='*60}")
    print(f"  Phase 5A figures complete")
    print(f"  Saved to: {FIG_DIR}")
    print(f"{'='*60}")

    # List all generated files
    p5a_figs = sorted(FIG_DIR.glob("p5a_*.png"))
    print(f"\n  Generated {len(p5a_figs)} figures:")
    for f in p5a_figs:
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<45} {size_kb:>5} KB")


if __name__ == "__main__":
    main()