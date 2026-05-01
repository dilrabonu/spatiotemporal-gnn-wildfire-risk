"""
Phase 5D v2 — Figure Generation for Corrected Intervention Results

Run AFTER: python scripts/phase5d_intervention_v2.py

USAGE
-----
    python scripts/phase5d_make_figures_v2.py
    python scripts/phase5d_make_figures_v2.py --arch GAT

OUTPUTS (paper figures)
-----------------------
    p5d_v2_{arch}_fuel_reduction_delta_map.png
    p5d_v2_{arch}_fuel_reduction_delta_hist.png
    p5d_v2_{arch}_firebreak_delta_map.png
    p5d_v2_{arch}_firebreak_delta_hist.png
    p5d_v2_{arch}_ignition_delta_map.png
    p5d_v2_{arch}_ignition_delta_hist.png
    p5d_v2_{arch}_all_scenarios_comparison.png    ← PAPER FIGURE
    p5d_v2_{arch}_high_risk_effect.png           ← PAPER FIGURE
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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config

config  = load_yaml_config(PROJECT_ROOT / "configs" / "gnn_config.yaml")
p       = config["paths"]

PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
TBL_DIR  = PROJECT_ROOT / "reports" / "tables"
FIG_DIR  = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"  ✓  {out.name}")


def load_effect(scenario_key: str, arch: str) -> dict | None:
    path = PRED_DIR / f"phase5d_v2_{scenario_key}_{arch.lower()}_effects.npz"
    if not path.exists():
        print(f"  ✗  {path.name} not found")
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_graph_test_pos():
    graph_path = PROJECT_ROOT / p["graph_data"]
    g   = torch.load(graph_path, map_location="cpu", weights_only=False)
    pos = g.pos[g.test_mask].numpy()   # (N_test, 2) [row, col]
    return pos


def plot_delta_map_and_hist(
    pos:      np.ndarray,
    effect:   dict,
    title:    str,
    arch:     str,
    key:      str,
) -> None:
    """Two-panel: spatial map + histogram. Primary paper figure."""
    delta   = effect["delta_bp"]
    sig     = effect["significant_mask"]
    lo      = effect["delta_bp_lo_90"]
    hi      = effect["delta_bp_hi_90"]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(delta), min(30_000, len(delta)), replace=False)

    # Dynamic colour scale
    p5, p95 = np.percentile(delta, [2, 98])
    vmax    = max(abs(p5), abs(p95), 0.003)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Spatial delta map
    ax = axes[0]
    sc = ax.scatter(
        pos[idx, 1], -pos[idx, 0],
        c=delta[idx], cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        s=2, alpha=0.6, rasterized=True
    )
    plt.colorbar(sc, ax=ax, label="Δ Burn Probability", shrink=0.8)
    ax.set_title(f"Spatial Δ BP\n{title}")
    ax.set_xlabel("Column (W→E)")
    ax.set_ylabel("Row (S→N)")

    # Panel 2: Only significant nodes
    ax2 = axes[1]
    sig_idx    = idx[sig[idx]]
    nonsig_idx = idx[~sig[idx]]
    ax2.scatter(pos[nonsig_idx,1], -pos[nonsig_idx,0],
                c="lightgrey", s=1, alpha=0.2, rasterized=True,
                label="Not significant")
    if len(sig_idx) > 0:
        sc2 = ax2.scatter(
            pos[sig_idx,1], -pos[sig_idx,0],
            c=delta[sig_idx], cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            s=3, alpha=0.9, rasterized=True
        )
        plt.colorbar(sc2, ax=ax2, label="Δ BP (significant)", shrink=0.8)
    ax2.set_title(f"Significant nodes only\n(90% CI ≠ 0, n={sig.sum():,})")
    ax2.set_xlabel("Column"); ax2.set_ylabel("Row")

    # Panel 3: Delta histogram with 90% CI
    ax3 = axes[2]
    ax3.hist(delta, bins=80, color="#2980B9", alpha=0.8, edgecolor="none",
             label="Δ BP distribution")
    ax3.axvline(0, color="black", lw=1.5, ls="--", label="No effect")
    ax3.axvline(delta.mean(), color="red", lw=2, ls="-",
                label=f"Mean={delta.mean():+.5f}")
    ax3.axvline(np.percentile(lo, 50), color="orange", lw=1.5, ls=":",
                label=f"Median 90% CI lo={np.percentile(lo,50):+.5f}")
    ax3.set_xlabel("Δ Burn Probability")
    ax3.set_ylabel("Node count")
    ax3.set_title("Distribution of intervention effects")
    ax3.legend(fontsize=8)

    pct_red = float((delta < 0).mean() * 100)
    pct_sig = float(sig.mean() * 100)
    fig.suptitle(
        f"Phase 5D v2 — {arch} — {title}\n"
        f"Mean Δ={delta.mean():+.5f}  Reduced: {pct_red:.1f}%  "
        f"Significant: {pct_sig:.1f}%",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, f"p5d_v2_{arch.lower()}_{key}_map_hist.png")


def plot_effect_vs_risk(
    effect:   dict,
    title:    str,
    arch:     str,
    key:      str,
) -> None:
    """Effect size vs original burn probability — shows where intervention helps most."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(effect["delta_bp"]),
                     min(20_000, len(effect["delta_bp"])), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(
        effect["y_orig_bp"][idx],
        effect["delta_bp"][idx],
        s=2, alpha=0.25, rasterized=True,
        c=effect["significant_mask"][idx].astype(float),
        cmap="RdYlGn_r"
    )
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Original Burn Probability")
    ax.set_ylabel("Δ Burn Probability (intervention effect)")
    ax.set_title(f"{title}\nEffect vs original risk\nGreen=significant")

    ax2 = axes[1]
    # Bin by original BP and show mean effect per bin
    bins   = np.linspace(0, effect["y_orig_bp"].max(), 15)
    bin_means  = []
    bin_lows   = []
    bin_highs  = []
    bin_centers = []
    for i in range(len(bins)-1):
        mask_b = (effect["y_orig_bp"] >= bins[i]) & \
                 (effect["y_orig_bp"] < bins[i+1])
        if mask_b.sum() < 5:
            continue
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        bin_means.append(float(effect["delta_bp"][mask_b].mean()))
        bin_lows.append(float(effect["delta_bp_lo_90"][mask_b].mean()))
        bin_highs.append(float(effect["delta_bp_hi_90"][mask_b].mean()))

    bc   = np.array(bin_centers)
    bm   = np.array(bin_means)
    blo  = np.array(bin_lows)
    bhi  = np.array(bin_highs)

    ax2.fill_between(bc, blo, bhi, alpha=0.3, color="#2980B9", label="90% CI")
    ax2.plot(bc, bm, "o-", color="#2980B9", lw=2, ms=5, label="Mean Δ BP")
    ax2.axhline(0, color="black", lw=1, ls="--")
    ax2.set_xlabel("Original Burn Probability")
    ax2.set_ylabel("Mean Δ BP (binned)")
    ax2.set_title("Mean effect by risk level\n(with 90% CI)")
    ax2.legend()

    plt.tight_layout()
    savefig(fig, f"p5d_v2_{arch.lower()}_{key}_effect_vs_risk.png")


def plot_all_scenarios_comparison(
    effects: dict[str, dict],
    pos:     np.ndarray,
    arch:    str,
) -> None:
    """Side-by-side spatial comparison — main paper figure."""
    labels = {
        "fuel_reduction_30pct":       "Fuel Reduction 30%\n(CFL × 0.70, all features)",
        "firebreak":                  "Firebreak Strip\n(CFL = 0 in strip)",
        "ignition_suppression_50pct": "Ignition Suppression 50%\n(Ign × 0.50, all features)",
    }

    n = len(effects)
    if n == 0:
        return

    all_deltas = np.concatenate([e["delta_bp"] for e in effects.values()])
    vmax = min(max(abs(np.percentile(all_deltas, [1,99])).max(), 0.003), 0.05)

    fig, axes = plt.subplots(1, n, figsize=(7*n, 7))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(pos), min(25_000, len(pos)), replace=False)

    for ax, (key, effect) in zip(axes, effects.items()):
        delta  = effect["delta_bp"]
        label  = labels.get(key, key)

        sc = ax.scatter(
            pos[idx, 1], -pos[idx, 0],
            c=delta[idx], cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            s=2, alpha=0.7, rasterized=True
        )
        plt.colorbar(sc, ax=ax, label="Δ BP", shrink=0.8)

        mean_d  = delta.mean()
        pct_r   = (delta < 0).mean() * 100
        pct_sig = effect["significant_mask"].mean() * 100

        ax.set_title(
            f"{label}\n"
            f"Mean Δ={mean_d:+.5f}  Red.: {pct_r:.0f}%  Sig.: {pct_sig:.1f}%",
            fontsize=10
        )
        ax.set_xlabel("Column (W→E)")
        ax.set_ylabel("Row (S→N)")

    fig.suptitle(
        f"Phase 5D v2 — {arch} Intervention Scenarios (Corrected)\n"
        "All derived features scaled · Calibrated 90% uncertainty bounds",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, f"p5d_v2_{arch.lower()}_all_scenarios.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="GAT")
    args   = parser.parse_args()
    arch   = args.arch

    print("\n" + "="*65)
    print(f"  Phase 5D v2 — Figure Generation  [{arch}]")
    print("="*65 + "\n")

    pos = load_graph_test_pos()
    print(f"  Test nodes: {len(pos):,}\n")

    scenarios = {
        "fuel_reduction_30pct":       "Fuel Reduction 30%",
        "firebreak":                  "Firebreak Strip",
        "ignition_suppression_50pct": "Ignition Suppression 50%",
    }

    available = {}
    for key, label in scenarios.items():
        print(f"  [{key}]")
        effect = load_effect(key, arch)
        if effect is None:
            continue
        available[key] = effect

        plot_delta_map_and_hist(pos, effect, label, arch, key)
        plot_effect_vs_risk(effect, label, arch, key)

    if len(available) > 1:
        print("\n  Generating all-scenarios comparison...")
        plot_all_scenarios_comparison(available, pos, arch)

    # Summary table from CSV
    csv_path = TBL_DIR / "phase5d_v2_intervention_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n{'='*65}")
        print("  FINAL RESULTS SUMMARY")
        print(f"{'='*65}")
        cols = ["scenario", "mean_delta_bp", "pct_reduced", "pct_significant"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))

    print(f"\n{'='*65}")
    print(f"  All Phase 5D v2 figures saved to: {FIG_DIR.name}/")
    for f in sorted(FIG_DIR.glob("p5d_v2_*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()