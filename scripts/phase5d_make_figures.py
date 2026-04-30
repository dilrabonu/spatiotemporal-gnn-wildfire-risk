"""
Phase 5D — Intervention Analysis Figure Generation

Run AFTER: python scripts/phase5d_intervention.py

USAGE
-----
    python scripts/phase5d_make_figures.py
    python scripts/phase5d_make_figures.py --arch GAT

OUTPUTS
-------
    p5d_fuel_reduction_delta_map.png     Spatial map of delta BP
    p5d_fuel_reduction_delta_hist.png    Distribution of delta BP
    p5d_fuel_reduction_uncertainty.png   Delta vs uncertainty
    p5d_firebreak_delta_map.png
    p5d_firebreak_delta_hist.png
    p5d_ignition_delta_map.png
    p5d_ignition_delta_hist.png
    p5d_all_scenarios_comparison.png     All 3 scenarios side-by-side
    p5d_significant_effects_map.png      Spatially significant effects
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
import matplotlib.colors as mcolors

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
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"  ✓  {path.name}")


def load_effect(scenario_key: str, arch: str) -> dict | None:
    path = PRED_DIR / f"phase5d_{scenario_key}_{arch.lower()}_effects.npz"
    if not path.exists():
        print(f"  ✗  {path.name} not found")
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_graph_pos_and_mask():
    graph_path = PROJECT_ROOT / p["graph_data"]
    g = torch.load(graph_path, map_location="cpu", weights_only=False)
    pos        = g.pos[g.test_mask].numpy()   # (N_test, 2) [row, col]
    return pos


def plot_delta_map(
    pos:           np.ndarray,
    delta_bp:      np.ndarray,
    title:         str,
    fname:         str,
    significant:   np.ndarray = None,
) -> None:
    """Spatial map of intervention effect coloured by delta BP."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(delta_bp), min(30_000, len(delta_bp)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: delta BP map
    ax = axes[0]
    vmax = max(abs(delta_bp[idx]).max(), 0.005)
    sc = ax.scatter(
        pos[idx, 1], -pos[idx, 0],   # col=x, -row=y (north up)
        c=delta_bp[idx],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        s=2, alpha=0.6, rasterized=True
    )
    plt.colorbar(sc, ax=ax, label="Δ Burn Probability")
    ax.set_xlabel("Column (west → east)")
    ax.set_ylabel("Row (south → north)")
    ax.set_title(f"{title}\nΔ BP: red=increase, blue=decrease")

    # Right: only significant effects
    ax2 = axes[1]
    if significant is not None:
        sig_idx = idx[significant[idx]]
        nonsig_idx = idx[~significant[idx]]
        ax2.scatter(
            pos[nonsig_idx, 1], -pos[nonsig_idx, 0],
            c="lightgrey", s=1, alpha=0.3, rasterized=True,
            label="Not significant"
        )
        if len(sig_idx) > 0:
            sc2 = ax2.scatter(
                pos[sig_idx, 1], -pos[sig_idx, 0],
                c=delta_bp[sig_idx],
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                s=3, alpha=0.8, rasterized=True
            )
            plt.colorbar(sc2, ax=ax2, label="Δ BP (significant)")
    ax2.set_title(f"{title}\nSignificant effects only (90% CI ≠ 0)")
    ax2.set_xlabel("Column"); ax2.set_ylabel("Row")

    pct_reduced = float((delta_bp < 0).mean() * 100)
    fig.suptitle(
        f"Phase 5D — {title}\n"
        f"Mean Δ BP={delta_bp.mean():+.5f}  "
        f"Cells reduced: {pct_reduced:.1f}%",
        fontsize=12
    )
    plt.tight_layout()
    savefig(fig, fname)


def plot_delta_histogram(
    delta_bp:   np.ndarray,
    delta_lo:   np.ndarray,
    delta_hi:   np.ndarray,
    title:      str,
    fname:      str,
) -> None:
    """Distribution of intervention effects with uncertainty bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: delta BP histogram
    ax = axes[0]
    ax.hist(delta_bp, bins=80, color="#2980B9", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="black", lw=1.5, ls="--", label="No effect")
    ax.axvline(delta_bp.mean(), color="red", lw=1.5, ls=":",
               label=f"Mean={delta_bp.mean():+.5f}")
    ax.set_xlabel("Δ Burn Probability (post - pre intervention)")
    ax.set_ylabel("Node count")
    ax.set_title(f"{title}\nDistribution of Δ BP")
    ax.legend()

    # Right: uncertainty bounds
    ax2 = axes[1]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(delta_bp), min(10_000, len(delta_bp)), replace=False)
    idx_sorted = idx[np.argsort(delta_bp[idx])]
    x = np.arange(len(idx_sorted))
    ax2.fill_between(x, delta_lo[idx_sorted], delta_hi[idx_sorted],
                     alpha=0.3, color="#2980B9", label="90% CI")
    ax2.plot(x, delta_bp[idx_sorted], color="#2980B9", lw=1, label="Mean Δ BP")
    ax2.axhline(0, color="black", lw=1, ls="--")
    ax2.set_xlabel("Test nodes (sorted by effect size)")
    ax2.set_ylabel("Δ Burn Probability")
    ax2.set_title(f"{title}\nCalibrated 90% Prediction Interval")
    ax2.legend()

    fig.suptitle(f"Phase 5D — {title}", fontsize=12)
    plt.tight_layout()
    savefig(fig, fname)


def plot_uncertainty_vs_effect(
    delta_bp:     np.ndarray,
    delta_std:    np.ndarray,
    y_orig_bp:    np.ndarray,
    title:        str,
    fname:        str,
) -> None:
    """Uncertainty vs effect size — shows where we are confident about reductions."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(delta_bp), min(20_000, len(delta_bp)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: effect vs uncertainty
    ax = axes[0]
    ax.scatter(delta_bp[idx], delta_std[idx],
               c=y_orig_bp[idx], cmap="YlOrRd",
               s=2, alpha=0.3, rasterized=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Δ Burn Probability")
    ax.set_ylabel("Uncertainty (std)")
    ax.set_title(f"{title}\nEffect size vs uncertainty\nColour = original BP")

    # Right: effect vs original BP
    ax2 = axes[1]
    ax2.scatter(y_orig_bp[idx], delta_bp[idx],
                s=2, alpha=0.3, rasterized=True)
    ax2.axhline(0, color="black", lw=1, ls="--")
    ax2.set_xlabel("Original Burn Probability")
    ax2.set_ylabel("Δ Burn Probability (intervention effect)")
    ax2.set_title(f"{title}\nEffect vs original risk level")

    fig.suptitle(f"Phase 5D — {title}", fontsize=12)
    plt.tight_layout()
    savefig(fig, fname)


def plot_all_scenarios_comparison(
    effects:   dict[str, dict],
    pos:       np.ndarray,
    arch:      str,
) -> None:
    """Side-by-side comparison of all 3 scenarios — paper figure."""
    n_scenarios = len(effects)
    if n_scenarios == 0:
        return

    scenario_labels = {
        "fuel_reduction_30pct":       "Fuel Reduction 30%\n(CFL × 0.70)",
        "firebreak":                  "Firebreak Strip\n(CFL = 0 in strip)",
        "ignition_suppression_50pct": "Ignition Suppression 50%\n(Ign × 0.50)",
    }

    fig, axes = plt.subplots(1, n_scenarios, figsize=(7*n_scenarios, 7))
    if n_scenarios == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(pos), min(25_000, len(pos)), replace=False)

    all_deltas = np.concatenate([e["delta_bp"] for e in effects.values()])
    vmax = min(max(abs(all_deltas).max(), 0.005), 0.05)

    for ax, (key, effect) in zip(axes, effects.items()):
        delta  = effect["delta_bp"]
        label  = scenario_labels.get(key, key)
        mean_d = delta.mean()
        pct_r  = (delta < 0).mean() * 100

        sc = ax.scatter(
            pos[idx, 1], -pos[idx, 0],
            c=delta[idx],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            s=2, alpha=0.7, rasterized=True
        )
        plt.colorbar(sc, ax=ax, label="Δ BP", shrink=0.8)
        ax.set_title(
            f"{label}\nMean Δ={mean_d:+.5f}  "
            f"Reduced: {pct_r:.1f}%",
            fontsize=10
        )
        ax.set_xlabel("Column"); ax.set_ylabel("Row (north up)")

    fig.suptitle(
        f"Phase 5D — {arch} Intervention Scenarios\n"
        "All with calibrated 90% uncertainty bounds",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, f"p5d_{arch.lower()}_all_scenarios_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="GAT")
    args = parser.parse_args()
    arch = args.arch

    print("\n" + "="*65)
    print(f"  Phase 5D — Figure Generation  [{arch}]")
    print("="*65 + "\n")

    # Load graph positions for spatial plots
    print("  Loading test node positions...")
    pos = load_graph_pos_and_mask()
    print(f"  Test nodes: {len(pos):,}")

    # Define scenarios
    scenarios = {
        "fuel_reduction_30pct":       "Fuel Reduction 30%",
        "firebreak":                  "Firebreak Strip",
        "ignition_suppression_50pct": "Ignition Suppression 50%",
    }

    available_effects = {}

    for scenario_key, scenario_label in scenarios.items():
        print(f"\n  Processing: {scenario_label}")
        effect = load_effect(scenario_key, arch)
        if effect is None:
            continue

        available_effects[scenario_key] = effect

        # 1. Spatial delta map
        plot_delta_map(
            pos, effect["delta_bp"],
            title       = scenario_label,
            fname       = f"p5d_{arch.lower()}_{scenario_key}_delta_map.png",
            significant = effect["significant_mask"],
        )

        # 2. Delta histogram + uncertainty
        plot_delta_histogram(
            effect["delta_bp"],
            effect["delta_bp_lo_90"],
            effect["delta_bp_hi_90"],
            title = scenario_label,
            fname = f"p5d_{arch.lower()}_{scenario_key}_delta_hist.png",
        )

        # 3. Uncertainty vs effect
        plot_uncertainty_vs_effect(
            effect["delta_bp"],
            effect["delta_std_bp"],
            effect["y_orig_bp"],
            title = scenario_label,
            fname = f"p5d_{arch.lower()}_{scenario_key}_uncertainty.png",
        )

    # 4. All scenarios comparison (paper figure)
    if len(available_effects) > 1:
        print("\n  Generating all-scenarios comparison figure...")
        plot_all_scenarios_comparison(available_effects, pos, arch)

    # Summary
    print(f"\n{'='*65}")
    print(f"  Phase 5D figures complete. Saved to: {FIG_DIR.name}/")
    print(f"{'='*65}")
    figs = sorted(FIG_DIR.glob("p5d_*.png"))
    for f in figs:
        print(f"    {f.name}")


if __name__ == "__main__":
    main()