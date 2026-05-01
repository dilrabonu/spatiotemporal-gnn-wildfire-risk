"""
Phase 5D v2 — Corrected Counterfactual Intervention Analysis

FIX APPLIED
-----------
v1 only modified the raw CFL/Ignition_Prob column (1 of ~11 related features).
The GAT learned to rely on pre-computed multi-scale statistics (CFL_mean_3x3,
CFL_mean_7x7, etc.) which were NOT modified → near-zero effects.

v2 finds ALL feature columns derived from each intervention variable and
scales them all consistently:

  Fuel reduction 30%:
    CFL (raw) × 0.70
    CFL_mean_3x3/7x7/15x15 × 0.70
    CFL_std_3x3/7x7/15x15 × 0.70
    CFL_grad_x, CFL_grad_y × 0.70
    interact_CFL_Ignition × 0.70 (CFL factor only — approximation)
    interact_FSP_CFL × 0.70

  Ignition suppression 50%:
    Ignition_Prob (raw) × 0.50
    Ignition_Prob_mean_3x3/7x7/15x15 × 0.50
    Ignition_Prob_std_3x3/7x7/15x15 × 0.50
    Ignition_Prob_grad_x, Ignition_Prob_grad_y × 0.50
    interact_CFL_Ignition × 0.50
    interact_Ignition_FSP × 0.50

  Firebreak:
    All CFL-derived features → 0.0 in strip rows
    (same column set as fuel reduction)

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/phase5d_intervention_v2.py
    python scripts/phase5d_intervention_v2.py --arch GAT
    python scripts/phase5d_make_figures_v2.py --arch GAT
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.models.gnn import build_model
from wildfire_gnn.models.intervention import (
    load_feature_names,
    build_row_band_mask,
    run_mc_inference,
    compute_intervention_effect,
    summarise_effect,
)

config   = load_yaml_config(PROJECT_ROOT / "configs" / "gnn_config.yaml")
p        = config["paths"]

GRAPH_PATH  = PROJECT_ROOT / p["graph_data"]
TRANS_PATH  = PROJECT_ROOT / p["target_transformer"]
FEAT_PATH   = PROJECT_ROOT / p["feature_names"]
CKPT_DIR    = PROJECT_ROOT / "checkpoints"
ARCHIVE_DIR = CKPT_DIR / "archive"
TBL_DIR     = PROJECT_ROOT / "reports" / "tables"
PRED_DIR    = PROJECT_ROOT / "reports" / "predictions"
TBL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

N_MC = 30


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5D v2 — Corrected Intervention")
    p.add_argument("--arch", default="GAT",
                   choices=["GAT", "GCN", "GraphSAGE"])
    p.add_argument("--n-mc",  type=int, default=N_MC)
    p.add_argument("--firebreak-row-min", type=int, default=5000)
    p.add_argument("--firebreak-row-max", type=int, default=5100)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Feature column discovery
# ════════════════════════════════════════════════════════════════════════════

def find_related_columns(
    keywords:      list[str],
    feature_names: list[str],
    exclude:       list[str] = None,
    verbose:       bool = True,
) -> list[int]:
    """
    Find all feature column indices whose name contains any of the keywords.

    Parameters
    ----------
    keywords      : list of strings to match (case-insensitive)
    feature_names : ordered list from feature_names.json
    exclude       : strings that EXCLUDE a column if present in name
    verbose       : print found columns

    Returns
    -------
    list of column indices
    """
    exclude = exclude or []
    cols    = []
    for i, name in enumerate(feature_names):
        name_upper = name.upper()
        matches_kw  = any(kw.upper() in name_upper for kw in keywords)
        matches_exc = any(ex.upper() in name_upper for ex in exclude)
        if matches_kw and not matches_exc:
            cols.append(i)

    if verbose:
        print(f"    Related columns ({len(cols)}):")
        for c in cols:
            print(f"      [{c:>2}] {feature_names[c]}")

    return cols


def scale_columns(
    x:       torch.Tensor,
    cols:    list[int],
    factor:  float,
    mask:    torch.Tensor = None,
) -> torch.Tensor:
    """
    Scale all specified columns by factor.
    If mask provided, only scale nodes where mask=True.

    Returns a new tensor — original is not modified.
    """
    x_new = x.clone()
    for col in cols:
        if mask is None:
            x_new[:, col] = x_new[:, col] * factor
        else:
            x_new[mask, col] = x_new[mask, col] * factor
    return x_new


def zero_columns(
    x:    torch.Tensor,
    cols: list[int],
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Set all specified columns to 0.0 (for firebreak — no fuel)."""
    x_new = x.clone()
    for col in cols:
        if mask is None:
            x_new[:, col] = 0.0
        else:
            x_new[mask, col] = 0.0
    return x_new


def print_feature_change_summary(
    x_orig:        torch.Tensor,
    x_new:         torch.Tensor,
    cols:          list[int],
    feature_names: list[str],
    scenario:      str,
) -> None:
    """Print before/after statistics for modified features."""
    print(f"\n  Feature change summary — {scenario}:")
    print(f"  {'Feature':<35} {'Original mean':>14} {'Modified mean':>14} {'Change':>8}")
    print(f"  {'-'*75}")
    for col in cols:
        orig_mean = float(x_orig[:, col].mean())
        new_mean  = float(x_new[:, col].mean())
        change    = (new_mean / orig_mean - 1) * 100 if abs(orig_mean) > 1e-9 else 0.0
        name      = feature_names[col]
        print(f"  {name:<35} {orig_mean:>14.4f} {new_mean:>14.4f} {change:>7.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# Model and data loading
# ════════════════════════════════════════════════════════════════════════════

def load_model(arch: str) -> torch.nn.Module | None:
    ckpt_name = f"phase5a_{arch.lower()}_best.pt"
    ckpt_path = ARCHIVE_DIR / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / f"gnn_{arch.lower()}_best.pt"
    if not ckpt_path.exists():
        print(f"  ✗  Checkpoint not found for {arch}")
        return None
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_cfg = ckpt.get("config", config)["model"]
    model = build_model(
        architecture = arch,
        in_channels  = m_cfg["in_channels"],
        hidden       = m_cfg["hidden_channels"],
        num_layers   = m_cfg.get("num_layers", 4),
        heads        = m_cfg.get("heads", 8),
        dropout      = m_cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    print(f"  ✓  {arch} loaded from {ckpt_path.name}")
    return model


def load_temperature(arch: str) -> float:
    temp_path = TBL_DIR / f"phase5b_{arch.lower()}_temperature.csv"
    if temp_path.exists():
        df = pd.read_csv(temp_path)
        T  = float(df["T"].iloc[0])
        print(f"  ✓  Temperature T={T:.4f}")
        return T
    print("  ⚠  Temperature not found — using T=1.0")
    return 1.0


def save_effects(arch: str, scenario_key: str, effect: dict) -> None:
    out = PRED_DIR / f"phase5d_v2_{scenario_key}_{arch.lower()}_effects.npz"
    np.savez_compressed(
        out,
        delta_bp         = effect["delta_bp"].astype(np.float32),
        delta_std_bp     = effect["delta_std_bp"].astype(np.float32),
        delta_bp_lo_90   = effect["delta_bp_lo_90"].astype(np.float32),
        delta_bp_hi_90   = effect["delta_bp_hi_90"].astype(np.float32),
        y_orig_bp        = effect["y_orig_bp"].astype(np.float32),
        y_new_bp         = effect["y_new_bp"].astype(np.float32),
        significant_mask = effect["significant_mask"],
        delta_samples_bp = effect["delta_samples_bp"].astype(np.float32),
    )
    size_mb = out.stat().st_size / 1024**2
    print(f"  ✓  Saved: {out.name}  ({size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════════════
# Intervention scenarios (v2 — all derived features)
# ════════════════════════════════════════════════════════════════════════════

def scenario_fuel_reduction(
    model, graph, transformer, feature_names, n_mc, temperature, mc_orig
) -> tuple[dict, dict]:
    """
    Fuel reduction 30%: scale ALL CFL-derived features by 0.70.

    CFL-derived columns: CFL, CFL_mean_3x3, CFL_std_3x3, CFL_mean_7x7,
    CFL_std_7x7, CFL_mean_15x15, CFL_std_15x15, CFL_grad_x, CFL_grad_y,
    interact_CFL_Ignition, interact_FSP_CFL
    """
    print("\n  ── Scenario 1 (v2): Fuel Reduction 30% — ALL CFL features ──")

    # Find all CFL-related columns
    cfl_cols = find_related_columns(
        keywords      = ["CFL"],
        feature_names = feature_names,
        verbose       = True,
    )

    x_modified = scale_columns(graph.x, cfl_cols, factor=0.70)
    print_feature_change_summary(
        graph.x, x_modified, cfl_cols, feature_names,
        "Fuel Reduction 30%"
    )

    print(f"\n  Running {n_mc} MC passes on modified graph...")
    mc_new = run_mc_inference(
        model, x_modified, graph.edge_index,
        graph.test_mask, n_mc, temperature
    )

    effect  = compute_intervention_effect(mc_orig, mc_new, transformer, temperature)
    summary = summarise_effect(
        effect, "Fuel Reduction 30% (all CFL features)",
        int(graph.test_mask.sum()), verbose=True
    )
    return effect, summary


def scenario_firebreak(
    model, graph, transformer, feature_names, n_mc, temperature,
    mc_orig, row_min, row_max
) -> tuple[dict, dict]:
    """
    Firebreak: set ALL CFL-derived features to 0 in the strip.
    Physical meaning: complete fuel removal → zero crown fire potential.
    """
    print(f"\n  ── Scenario 2 (v2): Firebreak Strip rows {row_min}–{row_max} ──")

    cfl_cols   = find_related_columns(["CFL"], feature_names, verbose=True)
    strip_mask = build_row_band_mask(graph.pos, row_min, row_max)
    n_strip    = int(strip_mask.sum())
    test_in_strip = strip_mask[graph.test_mask]

    print(f"  Strip covers {n_strip:,} nodes")
    print(f"  Test nodes in strip: {int(test_in_strip.sum()):,}")

    # Set all CFL features to 0 in strip (complete fuel removal)
    x_modified = zero_columns(graph.x, cfl_cols, mask=strip_mask)

    print(f"\n  CFL in strip BEFORE: {float(graph.x[strip_mask, cfl_cols[0]].mean()):.4f}")
    print(f"  CFL in strip AFTER : {float(x_modified[strip_mask, cfl_cols[0]].mean()):.4f}")

    print(f"\n  Running {n_mc} MC passes on modified graph...")
    mc_new = run_mc_inference(
        model, x_modified, graph.edge_index,
        graph.test_mask, n_mc, temperature
    )

    effect  = compute_intervention_effect(mc_orig, mc_new, transformer, temperature)
    summary = summarise_effect(
        effect, f"Firebreak (rows {row_min}-{row_max}, all CFL=0)",
        int(graph.test_mask.sum()), verbose=True
    )

    # Report effect in strip vs outside
    if int(test_in_strip.sum()) > 0:
        in_strip   = test_in_strip.numpy()
        delta_in   = effect["delta_bp"][in_strip]
        delta_out  = effect["delta_bp"][~in_strip]
        print(f"\n  Effect IN strip        : mean={delta_in.mean():+.5f}  "
              f"std={delta_in.std():.5f}")
        print(f"  Effect OUTSIDE strip   : mean={delta_out.mean():+.5f}  "
              f"std={delta_out.std():.5f}")
        print(f"  Significant IN strip   : "
              f"{int(effect['significant_mask'][in_strip].sum()):,}")
        summary["mean_delta_in_strip"]    = float(delta_in.mean())
        summary["mean_delta_out_strip"]   = float(delta_out.mean())
        summary["n_significant_in_strip"] = int(
            effect["significant_mask"][in_strip].sum())

    return effect, summary


def scenario_ignition_suppression(
    model, graph, transformer, feature_names, n_mc, temperature, mc_orig
) -> tuple[dict, dict]:
    """
    Ignition suppression 50%: scale ALL Ignition-derived features by 0.50.

    Ignition-derived columns: Ignition_Prob, Ignition_Prob_mean_3x3/7x7/15x15,
    Ignition_Prob_std_3x3/7x7/15x15, Ignition_Prob_grad_x/y,
    interact_CFL_Ignition, interact_Ignition_FSP
    """
    print("\n  ── Scenario 3 (v2): Ignition Suppression 50% — ALL Ign features ──")

    ign_cols = find_related_columns(
        keywords      = ["Ignition", "Ign"],
        feature_names = feature_names,
        verbose       = True,
    )

    x_modified = scale_columns(graph.x, ign_cols, factor=0.50)
    print_feature_change_summary(
        graph.x, x_modified, ign_cols, feature_names,
        "Ignition Suppression 50%"
    )

    print(f"\n  Running {n_mc} MC passes on modified graph...")
    mc_new = run_mc_inference(
        model, x_modified, graph.edge_index,
        graph.test_mask, n_mc, temperature
    )

    effect  = compute_intervention_effect(mc_orig, mc_new, transformer, temperature)
    summary = summarise_effect(
        effect, "Ignition Suppression 50% (all Ign features)",
        int(graph.test_mask.sum()), verbose=True
    )
    return effect, summary


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    arch = args.arch

    print("\n" + "="*65)
    print(f"  Phase 5D v2 — Corrected Intervention Analysis  [{arch}]")
    print(f"  Fix: all derived features scaled, not just raw feature")
    print("="*65 + "\n")

    # Load assets
    print("  Loading assets...")
    graph         = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    with open(TRANS_PATH, "rb") as f:
        transformer = pickle.load(f)
    feature_names = load_feature_names(FEAT_PATH)
    model         = load_model(arch)
    temperature   = load_temperature(arch)

    if model is None:
        sys.exit(1)

    print(f"\n  Graph : {graph.num_nodes:,} nodes  "
          f"Features={graph.num_node_features}")
    print(f"  Test  : {int(graph.test_mask.sum()):,}")
    print(f"  T     : {temperature:.4f}")
    print(f"\n  Total feature names: {len(feature_names)}")

    # Baseline
    print(f"\n  Getting baseline predictions ({args.n_mc} MC passes)...")
    from wildfire_gnn.models.intervention import run_mc_inference
    mc_orig = run_mc_inference(
        model, graph.x, graph.edge_index,
        graph.test_mask, args.n_mc, temperature
    )
    y_true_bp = graph.y_raw[graph.test_mask].numpy().ravel()
    y_orig_bp = transformer.inverse_transform(
        mc_orig["mean_pred"].reshape(-1,1)).ravel()
    ss_res = np.sum((y_true_bp - y_orig_bp)**2)
    ss_tot = np.sum((y_true_bp - y_true_bp.mean())**2)
    r2_check = float(1 - ss_res/ss_tot)
    print(f"  Baseline R² = {r2_check:.4f}  ✓")

    all_summaries = []

    # Scenario 1 — Fuel reduction
    effect1, summary1 = scenario_fuel_reduction(
        model, graph, transformer, feature_names,
        args.n_mc, temperature, mc_orig
    )
    save_effects(arch, "fuel_reduction_30pct", effect1)
    all_summaries.append(summary1)

    # Scenario 2 — Firebreak
    effect2, summary2 = scenario_firebreak(
        model, graph, transformer, feature_names,
        args.n_mc, temperature, mc_orig,
        row_min=args.firebreak_row_min,
        row_max=args.firebreak_row_max,
    )
    save_effects(arch, "firebreak", effect2)
    all_summaries.append(summary2)

    # Scenario 3 — Ignition suppression
    effect3, summary3 = scenario_ignition_suppression(
        model, graph, transformer, feature_names,
        args.n_mc, temperature, mc_orig
    )
    save_effects(arch, "ignition_suppression_50pct", effect3)
    all_summaries.append(summary3)

    # Save summary
    df = pd.DataFrame(all_summaries)
    df["arch"] = arch
    out_csv = TBL_DIR / "phase5d_v2_intervention_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  ✓  Summary saved: {out_csv.name}")

    # Final table
    print(f"\n{'='*70}")
    print(f"  PHASE 5D v2 INTERVENTION SUMMARY — {arch}")
    print(f"{'='*70}")
    print(f"  {'Scenario':<40} {'Mean Δ BP':>10} {'Reduced':>9} {'Significant':>13}")
    print(f"  {'-'*70}")
    for s in all_summaries:
        name = s['scenario'][:40]
        print(f"  {name:<40} {s['mean_delta_bp']:>+10.5f} "
              f"{s['pct_reduced']:>8.1f}% "
              f"{s['pct_significant']:>12.1f}%")
    print(f"{'='*70}")
    print(f"\n  Next: python scripts/phase5d_make_figures_v2.py --arch {arch}")
    print()


if __name__ == "__main__":
    main()