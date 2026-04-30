"""
Phase 5A — Save Proper Predictions NPZ from Archive Checkpoints

WHY THIS FILE EXISTS
--------------------
The phase5a_*_preds.npz files currently in reports/predictions/ were
saved from the BAD experimental runs (R²=0.354 etc.) NOT from the
confirmed good models in checkpoints/archive/.

Phase 5B (temperature scaling) needs the correct predictions:
  - y_true_bp        : true burn probability values (original scale)
  - y_pred_bp        : mean prediction (inverse-transformed)
  - mean_pred_t      : mean prediction in TRANSFORMED scale (for temperature scaling)
  - log_var          : predicted log-variance (for Gaussian NLL / aleatoric)
  - std_pred         : MC Dropout std (epistemic uncertainty)
  - aleatoric        : sqrt(exp(mean log_var)) — aleatoric uncertainty
  - total_unc        : combined uncertainty
  - samples          : all 30 MC Dropout raw predictions (transformed scale)
                       needed for Phase 5B calibration plots

This script loads each archive checkpoint, runs 30 MC Dropout passes,
and saves the FULL prediction package per architecture.

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/phase5a_save_predictions.py

OUTPUTS (saved to reports/predictions/)
----------------------------------------
    phase5a_gat_preds.npz        — overwrites old bad file
    phase5a_gcn_preds.npz        — overwrites old bad file
    phase5a_graphsage_preds.npz  — overwrites old bad file

Each NPZ contains these arrays:
    y_true_bp     (57531,)  true burn probability, original scale
    y_pred_bp     (57531,)  predicted BP, original scale (inverse-transformed)
    mean_pred_t   (57531,)  mean prediction, TRANSFORMED scale
    std_pred      (57531,)  epistemic uncertainty (MC Dropout std)
    aleatoric     (57531,)  aleatoric uncertainty (from log_var)
    total_unc     (57531,)  sqrt(epistemic² + aleatoric²)
    log_var_mean  (57531,)  mean log_var across MC passes (for temperature scaling)
    samples       (30, 57531) all MC pass predictions, transformed scale
    test_idx      (57531,)  indices of test nodes in full graph
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
import numpy as np
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.models.gnn import build_model, count_parameters

config   = load_yaml_config(PROJECT_ROOT / "configs" / "gnn_config.yaml")
p        = config["paths"]

GRAPH_PATH  = PROJECT_ROOT / p["graph_data"]
TRANS_PATH  = PROJECT_ROOT / p["target_transformer"]
CKPT_DIR    = PROJECT_ROOT / "checkpoints"
ARCHIVE_DIR = CKPT_DIR / "archive"
PRED_DIR    = PROJECT_ROOT / "reports" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

N_MC = 30


def load_graph():
    g = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    assert abs(float(g.y.mean())) < 0.5, "y not transformed"
    assert (g.train_mask & g.test_mask).sum() == 0
    return g


def load_transformer():
    with open(TRANS_PATH, "rb") as f:
        return pickle.load(f)


def load_model(arch: str) -> torch.nn.Module | None:
    """Try archive first, then main checkpoints directory."""
    ckpt_name  = f"phase5a_{arch.lower()}_best.pt"
    ckpt_path  = ARCHIVE_DIR / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / f"gnn_{arch.lower()}_best.pt"
    if not ckpt_path.exists():
        print(f"  ✗  No checkpoint found for {arch}")
        return None

    print(f"  Loading: {ckpt_path.name}")
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", config)
    m         = saved_cfg["model"]

    model = build_model(
        architecture = arch,
        in_channels  = m["in_channels"],
        hidden       = m["hidden_channels"],
        num_layers   = m.get("num_layers", 4),
        heads        = m.get("heads", 8),
        dropout      = m.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    print(f"  ✓  {arch} — {count_parameters(model):,} parameters")
    return model


def run_mc_dropout_full(
    model: torch.nn.Module,
    graph: "Data",
    n_samples: int = N_MC,
) -> dict[str, np.ndarray]:
    """
    Full-graph MC Dropout inference — dropout ON, no NeighborLoader.

    All 30 passes run on the complete graph so every test node sees
    its FULL neighborhood. This is correct for final predictions.

    Returns arrays for test nodes only (graph.test_mask).
    """
    model.train()   # dropout ON — critical

    sample_means   = []
    sample_logvars = []

    print(f"  Running {n_samples} MC Dropout passes...")
    with torch.no_grad():
        for i in range(n_samples):
            mean, lv = model(graph.x, graph.edge_index)
            sample_means.append(mean[graph.test_mask].numpy().copy())
            sample_logvars.append(lv[graph.test_mask].numpy().copy())
            if (i + 1) % 10 == 0:
                print(f"    Pass {i+1}/{n_samples}")

    samples      = np.stack(sample_means)    # (N_MC, N_test)
    logvar_stack = np.stack(sample_logvars)  # (N_MC, N_test)

    mean_pred_t  = samples.mean(axis=0)     # mean prediction (transformed scale)
    std_pred     = samples.std(axis=0)      # epistemic uncertainty
    log_var_mean = logvar_stack.mean(axis=0)
    aleatoric    = np.sqrt(np.exp(log_var_mean))
    total_unc    = np.sqrt(aleatoric**2 + std_pred**2)

    return {
        "samples":      samples,          # (30, N_test) — for calibration
        "mean_pred_t":  mean_pred_t,      # (N_test,)    — transformed
        "log_var_mean": log_var_mean,     # (N_test,)    — for temperature scaling
        "std_pred":     std_pred,         # (N_test,)    — epistemic
        "aleatoric":    aleatoric,        # (N_test,)    — aleatoric
        "total_unc":    total_unc,        # (N_test,)    — combined
    }


def save_predictions(
    arch:        str,
    mc:          dict,
    graph:       "Data",
    transformer,
) -> None:
    """Save full prediction package as NPZ."""
    # Inverse-transform mean prediction to original BP scale
    y_pred_bp = transformer.inverse_transform(
        mc["mean_pred_t"].reshape(-1, 1)
    ).ravel().astype(np.float32)

    # True burn probability (original scale)
    y_true_bp = graph.y_raw[graph.test_mask].numpy().ravel().astype(np.float32)

    # Test node indices in full graph
    test_idx  = np.where(graph.test_mask.numpy())[0].astype(np.int64)

    # Verify metrics match confirmed results
    ss_res = np.sum((y_true_bp - y_pred_bp)**2)
    ss_tot = np.sum((y_true_bp - y_true_bp.mean())**2)
    r2     = float(1 - ss_res / ss_tot)
    mae    = float(np.mean(np.abs(y_true_bp - y_pred_bp)))

    out_path = PRED_DIR / f"phase5a_{arch.lower()}_preds.npz"
    np.savez_compressed(
        out_path,
        # ── For metric reporting (original BP scale) ──────────────────
        y_true_bp    = y_true_bp,               # (57531,) original BP
        y_pred_bp    = y_pred_bp,               # (57531,) predicted BP
        # ── For Phase 5B temperature scaling (transformed scale) ──────
        mean_pred_t  = mc["mean_pred_t"].astype(np.float32),
        log_var_mean = mc["log_var_mean"].astype(np.float32),
        y_true_t     = graph.y[graph.test_mask].numpy().ravel().astype(np.float32),
        # ── Uncertainty decomposition ─────────────────────────────────
        std_pred     = mc["std_pred"].astype(np.float32),
        aleatoric    = mc["aleatoric"].astype(np.float32),
        total_unc    = mc["total_unc"].astype(np.float32),
        # ── All MC samples (for reliability diagrams in Phase 5B) ─────
        samples      = mc["samples"].astype(np.float32),
        # ── Node indices ──────────────────────────────────────────────
        test_idx     = test_idx,
    )

    size_mb = out_path.stat().st_size / 1024**2
    print(f"\n  ✓  Saved: {out_path.name}  ({size_mb:.1f} MB)")
    print(f"     y_true_bp  shape : {y_true_bp.shape}")
    print(f"     y_pred_bp  shape : {y_pred_bp.shape}")
    print(f"     samples    shape : {mc['samples'].shape}")
    print(f"     R² (verify)      : {r2:.4f}")
    print(f"     MAE (verify)     : {mae:.5f}")

    return r2


def main():
    print("\n" + "="*65)
    print("  Phase 5A — Save Confirmed Predictions from Archive")
    print("="*65 + "\n")

    graph       = load_graph()
    transformer = load_transformer()

    print(f"  Graph: {graph.num_nodes:,} nodes  "
          f"Test: {int(graph.test_mask.sum()):,}\n")

    arches  = ["GAT", "GCN", "GraphSAGE"]
    results = []

    for arch in arches:
        print(f"{'─'*65}")
        print(f"  {arch}")
        print(f"{'─'*65}")

        model = load_model(arch)
        if model is None:
            continue

        mc = run_mc_dropout_full(model, graph, n_samples=N_MC)
        r2 = save_predictions(arch, mc, graph, transformer)
        results.append((arch, r2))
        print()

    # Summary
    print("="*65)
    print("  CONFIRMED PREDICTIONS SAVED")
    print("="*65)
    print(f"  {'Architecture':<15} {'R² (verified)':>15}  File")
    print(f"  {'-'*55}")
    for arch, r2 in results:
        fname = f"phase5a_{arch.lower()}_preds.npz"
        print(f"  {arch:<15} {r2:>15.4f}  {fname}")

    print(f"""
  Each NPZ contains:
    y_true_bp    — true burn probability (original scale)
    y_pred_bp    — predicted burn probability (original scale)
    mean_pred_t  — mean prediction (TRANSFORMED scale) ← Phase 5B input
    log_var_mean — predicted log-variance              ← Phase 5B input
    y_true_t     — true y (TRANSFORMED scale)          ← Phase 5B input
    std_pred     — epistemic uncertainty (MC Dropout std)
    aleatoric    — aleatoric uncertainty (sqrt(exp(log_var)))
    total_unc    — combined uncertainty
    samples      — all 30 MC predictions (transformed) ← reliability diagrams
    test_idx     — node indices in full graph

  Phase 5B can now load phase5a_gat_preds.npz directly.
  No model retraining needed.
    """)


if __name__ == "__main__":
    main()