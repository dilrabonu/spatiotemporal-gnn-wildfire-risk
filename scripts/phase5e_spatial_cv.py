"""
Phase 5E — Spatial Cross-Validation (spatial block k-fold)

WHY
---
A single geographic split gives one number (GAT R^2 = 0.766 on the southern
hold-out). A reviewer cannot tell whether that is reliable or a lucky region.
Spatial k-fold rotates the held-out REGION across k geographic bands and
reports mean +/- std, proving the result is stable across all of Greece.

This is spatial block cross-validation (Roberts et al. 2017; Valavi et al.
2019) -- NOT random k-fold, which would leak spatially and contradict the
paper's central methodological argument.

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn

    # Smoke test first (2 folds x 3 epochs, ~10 min) -- verify it runs:
    python scripts/phase5e_spatial_cv.py --arch GAT --smoke-test

    # Real run: GAT-only, 5-fold (~9-10 hours, run overnight)
    python scripts/phase5e_spatial_cv.py --arch GAT --folds 5

    # Later, extend to the family (one at a time):
    python scripts/phase5e_spatial_cv.py --arch GCN --folds 5
    python scripts/phase5e_spatial_cv.py --arch GraphSAGE --folds 5

FOLD DESIGN
-----------
The row axis (graph.pos[:,0], range 6..7590, north->south) is divided into
k equal-width bands. In each fold, one band is the TEST region and the other
k-1 bands are TRAIN. A thin buffer (2% of the row span) between train and
test prevents 8-connected edges from leaking a test node's neighbour into
training.

OUTPUTS
-------
    reports/tables/phase5e_{arch}_cv_folds.csv     -- per-fold metrics
    reports/tables/phase5e_{arch}_cv_summary.csv   -- mean +/- std per metric
    checkpoints/cv/gnn_{arch}_fold{k}.pt           -- per-fold checkpoints
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config          import load_yaml_config
from wildfire_gnn.utils.reproducibility import set_seed
from wildfire_gnn.models.gnn            import build_model, count_parameters, gaussian_nll_loss
from wildfire_gnn.evaluation.metrics    import (
    r2_score, mae_score, spearman_rho, brier_score, expected_calibration_error,
)


def make_spatial_folds(pos_row, n_folds=5, buffer_frac=0.02):
    """Rotate equal-width ROW bands as the held-out test fold.
    Returns a list of (train_mask, test_mask) tensors, one per fold.
    A buffer band between train and test prevents spatial edge leakage."""
    row_min = float(pos_row.min())
    row_max = float(pos_row.max())
    span    = row_max - row_min
    buffer  = span * buffer_frac
    edges   = np.linspace(row_min, row_max, n_folds + 1)
    folds = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        test_mask  = (pos_row >= lo) & (pos_row < hi)
        excl_lo, excl_hi = lo - buffer, hi + buffer
        train_mask = (pos_row < excl_lo) | (pos_row >= excl_hi)
        folds.append((train_mask, test_mask, lo, hi))
    return folds


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5E -- Spatial cross-validation")
    p.add_argument("--config", default="configs/gnn_config.yaml")
    p.add_argument("--arch",   default="GAT",
                   choices=["GAT", "GCN", "GraphSAGE"])
    p.add_argument("--folds",  type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def train_one_fold(graph, train_mask, val_frac, arch, config, device,
                   epochs, smoke=False):
    """Train the model on train_mask, return the trained model.
    A small slice of the train band is held out as validation for early stop."""
    m_cfg = config["model"]; t_cfg = config["training"]

    # Carve a validation slice from the northern edge of the train region
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    n_val = int(len(train_idx) * val_frac)
    perm = torch.randperm(len(train_idx))
    val_sel   = train_idx[perm[:n_val]]
    train_sel = train_idx[perm[n_val:]]
    tr_mask = torch.zeros_like(train_mask); tr_mask[train_sel] = True
    va_mask = torch.zeros_like(train_mask); va_mask[val_sel]   = True

    model = build_model(
        architecture=arch, in_channels=m_cfg["in_channels"],
        hidden=m_cfg["hidden_channels"], num_layers=m_cfg.get("num_layers", 2),
        heads=m_cfg.get("heads", 4), dropout=m_cfg.get("dropout", 0.3),
    ).to(device)

    lr        = t_cfg.get("lr", 1e-3)
    wd        = t_cfg.get("weight_decay", 1e-5)
    patience  = 5 if smoke else t_cfg.get("patience", 15)
    grad_clip = t_cfg.get("gradient_clip", 1.0)
    bs        = int(t_cfg.get("batch_size") or 1024)
    neighbors = t_cfg.get("neighbors", [10, 5])

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tl = NeighborLoader(graph, num_neighbors=neighbors, batch_size=bs,
                        input_nodes=tr_mask, shuffle=True, num_workers=0)
    vl = NeighborLoader(graph, num_neighbors=neighbors, batch_size=bs * 2,
                        input_nodes=va_mask, shuffle=False, num_workers=0)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            mean, log_var = model(batch.x, batch.edge_index)
            n = batch.batch_size
            loss = gaussian_nll_loss(mean[:n], log_var[:n], batch.y[:n].squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                mean, log_var = model(batch.x, batch.edge_index)
                n = batch.batch_size
                vloss += gaussian_nll_loss(mean[:n], log_var[:n],
                                           batch.y[:n].squeeze()).item() * n
                vn += n
        vloss /= max(vn, 1)
        sched.step()
        if vloss < best_val - 1e-4:
            best_val, wait = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_fold(model, graph, test_mask, transformer, config, device):
    """Batched evaluation on the held-out test band (memory-safe)."""
    model.eval()
    nl = config["model"].get("num_layers", 2)
    loader = NeighborLoader(graph, num_neighbors=[-1] * nl, batch_size=512,
                            input_nodes=test_mask, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mean, _ = model(batch.x, batch.edge_index)
            preds.append(mean[:batch.batch_size].cpu().numpy())
    mean_pred = np.concatenate(preds)
    y_pred_bp = transformer.inverse_transform(mean_pred.reshape(-1, 1)).ravel()
    y_true_bp = graph.y_raw[test_mask].cpu().numpy().ravel()
    return {
        "r2":       r2_score(y_true_bp, y_pred_bp),
        "mae":      mae_score(y_true_bp, y_pred_bp),
        "spearman": spearman_rho(y_true_bp, y_pred_bp),
        "brier":    brier_score(y_true_bp, y_pred_bp),
        "ece":      expected_calibration_error(y_true_bp, y_pred_bp),
        "n_test":   int(test_mask.sum()),
    }


def main():
    args = parse_args()
    t0 = time.time()
    print("\n" + "=" * 65)
    print(f"  Phase 5E -- Spatial Cross-Validation  [{args.arch}, {args.folds}-fold]")
    print("=" * 65 + "\n")

    config = load_yaml_config(PROJECT_ROOT / args.config)
    set_seed(config["training"]["seed"])
    epochs = args.epochs or (3 if args.smoke_test else config["training"].get("epochs", 200))
    n_folds = 2 if args.smoke_test else args.folds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = config["paths"]
    graph = torch.load(PROJECT_ROOT / p["graph_data"], map_location="cpu",
                       weights_only=False)
    with open(PROJECT_ROOT / p["target_transformer"], "rb") as f:
        transformer = pickle.load(f)

    tbl_dir = PROJECT_ROOT / "reports" / "tables"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "cv"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pos_row = graph.pos[:, 0]
    folds = make_spatial_folds(pos_row, n_folds=n_folds, buffer_frac=0.02)
    print(f"  Graph: {graph.num_nodes:,} nodes | row range "
          f"{float(pos_row.min()):.0f}-{float(pos_row.max()):.0f}")
    print(f"  {n_folds} spatial folds, 2% buffer between train/test\n")

    results = []
    for k, (train_mask, test_mask, lo, hi) in enumerate(folds, 1):
        assert int((train_mask & test_mask).sum()) == 0, f"Fold {k} leakage!"
        print(f"  Fold {k}/{n_folds}: test band rows [{lo:.0f}, {hi:.0f}) | "
              f"train {int(train_mask.sum()):,} | test {int(test_mask.sum()):,}")
        tf = time.time()
        model = train_one_fold(graph, train_mask, val_frac=0.12, arch=args.arch,
                               config=config, device=device, epochs=epochs,
                               smoke=args.smoke_test)
        metrics = evaluate_fold(model, graph, test_mask, transformer, config, device)
        metrics.update({"fold": k, "row_lo": lo, "row_hi": hi,
                        "minutes": (time.time() - tf) / 60})
        results.append(metrics)
        torch.save({"model_state": model.state_dict(), "fold": k},
                   ckpt_dir / f"gnn_{args.arch.lower()}_fold{k}.pt")
        print(f"    R2={metrics['r2']:.4f}  MAE={metrics['mae']:.5f}  "
              f"ECE={metrics['ece']:.5f}  ({metrics['minutes']:.1f} min)\n")

    df = pd.DataFrame(results)
    df.to_csv(tbl_dir / f"phase5e_{args.arch.lower()}_cv_folds.csv", index=False)

    # Summary: mean +/- std across folds
    summary = {}
    for col in ["r2", "mae", "spearman", "brier", "ece"]:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"]  = df[col].std()
    summary["arch"] = args.arch
    summary["n_folds"] = n_folds
    pd.DataFrame([summary]).to_csv(
        tbl_dir / f"phase5e_{args.arch.lower()}_cv_summary.csv", index=False)

    print("=" * 65)
    print(f"  SPATIAL CV SUMMARY  [{args.arch}, {n_folds}-fold]")
    print("=" * 65)
    print(f"  R2       = {df['r2'].mean():.4f} +/- {df['r2'].std():.4f}")
    print(f"  MAE      = {df['mae'].mean():.5f} +/- {df['mae'].std():.5f}")
    print(f"  Spearman = {df['spearman'].mean():.4f} +/- {df['spearman'].std():.4f}")
    print(f"  Brier    = {df['brier'].mean():.5f} +/- {df['brier'].std():.5f}")
    print(f"  ECE      = {df['ece'].mean():.5f} +/- {df['ece'].std():.5f}")
    print(f"\n  Per-fold R2: {[round(r,3) for r in df['r2'].tolist()]}")
    print(f"  Single-split reference (Phase 5A): R2 = 0.7659")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 65)


