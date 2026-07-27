"""
Phase 5C — GAT Baselines Training Orchestrator (vanilla GAT + GATv2)

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn

    # Smoke test FIRST (5 epochs, ~2 min) — verify everything runs:
    python scripts/phase5c_train_gat_baselines.py --arch GAT_vanilla --smoke-test

    # Then the real runs:
    python scripts/phase5c_train_gat_baselines.py --arch GAT_vanilla
    python scripts/phase5c_train_gat_baselines.py --arch GATv2

WHAT THIS TRAINS
----------------
    GAT_vanilla : same 2-layer / 4-head / hidden-256 backbone as the full GAT,
                  but a single-output regression head trained with MSE.
                  This is the ABLATION baseline — full GAT vs vanilla GAT
                  isolates the contribution of the Gaussian NLL head.
    GATv2       : same backbone with GATv2Conv (dynamic attention).
                  Competitive baseline — answers "why GAT and not GATv2?".

PRE-CONDITIONS  (all already exist from Phase 3)
--------------
    data/processed/graph_data_enriched.pt
    data/features/target_transformer.pkl
    configs/gnn_config.yaml

OUTPUTS  (mirrors Phase 5A naming)
-------
    checkpoints/gnn_{arch}_best.pt
    reports/tables/phase5c_{arch}_metrics.csv
    reports/tables/phase5c_{arch}_history.csv
    reports/figures/p5c_{arch}_loss.png
    reports/predictions/phase5c_{arch}_preds.npz

NOTE ON LOSS
------------
    These are POINT-prediction baselines. The training loss is forced to MSE
    (config['uncertainty']['loss_function'] = 'mse') regardless of the config
    default, because a regression head has no log_variance to feed Gaussian NLL.
    Evaluation is a single deterministic forward pass in eval() mode — no MC
    Dropout — because a vanilla baseline produces point estimates only.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import NeighborLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config          import load_yaml_config
from wildfire_gnn.utils.reproducibility import set_seed
from wildfire_gnn.models.gnn            import build_model, count_parameters
from wildfire_gnn.evaluation.metrics    import (
    r2_score, mae_score, spearman_rho, brier_score,
    expected_calibration_error, binned_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5C — Train GAT baselines")
    p.add_argument("--config",     default="configs/gnn_config.yaml")
    p.add_argument("--arch",       default="GAT_vanilla",
                   choices=["GAT_vanilla", "GATv2"])
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--smoke-test", action="store_true", help="5-epoch smoke test")
    p.add_argument("--overwrite",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.time()

    print("\n" + "=" * 65)
    print(f"  Phase 5C — GAT Baseline Training  [{args.arch}]")
    print("=" * 65 + "\n")

    config = load_yaml_config(PROJECT_ROOT / args.config)
    set_seed(config["training"]["seed"])

    if args.epochs: config["training"]["epochs"] = args.epochs
    if args.lr:     config["training"]["lr"]     = args.lr
    if args.smoke_test:
        config["training"]["epochs"]  = 5
        config["training"]["patience"] = 5

    m_cfg = config["model"]
    t_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = config["paths"]
    graph_path       = PROJECT_ROOT / p["graph_data"]
    transformer_path = PROJECT_ROOT / p["target_transformer"]

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    tbl_dir  = PROJECT_ROOT / "reports" / "tables"
    fig_dir  = PROJECT_ROOT / "reports" / "figures"
    pred_dir = PROJECT_ROOT / "reports" / "predictions"
    for d in [ckpt_dir, tbl_dir, fig_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"gnn_{args.arch.lower()}_best.pt"
    if ckpt_path.exists() and not args.overwrite:
        print(f"  Checkpoint exists: {ckpt_path.name}  (use --overwrite)\n")
        sys.exit(0)

    # ── Load graph ────────────────────────────────────────────────────────
    print(f"  Loading graph: {graph_path.name}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    print(f"  Nodes {graph.num_nodes:,} | Features {graph.num_node_features} | "
          f"Train {int(graph.train_mask.sum()):,} | "
          f"Test {int(graph.test_mask.sum()):,}")
    assert graph.num_node_features == m_cfg["in_channels"], "Feature mismatch!"
    assert (graph.train_mask & graph.test_mask).sum() == 0, "Train/Test overlap!"

    # ── Build model (regression head, forced) ─────────────────────────────
    model = build_model(
        architecture = args.arch,
        in_channels  = m_cfg["in_channels"],
        hidden       = m_cfg["hidden_channels"],
        num_layers   = m_cfg.get("num_layers", 2),
        heads        = m_cfg.get("heads", 4),
        dropout      = m_cfg.get("dropout", 0.3),
        head_type    = "regression",
    ).to(device)
    print(f"  Model {model.name} | Parameters {count_parameters(model):,} | Device {device}")

    # ── Training (MSE, NeighborLoader — same protocol as Phase 5A) ─────────
    epochs     = t_cfg.get("epochs", 200)
    lr         = t_cfg.get("lr", 1e-3)
    wd         = t_cfg.get("weight_decay", 1e-5)
    patience   = t_cfg.get("patience", 15)
    min_delta  = t_cfg.get("min_delta", 1e-4)
    grad_clip  = t_cfg.get("gradient_clip", 1.0)
    batch_size = int(t_cfg.get("batch_size") or 1024)
    neighbors  = t_cfg.get("neighbors", [10, 5])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse       = nn.MSELoss()

    train_loader = NeighborLoader(graph, num_neighbors=neighbors,
                                  batch_size=batch_size, input_nodes=graph.train_mask,
                                  shuffle=True, num_workers=0)
    val_loader   = NeighborLoader(graph, num_neighbors=neighbors,
                                  batch_size=batch_size * 2, input_nodes=graph.val_mask,
                                  shuffle=False, num_workers=0)

    best_val, best_state, wait = float("inf"), None, 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}")
    print(f"  {'-'*32}")

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tn = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            mean, _ = model(batch.x, batch.edge_index)
            n = batch.batch_size
            loss = mse(mean[:n], batch.y[:n].squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tl += loss.item() * n; tn += n
        train_loss = tl / max(tn, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                mean, _ = model(batch.x, batch.edge_index)
                n = batch.batch_size
                vl += mse(mean[:n], batch.y[:n].squeeze()).item() * n; vn += n
        val_loss = vl / max(vn, 1)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}")

        if val_loss < best_val - min_delta:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  Early stopping at epoch {epoch} (best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  \u2713 Training done: {(time.time()-t0)/60:.1f} min  best_val={best_val:.4f}")

    # ── Evaluate: single deterministic forward pass (point predictor) ─────
    model.eval()
    with torch.no_grad():
        mean, _ = model(graph.x, graph.edge_index)
        mean_pred = mean[graph.test_mask].cpu().numpy()

    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f)
    y_pred_bp = transformer.inverse_transform(mean_pred.reshape(-1, 1)).ravel()
    y_true_bp = graph.y_raw[graph.test_mask].cpu().numpy().ravel()

    metrics = {
        "model":    model.name,
        "r2":       r2_score(y_true_bp, y_pred_bp),
        "mae":      mae_score(y_true_bp, y_pred_bp),
        "spearman": spearman_rho(y_true_bp, y_pred_bp),
        "brier":    brier_score(y_true_bp, y_pred_bp),
        "ece":      expected_calibration_error(y_true_bp, y_pred_bp),
        "n_test":   int(graph.test_mask.sum()),
    }
    print(f"\n  ── {model.name} Results (test split, original BP scale) ──")
    for k in ["r2", "mae", "spearman", "brier", "ece"]:
        print(f"    {k:<9}= {metrics[k]:.5f}")
    print(f"    vs full GAT (0.7659): "
          f"{'above' if metrics['r2'] > 0.7659 else 'below (expected for vanilla)'}")

    # ── Save everything (mirrors Phase 5A) ────────────────────────────────
    torch.save({"model_state": model.state_dict(), "model_name": model.name,
                "config": config, "history": history}, ckpt_path)
    print(f"\n  \u2713 Saved checkpoint: {ckpt_path.name}")

    pd.DataFrame([metrics]).to_csv(
        tbl_dir / f"phase5c_{args.arch.lower()}_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(
        tbl_dir / f"phase5c_{args.arch.lower()}_history.csv", index=False)

    binned = pd.DataFrame(binned_metrics(y_true_bp, y_pred_bp))
    binned["model"] = model.name
    binned.to_csv(tbl_dir / f"phase5c_{args.arch.lower()}_binned.csv", index=False)

    np.savez_compressed(
        pred_dir / f"phase5c_{args.arch.lower()}_preds.npz",
        y_true_bp=y_true_bp, y_pred_bp=y_pred_bp, mean_pred=mean_pred)
    print(f"  \u2713 Saved metrics, history, binned, predictions")

    # ── Loss curve figure ─────────────────────────────────────────────────
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h["epoch"], h["train_loss"], label="Train", lw=2)
    ax.plot(h["epoch"], h["val_loss"],   label="Val",   lw=2)
    best_epoch = h.loc[h["val_loss"].idxmin(), "epoch"]
    ax.axvline(best_epoch, color="red", ls="--", alpha=0.6,
               label=f"Best val (epoch {best_epoch:.0f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Phase 5C {model.name} Training Curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"p5c_{args.arch.lower()}_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713 Saved loss curve")

    print(f"\n{'='*65}")
    print(f"  Phase 5C [{args.arch}] complete — {(time.time()-t0)/60:.1f} min  "
          f"R\u00b2={metrics['r2']:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()