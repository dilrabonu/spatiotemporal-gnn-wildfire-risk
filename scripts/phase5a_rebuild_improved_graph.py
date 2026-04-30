"""
Phase 5A — Rebuild Graph with All 6 Improvements

This script rebuilds graph_data_enriched.pt with:
  1. Feature normalisation (StandardScaler — fit on train only)
  2. Positional encoding (normalised row/col as 2 extra features)
  3. Edge features (direction dx/dy between connected nodes)

Result: graph has 63 features instead of 61.
        gnn_config.yaml model.in_channels must be updated to 63.

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn

    python scripts/phase5a_rebuild_improved_graph.py

    # Then update gnn_config.yaml (in_channels: 63) and re-run:
    python scripts/phase5a_train_gnn.py --arch GraphSAGE --hidden 256 --dropout 0.25 --loss weighted_mse --overwrite
    python scripts/phase5a_train_gnn.py --arch GAT        --hidden 256 --dropout 0.3  --loss weighted_mse --overwrite
    python scripts/phase5a_train_gnn.py --arch GCN        --hidden 256 --dropout 0.3  --loss weighted_mse --overwrite

WHAT CHANGES vs original graph
--------------------------------
  Before:  graph.x shape = (327405, 61)   no normalisation, no pos
  After:   graph.x shape = (327405, 63)   normalised + pos_row + pos_col
           graph.edge_attr shape = (2511084, 2)  direction features
           graph.scaler_mean, graph.scaler_std saved for inverse at eval
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
from pathlib import Path

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config


def main():
    print("\n" + "="*65)
    print("  Phase 5A — Rebuild Improved Graph")
    print("  Adding: normalisation + positional encoding + edge features")
    print("="*65 + "\n")

    config     = load_yaml_config(PROJECT_ROOT / "configs" / "gnn_config.yaml")
    graph_path = PROJECT_ROOT / config["paths"]["graph_data"]

    if not graph_path.exists():
        print(f"  ✗ Graph not found: {graph_path}")
        print("  Run: python scripts/phase3_build_graph.py first")
        sys.exit(1)

    print(f"  Loading graph: {graph_path.name}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    print(f"  Original graph:")
    print(f"    num_nodes         : {graph.num_nodes:,}")
    print(f"    num_node_features : {graph.num_node_features}")
    print(f"    num_edges         : {graph.num_edges:,}")
    print(f"    train_mask.sum()  : {int(graph.train_mask.sum()):,}")
    print(f"    val_mask.sum()    : {int(graph.val_mask.sum()):,}")
    print(f"    test_mask.sum()   : {int(graph.test_mask.sum()):,}")

    X          = graph.x.numpy().astype(np.float64)
    train_mask = graph.train_mask.numpy()
    rows_idx   = graph.pos[:, 0].numpy()
    cols_idx   = graph.pos[:, 1].numpy()
    edge_index = graph.edge_index.numpy()

    # ── Improvement 5: Feature normalisation (train split only) ───────────
    print("\n  [1/3] Feature normalisation (StandardScaler, train only)...")
    from sklearn.preprocessing import StandardScaler

    X_train = X[train_mask]
    scaler  = StandardScaler()
    scaler.fit(X_train)

    X_scaled = scaler.transform(X)

    # Validate: train features should now be near zero mean, unit std
    X_check = X_scaled[train_mask]
    print(f"    Train mean after scaling : {X_check.mean():.4f}  (target: ~0)")
    print(f"    Train std  after scaling : {X_check.std():.4f}   (target: ~1)")
    assert abs(X_check.mean()) < 0.1, "Normalisation failed: mean too far from 0"
    print(f"  ✓  Features normalised: {X.shape[1]} features, fit on {train_mask.sum():,} train nodes")

    # ── Improvement 1: Positional encoding ────────────────────────────────
    print("\n  [2/3] Adding positional encoding (row/col normalised to [0,1])...")
    H, W = 7597, 7555

    pos_row = (rows_idx / H).reshape(-1, 1)  # (N, 1) — north=0, south=1
    pos_col = (cols_idx / W).reshape(-1, 1)  # (N, 1) — west=0, east=1

    X_enriched = np.hstack([X_scaled, pos_row, pos_col]).astype(np.float32)

    print(f"  ✓  Positional encoding added: {X.shape[1]} → {X_enriched.shape[1]} features")
    print(f"     pos_row: min={pos_row.min():.3f}  max={pos_row.max():.3f}")
    print(f"     pos_col: min={pos_col.min():.3f}  max={pos_col.max():.3f}")

    # ── Improvement 6: Edge features (direction dx/dy) ───────────────────
    print("\n  [3/3] Computing edge direction features...")
    src_nodes = edge_index[0]   # source node indices
    dst_nodes = edge_index[1]   # destination node indices

    # Direction from source to destination in raster space
    edge_dx = (cols_idx[dst_nodes] - cols_idx[src_nodes]).astype(np.float32)
    edge_dy = (rows_idx[dst_nodes] - rows_idx[src_nodes]).astype(np.float32)

    # Normalise to [-1, 1] — max distance = stride * sqrt(2) ≈ 8.5
    max_dist = float(np.sqrt(edge_dx**2 + edge_dy**2).max())
    edge_dx /= max_dist
    edge_dy /= max_dist

    edge_attr = np.stack([edge_dx, edge_dy], axis=1)  # (E, 2)
    print(f"  ✓  Edge features: shape={edge_attr.shape}")
    print(f"     dx range: [{edge_dx.min():.3f}, {edge_dx.max():.3f}]")
    print(f"     dy range: [{edge_dy.min():.3f}, {edge_dy.max():.3f}]")

    # ── Rebuild PyG Data object ────────────────────────────────────────────
    print("\n  Assembling improved graph...")
    improved_graph = Data(
        x           = torch.tensor(X_enriched,  dtype=torch.float32),
        y           = graph.y,
        y_raw       = graph.y_raw,
        pos         = graph.pos,
        edge_index  = graph.edge_index,
        edge_attr   = torch.tensor(edge_attr, dtype=torch.float32),
        train_mask  = graph.train_mask,
        val_mask    = graph.val_mask,
        test_mask   = graph.test_mask,
    )

    # Save scaler parameters in graph for reference
    improved_graph.scaler_mean = torch.tensor(
        scaler.mean_.astype(np.float32))
    improved_graph.scaler_std  = torch.tensor(
        scaler.scale_.astype(np.float32))

    # ── Assertions ─────────────────────────────────────────────────────────
    print("\n  Running assertions...")
    assert improved_graph.num_node_features == 63, \
        f"Expected 63 features, got {improved_graph.num_node_features}"
    assert improved_graph.edge_attr.shape == (graph.num_edges, 2), \
        f"Edge attr shape wrong: {improved_graph.edge_attr.shape}"
    assert not torch.isnan(improved_graph.x).any(), "NaN in x!"
    assert not torch.isinf(improved_graph.x).any(), "Inf in x!"
    assert (improved_graph.train_mask & improved_graph.val_mask).sum() == 0
    assert (improved_graph.train_mask & improved_graph.test_mask).sum() == 0
    assert improved_graph.val_mask.sum() > 0

    print(f"  ✓  num_node_features = {improved_graph.num_node_features}")
    print(f"  ✓  edge_attr shape   = {tuple(improved_graph.edge_attr.shape)}")
    print(f"  ✓  No NaN or Inf in features")
    print(f"  ✓  No geographic split overlap")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(improved_graph, graph_path)
    size_mb = graph_path.stat().st_size / 1024**2
    print(f"\n  ✓  Improved graph saved: {graph_path.name}  ({size_mb:.1f} MB)")

    # Update feature_names.json
    feat_names_path = PROJECT_ROOT / config["paths"]["feature_names"]
    if feat_names_path.exists():
        with open(feat_names_path) as f:
            feature_names = json.load(f)
        feature_names += ["pos_row_norm", "pos_col_norm"]
        with open(feat_names_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"  ✓  feature_names.json updated: {len(feature_names)} features")

    print(f"""
  ══════════════════════════════════════════════════════════
  DONE — Improved graph saved with:
    61 normalised features + 2 positional features = 63
    Edge direction attributes: shape (E, 2)

  REQUIRED: Update gnn_config.yaml before retraining:
    model:
      in_channels: 63   # was 61

  THEN RUN (in order):
    python scripts/phase5a_train_gnn.py --arch GraphSAGE --hidden 256 --dropout 0.25 --loss weighted_mse --overwrite
    python scripts/phase5a_train_gnn.py --arch GAT        --hidden 256 --dropout 0.3  --loss weighted_mse --overwrite
    python scripts/phase5a_train_gnn.py --arch GCN        --hidden 256 --dropout 0.3  --loss weighted_mse --overwrite
    python scripts/phase5a_make_figures.py --all
  ══════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()