"""
Phase 3 — Graph Construction (complete, runnable script)

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/phase3_build_graph.py
    python scripts/phase3_build_graph.py --stride 6     # default ~300k nodes
    python scripts/phase3_build_graph.py --stride 4     # ~600k nodes
    python scripts/phase3_build_graph.py --overwrite    # rebuild if exists
    python scripts/phase3_build_graph.py --no-dem       # skip DEM features

PRE-CONDITIONS (Phase 2 must be complete)
-----------------------------------------
    data/interim/aligned/Burn_Prob.tif          ← reference
    data/interim/aligned/CFL.tif
    data/interim/aligned/FSP_Index.tif
    data/interim/aligned/Fuel_Models.tif        ← nearest-neighbor resampled
    data/interim/aligned/Ignition_Prob.tif
    data/interim/aligned/Struct_Exp_Index.tif
    data/features/valid_cell_mask.npy           ← (7597,7555) bool
    data/features/target_transformer.pkl        ← QuantileTransformer

OUTPUTS
-------
    data/processed/graph_data_enriched.pt       ← PyG Data object (main output)
    data/features/splits_enriched.npz           ← split indices
    data/features/feature_names.json            ← ordered feature names

GRAPH STRUCTURE
---------------
    graph.x          : (N, 58) float32 — node features
    graph.y          : (N, 1)  float32 — quantile-transformed burn prob
    graph.y_raw      : (N, 1)  float32 — original burn probability
    graph.pos        : (N, 2)  float32 — (original_row, original_col)
    graph.edge_index : (2, E)  int64   — 8-connected pixel grid
    graph.train_mask : (N,)    bool
    graph.val_mask   : (N,)    bool
    graph.test_mask  : (N,)    bool

EXPECTED ASSERTIONS (all must pass before save)
-----------------------------------------------
    graph.num_nodes          ≈ 300,000  (±10k depending on stride)
    graph.num_node_features  == 58 (with DEM) or 53 (without DEM)
    graph.val_mask.sum()     >  0
    graph.test_mask.sum()    >  0
    (train_mask & val_mask).sum()  == 0
    (train_mask & test_mask).sum() == 0
    abs(graph.y.mean())      < 0.5   (near-Gaussian after transform)
    graph.y.std()            > 0.5   (not degenerate)

RUNTIME
-------
    ~5-15 minutes for stride=6 (most time: multi-scale stats computation)
"""

import os, sys, time, json, argparse
import numpy as np
from pathlib import Path

# ── Add src to path ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch_geometric.data import Data

from wildfire_gnn.utils import load_yaml_config, set_seed, get_logger
from wildfire_gnn.process.target_engineering import TargetTransformer
from wildfire_gnn.process.graph_builder import (
    spatial_grid_subsample,
    build_pixel_grid_edges,
    build_geographic_split,
)
from wildfire_gnn.features.feature_engineering import build_all_features

logger = get_logger("phase3")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 — Graph Construction")
    p.add_argument("--config",   default="configs/gnn_config.yaml")
    p.add_argument("--stride",   type=int, default=6)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--no-dem",   action="store_true")
    return p.parse_args()


def main():
    t0   = time.time()
    args = parse_args()

    print("\n" + "="*65)
    print("  Phase 3 — Graph Construction")
    print("="*65 + "\n")

    # ── Config ─────────────────────────────────────────────────────────────
    config = load_yaml_config(PROJECT_ROOT / args.config)
    set_seed(config["training"]["seed"])
    p = config["paths"]

    aligned_dir = PROJECT_ROOT / p["aligned_dir"]
    feat_dir    = PROJECT_ROOT / p["features_dir"]
    proc_dir    = PROJECT_ROOT / "data" / "processed"
    graph_out   = PROJECT_ROOT / p["graph_data"]
    split_out   = PROJECT_ROOT / p["spatial_split_path"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if graph_out.exists() and not args.overwrite:
        print(f"  Graph exists: {graph_out}")
        print("  Use --overwrite to rebuild.\n")
        sys.exit(0)

    # ── Pre-condition checks ───────────────────────────────────────────────
    print("  [0] Pre-condition checks...")
    required = [
        aligned_dir / "Burn_Prob.tif",
        aligned_dir / "CFL.tif",
        aligned_dir / "FSP_Index.tif",
        aligned_dir / "Fuel_Models.tif",
        aligned_dir / "Ignition_Prob.tif",
        aligned_dir / "Struct_Exp_Index.tif",
        PROJECT_ROOT / p["valid_cell_mask"],
        PROJECT_ROOT / p["target_transformer"],
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        print(f"\n  ✗  Missing files from Phase 2:")
        for m in missing:
            print(f"     {m}")
        print("\n  Run phase2_align_rasters.py first.\n")
        sys.exit(1)
    print(f"  ✓  All Phase 2 outputs found\n")

    # ── Load Phase 2 outputs ──────────────────────────────────────────────
    valid_mask  = np.load(PROJECT_ROOT / p["valid_cell_mask"])
    transformer = TargetTransformer.load(PROJECT_ROOT / p["target_transformer"])

    # DEM check
    dem_path = PROJECT_ROOT / p["dem"]
    use_dem  = dem_path.exists() and not args.no_dem
    if use_dem:
        print(f"  ✓  DEM found — terrain features will be included (58 total)")
    else:
        print(f"  ⚠  DEM not found at {dem_path}")
        print("     Run: python scripts/download_dem.py")
        print(f"     Continuing without DEM — 53 features total\n")

    # ── Step 1: Spatial subsampling ───────────────────────────────────────
    print(f"  [1/6] Spatial grid subsampling (stride={args.stride})...")
    rows_idx, cols_idx = spatial_grid_subsample(valid_mask, stride=args.stride)
    N = len(rows_idx)

    # ── Step 2: Feature matrix ────────────────────────────────────────────
    print(f"\n  [2/6] Feature engineering...")
    X, feature_names = build_all_features(
        rows        = rows_idx,
        cols        = cols_idx,
        aligned_dir = aligned_dir,
        stride      = args.stride,
        valid_mask  = valid_mask,
        use_dem     = use_dem,
        kernel_sizes= config["graph"]["feature_groups"]["multiscale_stats"],
    )

    # ── Step 3: Target variable ───────────────────────────────────────────
    print(f"\n  [3/6] Loading and transforming target variable...")
    import rasterio
    with rasterio.open(aligned_dir / "Burn_Prob.tif") as src:
        bp_arr = src.read(1).astype(np.float64)
        nd     = src.nodata
    if nd is not None:
        bp_arr[bp_arr == nd] = np.nan
    bp_arr[bp_arr < -1e30] = np.nan

    y_raw = bp_arr[rows_idx, cols_idx]
    y_raw = np.nan_to_num(y_raw, nan=0.0).astype(np.float32)

    y_t   = transformer.transform(y_raw.astype(np.float64)).astype(np.float32)
    transformer.validate(y_t)   # asserts mean≈0, std≈1

    print(f"  ✓  y_raw : min={y_raw.min():.5f}  max={y_raw.max():.4f}  "
          f"mean={y_raw.mean():.5f}")
    print(f"  ✓  y_t   : min={y_t.min():.3f}   max={y_t.max():.3f}   "
          f"mean={y_t.mean():.4f}")

    # ── Step 4: Edge index ────────────────────────────────────────────────
    print(f"\n  [4/6] Building 8-connected pixel grid edges...")
    edge_index = build_pixel_grid_edges(rows_idx, cols_idx, stride=args.stride)

    # ── Step 5: Geographic split ──────────────────────────────────────────
    print(f"\n  [5/6] Geographic block split...")
    split_cfg = config["split"]
    train_mask, val_mask, test_mask = build_geographic_split(
        rows_idx   = rows_idx,
        train_rows = split_cfg["train_rows"],
        val_rows   = split_cfg["val_rows"],
        test_rows  = [split_cfg["test_rows"][0], int(rows_idx.max())],
    )

    # ── Step 6: Assemble and save PyG graph ───────────────────────────────
    print(f"\n  [6/6] Assembling PyG Data object...")
    graph = Data(
        x          = torch.tensor(X,          dtype=torch.float32),
        y          = torch.tensor(y_t,        dtype=torch.float32).unsqueeze(1),
        y_raw      = torch.tensor(y_raw,      dtype=torch.float32).unsqueeze(1),
        pos        = torch.tensor(
                        np.stack([rows_idx, cols_idx], axis=1),
                        dtype=torch.float32),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        train_mask = torch.tensor(train_mask, dtype=torch.bool),
        val_mask   = torch.tensor(val_mask,   dtype=torch.bool),
        test_mask  = torch.tensor(test_mask,  dtype=torch.bool),
    )

    # ── Final assertions ──────────────────────────────────────────────────
    print("\n  Running final assertions...")

    assert graph.num_nodes == N, \
        f"num_nodes mismatch: {graph.num_nodes} vs {N}"
    assert graph.num_node_features == X.shape[1], \
        f"num_features mismatch: {graph.num_node_features} vs {X.shape[1]}"
    assert graph.val_mask.sum() > 0, \
        "val_mask is all zeros — geographic split is wrong!"
    assert graph.test_mask.sum() > 0, \
        "test_mask is all zeros — check test_rows range!"
    assert (graph.train_mask & graph.val_mask).sum() == 0, \
        "Train/Val geographic overlap detected!"
    assert (graph.train_mask & graph.test_mask).sum() == 0, \
        "Train/Test geographic overlap detected!"
    assert abs(float(graph.y.mean())) < 0.5, \
        f"y mean={graph.y.mean():.4f} — transform may not be applied"
    assert float(graph.y.std()) > 0.5, \
        f"y std={graph.y.std():.4f} — degenerate target"

    print(f"  ✓  num_nodes          = {graph.num_nodes:,}")
    print(f"  ✓  num_node_features  = {graph.num_node_features}")
    print(f"  ✓  num_edges          = {graph.num_edges:,}")
    print(f"  ✓  train_mask.sum()   = {graph.train_mask.sum():,}")
    print(f"  ✓  val_mask.sum()     = {graph.val_mask.sum():,}")
    print(f"  ✓  test_mask.sum()    = {graph.test_mask.sum():,}")
    print(f"  ✓  y mean = {float(graph.y.mean()):.4f}  "
          f"std = {float(graph.y.std()):.4f}")
    print(f"  ✓  No geographic overlap between splits")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(graph, graph_out)
    size_mb = graph_out.stat().st_size / 1024**2
    print(f"\n  ✓  Graph saved: {graph_out}  ({size_mb:.1f} MB)")

    np.savez(split_out,
             train_idx = np.where(train_mask)[0],
             val_idx   = np.where(val_mask)[0],
             test_idx  = np.where(test_mask)[0],
             rows_idx  = rows_idx,
             cols_idx  = cols_idx,
             stride    = np.array([args.stride]))
    print(f"  ✓  Splits saved: {split_out}")

    feat_path = PROJECT_ROOT / p["feature_names"]
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  ✓  Feature names: {feat_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Phase 3 Complete — {elapsed/60:.1f} min")
    print(f"{'='*65}")
    print(f"  Nodes      : {graph.num_nodes:,}  (stride={args.stride})")
    print(f"  Features   : {graph.num_node_features}")
    print(f"  Edges      : {graph.num_edges:,}")
    print(f"  Train/Val/Test: {graph.train_mask.sum():,} / "
          f"{graph.val_mask.sum():,} / {graph.test_mask.sum():,}")
    print(f"  Graph file : {graph_out.name}")
    print(f"\n  Load in notebook:")
    print(f"    graph = torch.load('{graph_out}')")
    print(f"    assert graph.num_nodes > 200_000")
    print(f"    assert graph.num_node_features >= 53")
    print()


if __name__ == "__main__":
    main()