"""
Fix the train/test imbalance from Phase 3 and rebuild graph_data_enriched.pt.

PROBLEM DIAGNOSED
-----------------
Phase 3 geographic split produced severe imbalance:
  Train: rows 0-1327  →  68,557 nodes  (20.9%)  ← TOO SMALL
  Val:   rows 1328-1517 → 11,352 nodes (3.5%)
  Test:  rows 1518-7590 → 247,496 nodes (75.6%) ← TOO LARGE

Root cause: northern Greece rows 0-1327 contain mostly sea and border
nodata, so very few cells are valid. Southern Greece (rows 1518+) covers
all major islands plus Peloponnese — much denser valid-cell coverage.

NEW SPLIT DESIGN
----------------
We keep strict geographic disjointness but redefine the row boundaries
to achieve approximately 70/8/22 proportion by actual node count.

Target: Train ~70%, Val ~8%, Test ~22%
Solution: Extend training to rows 0-4800 (covers northern + central Greece)
          Val: rows 4801-5200 (buffer band)
          Test: rows 5201-7590 (Peloponnese south, Crete, south Aegean)

WHY THIS IS STILL SCIENTIFICALLY VALID
---------------------------------------
The split remains geographically disjoint — no node appears in two splits.
Train covers: Macedonia, Thrace, Epirus, Thessaly, Sterea Ellada, N. Aegean
Test covers:  Southern Peloponnese, Crete, Dodecanese, South Cyclades

The model still cannot "see" any test node during training.
The evaluation is still honest geographic generalization.
The difference: model now has adequate training data (230k+ nodes).

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn

    # Step 1: Preview the new split without rebuilding
    python scripts/fix_split_rebuild_graph.py --preview

    # Step 2: Rebuild with new split
    python scripts/fix_split_rebuild_graph.py --rebuild

    # Step 3: Run Phase 4 with new graph
    python scripts/phase4_run_baselines.py
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preview",  action="store_true",
                   help="Show split statistics without rebuilding")
    p.add_argument("--rebuild",  action="store_true",
                   help="Rebuild graph with new split")
    p.add_argument("--train-end",  type=int, default=4800,
                   help="Last raster row in train split (default 4800)")
    p.add_argument("--val-end",    type=int, default=5200,
                   help="Last raster row in val split (default 5200)")
    p.add_argument("--config",  default="configs/gnn_config.yaml")
    return p.parse_args()


def preview_split(valid_mask: np.ndarray, train_end: int, val_end: int):
    """Show what the new split will look like before committing."""
    H, W = valid_mask.shape

    # Build row index for each valid cell
    rows_full, cols_full = np.where(valid_mask)

    # Current split (Phase 3)
    cur_train = (rows_full <= 1327)
    cur_val   = (rows_full >= 1328) & (rows_full <= 1517)
    cur_test  = (rows_full > 1517)

    # Proposed split
    new_train = (rows_full <= train_end)
    new_val   = (rows_full > train_end) & (rows_full <= val_end)
    new_test  = (rows_full > val_end)

    N = len(rows_full)

    print("\n" + "="*65)
    print("  Split Preview — Full Valid-Cell Population (11.8M nodes)")
    print("="*65)
    print(f"\n  {'Split':<12} {'CURRENT nodes':>15} {'CURRENT %':>10}  "
          f"{'NEW nodes':>12} {'NEW %':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Train':<12} {cur_train.sum():>15,} {100*cur_train.mean():>9.1f}%  "
          f"{new_train.sum():>12,} {100*new_train.mean():>7.1f}%")
    print(f"  {'Val':<12} {cur_val.sum():>15,} {100*cur_val.mean():>9.1f}%  "
          f"{new_val.sum():>12,} {100*new_val.mean():>7.1f}%")
    print(f"  {'Test':<12} {cur_test.sum():>15,} {100*cur_test.mean():>9.1f}%  "
          f"{new_test.sum():>12,} {100*new_test.mean():>7.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<12} {N:>15,}                {N:>12,}")
    print()

    # Row ranges
    print(f"  New row boundaries:")
    print(f"    Train : rows 0 – {train_end}   "
          f"(northern + central Greece)")
    print(f"    Val   : rows {train_end+1} – {val_end}  "
          f"(geographic buffer band)")
    print(f"    Test  : rows {val_end+1} – {rows_full.max()}  "
          f"(southern Greece + Crete + south Aegean)")
    print()

    # BP statistics per new split
    # We need BP values — load baseline CSV for this
    bp_csv = PROJECT_ROOT / "data" / "processed" / "baseline_dataset.csv"
    if bp_csv.exists():
        import pandas as pd
        df = pd.read_csv(bp_csv, usecols=["row", "target"])
        df_rows = df["row"].values
        df_bp   = df["target"].values

        for name, mask in [("Train", new_train), ("Val", new_val),
                            ("Test", new_test)]:
            # Match df rows to valid mask rows (approximate by filtering)
            split_bp = df_bp[mask[:len(df_bp)]] if len(df_bp) == N \
                       else df_bp[df_rows <= train_end if name=="Train"
                                  else (df_rows > train_end) & (df_rows <= val_end)
                                  if name=="Val" else df_rows > val_end]
            if len(split_bp) > 0:
                print(f"    {name} BP: mean={split_bp.mean():.5f}  "
                      f"std={split_bp.std():.5f}")
    print()

    # Check disjointness
    assert (new_train & new_val).sum() == 0,  "Train/Val overlap!"
    assert (new_train & new_test).sum() == 0, "Train/Test overlap!"
    assert (new_val & new_test).sum() == 0,   "Val/Test overlap!"
    covered = new_train.sum() + new_val.sum() + new_test.sum()
    assert covered == N, f"Not all nodes covered: {covered}/{N}"
    print("  ✓  New split is geographically disjoint")
    print("  ✓  All nodes covered")

    return new_train.sum(), new_val.sum(), new_test.sum()


def rebuild_graph(train_end: int, val_end: int, config: dict):
    """
    Rebuild graph_data_enriched.pt with new geographic split.

    Instead of re-running all feature engineering (slow), we:
    1. Load the existing graph
    2. Recompute split masks using new row boundaries
    3. Re-save with updated masks and updated splits_enriched.npz
    """
    import torch
    from wildfire_gnn.utils import set_seed

    set_seed(config["training"]["seed"])
    p = config["paths"]

    graph_path = PROJECT_ROOT / p["graph_data"]
    split_path = PROJECT_ROOT / p["spatial_split_path"]
    feat_path  = PROJECT_ROOT / p["feature_names"]

    print(f"\n  Loading existing graph: {graph_path.name}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    print(f"  Loaded: {graph.num_nodes:,} nodes, {graph.num_node_features} features")

    # Extract original row positions
    rows_idx = graph.pos[:, 0].numpy().astype(int)

    # Compute new masks
    train_mask = (rows_idx <= train_end)
    val_mask   = (rows_idx > train_end) & (rows_idx <= val_end)
    test_mask  = (rows_idx > val_end)

    # Assertions
    assert (train_mask & val_mask).sum() == 0,  "Train/Val overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train/Test overlap!"
    assert val_mask.sum() > 0,  "val_mask is empty!"
    assert test_mask.sum() > 0, "test_mask is empty!"
    covered = train_mask.sum() + val_mask.sum() + test_mask.sum()
    assert covered == graph.num_nodes, \
        f"Not all nodes covered: {covered}/{graph.num_nodes}"

    N = graph.num_nodes
    print(f"\n  New split:")
    print(f"    Train: {train_mask.sum():>8,}  ({100*train_mask.mean():.1f}%)  "
          f"rows 0–{train_end}")
    print(f"    Val  : {val_mask.sum():>8,}  ({100*val_mask.mean():.1f}%)  "
          f"rows {train_end+1}–{val_end}")
    print(f"    Test : {test_mask.sum():>8,}  ({100*test_mask.mean():.1f}%)  "
          f"rows {val_end+1}–{rows_idx.max()}")

    # Update graph object
    graph.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    graph.val_mask   = torch.tensor(val_mask,   dtype=torch.bool)
    graph.test_mask  = torch.tensor(test_mask,  dtype=torch.bool)

    # Save updated graph
    torch.save(graph, graph_path)
    print(f"\n  ✓  Graph updated and saved: {graph_path}")

    # Save updated splits
    np.savez(
        split_path,
        train_idx  = np.where(train_mask)[0],
        val_idx    = np.where(val_mask)[0],
        test_idx   = np.where(test_mask)[0],
        rows_idx   = rows_idx,
        cols_idx   = graph.pos[:, 1].numpy().astype(int),
        train_end  = np.array([train_end]),
        val_end    = np.array([val_end]),
        split_type = np.array(["balanced_geographic_block"]),
    )
    print(f"  ✓  Splits saved: {split_path}")

    # Final verification
    print(f"\n  Final assertions:")
    assert int(graph.train_mask.sum()) == train_mask.sum()
    assert int(graph.val_mask.sum())   == val_mask.sum()
    assert int(graph.test_mask.sum())  == test_mask.sum()
    assert (graph.train_mask & graph.val_mask).sum() == 0
    assert (graph.train_mask & graph.test_mask).sum() == 0
    print(f"  ✓  train_mask = {int(graph.train_mask.sum()):,}")
    print(f"  ✓  val_mask   = {int(graph.val_mask.sum()):,}")
    print(f"  ✓  test_mask  = {int(graph.test_mask.sum()):,}")
    print(f"  ✓  No overlap — geographically disjoint")

    return graph


def main():
    args = parse_args()

    print("\n" + "="*65)
    print("  Fix Train/Test Imbalance — Geographic Split Rebuild")
    print("="*65)

    # Load config
    from wildfire_gnn.utils import load_yaml_config
    config = load_yaml_config(PROJECT_ROOT / args.config)

    # Load valid mask for preview
    mask_path  = PROJECT_ROOT / config["paths"]["valid_cell_mask"]
    valid_mask = np.load(mask_path)
    print(f"\n  Valid-cell mask: {valid_mask.shape}  "
          f"valid={valid_mask.sum():,}")
    print(f"  Proposed split: train_end={args.train_end}, "
          f"val_end={args.val_end}")

    if args.preview or (not args.rebuild):
        n_train, n_val, n_test = preview_split(
            valid_mask, args.train_end, args.val_end
        )
        if not args.rebuild:
            print("  Run with --rebuild to apply this split.")
            print("  Run with --preview --rebuild to preview then rebuild.")
            return

    if args.rebuild:
        t0 = time.time()
        graph = rebuild_graph(args.train_end, args.val_end, config)
        elapsed = time.time() - t0

        print(f"\n{'='*65}")
        print(f"  Split fix complete — {elapsed:.1f}s")
        print(f"{'='*65}")
        print(f"\n  Graph is ready for Phase 4 baselines.")
        print(f"  Run: python scripts/phase4_run_baselines.py")
        print()


if __name__ == "__main__":
    main()