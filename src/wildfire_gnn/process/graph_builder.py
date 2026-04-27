"""
Graph Construction — Phase 3 of wildfire-uncertainty-gnn.

WHY A GRAPH (not a CNN or flat table)?
---------------------------------------
A graph lets us model the TOPOLOGY of wildfire spread explicitly.
Fire does not spread uniformly — it follows fuel corridors, terrain
channels, and wind patterns. These relationships are spatial:
a high-risk cell is dangerous partly because of what its NEIGHBORS
look like. Graph Neural Networks exploit this through message-passing:
each node aggregates information from its spatial neighbors, which in
turn aggregated from their neighbors.

A CNN on a raster grid also uses spatial context, but treats all
directions equally and uses fixed kernel weights. A GAT (Graph Attention
Network) learns WHICH neighbors matter most for each node — for example,
the upslope neighbor matters more than the downslope one for fire spread.

GRAPH SPECIFICATION
-------------------
Nodes : one per subsampled valid landscape cell (~300,000)
Edges : 8-connected pixel grid (horizontal, vertical, diagonal)
x     : (N, 58) node feature matrix
y     : (N, 1)  quantile-transformed burn probability
pos   : (N, 2)  original raster (row, col) coordinates
edge_index : (2, E) undirected edge pairs

WHY 8-CONNECTED (not 4-connected)?
------------------------------------
Fire spreads diagonally — it can travel across corners of cells.
The Rothermel (1972) fire spread model explicitly accounts for
all 8 directional spread pathways. Using 4-connected edges would
miss this diagonal spread mechanism entirely.

WHY SUBSAMPLING TO ~300,000 NODES?
--------------------------------------
Phase 2 confirmed 11,789,754 valid cells. A full-resolution GAT
graph would require:
  - Node feature matrix : 11.8M × 58 × 4 bytes = 2.7 GB
  - Edge index          : ~94M  × 2 × 8 bytes  = 1.5 GB
  - GAT attention maps  : multiply by 8 heads   = 8× overhead
Total: ~30 GB — exceeds all consumer and most research GPUs.

Spatial grid subsampling (stride=6) selects every 6th valid cell,
giving ~300,000 nodes. At 25m/pixel resolution, this means one
node per 150m × 150m area — still finer than most wildfire
management planning scales (typically 100-500m).

The subsampled graph preserves:
  - Geographic structure (nodes remain on a regular grid)
  - 8-neighbor adjacency (scaled by stride in raster space)
  - Spatial autocorrelation patterns in burn probability
  - Geographic disjointness of train/val/test splits

WHY GEOGRAPHIC SPLIT (not random)?
-------------------------------------
Burn probability has strong spatial autocorrelation — neighboring
cells have similar BP values. A random 80/10/10 split places many
test cells immediately adjacent to training cells. The GNN can
"cheat" by simply aggregating from its training neighbors.

Result of random split: inflated test R² by 0.3-0.5 compared to
the honest geographic evaluation. This was the core failure of the
previous project iteration.

Geographic block split (by raster row, north→south) ensures the
test region is geographically separated from training. Message-
passing cannot cross the split boundary. This is the honest
evaluation of whether the model has learned transferable spatial
patterns.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════
# Spatial subsampling
# ════════════════════════════════════════════════════════════════════════════

def spatial_grid_subsample(
    valid_mask: np.ndarray,
    stride:     int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select every stride-th valid cell from the full valid-cell mask.

    Strategy: build a regular grid of (row, col) candidates spaced by
    stride, then keep only candidates that fall inside the valid mask.

    Parameters
    ----------
    valid_mask : (H, W) bool array — True = valid cell (from Phase 2)
    stride     : spacing between selected nodes in raster pixels.
                 stride=6 → ~300k nodes from 11.8M valid cells.
                 stride=4 → ~600k nodes.
                 stride=1 → all 11.8M nodes (requires mini-batch).

    Returns
    -------
    rows_idx : 1D int array of selected node row positions
    cols_idx : 1D int array of selected node column positions

    Why grid subsampling vs random:
        Random sampling would give an irregular point cloud. Grid
        subsampling preserves the regular pixel structure, so 8-neighbor
        adjacency remains spatially meaningful (every node's neighbor is
        exactly 1 stride away, not at an arbitrary distance).
    """
    H, W = valid_mask.shape
    row_grid = np.arange(0, H, stride)
    col_grid = np.arange(0, W, stride)
    rr, cc   = np.meshgrid(row_grid, col_grid, indexing='ij')
    cand_rows = rr.ravel()
    cand_cols = cc.ravel()

    # Keep only candidates inside the valid mask
    ok        = valid_mask[cand_rows, cand_cols]
    rows_idx  = cand_rows[ok]
    cols_idx  = cand_cols[ok]

    print(f"  Spatial grid subsampling (stride={stride}):")
    print(f"    Grid candidates : {len(cand_rows):,}")
    print(f"    In valid mask   : {ok.sum():,}")
    print(f"    Selected nodes  : {len(rows_idx):,}")
    return rows_idx.astype(np.int64), cols_idx.astype(np.int64)


# ════════════════════════════════════════════════════════════════════════════
# Edge construction
# ════════════════════════════════════════════════════════════════════════════

def build_pixel_grid_edges(
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    stride:   int,
) -> np.ndarray:
    """
    Build 8-connected pixel grid edges between subsampled nodes.

    Two nodes are connected if they are exactly 1 stride apart in
    raster space (horizontally, vertically, or diagonally).
    This mirrors the 8 fire spread directions from Rothermel (1972).

    Parameters
    ----------
    rows_idx, cols_idx : node coordinates in original raster space
    stride             : subsampling stride (step between nodes)

    Returns
    -------
    edge_index : (2, E) int64 — [source_nodes, dest_nodes]
                 Undirected: each edge (i,j) has a corresponding (j,i)

    Implementation:
        1. Build hash map (row, col) → node_index for O(1) lookup
        2. For each node, check all 8 directional neighbors
        3. If neighbor exists in hash map, add both (i→j) and (j→i)
           Note: step 3 adds directed edges. PyG expects undirected edges
           as both directions.

    Edge count estimate:
        Interior node: 8 edges. Boundary node: 3-7 edges.
        For N=300k nodes: ~N × 5 average = ~1.5M edges.
    """
    print(f"\n  Building 8-connected edges (stride={stride})...")

    # O(1) lookup: (row, col) → node index
    node_lookup: dict[tuple[int,int], int] = {
        (int(r), int(c)): idx
        for idx, (r, c) in enumerate(zip(rows_idx.tolist(), cols_idx.tolist()))
    }

    # 8 directional offsets in raster space (scaled by stride)
    offsets = [
        (-stride, -stride), (-stride,  0), (-stride, stride),
        (0,       -stride),                (0,       stride),
        (stride,  -stride), (stride,   0), (stride,  stride),
    ]

    src_list: list[int] = []
    dst_list: list[int] = []

    for idx, (r, c) in enumerate(zip(rows_idx.tolist(), cols_idx.tolist())):
        for dr, dc in offsets:
            nbr_key = (r + dr, c + dc)
            if nbr_key in node_lookup:
                src_list.append(idx)
                dst_list.append(node_lookup[nbr_key])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    n_edges    = edge_index.shape[1]
    n_nodes    = len(rows_idx)
    print(f"    Edges built     : {n_edges:,}")
    print(f"    Avg per node    : {n_edges/n_nodes:.1f}")
    return edge_index


# ════════════════════════════════════════════════════════════════════════════
# Geographic split
# ════════════════════════════════════════════════════════════════════════════

def build_geographic_split(
    rows_idx:  np.ndarray,
    train_rows: list[int],
    val_rows:   list[int],
    test_rows:  list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each node to train/val/test based on original raster row.

    Uses the ORIGINAL raster row index (before subsampling) so that
    geographic disjointness is defined in real-world coordinates,
    not in subsampled node indices.

    Split design (north → south):
        Train: rows 0–1327     northern Greece + islands
        Val:   rows 1328–1517  narrow horizontal band (geographic buffer)
        Test:  rows 1518–7596  central/southern Greece + Peloponnese + Crete + Aegean

    Why a narrow val band?
        The val band acts as a geographic BUFFER between train and test.
        Without it, the boundary train nodes and test nodes would be
        adjacent, allowing partial information flow through their shared
        edges during training.

    Parameters
    ----------
    rows_idx   : original raster row for each node (N,)
    train/val/test_rows : [start, end] inclusive row ranges

    Returns
    -------
    train_mask, val_mask, test_mask : (N,) bool arrays

    Assertions enforced (fail loudly, never silently):
        - No node in both train and val
        - No node in both train and test
        - No node in both val and test
        - val_mask.sum() > 0 (not a zero placeholder)
        - All N nodes covered by exactly one split
    """
    train_mask = ((rows_idx >= train_rows[0]) & (rows_idx <= train_rows[1]))
    val_mask   = ((rows_idx >= val_rows[0])   & (rows_idx <= val_rows[1]))
    test_mask  = ((rows_idx >= test_rows[0])  & (rows_idx <= test_rows[1]))

    N = len(rows_idx)
    print(f"\n  Geographic split (north→south by raster row):")
    print(f"    Train rows {train_rows}: {train_mask.sum():>8,} nodes  "
          f"({100*train_mask.mean():.1f}%)")
    print(f"    Val   rows {val_rows}:   {val_mask.sum():>8,} nodes  "
          f"({100*val_mask.mean():.1f}%)")
    print(f"    Test  rows {test_rows}: {test_mask.sum():>8,} nodes  "
          f"({100*test_mask.mean():.1f}%)")

    # ── Hard assertions — these must NEVER fail silently ──────────────────
    assert (train_mask & val_mask).sum() == 0, \
        "GEOGRAPHIC LEAKAGE: Train and Val regions overlap!"
    assert (train_mask & test_mask).sum() == 0, \
        "GEOGRAPHIC LEAKAGE: Train and Test regions overlap!"
    assert (val_mask & test_mask).sum() == 0, \
        "GEOGRAPHIC LEAKAGE: Val and Test regions overlap!"
    assert val_mask.sum() > 0, \
        "val_mask is all zeros — val_rows range is empty or wrong!"
    assert test_mask.sum() > 0, \
        "test_mask is all zeros — test_rows range is empty or wrong!"

    covered = train_mask.sum() + val_mask.sum() + test_mask.sum()
    assert covered == N, (
        f"Not all nodes covered: {covered:,}/{N:,}. "
        f"Check test_rows[1] >= max raster row ({rows_idx.max()})."
    )

    print(f"  ✓  No geographic overlap. All {N:,} nodes covered.")
    return train_mask, val_mask, test_mask