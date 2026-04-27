"""
Feature Engineering — Phase 3 of wildfire-uncertainty-gnn.

WHY 58 FEATURES?
----------------
The GNN must predict burn probability for UNSEEN geographic regions
(the test split is geographically disjoint from training). A GNN with
only 4 raw raster features cannot generalize under a geographic split
because its message-passing cannot "reach across" the split boundary.

The solution: embed spatial context DIRECTLY into each node's feature
vector, so each node carries information about its local neighborhood
BEFORE message-passing even starts. Multi-scale statistics, terrain,
and interaction terms do exactly this.

Feature Groups (total = 58 with DEM, 53 without):
--------------------------------------------------
Group 1 — Base rasters (4):
    CFL, FSP_Index, Ignition_Prob, Struct_Exp_Index
    These are the direct FSim simulation inputs.

Group 2 — DEM terrain (5, requires dem_greece.tif):
    dem_elevation_m, dem_slope_deg, dem_aspect_sin, dem_aspect_cos, dem_twi
    Terrain is the strongest physical driver of fire spread rate
    (Rothermel 1972). Slope doubles fire spread every ~10° increase.
    Aspect controls fuel moisture (south-facing = drier = more flammable).
    WHY SIN/COS ENCODING: aspect is circular (0°=360°=North).
    If we use raw degrees, the model sees 1° and 359° as far apart,
    but both are near-North. Sin/cos makes it continuous:
        North: sin=0,  cos=1
        East:  sin=1,  cos=0
        South: sin=0,  cos=-1
        West:  sin=-1, cos=0

Group 3 — Fuel model one-hot (21):
    Binary indicator for each Scott-Burgan fuel model code present
    in the Greece dataset.
    WHY ONE-HOT NOT ORDINAL: Fuel codes (91, 98, 101...) are categorical.
    The numbers have no ordinal meaning — code 101 is not "more fuel"
    than code 98. One-hot encoding lets the model learn separate weights
    for each fuel type independently.

Group 4 — Interaction terms (3):
    CFL × Ignition_Prob, FSP_Index × CFL, Ignition_Prob × FSP_Index
    WHY: Fire risk is not additive. High fuel load (CFL) only becomes
    dangerous when combined with high ignition probability. A linear
    model cannot capture this — the product term explicitly encodes it.

Group 5 — Multi-scale neighborhood statistics (18):
    Mean and std of CFL, FSP_Index, Ignition_Prob at 3×3, 7×7, 15×15 windows
    = 3 features × 2 stats × 3 kernels = 18
    WHY: Wildfire spread is not a local phenomenon. A high-risk cell
    surrounded by low-risk cells behaves differently than one surrounded
    by high-risk cells. Multi-scale stats give each node awareness of its
    spatial context at 75m, 175m, and 375m radius (at ~25m/pixel).
    This is the most important feature group for geographic generalization.

Group 6 — Spatial gradients (6):
    dx, dy of CFL, FSP_Index, Ignition_Prob
    = 3 features × 2 directions = 6
    WHY: Fire spreads faster uphill and across fuel transitions.
    Spatial gradients in fuel load and ignition probability capture
    where sharp transitions occur — these are natural fire boundaries.

Group 7 — Node degree (1):
    Number of spatial neighbors (4-8 for interior nodes, fewer at boundaries)
    WHY: Boundary nodes have fewer neighbors, meaning less information
    from message-passing. The degree feature lets the model compensate
    by learning different behavior at spatial boundaries (coastlines,
    dataset edges). Required to reach exactly 58 features with DEM.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


# ── Constants ──────────────────────────────────────────────────────────────
NODATA_FLOAT = -9999.0     # used in aligned .tif files


def load_aligned(path: Path) -> np.ndarray:
    """Load aligned raster to float64, with nodata → nan."""
    with rasterio.open(path) as src:
        arr    = src.read(1).astype(np.float64)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -1e30] = np.nan
    return arr


def extract_at_nodes(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                     fill: float = 0.0) -> np.ndarray:
    """Extract values at node positions, filling nan with fill value."""
    vals = arr[rows, cols]
    return np.where(np.isnan(vals), fill, vals).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Group 1 — Base rasters
# ════════════════════════════════════════════════════════════════════════════

def add_base_rasters(
    rows: np.ndarray, cols: np.ndarray, aligned_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load 4 base FSim rasters and extract values at node positions.

    Also returns full raster arrays for use in multi-scale and gradient
    computation (avoids reloading from disk multiple times).

    Returns
    -------
    node_features : dict of name → 1D array (N,)
    full_arrays   : dict of name → 2D array (H, W) — needed downstream
    """
    names = ["CFL", "FSP_Index", "Ignition_Prob", "Struct_Exp_Index"]
    node_features = {}
    full_arrays   = {}

    for name in names:
        arr = load_aligned(aligned_dir / f"{name}.tif")
        node_features[name] = extract_at_nodes(arr, rows, cols)
        full_arrays[name]   = arr
        v = node_features[name]
        print(f"    ✓  {name:<24} mean={v.mean():.4f}  std={v.std():.4f}")

    return node_features, full_arrays


# ════════════════════════════════════════════════════════════════════════════
# Group 2 — DEM terrain features
# ════════════════════════════════════════════════════════════════════════════

def add_dem_features(
    rows: np.ndarray, cols: np.ndarray, aligned_dir: Path
) -> dict[str, np.ndarray]:
    """
    Extract 5 terrain features at node positions.

    Features: elevation_m, slope_deg, aspect_sin, aspect_cos, twi.
    These must already exist as aligned .tif files from Phase 2.
    Run scripts/download_dem.py + phase2_align_rasters.py --overwrite first.

    Returns empty dict if DEM files are not present (pipeline continues
    with 53 features instead of 58).
    """
    dem_names = [
        "dem_elevation_m", "dem_slope_deg", "dem_aspect_sin",
        "dem_aspect_cos", "dem_twi"
    ]
    features = {}
    for name in dem_names:
        path = aligned_dir / f"{name}.tif"
        if not path.exists():
            warnings.warn(f"DEM feature not found: {name}.tif — skipping")
            return {}   # return empty if any DEM file is missing
        arr = load_aligned(path)
        features[name] = extract_at_nodes(arr, rows, cols)
        v = features[name]
        print(f"    ✓  {name:<24} mean={v.mean():.4f}  std={v.std():.4f}")
    return features


# ════════════════════════════════════════════════════════════════════════════
# Group 3 — Fuel model one-hot encoding
# ════════════════════════════════════════════════════════════════════════════

def add_fuel_onehot(
    rows: np.ndarray, cols: np.ndarray, aligned_dir: Path
) -> tuple[dict[str, np.ndarray], list[int]]:
    """
    One-hot encode the Fuel_Models categorical raster.

    Each unique fuel code gets its own binary column: fuel_code_{code}.
    A cell is 1 in the column for its fuel code and 0 in all others.

    WHY ONE-HOT:
    Fuel codes are categorical labels (91=Short Grass, 98=Herbaceous,
    101=Timber Grass, etc.). The model must learn DIFFERENT behavior for
    each fuel type independently. One-hot encoding achieves this by giving
    each code its own learnable weight, while ordinal encoding would force
    the model to assume code 101 is "more" than code 91 in some sense.

    Returns
    -------
    features  : dict of "fuel_{code}" → binary array (N,)
    codes_used : list of unique codes found (for feature_names.json)
    """
    arr = load_aligned(aligned_dir / "Fuel_Models.tif")
    # Convert to int codes (stored as float32 after alignment)
    fuel_at_nodes = arr[rows, cols]
    fuel_at_nodes = np.where(np.isnan(fuel_at_nodes), 0,
                             fuel_at_nodes).astype(np.int32)

    # Unique valid codes (exclude 0 = nodata fill)
    codes = sorted(set(fuel_at_nodes.tolist()) - {0, -9999, 255})
    if len(codes) == 0:
        warnings.warn("No valid fuel codes found — check Fuel_Models alignment")
        codes = list(range(91, 131, 2))   # fallback

    features = {}
    for code in codes:
        features[f"fuel_{code}"] = (fuel_at_nodes == code).astype(np.float32)

    print(f"    ✓  Fuel_Models one-hot: {len(codes)} categories → "
          f"{len(codes)} binary features")
    return features, codes


# ════════════════════════════════════════════════════════════════════════════
# Group 4 — Interaction terms
# ════════════════════════════════════════════════════════════════════════════

def add_interactions(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute 3 multiplicative interaction terms between base features.

    WHY INTERACTIONS:
    Fire risk is multiplicative, not additive. High fuel load (CFL) is
    only dangerous when paired with high ignition probability. A linear
    model or GNN with only base features cannot capture this — each node
    feature is weighted independently. The interaction term
    CFL × Ignition_Prob gives the model an explicit signal for where
    BOTH conditions are simultaneously elevated.

    Features: CFL×Ignition, FSP×CFL, Ignition×FSP (all standardized first).
    """
    def z(x: np.ndarray) -> np.ndarray:
        """Standardize to zero mean, unit std before taking product."""
        std = x.std()
        return (x - x.mean()) / std if std > 0 else x

    cfl = z(base["CFL"])
    ign = z(base["Ignition_Prob"])
    fsp = z(base["FSP_Index"])

    features = {
        "interact_CFL_Ignition":  (cfl * ign).astype(np.float32),
        "interact_FSP_CFL":       (fsp * cfl).astype(np.float32),
        "interact_Ignition_FSP":  (ign * fsp).astype(np.float32),
    }
    print(f"    ✓  Interaction terms: 3 features (CFL×Ign, FSP×CFL, Ign×FSP)")
    return features


# ════════════════════════════════════════════════════════════════════════════
# Group 5 — Multi-scale neighborhood statistics
# ════════════════════════════════════════════════════════════════════════════

def add_multiscale_stats(
    rows:         np.ndarray,
    cols:         np.ndarray,
    full_arrays:  dict[str, np.ndarray],
    kernel_sizes: list[int] = [3, 7, 15],
) -> dict[str, np.ndarray]:
    """
    Compute mean and std of CFL, FSP_Index, Ignition_Prob at multiple
    spatial scales using box-filter approximation.

    WHY MULTI-SCALE STATS:
    This is the most critical feature group for geographic generalization.
    Under a strict geographic split, test nodes are spatially disconnected
    from training nodes. GNN message-passing cannot propagate information
    across the split boundary. Multi-scale statistics solve this by
    embedding neighborhood context DIRECTLY into each node's feature
    vector before training:
      - 3×3 kernel (≈75m): immediate neighbors
      - 7×7 kernel (≈175m): local landscape context
      - 15×15 kernel (≈375m): regional fire environment

    A node surrounded by high-CFL neighbors KNOWS it is in a high-risk
    region, even if its immediate neighbors are test-split nodes the GNN
    never sees during training.

    Features: 3 rasters × 2 stats × 3 kernels = 18
    """
    features = {}
    target_rasters = ["CFL", "FSP_Index", "Ignition_Prob"]

    for feat_name in target_rasters:
        arr       = full_arrays[feat_name]
        arr_clean = np.nan_to_num(arr, nan=0.0)

        for k in kernel_sizes:
            # Box-filter mean
            mean_map = uniform_filter(arr_clean, size=k, mode='reflect')

            # Box-filter std via E[x²] - E[x]² (fast, numerically stable)
            sq_map   = uniform_filter(arr_clean**2, size=k, mode='reflect')
            var_map  = np.maximum(sq_map - mean_map**2, 0.0)
            std_map  = np.sqrt(var_map)

            col_m = f"{feat_name}_mean_{k}x{k}"
            col_s = f"{feat_name}_std_{k}x{k}"
            features[col_m] = mean_map[rows, cols].astype(np.float32)
            features[col_s] = std_map[rows, cols].astype(np.float32)

    n = len(target_rasters) * len(kernel_sizes) * 2
    print(f"    ✓  Multi-scale stats: {len(target_rasters)} rasters × "
          f"{len(kernel_sizes)} kernels × 2 stats = {n} features")
    return features


# ════════════════════════════════════════════════════════════════════════════
# Group 6 — Spatial gradients
# ════════════════════════════════════════════════════════════════════════════

def add_spatial_gradients(
    rows:        np.ndarray,
    cols:        np.ndarray,
    full_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute row-direction and col-direction gradients of 3 key features.

    WHY GRADIENTS:
    Fire spread is driven by spatial transitions in fuel and risk, not
    just local values. A cell at the edge of a high-fuel patch (steep
    CFL gradient) is more likely to be a fire initiation point or spread
    boundary than a cell deep inside a uniform fuel region.
    Spatial gradients in CFL, FSP_Index, and Ignition_Prob explicitly
    encode these transitions for each node.

    Features: 3 rasters × 2 directions (x, y) = 6
    """
    features = {}
    grad_rasters = ["CFL", "FSP_Index", "Ignition_Prob"]

    for feat_name in grad_rasters:
        arr       = full_arrays[feat_name]
        arr_clean = np.nan_to_num(arr, nan=0.0)
        dy, dx    = np.gradient(arr_clean)

        features[f"{feat_name}_grad_x"] = dx[rows, cols].astype(np.float32)
        features[f"{feat_name}_grad_y"] = dy[rows, cols].astype(np.float32)

    n = len(grad_rasters) * 2
    print(f"    ✓  Spatial gradients: {len(grad_rasters)} rasters × 2 = {n} features")
    return features


# ════════════════════════════════════════════════════════════════════════════
# Group 7 — Node degree
# ════════════════════════════════════════════════════════════════════════════

def add_degree_feature(
    rows:       np.ndarray,
    cols:       np.ndarray,
    stride:     int,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute the number of valid 8-neighbors for each node.

    Interior nodes: 8 neighbors
    Edge nodes (coastline, dataset boundary): 3-7 neighbors

    WHY DEGREE:
    GNN message-passing aggregates information from all neighbors.
    A node with only 3 neighbors (coastline cell) receives less
    information than an interior node with 8. Without degree awareness,
    the GNN treats these equally, which is wrong. The degree feature
    lets the model learn different aggregation behavior at boundaries.
    Also: node degree reaches exactly 58 total features with DEM.

    Returns
    -------
    dict with "node_degree" → float32 array (N,), normalized to [0,1]
    """
    # Build quick lookup
    node_set = set(zip(rows.tolist(), cols.tolist()))
    offsets = [
        (-stride, -stride), (-stride, 0), (-stride, stride),
        (0, -stride),                      (0, stride),
        (stride, -stride),  (stride, 0),  (stride, stride),
    ]
    degrees = np.zeros(len(rows), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        deg = sum(1 for dr, dc in offsets if (r+dr, c+dc) in node_set)
        degrees[i] = deg

    # Normalize to [0, 1] (max = 8 neighbors)
    degrees_norm = degrees / 8.0
    print(f"    ✓  Node degree: mean={degrees_norm.mean():.3f}  "
          f"(interior=1.0, boundary<1.0)")
    return {"node_degree": degrees_norm}


# ════════════════════════════════════════════════════════════════════════════
# Master function
# ════════════════════════════════════════════════════════════════════════════

def build_all_features(
    rows:        np.ndarray,
    cols:        np.ndarray,
    aligned_dir: Path,
    stride:      int,
    valid_mask:  np.ndarray,
    use_dem:     bool = True,
    kernel_sizes: list[int] = [3, 7, 15],
) -> tuple[np.ndarray, list[str]]:
    """
    Build the complete (N, F) node feature matrix.

    Parameters
    ----------
    rows, cols  : node positions in original raster space
    aligned_dir : directory containing all aligned .tif files
    stride      : subsampling stride (used for degree computation)
    valid_mask  : full (H,W) valid-cell mask
    use_dem     : include DEM terrain features (requires dem_greece.tif)
    kernel_sizes: spatial scales for multi-scale statistics

    Returns
    -------
    X : np.ndarray (N, F) float32
    feature_names : list of F feature name strings
    """
    print(f"\n  Building feature matrix for {len(rows):,} nodes...")

    feature_dict: dict[str, np.ndarray] = {}
    feature_names: list[str]            = []

    # ── Group 1: Base rasters ──────────────────────────────────────────────
    print("\n  [1/7] Base rasters (4):")
    base_features, full_arrays = add_base_rasters(rows, cols, aligned_dir)
    feature_dict.update(base_features)
    feature_names += list(base_features.keys())

    # ── Group 2: DEM terrain ──────────────────────────────────────────────
    print(f"\n  [2/7] DEM terrain ({'5 features' if use_dem else 'SKIPPED'}):")
    if use_dem:
        dem_features = add_dem_features(rows, cols, aligned_dir)
        if dem_features:
            feature_dict.update(dem_features)
            feature_names += list(dem_features.keys())
        else:
            print("    ⚠  DEM files missing — skipping terrain features")
    else:
        print("    ⚠  DEM disabled — skipping terrain features")

    # ── Group 3: Fuel one-hot ─────────────────────────────────────────────
    print("\n  [3/7] Fuel model one-hot (21 expected):")
    fuel_features, _ = add_fuel_onehot(rows, cols, aligned_dir)
    feature_dict.update(fuel_features)
    feature_names += list(fuel_features.keys())

    # ── Group 4: Interactions ─────────────────────────────────────────────
    print("\n  [4/7] Interaction terms (3):")
    inter_features = add_interactions(base_features)
    feature_dict.update(inter_features)
    feature_names += list(inter_features.keys())

    # ── Group 5: Multi-scale stats ────────────────────────────────────────
    print("\n  [5/7] Multi-scale neighborhood statistics (18):")
    ms_features = add_multiscale_stats(rows, cols, full_arrays, kernel_sizes)
    feature_dict.update(ms_features)
    feature_names += list(ms_features.keys())

    # ── Group 6: Spatial gradients ────────────────────────────────────────
    print("\n  [6/7] Spatial gradients (6):")
    grad_features = add_spatial_gradients(rows, cols, full_arrays)
    feature_dict.update(grad_features)
    feature_names += list(grad_features.keys())

    # ── Group 7: Node degree ──────────────────────────────────────────────
    print("\n  [7/7] Node degree (1):")
    deg_features = add_degree_feature(rows, cols, stride, valid_mask)
    feature_dict.update(deg_features)
    feature_names += list(deg_features.keys())

    # ── Assemble matrix ───────────────────────────────────────────────────
    X = np.column_stack(
        [feature_dict[n] for n in feature_names]
    ).astype(np.float32)

    F = X.shape[1]
    print(f"\n  Feature matrix: {X.shape}  ({F} features)")

    # Count by group
    groups = {
        "Base rasters":    4,
        "DEM terrain":     len([n for n in feature_names if n.startswith("dem_")]),
        "Fuel one-hot":    len([n for n in feature_names if n.startswith("fuel_")]),
        "Interactions":    3,
        "Multi-scale":     len([n for n in feature_names if "_mean_" in n or "_std_" in n]),
        "Gradients":       len([n for n in feature_names if "_grad_" in n]),
        "Degree":          1,
    }
    print("  Feature breakdown:")
    for g, cnt in groups.items():
        print(f"    {g:<20} {cnt}")
    print(f"    {'TOTAL':<20} {sum(groups.values())}")

    return X, feature_names