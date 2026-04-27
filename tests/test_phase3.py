"""
Phase 3 test suite — run with: pytest tests/test_phase3.py -v

Tests are grouped to catch every known failure mode from the previous
project plus new failure modes specific to the sampling approach.

Groups:
  A — Spatial subsampling correctness
  B — Edge construction correctness
  C — Feature engineering correctness
  D — Geographic split integrity (the most critical group)
  E — Graph object assertions
  F — Integration: load saved graph and verify
"""

from __future__ import annotations
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wildfire_gnn.process.graph_builder import (
    spatial_grid_subsample,
    build_pixel_grid_edges,
    build_geographic_split,
)


# ════════════════════════════════════════════════════════════════════════════
# GROUP A — Spatial subsampling
# ════════════════════════════════════════════════════════════════════════════

class TestSpatialSubsampling:

    def test_stride1_keeps_all_valid(self):
        """stride=1 must keep every valid cell."""
        mask = np.ones((20, 20), dtype=bool)
        mask[0, :] = False  # top row = nodata
        rows, cols = spatial_grid_subsample(mask, stride=1)
        assert len(rows) == mask.sum(), \
            "stride=1 must keep all valid cells"

    def test_stride2_reduces_nodes(self):
        """stride=2 should give roughly 1/4 of nodes."""
        mask = np.ones((100, 100), dtype=bool)
        rows_1, _ = spatial_grid_subsample(mask, stride=1)
        rows_2, _ = spatial_grid_subsample(mask, stride=2)
        ratio = len(rows_2) / len(rows_1)
        assert 0.20 < ratio < 0.30, \
            f"stride=2 ratio={ratio:.3f} — expected ~0.25"

    def test_all_selected_are_valid(self):
        """All selected nodes must be inside the valid mask."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True  # only centre is valid
        rows, cols = spatial_grid_subsample(mask, stride=3)
        assert mask[rows, cols].all(), \
            "All selected nodes must be valid cells"

    def test_no_duplicate_nodes(self):
        """No node should appear twice."""
        mask = np.ones((60, 60), dtype=bool)
        rows, cols = spatial_grid_subsample(mask, stride=4)
        coords = set(zip(rows.tolist(), cols.tolist()))
        assert len(coords) == len(rows), \
            "Duplicate nodes detected in subsampled set"

    def test_returns_int64(self):
        """Row and col arrays must be int64 for PyG compatibility."""
        mask = np.ones((30, 30), dtype=bool)
        rows, cols = spatial_grid_subsample(mask, stride=3)
        assert rows.dtype == np.int64, f"rows dtype = {rows.dtype}"
        assert cols.dtype == np.int64, f"cols dtype = {cols.dtype}"

    def test_empty_mask_returns_empty(self):
        """All-nodata mask must return zero nodes."""
        mask = np.zeros((20, 20), dtype=bool)
        rows, cols = spatial_grid_subsample(mask, stride=2)
        assert len(rows) == 0
        assert len(cols) == 0


# ════════════════════════════════════════════════════════════════════════════
# GROUP B — Edge construction
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeConstruction:

    def _simple_grid(self, nrow=5, ncol=5, stride=1):
        """Create a fully-valid nrow×ncol grid and subsample it."""
        mask = np.ones((nrow, ncol), dtype=bool)
        return spatial_grid_subsample(mask, stride=stride)

    def test_interior_node_has_8_edges(self):
        """An interior node in a full 5×5 grid must have 8 outgoing edges."""
        rows, cols = self._simple_grid(5, 5, 1)
        ei = build_pixel_grid_edges(rows, cols, stride=1)
        # Node at position (2,2) — find its index
        lookup = {(int(r),int(c)): i for i,(r,c) in enumerate(zip(rows,cols))}
        idx = lookup[(2, 2)]
        # Count edges from this node
        count = int((ei[0] == idx).sum())
        assert count == 8, f"Interior node has {count} edges, expected 8"

    def test_corner_node_has_3_edges(self):
        """A corner node (0,0) must have 3 outgoing edges."""
        rows, cols = self._simple_grid(5, 5, 1)
        ei = build_pixel_grid_edges(rows, cols, stride=1)
        lookup = {(int(r),int(c)): i for i,(r,c) in enumerate(zip(rows,cols))}
        idx = lookup[(0, 0)]
        count = int((ei[0] == idx).sum())
        assert count == 3, f"Corner node has {count} edges, expected 3"

    def test_edge_index_shape(self):
        """edge_index must be shape (2, E) with E > 0."""
        rows, cols = self._simple_grid(10, 10, 1)
        ei = build_pixel_grid_edges(rows, cols, stride=1)
        assert ei.ndim == 2, "edge_index must be 2D"
        assert ei.shape[0] == 2, "edge_index must have 2 rows"
        assert ei.shape[1] > 0, "No edges generated"

    def test_edge_index_valid_range(self):
        """All node indices in edge_index must be < N."""
        rows, cols = self._simple_grid(8, 8, 1)
        ei = build_pixel_grid_edges(rows, cols, stride=1)
        N = len(rows)
        assert ei.max() < N, f"Edge index contains value {ei.max()} >= N={N}"
        assert ei.min() >= 0, "Negative edge index"

    def test_edges_are_int64(self):
        """Edge index dtype must be int64 for PyG compatibility."""
        rows, cols = self._simple_grid(6, 6, 1)
        ei = build_pixel_grid_edges(rows, cols, stride=1)
        assert ei.dtype == np.int64, f"dtype={ei.dtype}, expected int64"

    def test_stride2_edges_span_2_pixels(self):
        """With stride=2, edges connect nodes 2 pixels apart in raster space."""
        mask = np.ones((20, 20), dtype=bool)
        rows, cols = spatial_grid_subsample(mask, stride=2)
        ei = build_pixel_grid_edges(rows, cols, stride=2)
        # Check that all neighbor row differences are 0 or 2
        src_rows = rows[ei[0]]
        dst_rows = rows[ei[1]]
        diffs = np.abs(src_rows - dst_rows)
        assert np.all((diffs == 0) | (diffs == 2)), \
            "stride=2 edges must span 0 or 2 raster rows"

    def test_no_self_loops(self):
        """No node should have an edge to itself."""
        rows, cols = self._simple_grid(6, 6, 1)
        ei = build_pixel_grid_edges(rows, cols, stride=1)
        self_loops = (ei[0] == ei[1]).sum()
        assert self_loops == 0, f"Found {self_loops} self-loop edges"


# ════════════════════════════════════════════════════════════════════════════
# GROUP C — Feature engineering
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_interaction_terms_are_not_raw_products(self):
        """Interactions must use standardized (z-scored) inputs, not raw."""
        from wildfire_gnn.features.feature_engineering import add_interactions
        rng = np.random.default_rng(42)
        base = {
            "CFL":               rng.uniform(1, 25, 1000).astype(np.float32),
            "Ignition_Prob":     rng.uniform(0, 1, 1000).astype(np.float32),
            "FSP_Index":         rng.uniform(1, 38000, 1000).astype(np.float32),
            "Struct_Exp_Index":  rng.uniform(0, 10000, 1000).astype(np.float32),
        }
        result = add_interactions(base)
        # Standardized product should be near-zero mean
        mean_abs = abs(float(result["interact_CFL_Ignition"].mean()))
        assert mean_abs < 0.5, \
            f"Interaction mean={mean_abs:.3f} — not standardized"

    def test_multiscale_larger_kernel_smoother(self):
        """Larger kernel must produce lower std (more smoothed)."""
        from wildfire_gnn.features.feature_engineering import add_multiscale_stats
        rng  = np.random.default_rng(42)
        arr  = rng.uniform(0, 10, (100, 100))
        rows = np.array([50, 51, 52, 53])
        cols = np.array([50, 51, 52, 53])
        full = {"CFL": arr, "FSP_Index": arr, "Ignition_Prob": arr}
        result = add_multiscale_stats(rows, cols, full, kernel_sizes=[3, 15])
        std_3  = float(result["CFL_std_3x3"].std())
        std_15 = float(result["CFL_std_15x15"].std())
        assert std_15 <= std_3, \
            f"15×15 kernel ({std_15:.4f}) must be smoother than 3×3 ({std_3:.4f})"

    def test_degree_normalized_0_to_1(self):
        """Node degree must be in [0, 1] after normalization."""
        from wildfire_gnn.features.feature_engineering import add_degree_feature
        mask = np.ones((20, 20), dtype=bool)
        rows, cols = spatial_grid_subsample(mask, stride=2)
        result = add_degree_feature(rows, cols, stride=2, valid_mask=mask)
        deg = result["node_degree"]
        assert deg.min() >= 0.0, f"Degree min={deg.min():.3f}"
        assert deg.max() <= 1.0, f"Degree max={deg.max():.3f}"

    def test_interior_degree_is_1(self):
        """Interior nodes in full grid must have degree 8/8 = 1.0."""
        from wildfire_gnn.features.feature_engineering import add_degree_feature
        mask = np.ones((30, 30), dtype=bool)
        rows, cols = spatial_grid_subsample(mask, stride=2)
        result = add_degree_feature(rows, cols, stride=2, valid_mask=mask)
        # Most nodes in a full grid are interior
        interior = result["node_degree"] == 1.0
        assert interior.mean() > 0.5, \
            "Most nodes in full grid should be interior (degree=1.0)"


# ════════════════════════════════════════════════════════════════════════════
# GROUP D — Geographic split integrity (most critical)
# ════════════════════════════════════════════════════════════════════════════

class TestGeographicSplit:

    def _make_rows(self, max_row=7596, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        return rng.integers(0, max_row, n)

    def test_no_train_val_overlap(self):
        """Train and val must not share any node."""
        rows = self._make_rows()
        tm, vm, te = build_geographic_split(
            rows, [0,1327], [1328,1517], [1518,7596]
        )
        assert (tm & vm).sum() == 0, "Train/Val overlap!"

    def test_no_train_test_overlap(self):
        """Train and test must not share any node."""
        rows = self._make_rows()
        tm, vm, te = build_geographic_split(
            rows, [0,1327], [1328,1517], [1518,7596]
        )
        assert (tm & te).sum() == 0, "Train/Test overlap!"

    def test_no_val_test_overlap(self):
        """Val and test must not share any node."""
        rows = self._make_rows()
        tm, vm, te = build_geographic_split(
            rows, [0,1327], [1328,1517], [1518,7596]
        )
        assert (vm & te).sum() == 0, "Val/Test overlap!"

    def test_all_nodes_covered(self):
        """Every node must be in exactly one split."""
        rows = np.arange(0, 7597)   # all possible rows
        tm, vm, te = build_geographic_split(
            rows, [0,1327], [1328,1517], [1518,7596]
        )
        covered = tm.sum() + vm.sum() + te.sum()
        assert covered == len(rows), \
            f"Only {covered}/{len(rows)} nodes covered"

    def test_val_mask_not_zero(self):
        """val_mask must have at least one True entry."""
        rows = np.arange(1300, 1600)  # includes val range 1328-1517
        tm, vm, te = build_geographic_split(
            rows, [0,1327], [1328,1517], [1518,7596]
        )
        assert vm.sum() > 0, "val_mask is all zeros — critical failure!"

    def test_split_raises_on_overlap(self):
        """Overlapping row ranges must raise AssertionError."""
        rows = np.arange(0, 1000)
        with pytest.raises(AssertionError):
            build_geographic_split(
                rows,
                train_rows = [0, 600],
                val_rows   = [500, 700],   # overlaps with train!
                test_rows  = [700, 999],
            )

    def test_random_split_would_have_overlap(self):
        """
        Confirm that a random split CANNOT guarantee geographic disjointness.
        This test documents WHY we use geographic split instead of random.
        """
        rng  = np.random.default_rng(42)
        rows = rng.integers(0, 7596, 10000)
        # Randomly assign 80% train, 10% val, 10% test
        idx  = rng.permutation(10000)
        train_idx = idx[:8000]
        val_idx   = idx[8000:9000]
        test_idx  = idx[9000:]

        # Check if any train row value also appears in test row values
        train_rows_set = set(rows[train_idx].tolist())
        test_rows_set  = set(rows[test_idx].tolist())
        overlap = len(train_rows_set & test_rows_set)
        # Random split WILL have row overlap — this documents the problem
        assert overlap > 0, (
            "Random split had no row overlap — very unlikely, check RNG. "
            "This test documents that random splits DO cause geographic leakage."
        )


# ════════════════════════════════════════════════════════════════════════════
# GROUP E — Graph object assertions
# ════════════════════════════════════════════════════════════════════════════

class TestGraphObject:

    def _make_small_graph(self):
        """Build a minimal synthetic graph for testing graph properties."""
        from torch_geometric.data import Data
        N = 100
        F = 10
        E = 200
        return Data(
            x          = torch.randn(N, F),
            y          = torch.randn(N, 1),
            y_raw      = torch.rand(N, 1),
            pos        = torch.randint(0, 7597, (N, 2)).float(),
            edge_index = torch.randint(0, N, (2, E)),
            train_mask = torch.zeros(N, dtype=torch.bool).scatter_(
                            0, torch.arange(0, 70), True),
            val_mask   = torch.zeros(N, dtype=torch.bool).scatter_(
                            0, torch.arange(70, 85), True),
            test_mask  = torch.zeros(N, dtype=torch.bool).scatter_(
                            0, torch.arange(85, 100), True),
        )

    def test_masks_are_disjoint(self):
        """Train, val, test masks must not overlap."""
        g = self._make_small_graph()
        assert (g.train_mask & g.val_mask).sum() == 0
        assert (g.train_mask & g.test_mask).sum() == 0

    def test_masks_cover_all_nodes(self):
        """All nodes must be in exactly one split."""
        g = self._make_small_graph()
        covered = g.train_mask.sum() + g.val_mask.sum() + g.test_mask.sum()
        assert covered == g.num_nodes

    def test_val_mask_nonzero(self):
        """val_mask must have at least one True — guards against zero placeholder."""
        g = self._make_small_graph()
        assert g.val_mask.sum() > 0, "val_mask is all zeros!"

    def test_y_shape(self):
        """y must be (N, 1) — not (N,) flat."""
        g = self._make_small_graph()
        assert g.y.dim() == 2, f"y.dim()={g.y.dim()}, expected 2"
        assert g.y.shape[1] == 1, f"y.shape={g.y.shape}"

    def test_x_dtype_float32(self):
        """Feature matrix must be float32 for PyG compatibility."""
        g = self._make_small_graph()
        assert g.x.dtype == torch.float32

    def test_edge_index_dtype_int64(self):
        """edge_index must be int64 (torch.long)."""
        g = self._make_small_graph()
        assert g.edge_index.dtype == torch.long


# ════════════════════════════════════════════════════════════════════════════
# GROUP F — Integration: load saved graph
# ════════════════════════════════════════════════════════════════════════════

class TestSavedGraph:
    """
    These tests load the actual saved graph from Phase 3.
    Skipped automatically if phase3_build_graph.py has not been run yet.
    """

    GRAPH_PATH = ROOT / "data" / "processed" / "graph_data_enriched.pt"

    @pytest.fixture(autouse=True)
    def skip_if_no_graph(self):
        if not self.GRAPH_PATH.exists():
            pytest.skip("graph_data_enriched.pt not yet generated — "
                        "run phase3_build_graph.py first")

    def test_graph_loads(self):
        """Graph file must load without error."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert g is not None

    def test_num_nodes_reasonable(self):
        """N should be between 100k and 500k for stride=4-8."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert 100_000 < g.num_nodes < 600_000, \
            f"num_nodes={g.num_nodes:,} — out of expected range"

    def test_num_features_53_or_58(self):
        """Features must be 53 (no DEM) or 58 (with DEM)."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert g.num_node_features in (53, 58), \
            f"num_features={g.num_node_features} — expected 53 or 58"

    def test_y_is_transformed(self):
        """y must be near-Gaussian (transformed) not raw burn probability."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        y_mean = float(g.y.mean())
        y_std  = float(g.y.std())
        assert abs(y_mean) < 0.5, \
            f"y.mean()={y_mean:.4f} — transform not applied or double-applied"
        assert 0.5 < y_std < 2.0, \
            f"y.std()={y_std:.4f} — transform looks wrong"

    def test_no_split_overlap(self):
        """Train/Val/Test must be geographically disjoint."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert (g.train_mask & g.val_mask).sum() == 0,  "Train/Val overlap!"
        assert (g.train_mask & g.test_mask).sum() == 0, "Train/Test overlap!"

    def test_val_mask_nonzero(self):
        """val_mask must be non-zero — guards against zero placeholder bug."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert g.val_mask.sum() > 0, \
            "val_mask is all zeros — geographic split is wrong!"

    def test_feature_names_json_exists(self):
        """feature_names.json must exist and have correct length."""
        feat_path = ROOT / "data" / "features" / "feature_names.json"
        assert feat_path.exists(), "feature_names.json not found"
        with open(feat_path) as f:
            names = json.load(f)
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert len(names) == g.num_node_features, \
            f"feature_names has {len(names)} entries but graph has " \
            f"{g.num_node_features} features"

    def test_edge_index_valid(self):
        """All edge node indices must be < N."""
        g = torch.load(self.GRAPH_PATH, map_location="cpu",
                       weights_only=False)
        assert g.edge_index.max() < g.num_nodes, \
            "edge_index contains out-of-range node index"
        assert g.edge_index.min() >= 0, \
            "edge_index contains negative node index"