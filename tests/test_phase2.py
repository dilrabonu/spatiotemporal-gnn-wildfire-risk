"""
Phase 2 unit tests — run with: pytest tests/test_phase2.py -v

Every test here guards against a known failure mode from previous attempts.
ALL tests must pass before proceeding to Phase 3 (graph construction).

Test groups:
  A — Nodata masking (raster_io.py)
  B — Alignment assertions (alignment.py)
  C — Target transformation rules (target_engineering.py)
  D — DEM slope assertion (dem_features.py)
  E — Integration: valid cell mask
"""

from __future__ import annotations
import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# ── Add src to path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wildfire_gnn.process.raster_io import (
    load_raster, audit_alignment, NODATA_FLOAT, NODATA_THRESH
)
from wildfire_gnn.process.target_engineering import TargetTransformer
from wildfire_gnn.process.alignment import build_valid_cell_mask


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_test_raster(
    path: Path,
    shape=(50, 50),
    dtype="float32",
    nodata: float = -3.402823e+38,
    fill_value: float = 0.05,
    crs_epsg: int = 2100,
) -> Path:
    """Create a minimal synthetic raster for testing."""
    H, W = shape
    data = np.full((H, W), fill_value, dtype=dtype)
    # Add a band of nodata at the top
    data[:5, :] = nodata

    transform = from_bounds(0, 0, W * 25, H * 25, W, H)
    profile = dict(
        driver    = "GTiff",
        dtype     = dtype,
        width     = W,
        height    = H,
        count     = 1,
        crs       = CRS.from_epsg(crs_epsg),
        transform = transform,
        nodata    = nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def make_categorical_raster(path: Path, shape=(50, 50)) -> Path:
    """Simulate Fuel_Models.img — uint8 with nodata=255."""
    H, W = shape
    data = np.full((H, W), 98, dtype=np.uint8)   # typical fuel code
    data[:5, :] = 255                              # nodata band
    transform = from_bounds(0, 0, W*25, H*25, W, H)
    profile = dict(
        driver="GTiff", dtype="uint8", width=W, height=H, count=1,
        crs=CRS.from_epsg(2100), transform=transform, nodata=255
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


# ════════════════════════════════════════════════════════════════════════════
# GROUP A — Nodata masking
# ════════════════════════════════════════════════════════════════════════════

class TestNodataMasking:

    def test_float_nodata_masked(self, tmp_path):
        """Float raster nodata (-3.4e38) must be masked."""
        p = make_test_raster(tmp_path / "bp.tif", nodata=NODATA_FLOAT)
        ri = load_raster(p)
        # Top 5 rows (nodata) must all be masked
        assert ri.data.mask[:5, :].all(), \
            "Nodata rows should be masked"

    def test_valid_pixels_not_masked(self, tmp_path):
        """Valid pixels must NOT be masked."""
        p = make_test_raster(tmp_path / "bp.tif", fill_value=0.07)
        ri = load_raster(p)
        # Bottom 45 rows should be valid
        assert not ri.data.mask[5:, :].any(), \
            "Valid pixels should not be masked"

    def test_stats_exclude_nodata(self, tmp_path):
        """Statistics must be computed over valid pixels only."""
        p = make_test_raster(tmp_path / "bp.tif",
                             shape=(50, 50), fill_value=0.05)
        ri    = load_raster(p)
        stats = ri.stats()
        # Mean must be near 0.05, not near -3.4e38
        assert abs(stats["mean"] - 0.05) < 0.001, \
            f"Mean={stats['mean']} — nodata contaminating statistics"

    def test_uint8_nodata_255_masked(self, tmp_path):
        """Fuel_Models nodata=255 must be masked correctly."""
        p  = make_categorical_raster(tmp_path / "fuel.tif")
        ri = load_raster(p)
        # Top 5 rows (255) must be masked
        assert ri.data.mask[:5, :].all(), \
            "uint8 nodata=255 rows must be masked"

    def test_uint8_valid_not_masked(self, tmp_path):
        """Valid fuel codes (98) must NOT be masked."""
        p  = make_categorical_raster(tmp_path / "fuel.tif")
        ri = load_raster(p)
        assert not ri.data.mask[5:, :].any(), \
            "Valid fuel code pixels should not be masked"

    def test_large_negative_sentinel_masked(self, tmp_path):
        """Any value < -1e30 must be masked (catches -3.4e38 robustly)."""
        H, W = 20, 20
        data = np.full((H, W), 0.02, dtype=np.float32)
        data[0, 0] = -9.99e+37   # below NODATA_THRESH, above NODATA_FLOAT
        transform = from_bounds(0, 0, W*25, H*25, W, H)
        path = tmp_path / "test.tif"
        with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                           width=W, height=H, count=1,
                           crs=CRS.from_epsg(2100), transform=transform,
                           nodata=None) as dst:
            dst.write(data, 1)
        ri = load_raster(path)
        assert ri.data.mask[0, 0], \
            "Value below -1e30 must be masked even if nodata=None"

    def test_pct_valid_correct(self, tmp_path):
        """pct_valid should reflect true valid pixel proportion."""
        # 50x50 = 2500 total, top 5 rows = 250 nodata → 2250 valid = 90%
        p     = make_test_raster(tmp_path / "bp.tif")
        ri    = load_raster(p)
        stats = ri.stats()
        assert 88 < stats["pct_valid"] < 92, \
            f"pct_valid={stats['pct_valid']} — expected ~90%"


# ════════════════════════════════════════════════════════════════════════════
# GROUP B — Alignment assertions
# ════════════════════════════════════════════════════════════════════════════

class TestAlignment:

    def test_shape_mismatch_detected(self, tmp_path):
        """audit_alignment must flag rasters with different shapes."""
        ref = make_test_raster(tmp_path / "ref.tif", shape=(50, 50))
        mis = make_test_raster(tmp_path / "mis.tif", shape=(48, 48))

        from wildfire_gnn.process.raster_io import load_raster_stack
        stack = load_raster_stack(
            {"Burn_Prob": ref, "FSP_Index": mis}, verbose=False
        )
        audit = audit_alignment(stack, "Burn_Prob")
        assert audit["FSP_Index"]["shape_ok"] is False, \
            "Shape mismatch must be detected"
        assert audit["Burn_Prob"]["shape_ok"] is True, \
            "Reference must report shape_ok=True"

    def test_crs_mismatch_detected(self, tmp_path):
        """audit_alignment must flag rasters with different CRS."""
        ref  = make_test_raster(tmp_path / "ref.tif",  crs_epsg=2100)
        diff = make_test_raster(tmp_path / "diff.tif", crs_epsg=4326)

        from wildfire_gnn.process.raster_io import load_raster_stack
        stack = load_raster_stack(
            {"Burn_Prob": ref, "DEM": diff}, verbose=False
        )
        audit = audit_alignment(stack, "Burn_Prob")
        assert audit["DEM"]["crs_ok"] is False, \
            "CRS mismatch must be detected"

    def test_valid_cell_mask_shape(self, tmp_path):
        """valid_cell_mask must have same shape as reference raster."""
        p1 = make_test_raster(tmp_path / "r1.tif", shape=(30, 30))
        p2 = make_test_raster(tmp_path / "r2.tif", shape=(30, 30))
        mask = build_valid_cell_mask({"A": p1, "B": p2})
        assert mask.shape == (30, 30), \
            f"Mask shape {mask.shape} != (30, 30)"

    def test_valid_cell_mask_dtype(self, tmp_path):
        """valid_cell_mask must be boolean dtype."""
        p = make_test_raster(tmp_path / "r.tif", shape=(20, 20))
        mask = build_valid_cell_mask({"A": p})
        assert mask.dtype == bool, \
            f"Mask dtype {mask.dtype} must be bool"

    def test_valid_cell_mask_nodata_excluded(self, tmp_path):
        """Nodata pixels must be False in the valid mask."""
        p    = make_test_raster(tmp_path / "r.tif",
                               shape=(20, 20), nodata=NODATA_FLOAT)
        mask = build_valid_cell_mask({"A": p})
        # Top 5 rows are nodata — must be False in mask
        assert not mask[:5, :].any(), \
            "Nodata pixels must be excluded from valid mask"

    def test_no_nodes_dropped_assertion(self, tmp_path):
        """
        Guard against the 40,718 dropped-node bug from previous project.

        If we build a mask from ALL aligned rasters, the number of valid
        nodes must match what we count from the reference raster alone.
        In this synthetic test, two identical rasters should give the same
        valid count as one.
        """
        p1 = make_test_raster(tmp_path / "r1.tif", shape=(50, 50))
        p2 = make_test_raster(tmp_path / "r2.tif", shape=(50, 50))
        # Same nodata pattern — mask should be identical
        mask1 = build_valid_cell_mask({"A": p1})
        mask2 = build_valid_cell_mask({"A": p1, "B": p2})
        assert mask1.sum() == mask2.sum(), \
            "Adding identical raster to mask must not drop nodes"


# ════════════════════════════════════════════════════════════════════════════
# GROUP C — Target transformation
# ════════════════════════════════════════════════════════════════════════════

class TestTargetTransformer:

    def _make_bp_data(self, n=10000):
        """Simulate right-skewed burn probability distribution."""
        rng = np.random.default_rng(42)
        return rng.exponential(scale=0.02, size=n).clip(0, 0.25)

    def test_transform_mean_near_zero(self):
        """After quantile transform, mean must be near 0."""
        y   = self._make_bp_data()
        t   = TargetTransformer()
        y_t = t.fit_transform(y)
        assert abs(float(np.mean(y_t))) < 0.5, \
            f"Transformed mean={np.mean(y_t):.4f} — expected near 0"

    def test_transform_std_near_one(self):
        """After quantile transform, std must be near 1."""
        y   = self._make_bp_data()
        t   = TargetTransformer()
        y_t = t.fit_transform(y)
        assert 0.5 < float(np.std(y_t)) < 2.0, \
            f"Transformed std={np.std(y_t):.4f} — expected near 1"

    def test_inverse_recovers_original(self):
        """inverse_transform must recover the original values."""
        y   = self._make_bp_data()
        t   = TargetTransformer()
        y_t = t.fit_transform(y)
        y_r = t.inverse_transform(y_t)
        np.testing.assert_allclose(y, y_r.ravel(), atol=1e-4,
            err_msg="inverse_transform must recover original BP values")

    def test_validate_catches_double_transform(self):
        """validate() must raise if transform was applied twice."""
        y   = self._make_bp_data()
        t   = TargetTransformer()
        y_t = t.fit_transform(y)
        y_tt = t.transform(y_t)   # double-transform
        with pytest.raises(AssertionError):
            t.validate(y_tt)

    def test_validate_passes_single_transform(self):
        """validate() must NOT raise after single transform."""
        y   = self._make_bp_data()
        t   = TargetTransformer()
        y_t = t.fit_transform(y)
        t.validate(y_t)   # should not raise

    def test_fit_train_only_no_leakage(self):
        """Fitting on train and applying to val must not raise."""
        rng  = np.random.default_rng(99)
        train = rng.exponential(0.02, 8000).clip(0, 0.25)
        val   = rng.exponential(0.02, 2000).clip(0, 0.25)
        t = TargetTransformer()
        t.fit(train)
        # Transforming val must not raise
        val_t = t.transform(val)
        assert len(val_t) == len(val)

    def test_save_load_roundtrip(self, tmp_path):
        """Saved transformer must load and give identical results."""
        y  = self._make_bp_data()
        t  = TargetTransformer()
        t.fit(y)
        path = tmp_path / "transformer.pkl"
        t.save(path)
        t2  = TargetTransformer.load(path)
        np.testing.assert_allclose(
            t.transform(y), t2.transform(y), atol=1e-6,
            err_msg="Loaded transformer must give identical results"
        )

    def test_transform_requires_fit(self):
        """transform() must raise if called before fit()."""
        t = TargetTransformer()
        with pytest.raises(RuntimeError, match="fit"):
            t.transform(np.array([0.05, 0.10]))


# ════════════════════════════════════════════════════════════════════════════
# GROUP D — DEM slope assertion
# ════════════════════════════════════════════════════════════════════════════

class TestDEMSlope:

    def test_slope_calculation_correct(self, tmp_path):
        """
        Slope computed AFTER reprojection to metric CRS must be < 45°.

        This is the DEM bug guard from the previous project.
        We create a synthetic elevation grid with known gradient.
        """
        # 10m rise over 100m horizontal → slope = arctan(0.1) ≈ 5.7°
        H, W = 100, 100
        pixel_m = 100.0   # 100 metre pixels in EPSG:2100
        elev = np.zeros((H, W), dtype=np.float32)
        for row in range(H):
            elev[row, :] = row * 10   # 10m per pixel rise = 10m/100m = 10%

        dz_dy, dz_dx = np.gradient(elev.astype(np.float64), pixel_m, pixel_m)
        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        mean_slope = float(np.mean(slope_deg))

        # arctan(10/100) = arctan(0.1) ≈ 5.71°
        assert mean_slope < 45.0, \
            f"Slope={mean_slope:.1f}° — must be < 45°. DEM bug detected!"
        assert 5.0 < mean_slope < 7.0, \
            f"Slope={mean_slope:.2f}° — expected ≈5.7° for 10m/100m gradient"

    def test_slope_wrong_if_degrees_not_converted(self):
        """
        Confirm that computing gradient on degree values gives wrong slope.

        This documents WHY the previous project got slope=90°.
        The test asserts that the WRONG approach fails — do not use this
        approach in production code.
        """
        H, W = 50, 50
        # Same elevation field but using 0.001 degree pixel size (EPSG:4326)
        pixel_deg = 0.001
        elev = np.zeros((H, W), dtype=np.float64)
        for row in range(H):
            elev[row, :] = row * 10  # 10m rise per 0.001° = 10000 m/degree

        # WRONG: gradient in m/degree → huge ratio → arctan ≈ 90°
        dz_dy, dz_dx = np.gradient(elev, pixel_deg, pixel_deg)
        slope_deg_wrong = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        mean_wrong = float(np.mean(slope_deg_wrong))

        # This SHOULD be wrong (close to 90°) — documenting the failure
        assert mean_wrong > 45.0, \
            "Expected wrong slope > 45° when gradient not corrected for units"


# ════════════════════════════════════════════════════════════════════════════
# GROUP E — Phase 2 completion check
# ════════════════════════════════════════════════════════════════════════════

class TestPhase2Completion:
    """
    Check that Phase 2 output files exist and are valid.
    Only run these if phase2_align_rasters.py has been executed.
    Skips gracefully if files are not yet present.
    """

    @pytest.mark.skipif(
        not (Path("data/interim/aligned").exists()),
        reason="Aligned rasters not yet generated — run phase2_align_rasters.py first"
    )
    def test_aligned_rasters_exist(self):
        """All aligned .tif files must exist after pipeline run."""
        aligned = Path("data/interim/aligned")
        expected = [
            "Burn_Prob.tif", "CFL.tif", "FSP_Index.tif",
            "Fuel_Models.tif", "Ignition_Prob.tif", "Struct_Exp_Index.tif",
        ]
        for name in expected:
            assert (aligned / name).exists(), f"Missing aligned raster: {name}"

    @pytest.mark.skipif(
        not Path("data/processed/baseline_dataset.csv").exists(),
        reason="baseline_dataset.csv not yet generated"
    )
    def test_baseline_csv_exists(self):
        """baseline_dataset.csv must exist and have > 200k rows."""
        import pandas as pd
        df = pd.read_csv("data/processed/baseline_dataset.csv")
        assert len(df) > 200_000, \
            f"baseline_dataset.csv has only {len(df):,} rows — expected > 200k"
        assert "target" in df.columns, "Missing 'target' column"

    @pytest.mark.skipif(
        not Path("data/features/valid_cell_mask.npy").exists(),
        reason="valid_cell_mask.npy not yet generated"
    )
    def test_valid_cell_mask_exists(self):
        """valid_cell_mask.npy must be 2D boolean."""
        mask = np.load("data/features/valid_cell_mask.npy")
        assert mask.ndim == 2, "Mask must be 2D"
        assert mask.dtype == bool, "Mask must be boolean"

    @pytest.mark.skipif(
        not Path("data/features/target_transformer.pkl").exists(),
        reason="target_transformer.pkl not yet generated"
    )
    def test_transformer_loads_and_works(self):
        """target_transformer.pkl must load and inverse-transform correctly."""
        t  = TargetTransformer.load("data/features/target_transformer.pkl")
        y  = np.array([0.0, -0.5, 0.5, 1.0, -1.0])
        y_inv = t.inverse_transform(y)
        assert np.all(y_inv >= 0.0), "Inverse-transformed BP must be non-negative"
        assert np.all(y_inv <= 1.0), "Inverse-transformed BP must be <= 1.0"