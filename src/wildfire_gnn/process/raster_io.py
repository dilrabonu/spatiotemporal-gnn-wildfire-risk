"""
Raster I/O utilities — Phase 2 of wildfire-uncertainty-gnn.

WHAT THIS FILE DOES
-------------------
Loads FSim raster files (.img format) with correct nodata masking.
Returns masked numpy arrays safe for statistics and ML pipelines.

CRITICAL NODATA VALUES (confirmed from Phase 2 dataset audit)
-------------------------------------------------------------
  Continuous rasters (Burn_Prob, CFL, FSP_Index, Ignition_Prob,
  Struct_Exp_Index): nodata sentinel = -3.402823e+38

  Categorical raster (Fuel_Models): nodata sentinel = 255 (uint8)

  These MUST be masked BEFORE computing any statistic.
  If you skip masking, np.mean(array) returns ≈ -3.4e+38.

CONFIRMED SHAPES FROM DATASET AUDIT
-------------------------------------
  Burn_Prob.img        → (7597, 7555)  ← REFERENCE GRID
  CFL.img              → (7597, 7555)  ← already aligned
  FSP_Index.img        → (7592, 7541)  ← MISALIGNED — needs resample
  Fuel_Models.img      → (7932, 9039)  ← MISALIGNED — needs resample
  Ignition_Prob.img    → (7733, 9039)  ← MISALIGNED — needs resample
  Struct_Exp_Index.img → (7592, 7541)  ← MISALIGNED — needs resample

  CRS: All rasters are EPSG:2100 (Greek national grid).
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import rasterio

# ── Known nodata sentinels ─────────────────────────────────────────────────
NODATA_FLOAT  = -3.402823e+38   # FSim continuous raster sentinel
NODATA_UINT8  = 255             # Fuel_Models.img categorical sentinel
NODATA_THRESH = -1e+30          # anything below this is nodata in floats


class RasterInfo:
    """Container holding a loaded raster as a masked array + metadata."""

    def __init__(self, path: Path, data: np.ma.MaskedArray, profile: dict):
        self.path      = path
        self.data      = data           # shape: (height, width), masked
        self.profile   = profile
        self.name      = path.stem
        self.shape     = data.shape
        self.dtype     = str(data.dtype)
        self.crs       = profile.get("crs")
        self.transform = profile.get("transform")
        self.nodata    = profile.get("nodata")

        # Count valid vs nodata pixels
        if isinstance(data.mask, np.ndarray):
            self.n_valid  = int(np.sum(~data.mask))
        else:
            self.n_valid  = data.size if not data.mask else 0
        self.n_nodata = data.size - self.n_valid

    def valid_data(self) -> np.ndarray:
        """Return 1D array of all valid (non-masked) pixel values."""
        return self.data.compressed()

    def stats(self) -> dict:
        """Descriptive statistics over valid pixels only."""
        v = self.valid_data()
        if len(v) == 0:
            return {k: None for k in
                    ["min","max","mean","std","median","n_valid","n_nodata","pct_valid"]}
        return {
            "min":      float(np.min(v)),
            "max":      float(np.max(v)),
            "mean":     float(np.mean(v)),
            "std":      float(np.std(v)),
            "median":   float(np.median(v)),
            "n_valid":  self.n_valid,
            "n_nodata": self.n_nodata,
            "pct_valid": round(100.0 * self.n_valid / self.data.size, 2),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"RasterInfo({self.name})\n"
            f"  Shape   : {self.shape}\n"
            f"  CRS     : {self.crs}\n"
            f"  dtype   : {self.dtype}\n"
            f"  nodata  : {self.nodata}\n"
            f"  valid   : {self.n_valid:,} px  ({s['pct_valid']}%)\n"
            f"  min     : {s['min']:.6g}\n"
            f"  max     : {s['max']:.6g}\n"
            f"  mean    : {s['mean']:.6g}\n"
            f"  std     : {s['std']:.6g}\n"
        )


def load_raster(path: str | Path, band: int = 1) -> RasterInfo:
    """
    Load a single raster file and return a properly masked RasterInfo.

    Masking strategy (applied in order):
    1. Mask pixels equal to rasterio's reported nodata value.
    2. Mask float pixels below NODATA_THRESH = -1e30  (catches -3.4e38 robustly).
    3. Mask uint8 pixels equal to 255 (Fuel_Models sentinel).
    4. Mask any NaN or Inf remaining.

    Parameters
    ----------
    path : str | Path
        Path to raster file (.img or .tif).
    band : int
        Band index to read (1-indexed). Default 1.

    Returns
    -------
    RasterInfo
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        profile = dict(src.profile)
        raw     = src.read(band)
        nodata  = src.nodata

    mask = np.zeros(raw.shape, dtype=bool)   # True = invalid

    if nodata is not None:
        mask |= (raw == nodata)

    if raw.dtype.kind == "f":
        mask |= (raw < NODATA_THRESH)
        mask |= ~np.isfinite(raw)

    if raw.dtype == np.uint8:
        mask |= (raw == NODATA_UINT8)

    return RasterInfo(path=path,
                      data=np.ma.MaskedArray(raw, mask=mask),
                      profile=profile)


def load_raster_stack(
    paths: Dict[str, str | Path],
    verbose: bool = True,
) -> Dict[str, RasterInfo]:
    """
    Load multiple rasters by name dict.

    Parameters
    ----------
    paths : dict mapping name → file path
    verbose : print summary per raster

    Returns
    -------
    dict[str, RasterInfo]
    """
    results = {}
    for name, path in paths.items():
        try:
            ri = load_raster(path)
            results[name] = ri
            if verbose:
                s = ri.stats()
                print(f"  ✓  {name:<22} shape={ri.shape}  "
                      f"valid={ri.n_valid:>10,}  "
                      f"mean={s['mean']:.5f}")
        except FileNotFoundError:
            warnings.warn(f"  ✗  {name}: not found at {path}")
    return results


def audit_alignment(
    stack: Dict[str, RasterInfo],
    reference_name: str = "Burn_Prob",
) -> Dict[str, dict]:
    """
    Check which rasters share shape and CRS with the reference.

    Returns
    -------
    dict with shape_ok and crs_ok per raster.
    """
    ref = stack[reference_name]
    results = {}
    for name, ri in stack.items():
        results[name] = {
            "shape":    ri.shape,
            "shape_ok": ri.shape == ref.shape,
            "crs":      str(ri.crs),
            "crs_ok":   str(ri.crs) == str(ref.crs),
            "dtype":    ri.dtype,
            "pct_valid": ri.stats()["pct_valid"],
        }
    return results


def print_audit(audit: dict, reference_name: str = "Burn_Prob") -> None:
    """Pretty-print alignment audit results."""
    print(f"\n{'='*72}")
    print(f"  Raster Stack Alignment Audit  (reference = {reference_name})")
    print(f"{'='*72}")
    print(f"  {'Name':<24} {'Shape':<16} {'Aligned':<12} {'dtype':<10} {'Valid%'}")
    print(f"  {'-'*70}")
    for name, info in audit.items():
        aligned = "✓ OK" if info["shape_ok"] else "✗ MISMATCH"
        print(f"  {name:<24} {str(info['shape']):<16} {aligned:<12} "
              f"{info['dtype']:<10} {info['pct_valid']}%")
    misaligned = [n for n, a in audit.items() if not a["shape_ok"]]
    print(f"\n  Rasters needing resampling: {misaligned}")
    print(f"{'='*72}\n")