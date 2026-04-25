"""
Raster alignment pipeline — Phase 2 of wildfire-uncertainty-gnn.

WHAT THIS FILE DOES
-------------------
Reprojects and resamples all FSim feature rasters to a common reference
grid (Burn_Prob.img: 7597×7555, EPSG:2100) so that every feature raster
has identical shape and cell-to-cell correspondence before graph construction.

WHY THIS IS NECESSARY
---------------------
From the Phase 2 dataset audit, the six FSim rasters have 4 different shapes:
  (7597,7555) — Burn_Prob, CFL        → already aligned
  (7592,7541) — FSP_Index, Struct_Exp → needs +5 rows, +14 cols
  (7932,9039) — Fuel_Models           → needs −335 rows, −1484 cols
  (7733,9039) — Ignition_Prob         → needs −136 rows, −1484 cols

Without alignment, stacking these rasters into a node feature matrix
would assign different real-world locations to the same row/column index,
making all spatial relationships in the graph scientifically wrong.

RESAMPLING METHODS
------------------
  Continuous rasters  → Bilinear resampling (smooth, physically appropriate)
  Fuel_Models.img     → Nearest-neighbor ONLY (categorical — never interpolate)
    Reason: interpolating fuel category codes (e.g., 91 and 98) produces
    meaningless intermediate values (94.5) that do not correspond to any
    real fuel model. Always use nearest for categorical data.

OUTPUT
------
  All aligned rasters saved as GeoTIFF (.tif) in data/interim/aligned/
  All share: shape (7597,7555), CRS EPSG:2100, same transform.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

RESAMPLE_BILINEAR = Resampling.bilinear
RESAMPLE_NEAREST  = Resampling.nearest


def load_reference_profile(reference_path: str | Path) -> dict:
    """
    Load rasterio profile from the reference raster (Burn_Prob.img).

    This profile defines the target grid — every other raster
    will be resampled to match it exactly.
    """
    with rasterio.open(reference_path) as src:
        profile = dict(src.profile)
        profile["count"] = 1
    return profile


def align_raster(
    src_path:          str | Path,
    dst_path:          str | Path,
    reference_profile: dict,
    resampling:        Resampling = RESAMPLE_BILINEAR,
    overwrite:         bool = False,
) -> Path:
    """
    Reproject and resample one raster onto the reference grid.

    Parameters
    ----------
    src_path : path to raw source raster
    dst_path : path where aligned .tif will be written
    reference_profile : profile from load_reference_profile()
    resampling : Resampling.bilinear (continuous) or Resampling.nearest (categorical)
    overwrite : skip if file exists and overwrite=False

    Returns
    -------
    Path : dst_path
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    if dst_path.exists() and not overwrite:
        print(f"    SKIP (exists): {dst_path.name}")
        return dst_path

    dst_crs       = reference_profile["crs"]
    dst_transform = reference_profile["transform"]
    dst_height    = reference_profile["height"]
    dst_width     = reference_profile["width"]

    with rasterio.open(src_path) as src:
        src_data  = src.read(1).astype(np.float64)
        src_nodata = src.nodata

        # Mask nodata before reprojection
        if src_nodata is not None:
            src_data[src_data == src_nodata] = np.nan
        src_data[src_data < -1e30] = np.nan
        if src.dtypes[0] == "uint8":
            src_data[src_data == 255.0] = np.nan

        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float64)

        reproject(
            source        = src_data,
            destination   = dst_data,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = dst_transform,
            dst_crs       = dst_crs,
            resampling    = resampling,
            src_nodata    = np.nan,
            dst_nodata    = np.nan,
        )

        # Write output
        NODATA_OUT = -9999.0
        arr_out    = dst_data.astype(np.float32)
        arr_out[np.isnan(arr_out)] = NODATA_OUT

        out_profile = {
            "driver":    "GTiff",
            "dtype":     "float32",
            "width":     dst_width,
            "height":    dst_height,
            "count":     1,
            "crs":       dst_crs,
            "transform": dst_transform,
            "nodata":    NODATA_OUT,
            "compress":  "lzw",
            "tiled":     True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(arr_out, 1)

    print(f"    ✓  Aligned: {dst_path.name}  → ({dst_height}×{dst_width})")
    return dst_path


def align_all_rasters(
    raw_paths:      Dict[str, str | Path],
    aligned_dir:    str | Path,
    reference_name: str = "Burn_Prob",
    overwrite:      bool = False,
) -> Dict[str, Path]:
    """
    Align all FSim feature rasters to the reference grid.

    Automatically applies nearest-neighbor resampling to Fuel_Models
    and bilinear to all other rasters.

    Parameters
    ----------
    raw_paths : dict of name → raw file path (must include reference_name)
    aligned_dir : directory to write aligned .tif files
    reference_name : key in raw_paths that is the reference (Burn_Prob)
    overwrite : overwrite existing aligned files

    Returns
    -------
    dict of name → aligned .tif path
    """
    aligned_dir = Path(aligned_dir)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # Categorical rasters: MUST use nearest-neighbor
    CATEGORICAL = {"Fuel_Models"}

    ref_path    = Path(raw_paths[reference_name])
    ref_profile = load_reference_profile(ref_path)

    print(f"\n  Reference: {ref_path.name}")
    print(f"  Target shape : {ref_profile['height']} × {ref_profile['width']}")
    print(f"  Target CRS   : {ref_profile['crs']}\n")

    output_paths = {}
    for name, raw_path in raw_paths.items():
        raw_path = Path(raw_path)
        dst_path = aligned_dir / f"{name}.tif"

        if name == reference_name:
            # Copy reference to aligned dir as .tif
            if dst_path.exists() and not overwrite:
                print(f"    SKIP (exists): {dst_path.name}")
            else:
                with rasterio.open(raw_path) as s:
                    p = s.profile.copy()
                    d = s.read(1)
                p.update(driver="GTiff", compress="lzw",
                         tiled=True, blockxsize=512, blockysize=512)
                with rasterio.open(dst_path, "w", **p) as w:
                    w.write(d, 1)
                print(f"    ✓  Copied reference: {dst_path.name}")
            output_paths[name] = dst_path
            continue

        method = RESAMPLE_NEAREST if name in CATEGORICAL else RESAMPLE_BILINEAR
        label  = "nearest (CATEGORICAL)" if name in CATEGORICAL else "bilinear"
        print(f"  {name:<24} [{label}]")

        try:
            p = align_raster(raw_path, dst_path, ref_profile, method, overwrite)
            output_paths[name] = p
        except Exception as e:
            warnings.warn(f"  ✗  {name}: {e}")

    return output_paths


def verify_alignment(
    aligned_paths: Dict[str, Path],
    reference_name: str = "Burn_Prob",
) -> bool:
    """
    Assert all aligned rasters have identical shape and CRS.

    Returns True if all pass. Raises no exception — prints results.
    Call sys.exit(1) if this returns False before proceeding to Phase 3.
    """
    with rasterio.open(aligned_paths[reference_name]) as ref:
        ref_shape = (ref.height, ref.width)
        ref_crs   = ref.crs

    print(f"\n  Alignment Verification (reference={reference_name})")
    print(f"  Expected: shape={ref_shape}  crs={ref_crs}\n")

    all_ok = True
    for name, path in aligned_paths.items():
        with rasterio.open(path) as s:
            ok_shape = (s.height, s.width) == ref_shape
            ok_crs   = str(s.crs) == str(ref_crs)
        ok     = ok_shape and ok_crs
        status = "✓" if ok else "✗ FAILED"
        print(f"    {status}  {name:<25} shape_ok={ok_shape}  crs_ok={ok_crs}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("  ✓  All rasters aligned. Ready for Phase 3 — Graph Construction.\n")
    else:
        print("  ✗  ALIGNMENT FAILED — do not proceed to Phase 3.\n")
    return all_ok


def build_valid_cell_mask(aligned_paths: Dict[str, Path]) -> np.ndarray:
    """
    Build a 2D boolean mask: True where ALL rasters have valid data.

    A cell is invalid if ANY single raster has nodata at that location.
    This ensures the graph contains only fully-observed nodes.

    Returns
    -------
    np.ndarray shape (height, width), dtype=bool
    """
    first = next(iter(aligned_paths.values()))
    with rasterio.open(first) as s:
        H, W = s.height, s.width

    valid = np.ones((H, W), dtype=bool)

    for name, path in aligned_paths.items():
        with rasterio.open(path) as s:
            data   = s.read(1).astype(np.float64)
            nodata = s.nodata

        inv = np.zeros((H, W), dtype=bool)
        if nodata is not None:
            inv |= (data == nodata)
        if data.dtype.kind == "f":
            inv |= (data < -1e30)
            inv |= ~np.isfinite(data)
        if data.dtype == np.uint8:
            inv |= (data == 255)

        valid &= ~inv

    n   = int(np.sum(valid))
    tot = H * W
    print(f"\n  Valid-cell mask:")
    print(f"    Total cells  : {tot:,}")
    print(f"    Valid cells  : {n:,}  ({100*n/tot:.2f}%)")
    print(f"    Nodata cells : {tot-n:,}  ({100*(tot-n)/tot:.2f}%)\n")
    return valid