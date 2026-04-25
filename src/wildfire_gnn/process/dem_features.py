"""
DEM terrain feature extraction — Phase 2 of wildfire-uncertainty-gnn.

WHAT THIS FILE DOES
-------------------
Extracts 5 terrain features from a Digital Elevation Model (DEM) raster,
reprojected and resampled to match the FSim reference grid.

OUTPUT FEATURES (5 total — become columns in node feature matrix)
-----------------------------------------------------------------
  dem_elevation_m   raw elevation in metres
  dem_slope_deg     slope angle in degrees (0=flat, 90=vertical cliff)
  dem_aspect_sin    sin(aspect) — encodes N/S wind exposure
  dem_aspect_cos    cos(aspect) — encodes E/W wind exposure
  dem_twi           Topographic Wetness Index ≈ ln(1 / tan(slope))
                    higher TWI = wetter terrain = lower fire risk

WHY ASPECT IS ENCODED AS SIN/COS
----------------------------------
Aspect is a circular variable (0° = 360° = North). If you use raw degrees,
the model sees 1° and 359° as far apart, but they are both nearly-North.
Encoding as (sin, cos) pair makes the representation continuous:
  North (0°/360°): sin=0, cos=1
  East  (90°):     sin=1, cos=0
  South (180°):    sin=0, cos=-1
  West  (270°):    sin=-1, cos=0

CRITICAL BUG FIX — DEM SLOPE COMPUTATION
------------------------------------------
The DEM file (dem_greece.tif) may be in EPSG:4326 (geographic, degrees).
If it is, pixel size is ~0.001 degrees.

WRONG (produces slope ≈ 90° everywhere):
    dz_dy, dz_dx = np.gradient(elevation_array)
    # gradient units = metres / 0.001 degree — huge number
    # arctan(huge) ≈ 90°

CORRECT:
    After reprojecting to EPSG:2100 (metric), pixel size is in metres.
    np.gradient(elevation, pixel_size_m, pixel_size_m)
    gives dimensionless rise/run → slope in correct degrees.

VERIFICATION (always assert after extraction):
    assert slope_mean < 45° — Greece has mean slope ≈ 11°
    If you see slope_mean > 45°, the bug is back.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

NODATA_OUT = -9999.0


def extract_dem_features(
    dem_path:       str | Path,
    reference_path: str | Path,
    output_dir:     str | Path,
    overwrite:      bool = False,
) -> Dict[str, Path]:
    """
    Extract 5 terrain features from DEM, aligned to reference grid.

    Parameters
    ----------
    dem_path : path to DEM raster (any CRS — will be reprojected)
    reference_path : path to Burn_Prob.img (defines target grid)
    output_dir : directory to write 5 terrain .tif files
    overwrite : overwrite existing output files

    Returns
    -------
    dict of feature_name → output Path
    Empty dict if dem_path does not exist.
    """
    dem_path       = Path(dem_path)
    reference_path = Path(reference_path)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dem_path.exists():
        warnings.warn(
            f"DEM not found: {dem_path}\n"
            "  Download SRTM 90m: https://srtm.csi.cgiar.org/\n"
            "  Or EU-DEM 25m: https://www.eea.europa.eu/data-and-maps/data/copernicus-land-monitoring-service-eu-dem\n"
            "  Run with --skip-dem to continue without terrain features."
        )
        return {}

    print(f"  Extracting DEM features from: {dem_path.name}")

    # ── Load reference grid parameters ────────────────────────────────────
    with rasterio.open(reference_path) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width
        # Pixel size in metres (reference CRS is EPSG:2100 — metric)
        pixel_size_m  = abs(dst_transform.a)

    # ── Load and reproject elevation ───────────────────────────────────────
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float64)
        nd   = src.nodata
        if nd is not None:
            elev[elev == nd] = np.nan
        elev[elev < -500] = np.nan   # filter ocean/bad values

        src_transform = src.transform
        src_crs       = src.crs

    # Reproject to reference grid
    elev_ref = np.full((dst_height, dst_width), np.nan, dtype=np.float64)
    reproject(
        source        = elev,
        destination   = elev_ref,
        src_transform = src_transform,
        src_crs       = src_crs,
        dst_transform = dst_transform,
        dst_crs       = dst_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = np.nan,
        dst_nodata    = np.nan,
    )
    print(f"    Elevation: mean={np.nanmean(elev_ref):.1f}m, "
          f"max={np.nanmax(elev_ref):.1f}m")

    # ── Compute slope and aspect ───────────────────────────────────────────
    # Fill NaN with 0 for gradient computation (masked back after)
    nan_mask = np.isnan(elev_ref)
    elev_fill = np.where(nan_mask, 0.0, elev_ref)

    # np.gradient with pixel_size_m gives dimensionless (m/m) gradient
    dz_dy, dz_dx = np.gradient(elev_fill, pixel_size_m, pixel_size_m)
    dz_dy[nan_mask] = np.nan
    dz_dx[nan_mask] = np.nan

    # Slope: arctan of total gradient magnitude
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[nan_mask] = np.nan

    # ASSERT: slope must be physically realistic for Greece
    mean_slope = float(np.nanmean(slope_deg))
    if mean_slope >= 45.0:
        raise ValueError(
            f"DEM slope bug! mean_slope={mean_slope:.1f}° (expected ≈11° for Greece).\n"
            "  Check that elevation is in metres and pixel_size is in metres.\n"
            "  If DEM is in EPSG:4326, it must be reprojected first."
        )
    print(f"    Slope: mean={mean_slope:.2f}°  "
          f"[ASSERTION PASSED: mean < 45°]")

    # Aspect: north=0, east=90, south=180, west=270
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_sin = np.sin(aspect_rad)   # N/S component
    aspect_cos = np.cos(aspect_rad)   # E/W component
    aspect_sin[nan_mask] = np.nan
    aspect_cos[nan_mask] = np.nan

    # TWI: ln(1 / tan(slope)) — simplified local TWI
    slope_for_twi = np.maximum(slope_rad, np.radians(0.001))
    twi = np.log(1.0 / (np.tan(slope_for_twi) + 1e-10))
    twi[nan_mask] = np.nan

    # ── Save each feature as .tif ──────────────────────────────────────────
    base_profile = dict(
        driver="GTiff", dtype="float32",
        width=dst_width, height=dst_height, count=1,
        crs=dst_crs, transform=dst_transform,
        nodata=NODATA_OUT,
        compress="lzw", tiled=True, blockxsize=512, blockysize=512,
    )

    features = {
        "dem_elevation_m": elev_ref,
        "dem_slope_deg":   slope_deg,
        "dem_aspect_sin":  aspect_sin,
        "dem_aspect_cos":  aspect_cos,
        "dem_twi":         twi,
    }

    output_paths = {}
    for name, arr in features.items():
        out_path = output_dir / f"{name}.tif"
        if out_path.exists() and not overwrite:
            print(f"    SKIP (exists): {name}.tif")
            output_paths[name] = out_path
            continue

        out = arr.astype(np.float32)
        out[np.isnan(out)] = NODATA_OUT

        with rasterio.open(out_path, "w", **base_profile) as dst:
            dst.write(out, 1)

        v = out[out != NODATA_OUT]
        print(f"    ✓  {name:<22} mean={np.mean(v):.3f}  std={np.std(v):.3f}")
        output_paths[name] = out_path

    return output_paths


def validate_dem_features(paths: Dict[str, Path]) -> bool:
    """Assert DEM feature values are physically realistic."""
    NODATA = -9999.0
    bounds = {
        "dem_elevation_m": (0.0,   3000.0),
        "dem_slope_deg":   (0.0,   44.9),    # must stay below 45
        "dem_aspect_sin":  (-1.0,  1.0),
        "dem_aspect_cos":  (-1.0,  1.0),
        "dem_twi":         (-20.0, 20.0),
    }
    all_ok = True
    for name, (lo, hi) in bounds.items():
        path = paths.get(name)
        if path is None:
            print(f"  ✗  MISSING: {name}")
            all_ok = False
            continue
        with rasterio.open(path) as s:
            arr = s.read(1).astype(np.float64)
        v    = arr[arr != NODATA]
        mean = float(np.mean(v))
        ok   = lo <= mean <= hi
        sym  = "✓" if ok else "✗ FAILED"
        print(f"  {sym}  {name:<22} mean={mean:.3f}  expected=[{lo},{hi}]")
        all_ok = all_ok and ok
    return all_ok