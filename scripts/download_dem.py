"""
Download SRTM DEM for Greece and save to data/external/dem_greece.tif

USAGE
-----
    conda activate wildfire-gnn
    pip install elevation --quiet
    cd spatiotemporal_wildfire_gnn
    python scripts/download_dem.py

WHAT IT DOES
------------
    Downloads SRTM 90m elevation data for Greece bounding box,
    clips to the exact extent, and saves as dem_greece.tif
    in data/external/. File size: ~120 MB.

AFTER RUNNING
-------------
    Re-run Phase 2 alignment to add DEM features:
        python scripts/phase2_align_rasters.py --overwrite

    This adds 5 terrain features to the baseline dataset:
        dem_elevation_m, dem_slope_deg, dem_aspect_sin,
        dem_aspect_cos, dem_twi
"""

import sys
import subprocess
from pathlib import Path

# ── Greece bounding box (EPSG:4326, slightly padded) ──────────────────────
# lon_min, lat_min, lon_max, lat_max
BOUNDS = (19.2, 34.5, 29.0, 42.6)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "external" / "dem_greece.tif"
CACHE_DIR    = PROJECT_ROOT / "data" / "external" / ".srtm_cache"

def check_elevation_installed():
    try:
        import elevation
        return True
    except ImportError:
        return False

def main():
    print("\n" + "="*60)
    print("  DEM Download — SRTM 90m for Greece")
    print("="*60 + "\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Install elevation if missing ──────────────────────────────
    if not check_elevation_installed():
        print("  Installing elevation package...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "elevation", "--quiet"
        ])
        print("  ✓ elevation installed\n")

    import elevation

    # ── Step 2: Check if file already exists ──────────────────────────────
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1024**2
        print(f"  DEM already exists: {OUTPUT_PATH}")
        print(f"  Size: {size_mb:.1f} MB")
        print("  To re-download, delete the file and re-run.")
        return

    # ── Step 3: Download and clip ─────────────────────────────────────────
    print(f"  Downloading SRTM 90m for Greece...")
    print(f"  Bounding box: {BOUNDS}  (lon_min, lat_min, lon_max, lat_max)")
    print(f"  Output: {OUTPUT_PATH}")
    print("  This may take 2-10 minutes depending on connection speed.\n")

    elevation.clip(
        bounds    = BOUNDS,
        output    = str(OUTPUT_PATH),
        product   = "SRTM3",           # 90m resolution
        cache_dir = str(CACHE_DIR),
    )

    # ── Step 4: Verify ────────────────────────────────────────────────────
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1024**2
        print(f"\n  ✓ DEM saved: {OUTPUT_PATH}")
        print(f"  ✓ File size: {size_mb:.1f} MB")

        # Quick validation
        import rasterio
        import numpy as np
        with rasterio.open(OUTPUT_PATH) as src:
            print(f"  ✓ Shape  : {src.height} × {src.width}")
            print(f"  ✓ CRS    : {src.crs}")
            elev = src.read(1)
            valid = elev[elev != src.nodata]
            print(f"  ✓ Elev   : min={valid.min():.0f}m  "
                  f"max={valid.max():.0f}m  "
                  f"mean={valid.mean():.0f}m")

        print(f"""
  ── Next step ──────────────────────────────────────────────
  Re-run Phase 2 to add DEM terrain features:

      python scripts/phase2_align_rasters.py --overwrite

  This adds to baseline_dataset.csv:
      dem_elevation_m   dem_slope_deg   dem_aspect_sin
      dem_aspect_cos    dem_twi

  Feature count: 53 → 58 (required for full model)
  ────────────────────────────────────────────────────────────
        """)
    else:
        print("  ✗ Download failed — check internet connection")
        sys.exit(1)


if __name__ == "__main__":
    main()