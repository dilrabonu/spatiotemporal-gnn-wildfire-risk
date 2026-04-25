"""
Phase 2 — Full Raster Alignment Pipeline (Run this script once)

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn          ← your project root
    python scripts/phase2_align_rasters.py
    python scripts/phase2_align_rasters.py --overwrite   # redo everything
    python scripts/phase2_align_rasters.py --skip-dem    # no DEM features

WHAT IT DOES (7 steps)
-----------------------
  Step 1  Verify all raw rasters exist
  Step 2  Load and audit raw rasters (report shape mismatches)
  Step 3  Analyze burn probability target distribution
  Step 4  Align all rasters to Burn_Prob.img reference grid
  Step 4b Extract DEM terrain features (if DEM available)
  Step 5  Verify all aligned rasters match reference shape/CRS
  Step 6  Build valid-cell mask (True where all rasters are valid)
  Step 7  Build baseline_dataset.csv and fit QuantileTransformer

OUTPUTS
-------
  data/interim/aligned/
    Burn_Prob.tif           7597 x 7555  reference (copied)
    CFL.tif                 7597 x 7555  bilinear resample
    FSP_Index.tif           7597 x 7555  bilinear resample
    Fuel_Models.tif         7597 x 7555  NEAREST-NEIGHBOR (categorical!)
    Ignition_Prob.tif       7597 x 7555  bilinear resample
    Struct_Exp_Index.tif    7597 x 7555  bilinear resample
    dem_elevation_m.tif     7597 x 7555  bilinear (if DEM present)
    dem_slope_deg.tif       7597 x 7555
    dem_aspect_sin.tif      7597 x 7555
    dem_aspect_cos.tif      7597 x 7555
    dem_twi.tif             7597 x 7555

  data/processed/
    baseline_dataset.csv    ~300k rows x 9 columns

  data/features/
    valid_cell_mask.npy     shape (7597,7555) dtype bool
    target_transformer.pkl  QuantileTransformer fitted on all valid BP

EXPECTED RUNTIME
----------------
  5-20 minutes depending on machine and whether DEM is present.
  The alignment step is the slowest (rasterio warp on large rasters).

KNOWN ISSUES TO WATCH
---------------------
  1. DEM slope = 90° bug: slope is computed AFTER reprojection to EPSG:2100
     (metric CRS), so pixel_size is in metres. The bug from the previous
     project (computing gradient on EPSG:4326 degree values) cannot occur
     here because we reproject first. Assert: slope_mean < 45°.

  2. Fuel_Models nearest-neighbor: check the aligned raster still contains
     only valid category codes (91-189, plus nodata). Assert no values
     outside this range after alignment.

  3. Valid cell count: expect ~250,000 to ~320,000 valid cells. If you get
     < 100,000, the nodata mask logic is too aggressive.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# ── Add src/ to path ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils import load_yaml_config, set_seed, get_logger
from wildfire_gnn.process.raster_io import (
    load_raster_stack, audit_alignment, print_audit
)
from wildfire_gnn.process.alignment import (
    align_all_rasters, verify_alignment, build_valid_cell_mask
)
from wildfire_gnn.process.target_engineering import (
    TargetTransformer, analyze_target_distribution
)

logger = get_logger("phase2")


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2: raster alignment and target engineering"
    )
    p.add_argument("--config",    default="configs/gnn_config.yaml")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-dem",  action="store_true")
    return p.parse_args()


def main():
    t0   = time.time()
    args = parse_args()

    print("\n" + "="*65)
    print("  Phase 2 — Raster Alignment + Target Engineering")
    print("="*65 + "\n")

    # ── Config ─────────────────────────────────────────────────────────────
    config = load_yaml_config(PROJECT_ROOT / args.config)
    set_seed(config["training"]["seed"])
    p      = config["paths"]

    raw_dir     = PROJECT_ROOT / p["raw_dir"]
    aligned_dir = PROJECT_ROOT / p["aligned_dir"]
    feat_dir    = PROJECT_ROOT / p["features_dir"]
    proc_dir    = PROJECT_ROOT / "data" / "processed"
    dem_path    = PROJECT_ROOT / "data" / "external" / "dem_greece.tif"

    for d in [aligned_dir, feat_dir, proc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Verify raw files ────────────────────────────────────────────
    print("  [1/7] Verifying raw raster files...")
    raw_paths = {
        "Burn_Prob":        raw_dir / "Burn_Prob.img",
        "CFL":              raw_dir / "CFL.img",
        "FSP_Index":        raw_dir / "FSP_Index.img",
        "Fuel_Models":      raw_dir / "Fuel_Models.img",
        "Ignition_Prob":    raw_dir / "Ignition_Prob.img",
        "Struct_Exp_Index": raw_dir / "Struct_Exp_Index.img",
    }
    missing = [n for n, path in raw_paths.items() if not path.exists()]
    if missing:
        print(f"\n  ✗  Missing: {missing}")
        print(f"     Check config paths.raw_dir = {raw_dir}")
        sys.exit(1)
    print(f"  ✓  All {len(raw_paths)} raw rasters found\n")

    # ── Step 2: Audit raw rasters ───────────────────────────────────────────
    print("  [2/7] Loading and auditing raw rasters...")
    stack = load_raster_stack(raw_paths, verbose=True)
    audit = audit_alignment(stack, reference_name="Burn_Prob")
    print_audit(audit, reference_name="Burn_Prob")

    # ── Step 3: Target distribution ─────────────────────────────────────────
    print("  [3/7] Analyzing Burn_Prob target distribution...")
    bp_valid = stack["Burn_Prob"].valid_data()
    analyze_target_distribution(bp_valid)

    # ── Step 4: Align all rasters ───────────────────────────────────────────
    print("\n  [4/7] Aligning rasters to Burn_Prob.img reference grid...")
    aligned_paths = align_all_rasters(
        raw_paths      = raw_paths,
        aligned_dir    = aligned_dir,
        reference_name = "Burn_Prob",
        overwrite      = args.overwrite,
    )

    # ── Step 4b: DEM terrain features ──────────────────────────────────────
    if not args.skip_dem:
        if dem_path.exists():
            print("\n  [4b] Extracting DEM terrain features...")
            from wildfire_gnn.data.dem_features import (
                extract_dem_features, validate_dem_features
            )
            dem_paths = extract_dem_features(
                dem_path       = dem_path,
                reference_path = aligned_paths["Burn_Prob"],
                output_dir     = aligned_dir,
                overwrite      = args.overwrite,
            )
            if dem_paths:
                print("\n  Validating DEM features...")
                ok = validate_dem_features(dem_paths)
                if not ok:
                    print("  ✗  DEM validation failed — check slope assertion")
                    sys.exit(1)
                aligned_paths.update(dem_paths)
        else:
            print(f"\n  ⚠  DEM not at {dem_path} — skipping terrain features")
            print("     Download SRTM/EU-DEM and place at data/external/dem_greece.tif")

    # ── Step 5: Verify alignment ─────────────────────────────────────────────
    print("  [5/7] Verifying alignment...")
    fsim_aligned = {k: v for k, v in aligned_paths.items() if k in raw_paths}
    if not verify_alignment(fsim_aligned, reference_name="Burn_Prob"):
        print("  ✗  ALIGNMENT FAILED — do not proceed to Phase 3")
        sys.exit(1)

    # ── Step 6: Valid-cell mask ──────────────────────────────────────────────
    print("  [6/7] Building valid-cell mask...")
    valid_mask = build_valid_cell_mask(fsim_aligned)
    mask_path  = feat_dir / "valid_cell_mask.npy"
    np.save(mask_path, valid_mask)
    print(f"  ✓  Mask saved: {mask_path}")

    # ── Step 7: Baseline CSV + transformer ──────────────────────────────────
    print("\n  [7/7] Building baseline_dataset.csv...")
    import pandas as pd
    import rasterio

    rows_idx, cols_idx = np.where(valid_mask)
    data = {"row": rows_idx, "col": cols_idx}
    for name, path in fsim_aligned.items():
        with rasterio.open(path) as s:
            arr = s.read(1).astype(np.float64)
            nd  = s.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        data[name] = arr[valid_mask]

    df = pd.DataFrame(data)
    if "Burn_Prob" in df.columns:
        df = df.rename(columns={"Burn_Prob": "target"})

    # Fit transformer on all valid BP values
    transformer = TargetTransformer(n_quantiles=1000)
    y_raw       = df["target"].dropna().values
    y_t         = transformer.fit_transform(y_raw)
    transformer.validate(y_t)
    transformer.save(feat_dir / "target_transformer.pkl")

    df["target_transformed"] = transformer.transform(df["target"].values)
    csv_path = proc_dir / "baseline_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓  CSV saved: {csv_path}  [{df.shape[0]:,} rows × {df.shape[1]} cols]")

    # ── Final assertions ─────────────────────────────────────────────────────
    print("\n  Final assertions:")
    assert df.shape[0] > 200_000, f"Too few valid cells: {df.shape[0]:,}"
    assert "target" in df.columns
    assert valid_mask.dtype == bool
    assert valid_mask.ndim == 2
    print(f"  ✓  n_valid_cells > 200k  [{df.shape[0]:,}]")
    print(f"  ✓  target column present")
    print(f"  ✓  valid_mask is 2D bool")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Phase 2 Complete — {elapsed:.1f}s")
    print(f"  Ready for Phase 3: Graph Construction")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()