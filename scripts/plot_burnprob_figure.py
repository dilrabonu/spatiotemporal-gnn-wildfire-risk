"""
reports/paper_figures3/burn_prob_greece.png  (+ .pdf)

Run from project root:
    python scripts/plot_burnprob_figure.py
"""

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# ---- paths (relative to project root) --------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BURN = ROOT / "data/raw/FSim_Dataset_Greece_raw_files/Burn_Prob.img"
OUTDIR = ROOT / "reports/paper_figures3"
OUTDIR.mkdir(parents=True, exist_ok=True)   # created fresh, nothing overwritten elsewhere

# ---- split row bands (original raster rows, from your config) --------------
TRAIN_ROWS = (0, 4200)
VAL_ROWS   = (4201, 4800)
TEST_ROWS  = (4801, 7597)

# ---- read the raster -------------------------------------------------------
with rasterio.open(BURN) as src:
    arr = src.read(1).astype("float32")
    nodata = src.nodata
    print(f"Raster shape: {arr.shape}, nodata={nodata}")

# mask nodata + the FSim nodata sentinel used in your config
arr = np.where(arr <= -1e30, np.nan, arr)
if nodata is not None:
    arr = np.where(arr == nodata, np.nan, arr)
# burn prob is in [0, ~0.25]; clip stray values, keep NaN for water/coastline
arr = np.where((arr < 0) | (arr > 1.0), np.nan, arr)

valid = np.isfinite(arr)
print(f"Valid cells: {valid.sum():,}  "
      f"(min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}, "
      f"mean={np.nanmean(arr):.4f})")

# colour scale: cap at the 99.5th percentile so the thin high-risk tail
# doesn't wash out the map (your target max ≈ 0.25)
vmax = np.nanpercentile(arr, 99.5)
norm = Normalize(vmin=0.0, vmax=vmax)

# FIGURE A — burn-probability map alone

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
im = ax.imshow(arr, cmap="inferno", norm=norm, interpolation="nearest")
ax.set_title("FSim burn probability — Greece", fontsize=11)
ax.set_xlabel("Column (W–E)", fontsize=9)
ax.set_ylabel("Row (N–S)", fontsize=9)
ax.tick_params(labelsize=8)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Burn probability", fontsize=9)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "burn_prob_greece.png", bbox_inches="tight")
fig.savefig(OUTDIR / "burn_prob_greece.pdf", bbox_inches="tight")
plt.close(fig)

# FIGURE B — same map with the geographic block split marked
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
im = ax.imshow(arr, cmap="inferno", norm=norm, interpolation="nearest")
ncols = arr.shape[1]

def band(r0, r1, color, label):
    ax.add_patch(Rectangle((0, r0), ncols, r1 - r0, fill=False,
                           edgecolor=color, linewidth=1.5, label=label))
    ax.axhline(r1, color=color, linewidth=0.8, linestyle="--")

band(*TRAIN_ROWS, "cyan",   "Train (N Greece)")
band(*VAL_ROWS,   "lime",   "Validation (buffer)")
band(*TEST_ROWS,  "white",  "Test (S Greece)")

ax.set_title("Burn probability with geographic block split", fontsize=11)
ax.set_xlabel("Column (W–E)", fontsize=9)
ax.set_ylabel("Row (N–S)", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Burn probability", fontsize=9)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "burn_prob_greece_split.png", bbox_inches="tight")
fig.savefig(OUTDIR / "burn_prob_greece_split.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\nSaved to: {OUTDIR}")
print("  burn_prob_greece.png / .pdf")
print("  burn_prob_greece_split.png / .pdf")