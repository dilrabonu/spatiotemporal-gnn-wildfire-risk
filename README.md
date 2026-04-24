# Wildfire Burn Probability Prediction  
### Uncertainty-Calibrated, Intervention-Aware Spatiotemporal GNN

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![PyG 2.5](https://img.shields.io/badge/PyG-2.5-red.svg)](https://pyg.org)

---

## Research Motivation

Existing wildfire prediction models overlook three critical challenges:

| Gap | Problem | Our approach |
|-----|---------|--------------|
| **Gap 1 — Label uncertainty** | FSim simulation outputs are stochastic Monte Carlo estimates, not ground truth. Models treat them as noise-free targets. | Gaussian NLL loss with per-node aleatoric uncertainty head |
| **Gap 2 — Calibration** | Overconfident predictions are unreliable for high-stakes wildfire management decisions. | MC Dropout + temperature scaling; ECE / Brier evaluation |
| **Gap 3 — Intervention** | No existing model evaluates how landscape modifications (fuel reduction, firebreaks) affect fire spread under uncertainty. | Counterfactual graph manipulation + uncertainty propagation |

---

## Project Structure

```
wildfire-uncertainty-gnn/
│
├── configs/
│   └── gnn_config.yaml          # All hyperparameters and paths — single source of truth
│
├── data/
│   ├── raw/                     # FSim Dataset Greece (read-only, not committed)
│   ├── interim/aligned/         # Rasters aligned to common reference grid
│   ├── processed/               # graph_data_enriched.pt, baseline_dataset.csv
│   ├── features/                # Feature parquets, split indices, transformers
│   └── external/                # DEM (dem_greece.tif)
│
├── notebooks/
│   ├── 00_environment_validation.ipynb
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_gnn_experiments.ipynb
│   └── 05_uncertainty_calibration_analysis.ipynb
│
├── reports/
│   ├── figures/                 # All paper figures (generated, not committed)
│   └── tables/                  # Metric tables (CSV, generated)
│
├── scripts/                     # Standalone Python scripts (non-notebook)
│
├── src/wildfire_gnn/            # Source package (pip install -e .)
│   ├── data/                    # Raster loading, alignment, graph construction
│   ├── features/                # Feature engineering
│   ├── models/                  # GNN architectures + uncertainty heads
│   ├── evaluation/              # Metrics, calibration, intervention analysis
│   └── utils/                   # Config, logging, reproducibility
│
└── tests/                       # pytest — run before each phase
```

---

## Dataset

**FSim Dataset Greece** — wildfire simulation outputs from the Fire Simulation (FSim) system for Greece.

| File | Shape | Description |
|------|-------|-------------|
| `Burn_Prob.img` | 7597×7555 | **Target** — burn probability per pixel |
| `CFL.img` | 7597×7555 | Crown fire likelihood |
| `FSP_Index.img` | 7592×7541 | Fire spread potential index |
| `Fuel_Models.img` | 7932×9039 | Categorical fuel type (uint8) |
| `Ignition_Prob.img` | 7733×9039 | Ignition probability |
| `Struct_Exp_Index.img` | 7592×7541 | Structural exposure index |

**Critical note**: Rasters are not aligned. All must be resampled to the `Burn_Prob.img` reference grid (EPSG:2100) before graph construction.

---

## Installation

### Full install (recommended — handles GDAL/rasterio via conda)

```bash
git clone https://github.com/dilrabonu/wildfire-uncertainty-gnn.git
cd wildfire-uncertainty-gnn

# One-command setup (creates env + installs PyG + validates)
bash setup_env.sh

# Activate
conda activate wildfire-gnn
```

### Manual install (CPU only)

```bash
conda env create -f environment.yml
conda activate wildfire-gnn

# PyG extensions for CPU
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

pip install -e .
```

---

## Quickstart

```bash
conda activate wildfire-gnn
jupyter lab
# → notebooks/00_environment_validation.ipynb  (verify setup)
# → notebooks/01_dataset_exploration.ipynb     (Phase 2)
```

Run tests:
```bash
pytest tests/test_phase0_structure.py -v
```

---

## Research Phases

| Phase | Notebook | Output |
|-------|----------|--------|
| 0 — Setup | `00_environment_validation.ipynb` | Verified environment |
| 1 — Literature | (doc) | Gap statement, RQs |
| 2 — Data audit | `01_dataset_exploration.ipynb` | Aligned rasters, target analysis |
| 3 — Graph construction | `02_graph_construction.ipynb` | `graph_data_enriched.pt` (300k nodes, 58 features) |
| 4 — Baselines | `03_baseline_models.ipynb` | RF, XGBoost, CNN results |
| 5 — Main GNN | `04_gnn_experiments.ipynb` | GAT + uncertainty + calibration + intervention |
| 6 — Paper | `05_uncertainty_calibration_analysis.ipynb` + writing | Submission-ready paper |

---

## Key Design Decisions

**Spatial split** — Data is split by geographic block (raster row), not randomly. Random splits create leakage through spatial autocorrelation and inflate reported R² by 0.3–0.5.

**Enriched features** — 58 node features (vs 4 raw rasters) embed spatial context directly into each node. This is essential for generalization under geographic splits where GNN message-passing cannot "reach across" the train/test boundary.

**Gaussian NLL loss** — Directly addresses Gap 1. The model predicts mean + log-variance, treating each node's burn probability as a distribution rather than a point estimate.

**MC Dropout at inference** — 30 forward passes with dropout active gives epistemic uncertainty estimates. Combined with aleatoric output, this provides total predictive uncertainty.

---

## Citation

```bibtex
@article{khidirova2025wildfire,
  title   = {Uncertainty-Calibrated, Intervention-Aware Graph Neural Networks for Wildfire Burn Probability Prediction},
  author  = {Khidirova, Dilrabo},
  journal = {TBD},
  year    = {2025}
}
```

---

## License

MIT License — see `LICENSE` for details.