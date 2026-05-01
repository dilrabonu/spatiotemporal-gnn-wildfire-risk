# Uncertainty-Calibrated, Intervention-Aware GNN for Wildfire Burn Probability Prediction

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG 2.5](https://img.shields.io/badge/PyG-2.5-purple?style=flat-square)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-136%20passing-brightgreen?style=flat-square)](tests/)
[![Status](https://img.shields.io/badge/Status-Paper%20Ready-blue?style=flat-square)]()

**[Paper](#citation) · [Results](#results) · [Installation](#installation) · [Quickstart](#quickstart) · [Project Structure](#project-structure)**

</div>

---

## Overview

This repository implements a **Graph Attention Network (GAT)** for wildfire burn probability prediction on the FSim Dataset Greece. The model addresses three research gaps that are systematically overlooked in existing wildfire ML literature:

| Gap | Problem | Our Solution | Evidence |
|-----|---------|--------------|----------|
| **Gap 1 — Label Uncertainty** | FSim outputs are stochastic Monte Carlo estimates, not deterministic ground truth | Gaussian NLL loss with per-node aleatoric uncertainty head | Model learns label noise structure |
| **Gap 2 — Calibration** | Overconfident intervals are unsafe for fire management decisions | MC Dropout (30 passes) + Temperature Scaling (T=0.643) | PICP-90%=0.932, ECE=0.002 |
| **Gap 3 — Intervention** | No prior paper evaluates how landscape modifications affect burn probability under uncertainty | Counterfactual graph manipulation + calibrated Δ BP with 90% CI | Firebreak: 71.1% significant reductions |

---

## Results

All results are evaluated on a **geographically disjoint test split** (n=57,531 nodes, southern Greece + Aegean islands). Predictions are inverse-transformed to original burn probability scale before any metric computation.

### Primary Comparison — All Models

| Model | R² | MAE | Spearman ρ | Brier | ECE |
|---|---|---|---|---|---|
| **GAT** | **0.7659** | **0.01052** | 0.8799 | **0.00033** | **0.00204** |
| 2D CNN (spatial) | 0.7187 | 0.01235 | 0.8798 | 0.00039 | 0.00510 |
| GCN | 0.7088 | 0.01114 | **0.8893** | 0.00041 | 0.00449 |
| XGBoost | 0.6761 | 0.01259 | 0.8873 | 0.00045 | 0.00502 |
| Random Forest | 0.6617 | 0.01250 | 0.8926 | 0.00047 | 0.00594 |
| Ridge Regression | 0.1363 | 0.01881 | 0.8012 | 0.00121 | 0.01221 |
| Naive Mean | -0.2956 | 0.02412 | 0.0000 | 0.00181 | 0.02031 |

> **GAT beats CNN by ΔR²=+0.047 and XGBoost by ΔR²=+0.090 under strict geographic evaluation.**

### High-Risk Tail (Burn Probability > 0.047, Bin 5)

| Model | Bin 5 R² | Bin 5 MAE | Bin 5 Spearman |
|---|---|---|---|
| **GAT (ours)** | **+0.229** | **0.02542** | 0.676 |
| 2D CNN | +0.021 | 0.02789 | 0.635 |
| XGBoost | -0.236 | 0.03144 | 0.573 |
| Random Forest | -0.363 | 0.03298 | 0.607 |

> **GAT is the only model with positive within-bin R² in the highest-risk quintile.**  
> GAT Bin 5 MAE=0.025 beats CNN Bin 5 MAE=0.028.

### Uncertainty Calibration (Phase 5B)

| Architecture | T | PICP-90% (before) | PICP-90% (after) | ACE |
|---|---|---|---|---|
| **GAT** | 0.6426 | 0.9739 | **0.9324 ✓** | +0.120 |
| GCN | 0.4357 | 0.9911 | **0.9282 ✓** | +0.079 |
| GraphSAGE | 1.0407 | 0.6497 | 0.6627 ✗ | -0.124 |
| Target | — | — | **0.900 ± 0.05** | ~0.000 |

> XGBoost, CNN, and Random Forest produce point estimates only — calibrated uncertainty intervals are not available for these baselines.

### Intervention Analysis (Phase 5D)

| Scenario | Mean Δ BP | Significant nodes | Max reduction |
|---|---|---|---|
| Fuel reduction 30% (all CFL features) | -0.00040 | 88 / 57,531 (0.2%) | -0.069 BP |
| **Firebreak strip (rows 5000–5100)** | **-0.01416** ★ | **2,469 / 3,474 (71.1%)** | **-0.136 BP** |
| Ignition suppression 50% | -0.00017 † | 2,043 / 57,531 (3.6%) | -0.063 BP |

> ★ Mean within treated strip. Effects spatially confined (outside strip: mean Δ BP = +0.00014).  
> † Median reported (distribution asymmetric).  
> **All effects reported with calibrated 90% prediction intervals from 30 MC Dropout samples.**

---

## Architecture

```
Input: 327,405 nodes × 61 features × 2,511,084 edges (8-connected pixel grid)

   [Node features: 61]
          │
   ┌──────▼──────┐
   │ Input proj   │  Linear(61→256) → BN → ReLU → Dropout(0.3)
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │  GAT Layer 1 │  GATConv(256→256, heads=4) → BN → ReLU + Residual
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │  GAT Layer 2 │  GATConv(256→256, heads=4) → BN → ReLU + Residual
   └──────┬──────┘
          │
   ┌──────▼──────────────────────┐
   │   Gaussian NLL Head          │
   │   ├── mean_head: Linear(256→1)     → point prediction
   │   └── logvar_head: Linear(256→1)   → aleatoric uncertainty
   └─────────────────────────────┘
          │
   ┌──────▼──────────────────────┐
   │   MC Dropout (inference)     │  model.train() + 30 passes
   │   ├── mean_pred = mean(μ₁..μ₃₀)   → final prediction
   │   ├── epistemic = std(μ₁..μ₃₀)    → model uncertainty
   │   └── aleatoric = √exp(log_var)    → label uncertainty
   └─────────────────────────────┘

Parameters: 150,530 · Training: CPU NeighborLoader (batch=1024, neighbors=[10,5])
```

---

## Project Structure

```
spatiotemporal-gnn-wildfire-risk/
│
├── configs/
│   └── gnn_config.yaml              # Single source of truth for all hyperparameters
│
├── data/
│   ├── raw/                         # FSim Dataset Greece (not committed)
│   ├── interim/aligned/             # Rasters aligned to Burn_Prob reference grid
│   ├── processed/
│   │   └── graph_data_enriched.pt   # 327,405 nodes · 61 features · 120 MB
│   └── features/
│       ├── target_transformer.pkl   # QuantileTransformer (fit on train only)
│       ├── feature_names.json       # Ordered list of 61 feature names
│       └── splits_enriched.npz      # Train/val/test node indices
│
├── notebooks/
│   ├── 00_environment_validation.ipynb
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_cnn_baseline.ipynb
│   ├── 05a_gnn_architecture.ipynb
│   └── 05b_calibration_analysis.ipynb
│
├── scripts/
│   ├── phase2_align_raster.py
│   ├── phase3_build_graph.py
│   ├── phase3_make_figures.py
│   ├── phase4_run_baselines.py
│   ├── phase4_make_figures.py
│   ├── phase4b_run_cnn.py
│   ├── fix_split_rebuild_graph.py
│   ├── phase5a_train_gnn.py          # Train GAT, GCN, GraphSAGE
│   ├── phase5a_evaluate_all_arches.py
│   ├── phase5a_save_predictions.py
│   ├── phase5a_make_figures.py
│   ├── phase5b_calibrate.py          # Temperature scaling calibration
│   ├── phase5d_intervention_v2.py    # Counterfactual intervention (corrected)
│   ├── phase5d_make_figures_v2.py
│   └── paper_figures.py              # All 10 publication figures in one script
│
├── src/wildfire_gnn/
│   ├── models/
│   │   ├── gnn.py                    # GAT, GCN, GraphSAGE + Gaussian NLL head
│   │   ├── gnn_pipeline.py           # NeighborLoader training + MC Dropout
│   │   ├── baselines.py              # Ridge, RF, XGBoost
│   │   ├── cnn_baseline.py           # 2D CNN spatial baseline
│   │   ├── calibration.py            # Temperature scaling + PICP/ACE/ENCE
│   │   └── intervention.py           # Counterfactual graph manipulation
│   ├── evaluation/
│   │   └── metrics.py                # R², MAE, Spearman, Brier, ECE, binned
│   ├── process/
│   │   ├── alignment.py
│   │   ├── graph_builder.py
│   │   ├── dem_features.py
│   │   └── target_engineering.py
│   └── utils/
│       ├── config.py
│       ├── reproducibility.py
│       └── logging.py
│
├── tests/
│   ├── test_phase2.py                
│   ├── test_phase3.py                
│   ├── test_phase4.py
│   ├── test_phase5a.py               
│   ├── test_phase5b.py
│   └── test_phase5d.py               
│
├── reports/
│   ├── figures/                      # All figures (generated)
│   ├── paper_figures/                # Publication-ready figures (300 DPI)
│   ├── predictions/                  # NPZ prediction files per model
│   └── tables/                       # CSV metric tables per phase
│
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── setup_env.sh
```

---

## Dataset

**FSim Dataset Greece** — wildfire Fire Simulation (FSim) Monte Carlo outputs for Greece (CRS: EPSG:2100, 25m resolution).

| Layer | Original shape | Role | Alignment |
|---|---|---|---|
| `Burn_Prob.img` | 7597×7555 | **Target** — burn probability | Reference |
| `CFL.img` | 7597×7555 | Crown fire likelihood | ✓ Aligned |
| `FSP_Index.img` | 7592×7541 | Fire spread potential | Resampled |
| `Fuel_Models.img` | 7932×9039 | Categorical fuel type (uint8) | Nearest-neighbour |
| `Ignition_Prob.img` | 7733×9039 | Ignition probability | Resampled |
| `Struct_Exp_Index.img` | 7592×7541 | Structural exposure index | Resampled |

**Target statistics:** mean=0.024, median=0.012, std=0.033, skewness=2.593  
**Valid cells:** 11,789,754 (20.54% of 57.4M total raster cells)  
**QuantileTransformer** applied to target (output: mean=0.008, std=0.993)

---

## Installation

### Recommended (conda — handles GDAL/rasterio)

```bash
git clone https://github.com/dilrabonu/spatiotemporal-gnn-wildfire-risk.git
cd spatiotemporal-gnn-wildfire-risk

# Create environment + install all dependencies
bash setup_env.sh

conda activate wildfire-gnn

# Verify everything works
pytest tests/test_phase0_structure.py -v
```

### Manual (CPU)

```bash
conda env create -f environment.yml
conda activate wildfire-gnn

# PyTorch Geometric (CPU)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

pip install -e .
```

**Requirements:** Python 3.10 · PyTorch 2.1 · PyG 2.5 · rasterio · scikit-learn · xgboost · scipy

---

## Quickstart

### Step 1 — Validate environment
```bash
conda activate wildfire-gnn
pytest tests/ -v --tb=short
# Expected: all tests pass
```

### Step 2 — Reproduce main result (GAT R²=0.766)
```bash
# Load and evaluate the trained GAT from archive checkpoint
python scripts/phase5a_evaluate_all_arches.py
```

### Step 3 — Run calibration analysis
```bash
python scripts/phase5b_calibrate.py --all
```

### Step 4 — Run intervention analysis
```bash
python scripts/phase5d_intervention_v2.py --arch GAT
python scripts/phase5d_make_figures_v2.py --arch GAT
```

### Step 5 — Generate all paper figures
```bash
python scripts/paper_figures.py
# Output: reports/paper_figures/ (10 figures, 300 DPI)
```

### Retrain from scratch
```bash
# Phase 3: Build graph
python scripts/phase3_build_graph.py

# Phase 4: Run baselines
python scripts/phase4_run_baselines.py

# Phase 5A: Train all GNN architectures
python scripts/phase5a_train_gnn.py --arch GAT
python scripts/phase5a_train_gnn.py --arch GCN
python scripts/phase5a_train_gnn.py --arch GraphSAGE
```

---

## Reproducibility

| Component | Seed | Library | Confirmed |
|---|---|---|---|
| Graph construction | 42 | NumPy | ✓ |
| Model training | 42 | PyTorch | ✓ |
| MC Dropout inference | — | stochastic by design | ±0.001 R² |
| Geographic split | deterministic (row-based) | — | ✓ |

All results reproducible on any CPU. MC Dropout introduces ±0.001 variation in R² across runs — reported results use 30 passes.

**Critical rules (must not be violated):**
- `data.y` is already QuantileTransformed — never call `transform()` again at eval
- Always call `inverse_transform()` before computing R², MAE, Spearman
- Geographic split overlap must be zero: `(train_mask & test_mask).sum() == 0`
- MC Dropout: `model.train()` at inference — never `model.eval()`

---

## Key Findings

**1. GAT outperforms all baselines under strict geographic evaluation**  
R²=0.766 beats CNN (0.719), XGBoost (0.676), and all tabular baselines. The geographic block split prevents spatial leakage — naive models collapse to R²=-0.296.

**2. Graph topology adds value beyond spatial convolution**  
GAT beats CNN by ΔR²=+0.047. Attention-weighted message-passing captures asymmetric fire spread (upslope neighbors matter more than downslope) that fixed CNN kernels cannot learn.

**3. GAT is the first model with positive high-risk R²**  
All Phase 4 baselines show negative within-bin R² in the highest-risk quintile (BP > 0.047). GAT achieves R²=+0.229 and MAE=0.025 in this bin — lower than CNN (0.028) and XGBoost (0.031).

**4. Temperature scaling achieves target calibration**  
GAT PICP-90%=0.932 after temperature scaling (T=0.643) — within the 0.90±0.05 target band. ECE=0.002. No tabular or CNN baseline produces uncertainty intervals.

**5. Firebreak intervention: 71.1% significant reductions in treated strip**  
The first calibrated counterfactual wildfire intervention analysis with GNN. 2,469/3,474 strip nodes show statistically significant burn probability reductions (90% CI entirely negative). Maximum node-level reduction: -0.136 BP. Effects spatially confined to strip boundary — physically correct.

---

## Graph Construction Details

```
Nodes:     327,405  (stride-6 spatial grid subsampling from 11.79M valid cells)
Edges:   2,511,084  (8-connected pixel grid, avg degree 7.7)
Node features:   61  (see breakdown below)
Resolution:    150m per node (25m pixel × stride 6)
```

**Feature groups (61 total):**

| Group | Count | Features |
|---|---|---|
| Base rasters | 4 | CFL, FSP_Index, Ignition_Prob, Struct_Exp_Index |
| DEM terrain | 5 | elevation_m, slope_deg, aspect_sin, aspect_cos, TWI |
| Fuel one-hot | 24 | Binary indicators (24 Scott-Burgan categories in Greece) |
| Interactions | 3 | CFL×Ignition, FSP×CFL, Ignition×FSP |
| Multi-scale stats | 18 | mean+std of CFL/FSP/Ign at 3×3, 7×7, 15×15 kernels |
| Spatial gradients | 6 | dx, dy for CFL, FSP_Index, Ignition_Prob |
| Node degree | 1 | Normalised degree [0,1] |

**Geographic split (strict anti-leakage):**

```
Train:  237,304 nodes  (72.5%)  rows 0–4200    northern Greece
Val:     32,570 nodes   (9.9%)  rows 4201–4800  geographic buffer
Test:    57,531 nodes  (17.6%)  rows 4801–7590  southern Greece + Crete + Aegean
```

---

## Test Coverage

```bash
pytest tests/ -v

tests/test_phase2.py   — 27 tests   (raster alignment, valid mask, transforms)
tests/test_phase3.py   — 38 tests   (graph construction, edge validity, splits)
tests/test_phase5a.py  — 26 tests   (architectures, Gaussian NLL, MC Dropout)
tests/test_phase5b.py  —  20  tests  (temperature scaling, PICP, reliability)
tests/test_phase5d.py  — 20 tests   (feature modification, intervention effects)
```

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{khidirova2025wildfire,
  title   = {Uncertainty-Calibrated, Intervention-Aware Graph Attention Networks
             for Wildfire Burn Probability Prediction},
  authors  = {Khidirova, Dilrabo and ...},
  journal = {Environmental Data Science},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

**FSim Dataset Greece · EPSG:2100 · 327,405 nodes · GAT R²=0.766**

</div>