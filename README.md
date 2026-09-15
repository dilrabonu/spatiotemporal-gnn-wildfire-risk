# Uncertainty-Calibrated, Intervention-Aware GAT for Wildfire Burn-Probability Prediction

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG 2.5](https://img.shields.io/badge/PyG-2.5-purple?style=flat-square)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**[Paper](#citation) · [Results](#results) · [Installation](#installation) · [Quickstart](#quickstart) · [Project Structure](#project-structure)**

</div>

---

## Overview

This repository implements an **uncertainty-calibrated, intervention-aware Graph
Attention Network (GAT)** for wildfire burn-probability prediction on the openly
published Greek FSim dataset. The framework couples attention-based spatial graph
learning with a Gaussian negative-log-likelihood (NLL) head, Monte Carlo (MC)
Dropout, and post-hoc temperature scaling, and it is evaluated under a strict
geographic block split. It jointly addresses four themes that are rarely combined
in wildfire machine-learning studies:

| Theme | Problem | Approach |
|-------|---------|----------|
| **Label uncertainty** | The FSim target is a stochastic Monte Carlo estimate, not a deterministic measurement | Gaussian NLL head with a per-node aleatoric variance component |
| **Calibration** | Raw predictive intervals are miscalibrated under geographic shift | MC Dropout (30 passes) + single-scalar temperature scaling |
| **Geographic generalization** | Random splits leak spatial information across folds | Strict north-to-south geographic block split with a 60 km buffer |
| **Intervention analysis** | Few studies evaluate how landscape changes alter predicted risk under uncertainty | Counterfactual feature perturbation with calibrated ΔBP intervals |

> **Important — target and scope.** The prediction target is the FSim
> *burn-probability surface*, itself a modelled quantity produced by Monte Carlo
> fire-spread simulation. All metrics therefore measure **emulation fidelity to
> the simulator**, not agreement with observed fire. See the paper's Limitations
> section for the causal scope of the intervention experiments.

---

## Results

All metrics are computed on a **geographically disjoint test fold**
(*n* = 57,531 nodes; southern Greece, the Peloponnese, the Aegean islands, and
Crete). Predictions are inverse-transformed to the original burn-probability
scale before any metric is computed.

### Whole-test predictive performance

| Model | R² | MAE | Spearman ρ | ECE |
|---|---|---|---|---|
| **GAT (proposed)** | **0.7659** | **0.01052** | 0.8799 | **0.00204** |
| 2D CNN (spatial) | 0.7187 | 0.01235 | 0.8798 | 0.00510 |
| GCN | 0.7088 | 0.01114 | 0.8893 | 0.00449 |
| GATv2 | 0.6850 | 0.01250 | 0.8960 | 0.00780 |
| XGBoost | 0.6761 | 0.01259 | 0.8873 | 0.00502 |
| Random Forest | 0.6617 | 0.01250 | 0.8926 | 0.00594 |
| GAT (vanilla, MSE head) | 0.6543 | 0.01390 | **0.8996** | 0.01030 |
| GraphSAGE | 0.5043 | 0.01655 | 0.8095 | 0.01534 |
| Ridge Regression | 0.1363 | 0.01881 | 0.8012 | 0.01221 |
| Naive Mean | −0.0730 | 0.02412 | 0.0000 | 0.02031 |

> The proposed GAT attains the best R², MAE, and ECE. Its Spearman rank
> correlation (0.880) is mid-ranked, because Spearman rewards only monotone
> ordering and not the magnitude calibration that distinguishes the model.
> A naive training-mean predictor attains R² = −0.073 on the test fold — the
> hard lower bound imposed by the geographic distribution shift.

### High-risk tail (Bin 5: burn probability ∈ [0.047, 0.208], *n* = 11,507)

| Model | Bin 5 R² | Bin 5 MAE | Bin 5 Spearman |
|---|---|---|---|
| GAT (vanilla) | **+0.293** | **0.02365** | 0.729 |
| GATv2 | +0.130 | 0.02556 | 0.768 |
| 2D CNN | +0.021 | 0.02789 | 0.635 |
| GCN | −0.224 | 0.03154 | 0.653 |
| XGBoost | −0.236 | 0.03144 | 0.573 |
| Random Forest | −0.363 | 0.03298 | 0.607 |
| GraphSAGE | −1.026 | 0.04699 | 0.733 |
| **GAT (proposed)** | **−1.481** | **0.05062** | 0.520 |
| Ridge Regression | −1.900 | 0.04738 | 0.626 |
| Naive Mean | −5.097 | 0.08393 | — |

> **Honest trade-off (this is a central finding, not a weakness to hide).**
> The point-accuracy variants — the vanilla GAT and GATv2, trained with an MSE
> head — lead the high-risk tail. The proposed GAT records the weakest tail
> point-accuracy among the graph models: its Bin 5 MAE (0.051) exceeds the lower
> edge of the bin itself (0.047), and its Bin 5 Spearman (0.520) is the lowest of
> any learning model. This is a direct consequence of the Gaussian NLL objective,
> which down-weights the high-variance nodes that dominate the tail and expresses
> that region through wider calibrated intervals rather than sharper point
> predictions. The proposed model accepts this trade-off in exchange for the
> calibrated whole-distribution behaviour and the intervention analysis it makes
> possible.

### Uncertainty calibration via temperature scaling

| Architecture | T\* | PICP-90% (before) | PICP-90% (after) | ACE |
|---|---|---|---|---|
| **GAT** | 0.643 | 0.974 | **0.932 ✓** | +0.120 |
| GCN | 0.436 | 0.991 | **0.928 ✓** | +0.079 |
| GraphSAGE | 1.041 | 0.650 | 0.663 ✗ | −0.124 |
| Target | — | — | **0.900 ± 0.05** | ≈ 0.000 |

> The GAT and GCN were initially *underconfident* (intervals too wide). Because
> `T* < 1`, temperature scaling **narrows** the intervals (calibrated
> σ = T·σ̂), moving PICP-90% from 0.974 to 0.932 for the GAT — into the ±5%
> acceptance band around the 0.90 target. GraphSAGE fails calibration and is
> reported as such. The tabular and CNN baselines produce point estimates only,
> so interval-based calibration (PICP, ACE, ENCE) is not available for them.

### Counterfactual intervention (calibrated GAT, T = 0.643, 30 MC passes, 90% PIs)

| Scenario | Mean ΔBP | Significant nodes | Max reduction |
|---|---|---|---|
| Fuel-treatment proxy (all CFL-derived features × 0.70) | −0.00040 | 88 / 57,531 (0.2%) | −0.069 BP |
| **Firebreak-strip proxy (rows 5000–5100, CFL-derived → 0)** | **−0.01416** ★ | **2,469 / 3,474 (71.1%)** | **−0.136 BP** |
| Ignition suppression (all Ignition-derived features × 0.50) | −0.00017 † | 2,043 / 57,531 (3.6%) | −0.063 BP |

> ★ Mean within the treated strip; effects are spatially confined (outside the
> strip, mean ΔBP = +0.00014). † Median reported (the distribution is
> asymmetric). All effects use calibrated 90% prediction intervals from paired
> MC Dropout.
>
> **Causal scope.** FSim exposes only two genuine simulation *inputs* — the
> Scott–Burgan fuel model and the ignition probability. The conditional flame
> length (CFL), fire-size-potential, and structure-exposure layers are *outputs*
> of the same run that produced the target. The learned model draws most of its
> signal from CFL-derived features, so the fuel-treatment and firebreak scenarios
> perturb the CFL-derived feature group and are reported as **proxies /
> sensitivity analyses**, not strict causal interventions. Only ignition
> suppression acts on a genuine input. The two-hop receptive field (~1.2 km) is
> much smaller than the fires encoded in the target, so the model cannot express
> downstream treatment effects — the observed spatial confinement bounds, rather
> than confirms, their absence.

---

## Architecture

```
Input: 327,405 nodes × 61 features · 2,511,084 directed edge entries
       (1,255,542 undirected, 8-connected 600 m node lattice, mean degree 7.7)

   [Node features: 61]
          │
   Linear(61 → 256) → BatchNorm → ReLU → Dropout(0.3)
          │
   GAT Layer 1 : GATConv(4 heads × 64 = 256) → BatchNorm → ReLU  (+ residual)
          │
   GAT Layer 2 : GATConv(4 heads × 64 = 256) → BatchNorm → ReLU  (+ residual)
          │
   Gaussian head
     ├── mean_head   : Linear(256 → 1)   → predictive mean  μ̂
     └── logvar_head : Linear(256 → 1)   → aleatoric log-variance (clamped)
          │
   MC Dropout (inference) : model.train(), BN in eval, 30 stochastic passes
     ├── μ̂        = mean of the 30 means            → point prediction
     ├── epistemic = variance of the 30 means
     └── aleatoric = mean of the 30 head variances
          │
   Temperature scaling : calibrated σ = T·σ̂   (T* = 0.643, fit on validation)

Parameters: 150,530 · Training: CPU NeighborLoader (batch 1024, neighbours [10, 5])
```

Two graph-attention layers, hidden width 256, four attention heads per layer
(each head 64 channels; concatenation restores 256). During a 10-epoch warm-up
the predicted log-variance is frozen (the objective reduces to MSE on μ̂); the
full Gaussian NLL is minimised thereafter.

---

## Dataset

**Greek FSim dataset** — country-scale Monte Carlo fire-spread simulation outputs
for Greece, published by Palaiologou et al. (2026), *Data in Brief* 64, 112304,
under a **CC BY-NC 4.0** licence (Zenodo: `10.5281/zenodo.17579289`). CRS
EPSG:2100. **The six co-registered raster layers are at 100 m resolution.** We
did not run any new fire simulations; all modelling begins from the published
rasters.

| Layer | Role | Meaning |
|---|---|---|
| `Burn_Prob` | **Target** | Annual burn probability = (times a cell burns) / (simulated fire seasons) |
| `CFL` | Output | **Conditional flame length** (m), probability-weighted over 20 flame-length bins |
| `FSP_Index` | Output | **Fire Size Potential Index** — mean size (ha) of fires ignited in a cell |
| `Struct_Exp_Index` | Output | **Structure Exposure Index** — non-annualised count of buildings reached |
| `Fuel_Models` | **Input** | Categorical Scott–Burgan fuel model codes (nearest-neighbour resampled) |
| `Ignition_Prob` | **Input** | Per-cell human ignition probability (itself a Random Forest output) |

> Only `Fuel_Models` and `Ignition_Prob` are genuine inputs to the FSim run;
> `CFL`, `FSP_Index`, and `Struct_Exp_Index` are sibling outputs of the same run
> that produced the target — a distinction that governs the intervention analysis.

**Target statistics (11,789,754 valid cells, 20.54% of the 57.4 M raster
extent):** mean 0.0242, median 0.0121, std 0.0328, skewness 2.59, range
[3.6 × 10⁻⁶, 0.251]. The target was mapped to an approximately standard-normal
distribution with a `QuantileTransformer` fit **on the training split only**
(transformed training-fold mean ≈ 0, std ≈ 1); all metrics are computed after
inverting the transform.

> **Note on the published target.** Because island Pyromes received far more
> simulated fires than mainland Pyromes, the dataset authors rescaled the burn
> probability per Pyrome to a common density of 17 ignitions/km². Our target is
> this published, Pyrome-corrected surface; we inherit the rescaling and do not
> reverse it.

---

## Graph construction

```
Nodes:          327,405   (stride-6 spatial subsampling of 11.79 M valid cells)
Edges:        2,511,084   directed entries (1,255,542 undirected), 8-connected
Node value:   a single 100 m cell   (NOT a 6×6 block average; 35 of 36 discarded)
Node spacing: 600 m       (100 m pixel × stride 6)
Node features:     61
```

**Feature groups (61 total):**

| Group | Count | Features |
|---|---|---|
| Base rasters | 4 | CFL, FSP_Index, Ignition_Prob, Struct_Exp_Index |
| DEM terrain | 5 | elevation, slope, sin(aspect), cos(aspect), TWI (from a 25 m DEM resampled to the analysis grid) |
| Fuel one-hot | 24 | Scott–Burgan fuel categories present in Greece |
| Interactions | 3 | CFL×Ignition, FSP×CFL, Ignition×FSP |
| Multi-scale stats | 18 | mean + std of the base rasters over 3×3, 7×7, 15×15 kernels (300 / 700 / 1500 m windows), computed on the full-resolution raster before subsampling |
| Spatial gradients | 6 | x- and y-derivatives of three primary rasters |
| Node degree | 1 | number of eight-connected neighbours |

**Geographic block split (strict, north → south in EPSG:2100):**

```
Train:  237,304 nodes  (72.5%)  rows 0–4200      northern Greece
Val:     32,570 nodes   (9.9%)  rows 4201–4800   ~60 km buffer band
Test:    57,531 nodes  (17.6%)  rows 4801–7597   southern Greece + Peloponnese + Aegean + Crete
```

No node is shared between folds and no graph edge crosses the train–test
boundary, so no information leaks through message passing. The widest
neighbourhood feature (the 15×15 kernel) reaches only 1.5 km, far inside the
~60 km buffer. The split also exposes a genuine distribution shift: the mean burn
probability is 0.019 in the training fold and 0.029 in the test fold.

---

## Project structure

```
spatiotemporal-gnn-wildfire-risk/
│
├── configs/
│   └── gnn_config.yaml              # Single source of truth for all hyperparameters
│
├── data/
│   ├── raw/                         # FSim Dataset Greece (not committed)
│   ├── interim/aligned/             # Rasters aligned to the Burn_Prob reference grid
│   ├── processed/
│   │   └── graph_data_enriched.pt   # 327,405 nodes · 61 features
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
│   ├── phase4_run_baselines.py       # Ridge, RF, XGBoost, Naive Mean
│   ├── phase4b_run_cnn.py            # 2D CNN spatial baseline
│   ├── phase5a_train_gnn.py          # Train GAT (Gaussian NLL head)
│   ├── phase5a_evaluate_all_arches.py# Table: whole-test metrics
│   ├── phase5b_calibrate.py          # Temperature scaling (GAT, GCN, GraphSAGE)
│   ├── phase5c_train_gat_baselines.py# Vanilla GAT (MSE head) and GATv2 (MSE head)
│   ├── phase5d_intervention_v2.py    # Counterfactual intervention (all derived features)
│   ├── phase5d_make_figures_v2.py
│   └── paper_figures.py              # Publication figures
│
├── src/wildfire_gnn/
│   ├── models/
│   │   ├── gnn.py                    # GAT, GCN, GraphSAGE, GATv2 + Gaussian NLL / regression heads
│   │   ├── baselines.py             # Ridge, RF, XGBoost
│   │   ├── cnn_baseline.py          # 2D CNN spatial baseline
│   │   ├── calibration.py           # Temperature scaling + PICP / ACE / ENCE
│   │   └── intervention.py          # Counterfactual feature perturbation
│   ├── evaluation/
│   │   └── metrics.py               # R², MAE, Spearman, ECE, binned metrics
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
├── reports/
│   ├── figures/                      # Generated figures
│   ├── paper_figures/                # Publication-ready figures
│   ├── predictions/                  # NPZ prediction files per model
│   └── tables/                       # CSV metric tables per phase
│
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── setup_env.sh
```

---

## Installation

### Recommended (conda — handles GDAL / rasterio)

```bash
git clone https://github.com/dilrabonu/spatiotemporal-gnn-wildfire-risk.git
cd spatiotemporal-gnn-wildfire-risk

bash setup_env.sh
conda activate wildfire-gnn
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

**Requirements:** Python 3.10 · PyTorch 2.1 · PyG 2.5 · rasterio · scikit-learn ·
xgboost · scipy.

---

## Quickstart

```bash
conda activate wildfire-gnn

# Reproduce the whole-test metrics table (GAT R² = 0.766)
python scripts/phase5a_evaluate_all_arches.py

# Calibration analysis (temperature scaling)
python scripts/phase5b_calibrate.py --all

# Counterfactual intervention analysis
python scripts/phase5d_intervention_v2.py --arch GAT
python scripts/phase5d_make_figures_v2.py --arch GAT

# Regenerate publication figures
python scripts/paper_figures.py
```

### Retrain from scratch

```bash
python scripts/phase3_build_graph.py            # build the graph
python scripts/phase4_run_baselines.py          # Ridge, RF, XGBoost, Naive Mean
python scripts/phase4b_run_cnn.py               # 2D CNN
python scripts/phase5a_train_gnn.py --arch GAT  # proposed GAT (Gaussian NLL)
python scripts/phase5c_train_gat_baselines.py --arch GAT_vanilla   # MSE head
python scripts/phase5c_train_gat_baselines.py --arch GATv2         # MSE head
```

---

## Reproducibility

| Component | Seed | Notes |
|---|---|---|
| Graph construction | 42 | Deterministic |
| Model training | 42 | Adam, cosine schedule, grad-clip 1.0 |
| Geographic split | — | Deterministic, row-based |
| MC Dropout inference | — | Stochastic by design; 30 passes; ≈ ±0.001 R² across runs |

**Critical rules (must not be violated):**

- `data.y` is already QuantileTransformed — never call `transform()` again at
  evaluation.
- Always call `inverse_transform()` before computing R², MAE, or Spearman.
- Geographic-split overlap must be zero: `(train_mask & test_mask).sum() == 0`.
- MC Dropout uses `model.train()` at inference (dropout active) while batch
  normalisation uses its running statistics (eval mode).

---

## Key findings

1. **GAT leads whole-test predictive performance under strict geographic
   evaluation.** R² = 0.766, ahead of the 2D CNN (0.719) and XGBoost (0.676).
   A naive predictor collapses to R² = −0.073, the hard lower bound set by the
   distribution shift.

2. **Graph topology adds value beyond fixed-kernel convolution.** The GAT
   improves on the 2D CNN by ΔR² = +0.047 and on XGBoost by ΔR² = +0.090.

3. **The Gaussian head trades tail point-accuracy for calibration.** The
   MSE-head variants (vanilla GAT, GATv2) lead the high-risk tail, whereas the
   proposed GAT is deliberately worst there (Bin 5 R² = −1.481) because the NLL
   objective represents high-variance nodes through wider intervals. This
   trade-off is reported openly.

4. **Temperature scaling reaches the target calibration band.** GAT
   PICP-90% = 0.932 after scaling (T* = 0.643), ECE = 0.002. GraphSAGE fails and
   is reported as such.

5. **Calibrated counterfactual intervention.** In the firebreak-strip proxy,
   71.1% of treated nodes show reductions whose full 90% interval lies below
   zero, with effects spatially confined to the strip. These scenarios are
   framed as proxies/sensitivity analyses (see *Causal scope* above), not strict
   causal interventions.

---

## Citation

```bibtex
@article{author2026wildfire,
  title   = {Uncertainty-Calibrated, Intervention-Aware Graph Attention Networks
             for Wildfire Probability Prediction},
  author  = {Author, First A. and Chatterjee, Ayan and Khidirova, Dilrabo},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

**Dataset:** Palaiologou, P. et al. (2026). *A dataset to support wildland fire
and fuel management in Greece created with stochastic wildfire simulations.*
Data in Brief 64, 112304. CC BY-NC 4.0.
Zenodo: https://zenodo.org/records/17579289 ·
Article: https://www.sciencedirect.com/science/article/pii/S2352340925010248

---

## License

MIT License — see [`LICENSE`](LICENSE). Note that the **Greek FSim dataset is
licensed CC BY-NC 4.0** (non-commercial); this repository's MIT licence covers
the code only, not the dataset.

---

<div align="center">

**Greek FSim dataset · EPSG:2100 · 100 m rasters · 327,405-node 600 m graph · GAT R² = 0.766**

Dilrabo Khidirova · [LinkedIn](https://www.linkedin.com/in/dilrabo-khidirova-3144b8244/)

</div>
