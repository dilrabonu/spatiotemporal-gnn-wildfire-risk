"""
wildfire_gnn
============
Uncertainty-calibrated, intervention-aware spatiotemporal GNN
for wildfire burn probability prediction (FSim dataset, Greece).

Research gaps addressed
-----------------------
Gap 1  — Label uncertainty: simulation outputs are stochastic MC estimates.
Gap 2  — Calibration: overconfident predictions invalid for risk decisions.
Gap 3  — Intervention: counterfactual analysis of fuel / firebreak scenarios.

Sub-packages
------------
data        — raster loading, alignment, graph construction, spatial splits
features    — feature engineering (terrain, multi-scale, interactions)
models      — GNN architectures (GAT, GraphSAGE, GCN) + uncertainty heads
evaluation  — metrics, calibration, reliability diagrams, intervention analysis
utils       — config loading, logging, reproducibility helpers
"""

__version__ = "0.1.0"
__author__ = "Dilrabo Khidirova"