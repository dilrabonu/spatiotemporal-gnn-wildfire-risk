"""
Counterfactual Intervention Analysis — Phase 5D.

WHY THIS IS THE MOST IMPORTANT PHASE FOR PUBLICATION
------------------------------------------------------
Every prior wildfire ML paper asks: "Can we predict burn probability?"
No prior paper asks: "What happens to burn probability if we intervene?"

This phase introduces COUNTERFACTUAL REASONING to wildfire prediction:
  - What if we reduced Crown Fire Likelihood (CFL) by 30% through fuel treatment?
  - What if we added a firebreak (CFL = 0) across a landscape strip?
  - What if we suppressed ignition sources by 50%?

The GNN answers these questions by modifying graph node features and
re-running inference. This is possible ONLY with a graph model — XGBoost
and CNN cannot propagate intervention effects through spatial topology.

SCIENTIFIC NOVELTY
------------------
When we reduce CFL in cell A, the GAT also sees the effect in cells B,C,D
that are spatially connected to A — because their neighborhood features
change. This spatial propagation of intervention effects through the graph
is what makes GNN intervention analysis fundamentally different from:
  - Simply computing f(x_modified) in XGBoost (no propagation)
  - CNN: propagation is local (fixed 7×7 kernel), no topology awareness
  - GNN: propagation follows the actual fire-risk network topology

UNCERTAINTY IN INTERVENTION EFFECTS
------------------------------------
The intervention effect delta_y = y_new - y_orig is reported WITH
calibrated uncertainty bounds (using T=0.643 from Phase 5B):

  delta_y ± uncertainty(delta_y)

This allows statements like:
  "Fuel reduction (30% CFL) reduces burn probability by 0.008 ± 0.003
   (90% PI: [-0.014, -0.002]) in the treated region."

THREE INTERVENTION SCENARIOS
-----------------------------
1. fuel_reduction_30pct
   - Reduce CFL by 30% across all nodes (or a target region)
   - Models systematic fuel treatment programme
   - Expected effect: moderate BP reduction in high-CFL areas

2. firebreak_strip
   - Set CFL to 0 in a horizontal raster row band
   - Models a physical firebreak across the landscape
   - Expected effect: local BP reduction within strip + edge effects

3. ignition_suppression_50pct
   - Reduce Ignition_Prob by 50% across all nodes
   - Models improved fire detection + suppression response
   - Expected effect: BP reduction, especially in high-ignition areas

FEATURE INDEX LOOKUP
--------------------
Feature indices depend on graph.feature_names.json from Phase 3.
The intervention module looks up feature names → indices at runtime.
This is robust to any feature order changes.
"""

from __future__ import annotations
import json
import copy
import numpy as np
import torch
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════
# Feature index lookup
# ════════════════════════════════════════════════════════════════════════════

def get_feature_index(feature_name: str, feature_names: list[str]) -> int:
    """
    Look up the column index of a named feature in the 61-feature matrix.

    Parameters
    ----------
    feature_name  : exact name as in feature_names.json (e.g. "CFL")
    feature_names : ordered list from feature_names.json

    Returns
    -------
    int : column index in graph.x
    """
    if feature_name not in feature_names:
        available = [f for f in feature_names if feature_name.lower() in f.lower()]
        raise ValueError(
            f"Feature '{feature_name}' not found in feature_names.\n"
            f"Similar features: {available}\n"
            f"All features: {feature_names[:10]}..."
        )
    return feature_names.index(feature_name)


def load_feature_names(feature_names_path: Path) -> list[str]:
    """Load feature names from JSON file saved in Phase 3."""
    with open(feature_names_path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# Graph modification functions
# ════════════════════════════════════════════════════════════════════════════

def modify_feature_multiplicative(
    graph_x:        torch.Tensor,
    feature_idx:    int,
    delta_fraction: float,
    node_mask:      Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Modify a feature by a multiplicative factor: x_new = x * (1 + delta_fraction)

    Example: delta_fraction=-0.30 → 30% reduction in CFL

    Parameters
    ----------
    graph_x        : (N, F) node feature matrix (will NOT be modified in place)
    feature_idx    : column index of feature to modify
    delta_fraction : fractional change, e.g. -0.30 for 30% reduction
    node_mask      : (N,) bool tensor, apply only to True nodes.
                     If None, applies to all nodes.

    Returns
    -------
    x_modified : (N, F) new feature matrix (copy)
    """
    x_new = graph_x.clone()

    if node_mask is None:
        x_new[:, feature_idx] = x_new[:, feature_idx] * (1.0 + delta_fraction)
    else:
        x_new[node_mask, feature_idx] = (
            x_new[node_mask, feature_idx] * (1.0 + delta_fraction)
        )

    return x_new


def modify_feature_absolute(
    graph_x:      torch.Tensor,
    feature_idx:  int,
    new_value:    float,
    node_mask:    Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Set a feature to an absolute value: x_new = new_value

    Example: new_value=0.0 → firebreak (CFL=0 everywhere in strip)

    Parameters
    ----------
    graph_x     : (N, F) node feature matrix
    feature_idx : column index
    new_value   : value to set (original scale)
    node_mask   : (N,) bool tensor, apply only to True nodes
    """
    x_new = graph_x.clone()

    if node_mask is None:
        x_new[:, feature_idx] = new_value
    else:
        x_new[node_mask, feature_idx] = new_value

    return x_new


def build_row_band_mask(
    graph_pos:   torch.Tensor,
    row_min:     int,
    row_max:     int,
) -> torch.Tensor:
    """
    Create a node mask for a horizontal band of raster rows.

    Used for firebreak scenario: apply intervention only to nodes
    in rows [row_min, row_max] of the original raster.

    Parameters
    ----------
    graph_pos : (N, 2) tensor of [row, col] positions
    row_min   : minimum raster row (inclusive)
    row_max   : maximum raster row (inclusive)

    Returns
    -------
    mask : (N,) bool tensor, True for nodes in the row band
    """
    rows = graph_pos[:, 0]
    return (rows >= row_min) & (rows <= row_max)


def build_region_mask(
    graph_pos: torch.Tensor,
    row_min:   int,
    row_max:   int,
    col_min:   int,
    col_max:   int,
) -> torch.Tensor:
    """
    Create a node mask for a rectangular raster region.

    Used for spatially targeted interventions.
    """
    rows = graph_pos[:, 0]
    cols = graph_pos[:, 1]
    return (
        (rows >= row_min) & (rows <= row_max) &
        (cols >= col_min) & (cols <= col_max)
    )


# ════════════════════════════════════════════════════════════════════════════
# MC Dropout inference on modified graph
# ════════════════════════════════════════════════════════════════════════════

def run_mc_inference(
    model:       "torch.nn.Module",
    graph_x:     torch.Tensor,
    edge_index:  torch.Tensor,
    test_mask:   torch.Tensor,
    n_samples:   int = 30,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Run MC Dropout inference and return mean + calibrated uncertainty.

    Parameters
    ----------
    model       : trained GNN (dropout ON for MC Dropout)
    graph_x     : (N, F) feature matrix (may be modified for intervention)
    edge_index  : (2, E) edge index
    test_mask   : (N,) bool — which nodes to report
    n_samples   : number of MC Dropout passes
    temperature : T from Phase 5B temperature scaling

    Returns
    -------
    dict with:
        mean_pred  (N_test,) mean of MC samples (transformed scale)
        std_pred   (N_test,) MC std (epistemic, calibrated by T)
        aleatoric  (N_test,) aleatoric uncertainty from log_var
        total_unc  (N_test,) combined, temperature-scaled
        samples    (n_samples, N_test) all MC raw predictions
    """
    model.train()   # dropout ON — critical for MC Dropout

    sample_means   = []
    sample_logvars = []

    with torch.no_grad():
        for _ in range(n_samples):
            mean, lv = model(graph_x, edge_index)
            sample_means.append(mean[test_mask].cpu().numpy())
            sample_logvars.append(lv[test_mask].cpu().numpy())

    samples      = np.stack(sample_means)    # (n_samples, N_test)
    lv_stack     = np.stack(sample_logvars)

    mean_pred    = samples.mean(axis=0)
    std_pred     = samples.std(axis=0) * temperature   # calibrated epistemic
    aleatoric    = np.sqrt(np.exp(lv_stack.mean(axis=0)))
    total_unc    = np.sqrt((temperature * std_pred)**2 + aleatoric**2)

    return {
        "mean_pred": mean_pred,
        "std_pred":  std_pred,
        "aleatoric": aleatoric,
        "total_unc": total_unc,
        "samples":   samples,
    }


# ════════════════════════════════════════════════════════════════════════════
# Intervention effect computation
# ════════════════════════════════════════════════════════════════════════════

def compute_intervention_effect(
    mc_orig:     dict[str, np.ndarray],
    mc_new:      dict[str, np.ndarray],
    transformer,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Compute delta_y = y_new - y_orig with uncertainty bounds.

    Both predictions are in transformed (near-Gaussian) space.
    We compute the delta in transformed space and then propagate
    through the inverse transform.

    The uncertainty on delta_y accounts for uncertainty in BOTH
    y_orig and y_new (combined via root sum of squares because
    the MC samples are independent):
        std(delta) ≈ sqrt(std(y_new)² + std(y_orig)²)

    Parameters
    ----------
    mc_orig     : MC inference results on original graph
    mc_new      : MC inference results on modified graph
    transformer : QuantileTransformer for inverse transform
    temperature : T from Phase 5B

    Returns
    -------
    dict with:
        delta_bp         : mean intervention effect (original BP scale)
        delta_bp_lo_90   : 5th percentile of delta (lower 90% CI bound)
        delta_bp_hi_90   : 95th percentile of delta (upper 90% CI bound)
        delta_std_bp     : std of delta (original BP scale)
        y_orig_bp        : original predictions (BP scale)
        y_new_bp         : post-intervention predictions (BP scale)
        significant_mask : nodes where 90% CI does not contain 0
    """
    # Compute per-sample delta in transformed space
    # Both sample arrays: (30, N_test)
    samples_orig = mc_orig["samples"]   # (30, N_test)
    samples_new  = mc_new["samples"]    # (30, N_test)

    # Delta per MC sample
    delta_samples = samples_new - samples_orig  # (30, N_test)

    # Mean delta in transformed space
    delta_t_mean = delta_samples.mean(axis=0)

    # Inverse transform mean predictions to BP scale
    y_orig_bp = transformer.inverse_transform(
        mc_orig["mean_pred"].reshape(-1, 1)
    ).ravel()
    y_new_bp = transformer.inverse_transform(
        mc_new["mean_pred"].reshape(-1, 1)
    ).ravel()

    # Delta in original BP scale (using mean predictions)
    delta_bp = y_new_bp - y_orig_bp

    # Uncertainty on delta: propagate through inverse transform
    # Use local derivative of inverse transform
    eps      = 0.01
    mean_ref = mc_orig["mean_pred"]
    up       = transformer.inverse_transform((mean_ref + eps).reshape(-1,1)).ravel()
    dn       = transformer.inverse_transform((mean_ref - eps).reshape(-1,1)).ravel()
    deriv    = np.abs((up - dn) / (2 * eps))

    # Std of delta in transformed space
    delta_std_t = delta_samples.std(axis=0)

    # Convert to BP scale
    delta_std_bp = delta_std_t * deriv

    # 90% prediction interval for delta (5th, 95th percentile of MC samples)
    # Convert each sample's delta to BP scale
    delta_samples_bp = np.array([
        transformer.inverse_transform(
            samples_new[i].reshape(-1,1)
        ).ravel() -
        transformer.inverse_transform(
            samples_orig[i].reshape(-1,1)
        ).ravel()
        for i in range(delta_samples.shape[0])
    ])  # (30, N_test)

    delta_bp_lo_90 = np.percentile(delta_samples_bp, 5,  axis=0)
    delta_bp_hi_90 = np.percentile(delta_samples_bp, 95, axis=0)

    # Significant effect: 90% CI does not include 0
    significant_mask = (delta_bp_lo_90 > 0) | (delta_bp_hi_90 < 0)

    return {
        "delta_bp":         delta_bp,
        "delta_std_bp":     delta_std_bp,
        "delta_bp_lo_90":   delta_bp_lo_90,
        "delta_bp_hi_90":   delta_bp_hi_90,
        "y_orig_bp":        y_orig_bp,
        "y_new_bp":         y_new_bp,
        "significant_mask": significant_mask,
        "delta_samples_bp": delta_samples_bp,   # (30, N_test) for paper figures
    }


def summarise_effect(
    effect:       dict[str, np.ndarray],
    scenario_name: str,
    test_mask_sum: int,
    intervention_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute summary statistics for an intervention scenario.

    Parameters
    ----------
    effect            : output of compute_intervention_effect()
    scenario_name     : name of the scenario (for printing)
    test_mask_sum     : total number of test nodes
    intervention_mask : (N_test,) bool — which test nodes were treated
                        If None, uses all test nodes

    Returns
    -------
    dict of scalar summary statistics
    """
    delta = effect["delta_bp"]
    sig   = effect["significant_mask"]

    # Use intervention nodes if provided, else all test nodes
    if intervention_mask is not None:
        eval_mask = intervention_mask
    else:
        eval_mask = np.ones(len(delta), dtype=bool)

    delta_eval = delta[eval_mask]
    sig_eval   = sig[eval_mask]

    summary = {
        "scenario":           scenario_name,
        "n_test":             int(test_mask_sum),
        "n_treated":          int(eval_mask.sum()),
        "mean_delta_bp":      float(delta_eval.mean()),
        "median_delta_bp":    float(np.median(delta_eval)),
        "std_delta_bp":       float(delta_eval.std()),
        "mean_delta_pct":     float(
            (delta_eval / (effect["y_orig_bp"][eval_mask] + 1e-8)).mean() * 100
        ),
        "n_significant":      int(sig_eval.sum()),
        "pct_significant":    float(sig_eval.mean() * 100),
        "pct_reduced":        float((delta_eval < 0).mean() * 100),
        "pct_increased":      float((delta_eval > 0).mean() * 100),
        "max_reduction_bp":   float(delta_eval.min()),
        "max_increase_bp":    float(delta_eval.max()),
        "p5_delta_bp":        float(np.percentile(delta_eval, 5)),
        "p95_delta_bp":       float(np.percentile(delta_eval, 95)),
    }

    if verbose:
        print(f"\n  ── {scenario_name} ──")
        print(f"    Treated nodes    : {summary['n_treated']:,}"
              f"  ({100*summary['n_treated']/test_mask_sum:.1f}% of test)")
        print(f"    Mean Δ BP        : {summary['mean_delta_bp']:+.5f}"
              f"  ({summary['mean_delta_pct']:+.2f}%)")
        print(f"    Median Δ BP      : {summary['median_delta_bp']:+.5f}")
        print(f"    Std(Δ BP)        : {summary['std_delta_bp']:.5f}")
        print(f"    90% CI           : [{summary['p5_delta_bp']:+.5f},"
              f" {summary['p95_delta_bp']:+.5f}]")
        print(f"    Nodes reduced    : {summary['pct_reduced']:.1f}%")
        print(f"    Significant eff. : {summary['n_significant']:,}"
              f"  ({summary['pct_significant']:.1f}%)")
        print(f"    Max reduction    : {summary['max_reduction_bp']:+.5f} BP")

    return summary