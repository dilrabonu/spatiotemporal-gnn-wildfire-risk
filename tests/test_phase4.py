"""
Phase 4 tests.

Run:
    pytest tests/test_phase4.py -q
"""

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.evaluation.metrics import (
    r2_score,
    mae_score,
    spearman_rho,
    expected_calibration_error,
    compute_all_metrics,
)


def test_phase4_graph_exists():
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    assert graph_path.exists(), "Missing graph_data_enriched.pt"


def test_phase4_graph_has_required_masks():
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    for attr in ["train_mask", "val_mask", "test_mask", "x", "y", "y_raw", "pos"]:
        assert hasattr(graph, attr), f"Missing graph attribute: {attr}"

    assert int(graph.train_mask.sum()) > 0
    assert int(graph.val_mask.sum()) > 0
    assert int(graph.test_mask.sum()) > 0

    assert int((graph.train_mask & graph.val_mask).sum()) == 0
    assert int((graph.train_mask & graph.test_mask).sum()) == 0
    assert int((graph.val_mask & graph.test_mask).sum()) == 0


def test_phase4_feature_count_is_61():
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    assert graph.num_node_features == 61


def test_phase4_metric_functions_basic():
    y_true = np.array([0.01, 0.02, 0.03, 0.04])
    y_pred = np.array([0.01, 0.021, 0.029, 0.039])

    assert r2_score(y_true, y_pred) > 0.95
    assert mae_score(y_true, y_pred) < 0.002
    assert spearman_rho(y_true, y_pred) > 0.9
    assert expected_calibration_error(y_true, y_pred) >= 0.0

    metrics = compute_all_metrics(y_true, y_pred, model_name="test", verbose=False)

    for key in ["model", "r2", "mae", "spearman", "brier", "ece", "n_test", "binned"]:
        assert key in metrics


def test_phase4_output_dirs_exist_or_creatable():
    for rel in ["reports/tables", "reports/figures", "reports/predictions"]:
        path = PROJECT_ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists()