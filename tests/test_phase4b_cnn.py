"""
Tests for Phase 4B CNN baseline.

Run:
    pytest tests/test_phase4b_cnn.py -q
"""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.models.cnn_baseline import build_cnn_model


def test_cnn_model_forward_pass():
    import torch

    model = build_cnn_model(in_channels=4, patch_size=7)
    x = torch.randn(8, 4, 7, 7)

    y = model(x)

    assert y.shape == (8,)
    assert torch.isfinite(y).all()


def test_phase4b_required_rasters_exist():
    aligned_dir = PROJECT_ROOT / "data" / "interim" / "aligned"

    required = [
        "CFL.tif",
        "FSP_Index.tif",
        "Ignition_Prob.tif",
        "Struct_Exp_Index.tif",
    ]

    missing = [name for name in required if not (aligned_dir / name).exists()]
    assert not missing, f"Missing CNN raster inputs: {missing}"


def test_phase4b_graph_exists():
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    assert graph_path.exists(), "Missing graph_data_enriched.pt"


def test_phase4b_output_dirs_creatable():
    for rel in [
        "reports/tables",
        "reports/figures",
        "reports/predictions",
        "reports/checkpoints",
    ]:
        path = PROJECT_ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists()