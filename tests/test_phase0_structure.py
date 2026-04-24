"""
Phase 0 structural tests.
Run: pytest tests/test_phase0_structure.py -v

These tests assert that the project is correctly set up before
any modeling work begins. Every test must pass before Phase 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ── Project root detection ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Directory structure
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/interim/aligned",
    "data/processed",
    "data/features",
    "data/external",
    "notebooks",
    "reports/figures",
    "reports/tables",
    "scripts",
    "src/wildfire_gnn",
    "src/wildfire_gnn/data",
    "src/wildfire_gnn/features",
    "src/wildfire_gnn/models",
    "src/wildfire_gnn/evaluation",
    "src/wildfire_gnn/utils",
    "tests",
]


@pytest.mark.parametrize("directory", REQUIRED_DIRS)
def test_directory_exists(directory: str) -> None:
    """Every required directory must exist."""
    assert (ROOT / directory).is_dir(), (
        f"Missing directory: {directory}\n"
        f"Run bash setup_env.sh to create the full structure."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Required files
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = [
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "configs/gnn_config.yaml",
    "src/wildfire_gnn/__init__.py",
    "src/wildfire_gnn/utils/__init__.py",
    "src/wildfire_gnn/utils/config.py",
    "src/wildfire_gnn/utils/reproducibility.py",
    "src/wildfire_gnn/utils/logging.py",
    "notebooks/00_environment_validation.ipynb",
]


@pytest.mark.parametrize("filepath", REQUIRED_FILES)
def test_file_exists(filepath: str) -> None:
    """Every required file must exist."""
    assert (ROOT / filepath).is_file(), f"Missing file: {filepath}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Config validation
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def config() -> dict:
    config_path = ROOT / "configs" / "gnn_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


REQUIRED_CONFIG_KEYS = [
    "paths",
    "raster",
    "graph",
    "split",
    "target",
    "model",
    "uncertainty",
    "training",
    "evaluation",
    "intervention",
]


def test_config_top_level_keys(config: dict) -> None:
    """Config must contain all required top-level sections."""
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    assert not missing, f"Config missing keys: {missing}"


def test_config_graph_node_count(config: dict) -> None:
    """Graph node count must be set to 300,000."""
    assert config["graph"]["n_nodes"] == 300000, (
        "graph.n_nodes must be 300000 — changing this risks silent node dropping"
    )


def test_config_graph_features(config: dict) -> None:
    """Graph feature dimension must be set to 58."""
    assert config["graph"]["node_features"] == 58


def test_config_split_method(config: dict) -> None:
    """Spatial split must use geographic block — never random."""
    assert config["split"]["method"] == "geographic_block", (
        "split.method must be 'geographic_block'. "
        "Random splits cause geographic leakage on spatial data."
    )


def test_config_split_no_overlap(config: dict) -> None:
    """Train, val, and test row ranges must not overlap."""
    train_end = config["split"]["train_rows"][1]
    val_start = config["split"]["val_rows"][0]
    val_end = config["split"]["val_rows"][1]
    test_start = config["split"]["test_rows"][0]

    assert train_end < val_start, "Train and val row ranges overlap!"
    assert val_end < test_start, "Val and test row ranges overlap!"


def test_config_seed(config: dict) -> None:
    """A reproducibility seed must be set."""
    assert "seed" in config["training"]
    assert isinstance(config["training"]["seed"], int)


def test_config_loss_function(config: dict) -> None:
    """Loss function must support uncertainty (not plain MSE)."""
    allowed = {"gaussian_nll", "mse", "bce"}
    loss = config["uncertainty"]["loss_function"]
    assert loss in allowed, f"Unknown loss: {loss}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Package imports
# ══════════════════════════════════════════════════════════════════════════════

def test_wildfire_gnn_importable() -> None:
    """The wildfire_gnn package must be importable."""
    import wildfire_gnn  # noqa: F401


def test_utils_importable() -> None:
    """All utils must be importable."""
    from wildfire_gnn.utils import (  # noqa: F401
        load_yaml_config,
        resolve_paths,
        get_project_root,
        set_seed,
        get_device,
        describe_device,
        get_logger,
        section,
        success,
        warn,
        error,
    )


def test_config_loader_works() -> None:
    """Config loader must load gnn_config.yaml without errors."""
    from wildfire_gnn.utils import load_yaml_config

    config = load_yaml_config(ROOT / "configs" / "gnn_config.yaml")
    assert isinstance(config, dict)
    assert len(config) > 0


def test_seed_setter_runs() -> None:
    """set_seed must run without errors."""
    from wildfire_gnn.utils import set_seed

    set_seed(42)  # should not raise


def test_get_device_returns_device() -> None:
    """get_device must return a valid torch.device."""
    import torch
    from wildfire_gnn.utils import get_device

    device = get_device()
    assert isinstance(device, torch.device)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Critical dependency versions
# ══════════════════════════════════════════════════════════════════════════════

def test_torch_version() -> None:
    """PyTorch version must be 2.x (not 1.x)."""
    import torch

    major = int(torch.__version__.split(".")[0])
    assert major >= 2, f"PyTorch 2.x required, got {torch.__version__}"


def test_pyg_importable() -> None:
    """PyTorch Geometric must be importable."""
    import torch_geometric  # noqa: F401
    from torch_geometric.data import Data  # noqa: F401


def test_rasterio_importable() -> None:
    """rasterio must be importable (required for all raster operations)."""
    import rasterio  # noqa: F401


def test_geopandas_importable() -> None:
    """geopandas must be importable."""
    import geopandas  # noqa: F401


def test_gitignore_excludes_data() -> None:
    """.gitignore must exclude the data/ directory (protect raw data)."""
    gitignore = ROOT / ".gitignore"
    content = gitignore.read_text()
    assert "data/raw/" in content, ".gitignore must exclude data/raw/"
    assert "*.img" in content, ".gitignore must exclude *.img files"
    assert "*.pt" in content, ".gitignore must exclude model .pt files"