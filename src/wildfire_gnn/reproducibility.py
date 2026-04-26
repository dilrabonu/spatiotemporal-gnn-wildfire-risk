"""Reproducibility helpers — seed everything before any experiment."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Call this at the top of every notebook and training script
    before any data loading or model initialization.

    Parameters
    ----------
    seed : int
        Random seed. Default matches config.training.seed = 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic ops (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed for dict/set ordering
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    prefer_cuda : bool
        If True (default), use CUDA when available.

    Returns
    -------
    torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif prefer_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")
    return device


def describe_device(device: torch.device) -> str:
    """Return a human-readable device description for logging."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"CUDA — {name} ({mem:.1f} GB)"
    elif device.type == "mps":
        return "MPS — Apple Silicon GPU"
    else:
        return "CPU"