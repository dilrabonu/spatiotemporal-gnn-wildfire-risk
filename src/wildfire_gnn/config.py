"""Config loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a plain dict.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file, relative to project root or absolute.

    Returns
    -------
    dict[str, Any]
        Parsed config dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Return the project root directory.

    Walks upward from this file until it finds pyproject.toml.
    Falls back to the current working directory if not found.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(os.getcwd())


def resolve_paths(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Make all relative paths in config['paths'] absolute from project root.

    Parameters
    ----------
    config : dict
        Parsed config dict (must contain a 'paths' key).
    root : Path, optional
        Project root. Defaults to get_project_root().

    Returns
    -------
    dict
        Config dict with absolute paths under config['paths'].
    """
    if root is None:
        root = get_project_root()

    paths = config.get("paths", {})
    resolved: dict[str, str] = {}

    for key, value in paths.items():
        if isinstance(value, str):
            p = Path(value)
            resolved[key] = str(root / p) if not p.is_absolute() else value
        else:
            resolved[key] = value

    config["paths"] = resolved
    return config