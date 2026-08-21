"""Configuration and project-path helpers.

All paths resolve relative to the project root (the parent of ``src/``), so scripts can be run
from anywhere. YAML configs live in ``configs/``.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

import yaml


@functools.lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the project root (the directory that contains ``configs/`` and ``src/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    # Fallback: two levels up from src/utils/.
    return here.parents[2]


def path(*parts: str) -> Path:
    """Build an absolute path under the project root."""
    return project_root().joinpath(*parts)


def load_yaml(rel_path: str) -> Dict[str, Any]:
    """Load a YAML file given a path relative to the project root."""
    with open(path(rel_path), "r") as f:
        return yaml.safe_load(f)


def load_configs() -> Dict[str, Dict[str, Any]]:
    """Load all four Stage-1 config files into a single dict."""
    return {
        "datasets": load_yaml("configs/datasets.yaml"),
        "models": load_yaml("configs/models.yaml"),
        "preprocessing": load_yaml("configs/preprocessing.yaml"),
        "classifier": load_yaml("configs/classifier.yaml"),
    }


def ensure_dir(p: Path) -> Path:
    """Create a directory (and parents) if missing; return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p
