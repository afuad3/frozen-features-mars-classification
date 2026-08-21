"""Deterministic seeding across Python, NumPy, and PyTorch (CPU + MPS).

Embedding extraction is deterministic regardless (eval mode, no dropout, no random augmentation),
but we still fix all seeds so any incidental randomness (e.g. sklearn solvers, bootstrap resampling,
UMAP init) is reproducible.
"""
from __future__ import annotations

import os
import random
import warnings

import numpy as np

DEFAULT_SEED = 42


def suppress_spurious_matmul_warnings() -> None:
    """Silence numpy 2.0 + Apple Accelerate spurious matmul FPE warnings.

    On macOS arm64, numpy 2.0.2 (the last release supporting Python 3.9) is built against the
    Accelerate BLAS, which spuriously raises the floating-point 'divide by zero' / 'overflow' /
    'invalid value' flags during matmul even though the numerical results are CORRECT. We verified
    correctness against torch's independent BLAS (exact match) and a float64 reference (fp32
    precision). The issue is fixed in numpy >= 2.1 (which requires Python >= 3.10). These filters
    hide ONLY those three exact spurious messages; all other warnings still surface.
    """
    for msg in ("divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul"):
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)
    # DINOv2's bicubic positional-encoding interpolation is not implemented on MPS; it falls back to
    # CPU (correct, negligible cost). Silence the once-per-batch UserWarning.
    warnings.filterwarnings("ignore", message=r".*upsample_bicubic2d.*MPS.*", category=UserWarning)


def set_global_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy, and (if available) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    suppress_spurious_matmul_warnings()
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():  # not on this machine, but harmless
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def new_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh NumPy Generator (preferred for bootstrap/resampling)."""
    return np.random.default_rng(seed)
