"""Bootstrap confidence intervals over the test set.

Resample indices are generated once and SHARED across models so model differences use the SAME
resampled test sets (paired bootstrap) - this supports honest difference CIs rather than comparing
tiny numerical gaps.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)

METRIC_FNS: Dict[str, Callable] = {
    "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
    "macro_f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    "weighted_f1": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
}


def make_resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples x n) array of bootstrap row indices, reproducible and shareable."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: str,
                 indices: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    """Point estimate + percentile CI for a metric using shared resample indices."""
    fn = METRIC_FNS[metric]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    point = float(fn(y_true, y_pred))
    vals = np.array([fn(y_true[idx], y_pred[idx]) for idx in indices])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(vals, [alpha, 1 - alpha])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "boot_mean": float(vals.mean()), "boot_std": float(vals.std())}


def paired_difference_ci(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                         metric: str, indices: np.ndarray,
                         confidence: float = 0.95) -> Dict[str, float]:
    """CI for (model_a - model_b) on the same resampled test sets (paired)."""
    fn = METRIC_FNS[metric]
    y_true = np.asarray(y_true)
    diffs = np.array([fn(y_true[idx], np.asarray(y_pred_a)[idx])
                      - fn(y_true[idx], np.asarray(y_pred_b)[idx]) for idx in indices])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    point = float(fn(y_true, y_pred_a) - fn(y_true, y_pred_b))
    # Two-sided bootstrap p-value that the difference is 0.
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": min(1.0, p), "significant_95": bool(lo > 0 or hi < 0)}
