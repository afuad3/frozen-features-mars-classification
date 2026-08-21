"""Calibration diagnostics (§20).

Recorded for analysis only - abstention is NOT applied in the primary comparison. Confidence is the
max predicted class probability.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def confidences_from_proba(proba: np.ndarray) -> np.ndarray:
    return proba.max(axis=1)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                               confidences: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE with equal-width confidence bins."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray,
                      n_bins: int = 15) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(mask.sum()))
        accs.append(float(correct[mask].mean()) if mask.sum() else np.nan)
        confs.append(float(confidences[mask].mean()) if mask.sum() else np.nan)
    return {"bin_center": np.array(centers), "accuracy": np.array(accs),
            "confidence": np.array(confs), "count": np.array(counts)}


def coverage_accuracy_curve(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray,
                            thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """For reference only (§20/§21): accuracy vs coverage at confidence thresholds.

    Kept SEPARATE from the primary 100%-coverage comparison. Accuracy is computed on the retained
    (non-abstained) subset; coverage is the retained fraction.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).astype(float)
    covs, accs, absten = [], [], []
    for t in thresholds:
        keep = confidences >= t
        cov = keep.mean()
        covs.append(float(cov))
        absten.append(float(1 - cov))
        accs.append(float(correct[keep].mean()) if keep.sum() else np.nan)
    return {"threshold": np.asarray(thresholds), "coverage": np.array(covs),
            "abstention": np.array(absten), "accuracy": np.array(accs)}
