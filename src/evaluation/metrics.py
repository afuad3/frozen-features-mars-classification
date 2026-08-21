"""Classification metrics (§15). Macro-F1 is the primary metric."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, precision_score,
                             recall_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    labels: Optional[List[int]] = None) -> Dict[str, float]:
    """Return the full headline metric set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int],
                    class_names: Dict[int, str]) -> pd.DataFrame:
    """Per-class precision/recall/F1/support table."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({
        "class_id": labels,
        "class_name": [class_names.get(i, str(i)) for i in labels],
        "precision": p, "recall": r, "f1": f1, "support": support,
    })


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int],
              normalize: Optional[str] = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def most_common_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent TRAIN class (sanity check vs paper)."""
    vals, counts = np.unique(y_train, return_counts=True)
    majority = vals[int(np.argmax(counts))]
    return float(np.mean(y_test == majority))
