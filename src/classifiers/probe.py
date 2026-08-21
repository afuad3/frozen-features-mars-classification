"""Linear probe: StandardScaler -> multinomial L2 logistic regression.

Protocol (identical for every representation):
  * StandardScaler is fit on TRAIN only (inside the sklearn Pipeline).
  * Hyperparameters (C, class_weight) are selected by VALIDATION Macro-F1 ONLY.
  * The final model is trained on TRAIN only with the selected hyperparameters.
  * The TEST set is evaluated exactly once and never used for selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_pipeline(C: float, class_weight: Optional[str], solver: str, max_iter: int,
                   seed: int) -> Pipeline:
    # multi_class is left at default ('auto' -> multinomial for lbfgs, >2 classes): future-proof.
    clf = LogisticRegression(penalty="l2", C=C, class_weight=class_weight, solver=solver,
                             max_iter=max_iter, random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


@dataclass
class ProbeResult:
    best_params: Dict
    pipeline: Pipeline
    val_results: List[Dict] = field(default_factory=list)
    classes_: np.ndarray = None


def select_and_fit(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   grid: Dict, solver: str = "lbfgs", max_iter: int = 5000,
                   seed: int = 42, logger=None) -> ProbeResult:
    """Grid-search (C x class_weight), select by validation Macro-F1, refit-on-train best."""
    best = None
    results: List[Dict] = []
    for C in grid["C"]:
        for cw in grid["class_weight"]:
            pipe = _make_pipeline(C, cw, solver, max_iter, seed)
            pipe.fit(X_train, y_train)
            val_pred = pipe.predict(X_val)
            macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            row = {"C": C, "class_weight": cw, "val_macro_f1": float(macro_f1)}
            results.append(row)
            if logger:
                logger.info(f"    C={C:<7} class_weight={str(cw):<9} val_macro_f1={macro_f1:.4f}")
            if best is None or macro_f1 > best["val_macro_f1"]:
                best = {**row, "pipeline": pipe}

    # 'best' pipeline is already trained on TRAIN only -> use it directly as the final model.
    return ProbeResult(
        best_params={k: best[k] for k in ("C", "class_weight", "val_macro_f1")},
        pipeline=best["pipeline"],
        val_results=results,
        classes_=best["pipeline"].named_steps["clf"].classes_,
    )


def predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict(X)


def predict_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict_proba(X)
