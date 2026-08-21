"""PCA and UMAP projections for representation visualization (§18).

Explanatory only - visual separation is not quantitative evidence of classifier superiority.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def pca_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def umap_2d(X: np.ndarray, seed: int = 42, n_neighbors: int = 15,
            min_dist: float = 0.1, metric: str = "cosine") -> Optional[np.ndarray]:
    """UMAP 2-D embedding. Returns None if umap-learn is unavailable."""
    try:
        import umap
    except Exception:
        return None
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=seed)
    return reducer.fit_transform(X)
