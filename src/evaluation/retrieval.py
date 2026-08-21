"""Nearest-neighbor retrieval analysis (§19).

For each query TEST embedding, retrieve nearest TRAIN embeddings in representation space (cosine on
L2-normalized features). Test labels are used ONLY after retrieval to score agreement/purity.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def knn_retrieval(train_emb_l2: np.ndarray, train_labels: np.ndarray,
                  query_emb_l2: np.ndarray, query_labels: np.ndarray,
                  k: int = 5) -> Dict:
    """Return top-1/top-5 agreement + retrieval purity for the query set.

    Cosine similarity == dot product on L2-normalized vectors.
    """
    sims = query_emb_l2 @ train_emb_l2.T                 # (Q x N)
    topk = np.argsort(-sims, axis=1)[:, :k]              # indices of k nearest train items
    neigh_labels = train_labels[topk]                    # (Q x k)

    top1 = neigh_labels[:, 0]
    top1_agree = float(np.mean(top1 == query_labels))
    top5_agree = float(np.mean(np.any(neigh_labels == query_labels[:, None], axis=1)))
    purity = float(np.mean(neigh_labels == query_labels[:, None]))  # fraction of neighbors matching
    return {
        "k": k,
        "top1_agreement": top1_agree,
        "topk_agreement": top5_agree,
        "retrieval_purity": purity,
        "neighbors_idx": topk,
        "neighbors_labels": neigh_labels,
    }
