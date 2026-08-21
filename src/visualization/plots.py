"""High-resolution figures (§16, §18, §26). Matplotlib; seaborn only for styling if present."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 200


def _style():
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
    except Exception:
        plt.style.use("default")


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(names: List[str], counts: List[int], title: str, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    order = np.argsort(counts)[::-1]
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]
    ax.bar(range(len(names)), counts, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=7)
    save(fig, out)


def plot_model_comparison(models: List[str], metric_values: Dict[str, List[float]],
                          title: str, out: Path):
    """Grouped bar chart of several metrics across models."""
    _style()
    metrics = list(metric_values.keys())
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.8 / max(1, len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, m in enumerate(metrics):
        ax.bar(x + j * width, metric_values[m], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=len(metrics))
    save(fig, out)


def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, out: Path,
                   normalized: bool = False):
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=(1.0 if normalized else None))
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = ".2f" if normalized else "d"
    thresh = (cm.max() if cm.size else 1) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalized or val > 0:
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=6,
                        color="white" if val > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, out)


def plot_per_class_f1(class_names: List[str], f1_by_model: Dict[str, List[float]],
                      title: str, out: Path):
    _style()
    x = np.arange(len(class_names))
    models = list(f1_by_model.keys())
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(9, len(class_names) * 0.9), 5.5))
    for j, m in enumerate(models):
        ax.bar(x + j * width, f1_by_model[m], width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend()
    save(fig, out)


def plot_embedding_scatter(coords: np.ndarray, labels: np.ndarray, label_names: Dict,
                           title: str, out: Path, max_points: int = 4000, seed: int = 42):
    _style()
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords, labels = coords[idx], labels[idx]
    fig, ax = plt.subplots(figsize=(8, 7))
    uniq = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    for k, lab in enumerate(uniq):
        m = labels == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6,
                   color=cmap(k % cmap.N), label=str(label_names.get(lab, lab)))
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    save(fig, out)


def plot_reliability(bin_center: np.ndarray, accuracy: np.ndarray, title: str, out: Path):
    _style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(bin_center, accuracy, "-o", color="#C44E52", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    save(fig, out)


def plot_nn_panel(rows: List[Dict], title: str, out: Path):
    """rows: [{query_path, query_label, neighbors: [(path,label),...]}]. Renders a grid."""
    _style()
    from PIL import Image
    n_rows = len(rows)
    n_cols = 1 + max(len(r["neighbors"]) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.7))
    if n_rows == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        with Image.open(r["query_path"]) as im:
            axes[i, 0].imshow(np.asarray(im.convert("L")), cmap="gray")
        axes[i, 0].set_title(f"query\n{r['query_label']}", fontsize=6)
        axes[i, 0].axis("off")
        for j, (pth, lab) in enumerate(r["neighbors"], start=1):
            with Image.open(pth) as im:
                axes[i, j].imshow(np.asarray(im.convert("L")), cmap="gray")
            match = "✓" if lab == r["query_label"] else "✗"
            axes[i, j].set_title(f"{match} {lab}", fontsize=6)
            axes[i, j].axis("off")
        for j in range(len(r["neighbors"]) + 1, n_cols):
            axes[i, j].axis("off")
    fig.suptitle(title)
    save(fig, out)
