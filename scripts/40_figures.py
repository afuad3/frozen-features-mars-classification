"""Phase H: assemble the required figures (§26) from saved artifacts. Skips any figure whose
inputs are missing (logs a warning) so it can run incrementally."""
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src.evaluation.metrics import confusion
from src.models.registry import PRIMARY_MODELS
from src.utils.config import ensure_dir, load_yaml, path
from src.utils.logging_utils import get_logger
from src.visualization import plots

TAB = path("results", "tables")
FIG = path("results", "figures")
AUD = path("results", "audits")


def _models_with_metrics(dataset):
    return [m for m in PRIMARY_MODELS if (TAB / f"{dataset}_{m}_metrics.json").exists()]


def _metrics(dataset, model):
    with open(TAB / f"{dataset}_{model}_metrics.json") as f:
        return json.load(f)


def _best(dataset, models):
    return max(models, key=lambda m: _metrics(dataset, m)["metrics"]["macro_f1"])


def fig_class_distribution(dataset, index_csv, title, out, log):
    fp = AUD / index_csv
    if not fp.exists():
        log.warning(f"missing {fp}; skip {out}")
        return
    df = pd.read_csv(fp)
    vc = df["class_name"].value_counts()
    plots.plot_class_distribution(vc.index.tolist(), vc.values.tolist(), title, FIG / out)
    log.info(f"wrote {out}")


def fig_model_comparison(dataset, title, out, log):
    models = _models_with_metrics(dataset)
    if not models:
        log.warning(f"no metrics for {dataset}; skip {out}")
        return
    metric_keys = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    values = {k: [_metrics(dataset, m)["metrics"][k] for m in models] for k in metric_keys}
    plots.plot_model_comparison(models, values, title, FIG / out)
    log.info(f"wrote {out}")


def fig_confusion(dataset, title, out, log, normalized=False):
    models = _models_with_metrics(dataset)
    if not models:
        log.warning(f"no metrics for {dataset}; skip {out}")
        return
    best = _best(dataset, models)
    preds = pd.read_parquet(TAB / f"{dataset}_{best}_test_predictions.parquet")
    labels = sorted(preds["y_true"].unique().tolist())
    names = [preds.loc[preds["y_true"] == l, "class_name"].iloc[0] for l in labels]
    cm = confusion(preds["y_true"].values, preds["y_pred"].values, labels,
                   normalize="true" if normalized else None)
    plots.plot_confusion(cm, names, f"{title} - {best}" + (" (norm)" if normalized else ""),
                         FIG / out, normalized=normalized)
    log.info(f"wrote {out}")


def fig_per_class_f1(dataset, title, out, log):
    fp = TAB / f"{dataset}_perclass_f1_by_model.csv"
    if not fp.exists():
        log.warning(f"missing {fp}; skip {out}")
        return
    df = pd.read_csv(fp)
    class_names = df["class_name"].tolist()
    f1_by_model = {c.replace("f1_", ""): df[c].tolist() for c in df.columns if c.startswith("f1_")}
    plots.plot_per_class_f1(class_names, f1_by_model, title, FIG / out)
    log.info(f"wrote {out}")


def fig_embedding(dataset, title, out, log):
    models = _models_with_metrics(dataset)
    if not models:
        log.warning(f"no models for {dataset}; skip {out}")
        return
    best = _best(dataset, models)
    fp = TAB / f"{dataset}_{best}_coords.parquet"
    if not fp.exists():
        log.warning(f"missing {fp}; skip {out}")
        return
    df = pd.read_parquet(fp)
    label_names = dict(zip(df["class_id"], df["class_name"]))
    xcol, ycol, tag = ("umap_x", "umap_y", "UMAP") if "umap_x" in df.columns else ("pca_x", "pca_y", "PCA")
    coords = df[[xcol, ycol]].values
    plots.plot_embedding_scatter(coords, df["class_id"].values, label_names,
                                 f"{title} - {best} ({tag}, test)", FIG / out)
    log.info(f"wrote {out} ({tag})")


def fig_nn_panel(dataset, out, log):
    fp = TAB / f"{dataset}_nn_panel_spec.json"
    if not fp.exists():
        log.warning(f"missing {fp}; skip {out}")
        return
    spec = json.load(open(fp))
    rows = [{"query_path": s["query_path"], "query_label": s["query_label"],
             "neighbors": [(n["path"], n["label"]) for n in s["neighbors"]]} for s in spec["spec"]]
    plots.plot_nn_panel(rows, f"Nearest-neighbor retrieval - {dataset} ({spec['model']})", FIG / out)
    log.info(f"wrote {out}")


def main():
    log = get_logger("figures")
    ensure_dir(FIG)
    # 1-2 class distributions
    fig_class_distribution("hirise", "hirise_originals_index.csv",
                           "HiRISE class distribution (originals)", "fig01_hirise_class_dist.png", log)
    fig_class_distribution("msl", "msl_index.csv", "MSL class distribution",
                           "fig02_msl_class_dist.png", log)
    # 3-4 model comparison
    fig_model_comparison("hirise", "HiRISE - frozen representation comparison",
                         "fig03_hirise_model_comparison.png", log)
    fig_model_comparison("msl", "MSL - frozen representation comparison",
                         "fig04_msl_model_comparison.png", log)
    # 5-6 confusion (best) - raw + normalized
    fig_confusion("hirise", "HiRISE confusion", "fig05_hirise_confusion.png", log, normalized=False)
    fig_confusion("hirise", "HiRISE confusion", "fig05_hirise_confusion_norm.png", log, normalized=True)
    fig_confusion("msl", "MSL confusion", "fig06_msl_confusion.png", log, normalized=False)
    fig_confusion("msl", "MSL confusion", "fig06_msl_confusion_norm.png", log, normalized=True)
    # 7-8 per-class F1
    fig_per_class_f1("hirise", "HiRISE per-class F1", "fig07_hirise_perclass_f1.png", log)
    fig_per_class_f1("msl", "MSL per-class F1", "fig08_msl_perclass_f1.png", log)
    # 9-10 PCA/UMAP
    fig_embedding("hirise", "HiRISE representation", "fig09_hirise_embedding.png", log)
    fig_embedding("msl", "MSL representation", "fig10_msl_embedding.png", log)
    # 11 NN retrieval (HiRISE as representative)
    fig_nn_panel("hirise", "fig11_hirise_nn_retrieval.png", log)


if __name__ == "__main__":
    main()
