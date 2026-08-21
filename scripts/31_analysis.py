"""Phase G: cross-model analysis. Consumes saved embeddings + predictions; writes analysis artifacts.

Produces (per dataset):
  * paired bootstrap difference CIs between models (macro_f1, accuracy)   -> *_paired_bootstrap.csv
  * PCA + UMAP 2-D coordinates per model (on the test split)             -> *_{model}_coords.parquet
  * nearest-neighbor retrieval metrics + a reproducible qualitative panel spec
  * calibration reliability-curve data for the best model
  * error analysis: confusion pairs, easiest/hardest classes, FM-vs-ResNet per-class deltas,
    reproducible representative-error samples
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src.embeddings.io import embeddings_exist, load_embeddings
from src.evaluation.bootstrap import make_resample_indices, paired_difference_ci
from src.evaluation.calibration import reliability_curve
from src.evaluation.metrics import confusion
from src.evaluation.retrieval import knn_retrieval
from src.visualization.dimreduce import pca_2d, umap_2d
from src.models.registry import PRIMARY_MODELS
from src.utils.config import ensure_dir, load_configs, path
from src.utils.logging_utils import get_logger
from src.utils.seeds import new_rng, set_global_seed

TAB = path("results", "tables")


def _available_models(dataset):
    return [m for m in PRIMARY_MODELS if embeddings_exist(dataset, m)
            and (TAB / f"{dataset}_{m}_metrics.json").exists()]


def _load_metrics(dataset, model):
    with open(TAB / f"{dataset}_{model}_metrics.json") as f:
        return json.load(f)


def paired_bootstrap(dataset, models, cfg, log):
    # Align predictions across models by image_id.
    preds = {m: pd.read_parquet(TAB / f"{dataset}_{m}_test_predictions.parquet") for m in models}
    base = preds[models[0]][["image_id", "y_true"]].copy()
    for m in models:
        base = base.merge(preds[m][["image_id", "y_pred"]].rename(columns={"y_pred": f"pred_{m}"}),
                          on="image_id")
    y_true = base["y_true"].values
    idxs = make_resample_indices(len(y_true), cfg["classifier"]["evaluation"]["bootstrap"]["n_resamples"],
                                 seed=cfg["classifier"]["seed"])
    rows = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i >= j:
                continue
            a, b = models[i], models[j]
            for metric in ["macro_f1", "accuracy"]:
                r = paired_difference_ci(y_true, base[f"pred_{a}"].values, base[f"pred_{b}"].values,
                                         metric, idxs)
                rows.append({"model_a": a, "model_b": b, "metric": metric, **r})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / f"{dataset}_paired_bootstrap.csv", index=False)
    log.info(f"[{dataset}] wrote paired bootstrap ({len(out)} comparisons)")
    return out


def coords_and_retrieval(dataset, models, cfg, log):
    retr_rows = []
    for m in models:
        raw, l2, meta, _ = load_embeddings(dataset, m)
        test_mask = meta["split"].values == "test"
        train_mask = meta["split"].values == "train"

        # PCA / UMAP on test split (clean).
        Xte = raw[test_mask]
        pca = pca_2d(Xte, seed=cfg["classifier"]["seed"])
        um = umap_2d(Xte, seed=cfg["classifier"]["seed"])
        coords = meta.loc[test_mask, ["image_id", "class_id", "class_name", "instrument",
                                      "source_id"]].reset_index(drop=True)
        coords["pca_x"], coords["pca_y"] = pca[:, 0], pca[:, 1]
        if um is not None:
            coords["umap_x"], coords["umap_y"] = um[:, 0], um[:, 1]
        coords.to_parquet(TAB / f"{dataset}_{m}_coords.parquet", index=False)

        # Nearest-neighbor retrieval (cosine on L2 features).
        tr_lab = meta.loc[train_mask, "class_id"].values.astype(int)
        te_lab = meta.loc[test_mask, "class_id"].values.astype(int)
        r = knn_retrieval(l2[train_mask], tr_lab, l2[test_mask], te_lab, k=5)
        retr_rows.append({"model": m, "top1_agreement": r["top1_agreement"],
                          "top5_agreement": r["topk_agreement"],
                          "retrieval_purity": r["retrieval_purity"]})
        log.info(f"[{dataset}/{m}] NN top1={r['top1_agreement']:.3f} top5={r['topk_agreement']:.3f}")
    pd.DataFrame(retr_rows).to_csv(TAB / f"{dataset}_retrieval.csv", index=False)


def best_model(dataset, models):
    return max(models, key=lambda m: _load_metrics(dataset, m)["metrics"]["macro_f1"])


def nn_panel_spec(dataset, model, cfg, log, n_queries=6, k=5):
    raw, l2, meta, _ = load_embeddings(dataset, model)
    test_mask = (meta["split"].values == "test")
    train_mask = (meta["split"].values == "train")
    meta_te = meta.loc[test_mask].reset_index(drop=True)
    meta_tr = meta.loc[train_mask].reset_index(drop=True)
    tr_lab = meta_tr["class_id"].values.astype(int)
    te_lab = meta_te["class_id"].values.astype(int)
    r = knn_retrieval(l2[train_mask], tr_lab, l2[test_mask], te_lab, k=k)

    rng = new_rng(cfg["classifier"]["seed"])
    sample = sorted(rng.choice(len(meta_te), size=min(n_queries, len(meta_te)), replace=False).tolist())
    spec = []
    for qi in sample:
        neigh = [{"path": meta_tr.iloc[ni]["filepath"], "label": meta_tr.iloc[ni]["class_name"]}
                 for ni in r["neighbors_idx"][qi]]
        spec.append({"query_path": meta_te.iloc[qi]["filepath"],
                     "query_label": meta_te.iloc[qi]["class_name"], "neighbors": neigh})
    with open(TAB / f"{dataset}_nn_panel_spec.json", "w") as f:
        json.dump({"model": model, "spec": spec}, f, indent=2)
    log.info(f"[{dataset}] NN panel spec for best model '{model}' ({len(spec)} queries)")


def error_analysis(dataset, models, cfg, log):
    best = best_model(dataset, models)
    preds = pd.read_parquet(TAB / f"{dataset}_{best}_test_predictions.parquet")
    labels = sorted(preds["y_true"].unique().tolist())
    name_by_id = dict(zip(preds["y_true"], preds["class_name"]))
    cm = confusion(preds["y_true"].values, preds["y_pred"].values, labels)

    # Most frequent confusion pairs (off-diagonal).
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": name_by_id.get(a, a), "pred": name_by_id.get(b, b),
                              "count": int(cm[i, j])})
    pairs = sorted(pairs, key=lambda d: -d["count"])[:15]
    pd.DataFrame(pairs).to_csv(TAB / f"{dataset}_confusion_pairs.csv", index=False)

    # FM-vs-ResNet per-class F1 deltas.
    if "resnet50" in models:
        rn = pd.read_csv(TAB / f"{dataset}_resnet50_perclass.csv")[["class_name", "f1"]]
        merged = rn.rename(columns={"f1": "f1_resnet50"})
        for m in models:
            if m == "resnet50":
                continue
            pc = pd.read_csv(TAB / f"{dataset}_{m}_perclass.csv")[["class_name", "f1"]]
            merged = merged.merge(pc.rename(columns={"f1": f"f1_{m}"}), on="class_name")
        merged.to_csv(TAB / f"{dataset}_perclass_f1_by_model.csv", index=False)

    # Reproducible representative errors from the best model.
    rng = new_rng(cfg["classifier"]["seed"])
    errs = preds[preds["y_true"] != preds["y_pred"]].reset_index(drop=True)
    if len(errs):
        take = sorted(rng.choice(len(errs), size=min(12, len(errs)), replace=False).tolist())
        errs.iloc[take].to_csv(TAB / f"{dataset}_representative_errors.csv", index=False)

    # Calibration reliability for the best model.
    proba = np.load(TAB / f"{dataset}_{best}_test_proba.npy")
    conf = proba.max(axis=1)
    rc = reliability_curve(preds["y_true"].values, preds["y_pred"].values, conf,
                           n_bins=cfg["classifier"]["evaluation"]["calibration"]["n_bins"])
    pd.DataFrame({"bin_center": rc["bin_center"], "accuracy": rc["accuracy"],
                  "confidence": rc["confidence"], "count": rc["count"]}).to_csv(
        TAB / f"{dataset}_{best}_reliability.csv", index=False)
    log.info(f"[{dataset}] error analysis done; best model = {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hirise", "msl"])
    args = ap.parse_args()
    log = get_logger(f"analysis_{args.dataset}")
    set_global_seed()
    cfg = load_configs()
    models = _available_models(args.dataset)
    if not models:
        log.error("No models with embeddings+metrics found; run scripts/20 and 30 first.")
        return
    log.info(f"[{args.dataset}] models: {models}")
    if len(models) >= 2:
        paired_bootstrap(args.dataset, models, cfg, log)
    coords_and_retrieval(args.dataset, models, cfg, log)
    best = best_model(args.dataset, models)
    nn_panel_spec(args.dataset, best, cfg, log)
    error_analysis(args.dataset, models, cfg, log)


if __name__ == "__main__":
    main()
