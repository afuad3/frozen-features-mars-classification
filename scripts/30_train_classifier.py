"""Phase G: train the linear probe and evaluate on the untouched test set.

For each representation: select (C, class_weight) on VALIDATION Macro-F1, refit-on-train, evaluate
TEST once. Saves per-model metrics (+ bootstrap CIs), per-class table, validation-selection table,
and test predictions (with probabilities) for downstream analysis.

    python scripts/30_train_classifier.py --dataset hirise --model all
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src.classifiers.probe import predict, predict_proba, select_and_fit
from src.embeddings.io import embeddings_exist, load_embeddings
from src.evaluation.bootstrap import bootstrap_ci, make_resample_indices
from src.evaluation.calibration import (confidences_from_proba, expected_calibration_error)
from src.evaluation.metrics import (compute_metrics, most_common_class_baseline, per_class_table)
from src.models.registry import PRIMARY_MODELS
from src.utils.config import ensure_dir, load_configs, path
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed

TAB = path("results", "tables")


def _split_arrays(raw, meta, split):
    m = meta["split"].values == split
    return raw[m], meta.loc[m, "class_id"].values.astype(int), meta.loc[m]


def run_one(dataset, model, cfg, log):
    set_global_seed(cfg["classifier"]["seed"])
    raw, l2, meta, attrs = load_embeddings(dataset, model)
    labels = sorted(meta["class_id"].astype(int).unique().tolist())
    classmap = {int(r.class_id): r.class_name for r in meta.itertuples()}

    Xtr, ytr, _ = _split_arrays(raw, meta, "train")
    Xva, yva, _ = _split_arrays(raw, meta, "val")
    Xte, yte, meta_te = _split_arrays(raw, meta, "test")
    log.info(f"[{dataset}/{model}] train={len(ytr)} val={len(yva)} test={len(yte)} dim={raw.shape[1]}")

    ccfg = cfg["classifier"]["primary"]
    probe = select_and_fit(Xtr, ytr, Xva, yva, grid=ccfg["grid"], solver=ccfg["solver"],
                           max_iter=ccfg["max_iter"], seed=cfg["classifier"]["seed"], logger=log)
    log.info(f"[{dataset}/{model}] selected {probe.best_params}")

    y_pred = predict(probe.pipeline, Xte)
    proba = predict_proba(probe.pipeline, Xte)
    conf = confidences_from_proba(proba)

    metrics = compute_metrics(yte, y_pred, labels=labels)
    metrics["baseline_most_common_acc"] = most_common_class_baseline(ytr, yte)
    metrics["ece"] = expected_calibration_error(yte, y_pred, conf,
                                                n_bins=cfg["classifier"]["evaluation"]["calibration"]["n_bins"])

    # Bootstrap CIs (shared resample indices per dataset test size).
    bcfg = cfg["classifier"]["evaluation"]["bootstrap"]
    idxs = make_resample_indices(len(yte), bcfg["n_resamples"], seed=cfg["classifier"]["seed"])
    cis = {m: bootstrap_ci(yte, y_pred, m, idxs, bcfg["confidence"])
           for m in ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]}

    ensure_dir(TAB)
    # per-class
    pct = per_class_table(yte, y_pred, labels, classmap)
    pct.to_csv(TAB / f"{dataset}_{model}_perclass.csv", index=False)
    # validation selection
    pd.DataFrame(probe.val_results).to_csv(TAB / f"{dataset}_{model}_val_selection.csv", index=False)
    # predictions + probabilities
    preds = meta_te[["image_id", "class_id", "class_name"]].copy()
    preds = preds.rename(columns={"class_id": "y_true"})
    preds["y_pred"] = y_pred
    preds["confidence"] = conf
    preds.reset_index(drop=True).to_parquet(TAB / f"{dataset}_{model}_test_predictions.parquet",
                                            index=False)
    np.save(TAB / f"{dataset}_{model}_test_proba.npy", proba.astype(np.float32))
    np.save(TAB / f"{dataset}_{model}_proba_classes.npy", probe.classes_)

    result = {
        "dataset": dataset, "model": model,
        "checkpoint": str(attrs.get("checkpoint")),
        "embedding_dim": int(raw.shape[1]),
        "classifier": cfg["classifier"]["primary"]["display_name"],
        "best_params": probe.best_params,
        "metrics": metrics, "ci": cis,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
    }
    with open(TAB / f"{dataset}_{model}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"[{dataset}/{model}] macro_f1={metrics['macro_f1']:.4f} "
             f"bal_acc={metrics['balanced_accuracy']:.4f} acc={metrics['accuracy']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hirise", "msl"])
    ap.add_argument("--model", default="all")
    args = ap.parse_args()
    log = get_logger(f"classify_{args.dataset}")
    cfg = load_configs()
    models = PRIMARY_MODELS if args.model == "all" else [args.model]
    for m in models:
        if not embeddings_exist(args.dataset, m):
            log.warning(f"Missing embeddings for {args.dataset}/{m}; run scripts/20 first. Skipping.")
            continue
        run_one(args.dataset, m, cfg, log)


if __name__ == "__main__":
    main()
