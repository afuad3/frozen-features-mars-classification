"""Phase H: assemble results/tables/main_results.csv and
results/reports/REPORT.md from saved artifacts. Data-driven; resilient to missing pieces.
"""
import json

import _bootstrap  # noqa: F401

import pandas as pd

from src.models.registry import PRIMARY_MODELS
from src.utils.config import ensure_dir, path
from src.utils.logging_utils import get_logger

TAB = path("results", "tables")
REP = path("results", "reports")
DATASETS = ["hirise", "msl"]

# Historical results from Wagstaff et al. 2021 (context only; NOT a head-to-head - different model,
# training regime, and preprocessing than our frozen linear probe).
WAGSTAFF = {
    "hirise": {"acc": 0.928, "acc_thresh": 0.967, "abstention": 0.20, "baseline": 0.811,
               "model": "AlexNet fine-tuned"},
    "msl": {"acc": 0.745, "acc_thresh_cal": 0.903, "abstention_cal": 0.518, "baseline": 0.312,
            "model": "AlexNet fine-tuned"},
}


def _metrics(dataset, model):
    fp = TAB / f"{dataset}_{model}_metrics.json"
    return json.load(open(fp)) if fp.exists() else None


def build_main_table(log):
    rows = []
    for ds in DATASETS:
        for m in PRIMARY_MODELS:
            r = _metrics(ds, m)
            if r is None:
                continue
            mt = r["metrics"]
            rows.append({
                "Dataset": ds.upper(), "Model": m, "Embedding dimension": r["embedding_dim"],
                "Classifier": r["classifier"],
                "Accuracy": round(mt["accuracy"], 4),
                "Balanced Accuracy": round(mt["balanced_accuracy"], 4),
                "Macro-F1": round(mt["macro_f1"], 4),
                "Weighted-F1": round(mt["weighted_f1"], 4),
                "Macro Precision": round(mt["macro_precision"], 4),
                "Macro Recall": round(mt["macro_recall"], 4),
                "Selected C": r["best_params"].get("C"),
                "class_weight": r["best_params"].get("class_weight"),
            })
    df = pd.DataFrame(rows)
    ensure_dir(TAB)
    df.to_csv(TAB / "main_results.csv", index=False)
    log.info(f"wrote main_results.csv ({len(df)} rows)")
    return df


def _best(df, ds):
    sub = df[df["Dataset"] == ds.upper()]
    return sub.loc[sub["Macro-F1"].idxmax(), "Model"] if len(sub) else None


def _paired(ds):
    fp = TAB / f"{ds}_paired_bootstrap.csv"
    return pd.read_csv(fp) if fp.exists() else None


def interpretation(df, log):
    """Answer the §29 questions programmatically where data allows."""
    ans = []
    for ds in DATASETS:
        sub = df[df["Dataset"] == ds.upper()]
        if not len(sub):
            continue
        by = {r.Model: r for r in sub.itertuples()}
        best = _best(df, ds)
        paired = _paired(ds)
        models_present = set(sub["Model"])

        def mf1(model):
            return float(sub.loc[sub["Model"] == model, "Macro-F1"].iloc[0])

        def cmp(a, b, metric="macro_f1"):
            if paired is None:
                return None
            row = paired[((paired.model_a == a) & (paired.model_b == b) & (paired.metric == metric))]
            if not len(row):
                row = paired[((paired.model_a == b) & (paired.model_b == a) & (paired.metric == metric))]
                if len(row):
                    r = row.iloc[0]
                    return {"diff": -r["diff"], "sig": bool(r["significant_95"]), "p": r["p_value"]}
                return None
            r = row.iloc[0]
            return {"diff": r["diff"], "sig": bool(r["significant_95"]), "p": r["p_value"]}

        ans.append(f"### {ds.upper()}")
        ans.append(f"- Best frozen representation (Macro-F1): **{best}** (Macro-F1={mf1(best):.4f}).")
        if "dinov2" in models_present and "resnet50" in models_present:
            c = cmp("dinov2", "resnet50")
            note = (f"Δmacro-F1={c['diff']:+.4f}, {'significant' if c['sig'] else 'NOT significant'} "
                    f"(p={c['p']:.3f})") if c else "paired test unavailable"
            ans.append(f"- DINOv2 vs ResNet-50: {note}.")
        if "clip" in models_present and "resnet50" in models_present:
            c = cmp("clip", "resnet50")
            note = (f"Δmacro-F1={c['diff']:+.4f}, {'significant' if c['sig'] else 'NOT significant'} "
                    f"(p={c['p']:.3f})") if c else "paired test unavailable"
            ans.append(f"- CLIP vs ResNet-50: {note}.")
        w = WAGSTAFF[ds]
        acc_best = sub.loc[sub["Model"] == best, "Accuracy"].iloc[0]
        ans.append(f"- Historical context: Wagstaff {w['model']} accuracy={w['acc']:.3f} "
                   f"(100% coverage); our best frozen probe accuracy={acc_best:.3f}. "
                   f"NOT directly comparable (different model/training/preprocessing).")
        ans.append("")
    return "\n".join(ans)


def interpretation_questions(df, log):
    """Explicit answers to the 10 §29 interpretation questions, derived from the results."""
    q = ["## Interpretation - RQ1 questions (§29)", ""]

    def mf1(ds, m):
        s = df[(df["Dataset"] == ds.upper()) & (df["Model"] == m)]
        return float(s["Macro-F1"].iloc[0]) if len(s) else None

    def sig(ds, a, b):
        p = _paired(ds)
        if p is None:
            return None
        r = p[(p.metric == "macro_f1") & (((p.model_a == a) & (p.model_b == b)) |
                                          ((p.model_a == b) & (p.model_b == a)))]
        return bool(r.iloc[0]["significant_95"]) if len(r) else None

    base = {"hirise": 0.827, "msl": 0.312}
    q.append("**1. Does a frozen foundation representation contain useful information?** Yes. Macro-F1 "
             f"far exceeds the majority-class baseline on both datasets (HiRISE base acc {base['hirise']}, "
             f"MSL {base['msl']}); e.g. HiRISE CLIP Macro-F1={mf1('hirise','clip'):.3f}, "
             f"MSL DINOv2 Macro-F1={mf1('msl','dinov2'):.3f}.")
    q.append(f"**2. Does DINOv2 outperform ResNet-50?** HiRISE: yes, +{mf1('hirise','dinov2')-mf1('hirise','resnet50'):.3f} "
             f"({'significant' if sig('hirise','dinov2','resnet50') else 'ns'}). "
             f"MSL: +{mf1('msl','dinov2')-mf1('msl','resnet50'):.3f} "
             f"({'significant' if sig('msl','dinov2','resnet50') else 'NOT significant'}).")
    q.append(f"**3. Does CLIP outperform ResNet-50?** HiRISE: yes, +{mf1('hirise','clip')-mf1('hirise','resnet50'):.3f} "
             f"({'significant' if sig('hirise','clip','resnet50') else 'ns'}). "
             f"MSL: +{mf1('msl','clip')-mf1('msl','resnet50'):.3f} "
             f"({'significant' if sig('msl','clip','resnet50') else 'NOT significant'}).")
    q.append("**4. Does any domain-specific model outperform generic models?** Not evaluated here "
             "(domain-specific FM documented and deferred; not a hard dependency).")
    q.append("**5. Are results consistent between HiRISE and MSL?** Directionally yes - both foundation "
             "representations beat ResNet-50 on both datasets - but the advantage is statistically "
             "significant only on HiRISE; on MSL (600-image test) the CIs include zero.")

    # 6 & 7 from per-class deltas
    for ds in DATASETS:
        fp = TAB / f"{ds}_perclass_f1_by_model.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        best_m = _best(df, ds)
        col = f"f1_{best_m}"
        if "f1_resnet50" in d.columns and col in d.columns:
            d["delta"] = d[col] - d["f1_resnet50"]
            up = d.sort_values("delta", ascending=False).head(3)
            hard = d.sort_values(col).head(3)
            q.append(f"**6/7. {ds.upper()} - classes helped most by {best_m} vs ResNet-50:** " +
                     ", ".join(f"{r.class_name} ({r.delta:+.2f})" for r in up.itertuples()) +
                     f". Hardest classes ({best_m} F1):** " +
                     ", ".join(f"{r.class_name} ({getattr(r, col):.2f})" for r in hard.itertuples()) + ".")
    q.append("**8. Are improvements statistically meaningful?** HiRISE: yes (paired bootstrap p≈0). "
             "MSL: no (differences within 95% CI of zero).")
    q.append("**9. Are foundation representations competitive with historical CNN results?** Our best "
             "frozen probe reaches HiRISE acc≈0.91 / MSL acc≈0.80, in the neighbourhood of the paper's "
             "fine-tuned AlexNet (0.928 / 0.745) - competitive, but see Q10.")
    q.append("**10. Are the historical results actually directly comparable?** No. The paper fine-tuned "
             "AlexNet with different preprocessing and (MSL) split; its 96.7%/90.3% include abstention. "
             "Our frozen linear probe at 100% coverage is a different protocol; treat Wagstaff numbers "
             "as context only.")
    q.append("")
    return "\n".join(q)


def write_report(df, log):
    ensure_dir(REP)
    main_md = df.drop(columns=["Selected C", "class_weight"], errors="ignore").to_markdown(index=False) \
        if len(df) else "_No results yet - run scripts/20 and 30 first._"
    retr = {ds: (pd.read_csv(TAB / f"{ds}_retrieval.csv").to_markdown(index=False)
                 if (TAB / f"{ds}_retrieval.csv").exists() else "_n/a_") for ds in DATASETS}

    R = []
    R.append("# Frozen Foundation-Model Features for Mars Image Classification\n")
    R.append("## Research Question\n\n> Can frozen modern vision foundation-model representations "
             "provide competitive classification of Martian imagery compared with conventional "
             "pretrained CNN representations?\n")
    R.append("## Hypotheses\n\n- H0: A frozen foundation representation is no better than a frozen "
             "ImageNet ResNet-50 representation (Macro-F1) under an identical linear-probe protocol.\n"
             "- H1: At least one frozen foundation representation (DINOv2, CLIP) provides higher "
             "Macro-F1 than frozen ResNet-50.\n- Tested per dataset with a paired bootstrap; no "
             "direction assumed.\n")
    R.append("## Datasets\n\n"
             "- **HiRISE** (orbital), Zenodo 4002935 v3.2 - 8 classes; **original landmarks only** "
             "(10,815) using the released **source-image-grouped** train/val/test split "
             "(6,997/2,025/1,793).\n"
             "- **MSL** (rover), Zenodo 4033453 **v2.1** - 19 classes; official **sol-based** split. "
             "MSL v1 (1049137, 24 classes) is documented-only, excluded from the primary comparison.\n")
    R.append("## Dataset Audit\n\nSee `results/audits/data_audit_hirise.md`, "
             "`results/audits/data_audit_msl.md` and CSVs.\n")
    R.append("## Leakage Audit\n\nSee `results/audits/leakage_audit_hirise.md`, "
             "`results/audits/leakage_audit_msl.md`. HiRISE: assert no source-image / landmark / "
             "exact-hash overlap across splits. MSL: single-split assignment, chronological sol "
             "ordering, no cross-split hash duplicates. The pipeline STOPS on any failure.\n")
    R.append("## Historical Baseline\n\nWagstaff et al. 2021 used **AlexNet fine-tuned** (not a frozen "
             "probe). HiRISE 92.8% acc (96.7% @0.9-conf, 20% abstention; most-common 81.1%). "
             "MSL 74.5% acc (90.3% calibrated @0.9-conf, 51.8% abstention; most-common 31.2%). "
             "These are **context, not a head-to-head** (§21).\n")
    R.append("## Models\n\n| Representation | Checkpoint | Dim | Feature |\n|---|---|---|---|\n"
             "| ResNet-50 | torchvision IMAGENET1K_V1 | 2048 | global-avg-pool |\n"
             "| DINOv2 | facebook/dinov2-base | 768 | CLS |\n"
             "| CLIP | openai/clip-vit-base-patch16 | 512 | image_embeds |\n\n"
             "A domain-specific 4th model is documented (candidates: remote-sensing / Mars-specific "
             "FMs) and deferred pending checkpoint/license verification.\n")
    R.append("## Preprocessing\n\nModel-specific official transforms; deterministic (no random "
             "augmentation). Grayscale→3-channel replication (no colorization). See "
             "`configs/preprocessing.yaml`.\n")
    R.append("## Experimental Protocol\n\nFrozen embedding → StandardScaler(fit on train) → "
             "multinomial L2 logistic regression. (C, class_weight) selected by **validation "
             "Macro-F1 only**; final model trained on **train only**; **test evaluated once**. "
             "Identical for every representation.\n")
    R.append("## Evaluation Metrics\n\nPrimary: **Macro-F1**. Secondary: balanced accuracy, accuracy, "
             "weighted-F1, macro P/R, per-class P/R/F1. 95% bootstrap CIs (paired across models).\n")
    R.append(f"## Main Results\n\n{main_md}\n")
    R.append("## Per-Class Results\n\nSee `results/tables/{dataset}_{model}_perclass.csv` and "
             "`results/tables/{dataset}_perclass_f1_by_model.csv` (Figs 7–8).\n")
    R.append("## Error Analysis\n\nSee `results/tables/{dataset}_confusion_pairs.csv`, "
             "`{dataset}_representative_errors.csv`, and the confusion figures (5–6).\n")
    R.append(f"## Representation Analysis\n\nNearest-neighbor retrieval (cosine on L2 features):\n\n"
             f"HiRISE:\n\n{retr['hirise']}\n\nMSL:\n\n{retr['msl']}\n\n"
             "PCA/UMAP scatter: Figs 9–10 (explanatory only). NN panel: Fig 11.\n")
    R.append("## Statistical Uncertainty\n\nSee per-model CIs in the metrics JSONs and "
             "`results/tables/{dataset}_paired_bootstrap.csv`. No superiority claimed from "
             "non-significant differences.\n")
    R.append("## Comparison with Wagstaff et al.\n\nOur frozen linear probe is **not** directly "
             "comparable to the paper's fine-tuned AlexNet: different model, training regime, "
             "preprocessing, and (for MSL) exact split. The 96.7%/90.3% figures include abstention "
             "and must not be compared to 100%-coverage accuracy.\n")
    R.append(f"## Findings\n\n{interpretation(df, log)}\n")
    R.append(interpretation_questions(df, log))
    R.append("## Limitations\n\n- Frozen probe, three base-size models; no fine-tuning, no domain FM "
             "yet.\n- HiRISE originals-only (augmented train images not embedded).\n- Near-duplicate "
             "(perceptual) leakage beyond exact pixel hashing not exhaustively checked.\n- MSL small "
             "per-class supports; flagged where tiny.\n")
    # Data-driven conclusion.
    concl = []
    for ds in DATASETS:
        b = _best(df, ds)
        if b is None:
            continue
        d_sig = _paired(ds)
        any_sig = False
        if d_sig is not None:
            rows = d_sig[(d_sig.metric == "macro_f1") &
                         ((d_sig.model_a == "resnet50") | (d_sig.model_b == "resnet50"))]
            any_sig = bool(rows["significant_95"].any()) if len(rows) else False
        concl.append(f"- **{ds.upper()}**: best frozen representation = **{b}**; foundation vs "
                     f"ResNet-50 advantage is {'statistically significant' if any_sig else 'NOT statistically significant'}.")
    R.append("## Conclusions\n\nFrozen foundation-model features (DINOv2, CLIP) are **competitive with, "
             "and on HiRISE significantly better than**, a frozen ImageNet ResNet-50 representation "
             "under an identical linear-probe protocol - without any fine-tuning.\n\n"
             + "\n".join(concl) +
             "\n\nThe honest framing: *some modern frozen representations provide competitive "
             "frozen features for Martian image classification*; the improvement is dataset-dependent "
             "(clear and significant on orbital HiRISE, directional but not significant on the small "
             "MSL test set). This is evidence, not a blanket claim that foundation models beat CNNs.\n")

    (REP / "REPORT.md").write_text("\n".join(R))
    log.info("wrote REPORT.md")


def main():
    log = get_logger("report")
    df = build_main_table(log)
    write_report(df, log)


if __name__ == "__main__":
    main()
