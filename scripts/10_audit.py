"""Phase B: dataset audit (§5) -> results/audits/data_audit_{dataset}.md + CSVs.

Reports totals, class distribution, image dims/channels, per-split distributions, and
(HiRISE) source-image distribution / (MSL) instrument + sol distribution. Reads everything from the
actual files; prints sample filenames so parsing conventions can be verified.
"""
import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from PIL import Image

from src.data import hirise as H
from src.data import msl as M
from src.utils.config import ensure_dir, load_yaml, path
from src.utils.logging_utils import get_logger

AUDIT_DIR = path("results", "audits")


def _sample_image_props(paths, n=50):
    sizes, modes = Counter(), Counter()
    for p in paths[:n]:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception:
            modes["<open-failed>"] += 1
    return sizes, modes


def _split_class_table(df, class_col="class_name"):
    return pd.crosstab(df[class_col], df["split"]).reset_index()


def audit_hirise(cfg, log):
    idx = H.build_index(cfg, originals_only=cfg.get("originals_only", True))
    full, orig = idx.table, idx.originals
    ensure_dir(AUDIT_DIR)

    n_total_rows = len(full)
    n_originals = int(full["is_original"].sum())
    n_augmented = n_total_rows - n_originals
    landmark_counts = orig["split"].value_counts().to_dict()

    sizes, modes = _sample_image_props(orig["filepath"].tolist())
    exp = cfg["expected_landmark_counts"]
    counts_ok = all(landmark_counts.get(s, 0) == exp[s] for s in ["train", "val", "test"])

    # Save machine-readable CSVs
    orig.to_csv(AUDIT_DIR / "hirise_originals_index.csv", index=False)
    _split_class_table(orig).to_csv(AUDIT_DIR / "hirise_class_by_split.csv", index=False)
    src_by_split = (orig.groupby("split")["source_id"].nunique().rename("n_source_images")
                    .reset_index())
    src_by_split.to_csv(AUDIT_DIR / "hirise_sourceimages_by_split.csv", index=False)

    class_counts = orig["class_name"].value_counts()
    lines = [f"# HiRISE data audit - {cfg['full_name']}", "",
             f"- Zenodo record: {cfg['zenodo_record']}",
             f"- Total rows in split file: **{n_total_rows}** (expected released images "
             f"{cfg['expected_image_counts']['total_released']})",
             f"- Original (unaugmented) landmarks: **{n_originals}**",
             f"- Augmented rows: **{n_augmented}**",
             f"- Classes: **{cfg['num_classes']}**",
             f"- Unmatched source-id filenames: **{len(idx.unmatched_source)}** "
             f"(examples: {idx.unmatched_source[:3]})",
             "",
             "## Originals-only landmark split counts",
             f"- train: {landmark_counts.get('train', 0)} (expected {exp['train']})",
             f"- val:   {landmark_counts.get('val', 0)} (expected {exp['val']})",
             f"- test:  {landmark_counts.get('test', 0)} (expected {exp['test']})",
             f"- **Counts match paper/dataset:** {'YES' if counts_ok else 'NO - INVESTIGATE'}",
             "",
             "## Image properties (sampled)",
             f"- Sizes: {dict(sizes)}",
             f"- Modes/channels: {dict(modes)}",
             "",
             "## Class distribution (originals)"]
    for name, c in class_counts.items():
        lines.append(f"- {name}: {c} ({100*c/len(orig):.1f}%)")
    lines += ["", "## Source-image count per split", src_by_split.to_markdown(index=False),
              "", "## Sample filenames",
              "```", "\n".join(orig["filename"].head(8).tolist()), "```"]
    (AUDIT_DIR / "data_audit_hirise.md").write_text("\n".join(str(x) for x in lines))
    log.info(f"counts_ok={counts_ok}; originals={n_originals}; wrote data_audit_hirise.md")
    if not counts_ok:
        log.warning("HiRISE landmark counts do NOT match expected 6997/2025/1793. "
                    "Inspect sample filenames and adjust AUG_MARKERS in src/data/hirise.py.")


def audit_msl(cfg, log):
    full = M.build_index(cfg, originals_only=False, dedup_train=False)  # all rows incl augmented
    orig = M.build_index(cfg, originals_only=True, dedup_train=False)   # paper's 2,900 (pre-dedup)
    df = M.build_index(cfg, originals_only=True, dedup_train=True)      # the set we embed (deduped)
    ensure_dir(AUDIT_DIR)

    df.to_csv(AUDIT_DIR / "msl_index.csv", index=False)  # originals-only index (embedded set)
    _split_class_table(df).to_csv(AUDIT_DIR / "msl_class_by_split.csv", index=False)
    instr_by_split = pd.crosstab(df["instrument"], df["split"]).reset_index()
    instr_by_split.to_csv(AUDIT_DIR / "msl_instrument_by_split.csv", index=False)

    sizes, modes = _sample_image_props(df["filepath"].tolist())
    full_counts = full["split"].value_counts().to_dict()
    orig_counts = orig["split"].value_counts().to_dict()          # pre-dedup (paper's 2,900)
    embed_counts = df["split"].value_counts().to_dict()           # post-dedup (embedded)
    exp = cfg.get("expected_originals_counts", {})
    counts_ok = all(orig_counts.get(s, 0) == exp.get(s) for s in ["train", "val", "test"]) if exp else None
    n_bad_sol = int(df["sol"].isna().sum())
    sol_ranges = (df.dropna(subset=["sol"]).groupby("split")["sol"]
                  .agg(["min", "max", "count"]).reset_index())

    class_counts = df["class_name"].value_counts()
    lines = [f"# MSL data audit - {cfg['full_name']}", "",
             f"- Zenodo record: {cfg['zenodo_record']}",
             f"- Total released images (incl. augmented train): **{len(full)}** {full_counts}",
             f"- Originals (pre-dedup, paper's benchmark): **{len(orig)}** {orig_counts}",
             f"  - expected {exp} -> **{'MATCH' if counts_ok else ('MISMATCH - INVESTIGATE' if counts_ok is not None else 'n/a')}**",
             f"- **Embedded set (post train-dedup): {len(df)}** {embed_counts} "
             f"(dropped {len(orig)-len(df)} train duplicate(s) of val/test per leakage policy)",
             f"- Classes present: **{df['class_id'].nunique()}** (declared {cfg['num_classes']})",
             f"- Filenames without parseable sol/instrument: **{n_bad_sol}**",
             "",
             "## Image properties (sampled, originals)",
             f"- Sizes: {dict(sizes)}",
             f"- Modes/channels: {dict(modes)}",
             "",
             "## Sol range per split (chronological check, originals)",
             sol_ranges.to_markdown(index=False),
             "",
             "## Instrument distribution per split (originals)",
             instr_by_split.to_markdown(index=False),
             "",
             "## Class distribution (originals)"]
    for name, c in class_counts.items():
        lines.append(f"- {name}: {c} ({100*c/len(df):.1f}%)")
    lines += ["", "## Sample filenames",
              "```", "\n".join(df["filepath"].map(lambda p: Path(p).name).head(8).tolist()), "```"]
    (AUDIT_DIR / "data_audit_msl.md").write_text("\n".join(str(x) for x in lines))
    log.info(f"MSL originals={len(df)} {orig_counts} counts_ok={counts_ok}; wrote data_audit_msl.md")
    if counts_ok is False:
        log.warning("MSL originals counts do NOT match expected 2000/300/600 - investigate.")
    if n_bad_sol:
        log.warning(f"{n_bad_sol} MSL filenames failed sol/instrument parsing - inspect samples.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hirise", "msl"])
    args = ap.parse_args()
    log = get_logger(f"audit_{args.dataset}")
    cfg = load_yaml("configs/datasets.yaml")[args.dataset]
    (audit_hirise if args.dataset == "hirise" else audit_msl)(cfg, log)


if __name__ == "__main__":
    main()
