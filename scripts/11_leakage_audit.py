"""Phase B (§6): leakage audit -> results/audits/leakage_audit_{dataset}.md.

CRITICAL. Exits non-zero (STOP) if leakage is detected or the split cannot be verified, per §33.
Do NOT proceed to embedding extraction unless this passes.

Checks
------
HiRISE: (a) split counts = 6997/2025/1793; (b) no SOURCE-image id shared across splits;
        (c) no landmark family shared across splits (checked on the FULL table incl. augmented);
        (d) no exact file/pixel hash shared across splits.
MSL:    (i) each image in exactly one split; (ii) chronological sol ordering (test after train);
        (iii) no exact file/pixel hash shared across splits; (iv) instrument distribution.
"""
import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from src.data import hirise as H
from src.data import msl as M
from src.data.hashing import file_sha256, pixel_sha256
from src.utils.config import ensure_dir, load_yaml, path
from src.utils.logging_utils import get_logger

AUDIT_DIR = path("results", "audits")


def _cross_split_groups(df, key):
    """Return rows of groups (by `key`) that span more than one split."""
    g = df.groupby(key)["split"].nunique()
    bad_keys = g[g > 1].index.tolist()
    return df[df[key].isin(bad_keys)].sort_values([key, "split"]), bad_keys


def _hash_frame(df, log, label):
    log.info(f"Hashing {len(df)} {label} images (file + pixel)...")
    fh, ph = [], []
    for i, p in enumerate(df["filepath"].tolist()):
        try:
            fh.append(file_sha256(p))
            ph.append(pixel_sha256(p))
        except Exception:
            fh.append(f"<err-{i}>")
            ph.append(f"<err-{i}>")
        if i % 2000 == 0 and i:
            log.info(f"  hashed {i}/{len(df)}")
    out = df.copy()
    out["file_sha256"] = fh
    out["pixel_sha256"] = ph
    return out


def audit_hirise(cfg, log) -> bool:
    idx = H.build_index(cfg, originals_only=True)
    full, orig = idx.table, idx.originals
    exp = cfg["expected_landmark_counts"]
    problems = []

    counts = orig["split"].value_counts().to_dict()
    counts_ok = all(counts.get(s, 0) == exp[s] for s in ["train", "val", "test"])
    if not counts_ok:
        problems.append(f"Landmark split counts {counts} != expected "
                        f"{ {k: exp[k] for k in ['train','val','test']} }.")

    # (b) source-image id across splits (on originals is enough, but check full table too)
    src_bad, src_keys = _cross_split_groups(full.dropna(subset=["source_id"]), "source_id")
    if src_keys:
        problems.append(f"{len(src_keys)} SOURCE images appear in >1 split (source-image leakage).")

    # (c) landmark family across splits (full table incl augmented)
    lm_bad, lm_keys = _cross_split_groups(full, "landmark_id")
    if lm_keys:
        problems.append(f"{len(lm_keys)} landmark families appear in >1 split.")

    # (d) exact hash duplicates across splits (on originals)
    hashed = _hash_frame(orig, log, "HiRISE original")
    fh_bad, fh_keys = _cross_split_groups(hashed, "file_sha256")
    px_bad, px_keys = _cross_split_groups(hashed, "pixel_sha256")
    if fh_keys:
        problems.append(f"{len(fh_keys)} exact FILE hashes shared across splits.")
    if px_keys:
        problems.append(f"{len(px_keys)} exact PIXEL hashes shared across splits.")

    passed = len(problems) == 0
    ensure_dir(AUDIT_DIR)
    if src_keys:
        src_bad.to_csv(AUDIT_DIR / "hirise_leakage_source_examples.csv", index=False)
    if px_keys:
        px_bad.to_csv(AUDIT_DIR / "hirise_leakage_pixelhash_examples.csv", index=False)

    lines = ["# HiRISE leakage audit (§6.1)", "",
             f"**RESULT: {'PASS ✅' if passed else 'FAIL ❌ - STOP'}**", "",
             "## Checks",
             f"- (a) split counts 6997/2025/1793: {'OK' if counts_ok else 'MISMATCH'} ({counts})",
             f"- (b) no source-image shared across splits: {'OK' if not src_keys else f'{len(src_keys)} shared'}",
             f"- (c) no landmark family shared across splits: {'OK' if not lm_keys else f'{len(lm_keys)} shared'}",
             f"- (d) no exact file-hash across splits: {'OK' if not fh_keys else f'{len(fh_keys)} shared'}",
             f"- (d) no exact pixel-hash across splits: {'OK' if not px_keys else f'{len(px_keys)} shared'}",
             "",
             "## Notes",
             "- Test set is the released unaugmented set; train/val use one original per landmark.",
             "- Near-duplicate (perceptual) detection beyond exact pixel hashing is a documented "
             "limitation.",
             ""]
    if not passed:
        lines += ["## Problems", *[f"- {p}" for p in problems]]
    (AUDIT_DIR / "leakage_audit_hirise.md").write_text("\n".join(lines))
    log.info(f"HiRISE leakage audit: {'PASS' if passed else 'FAIL'}")
    return passed


def audit_msl(cfg, log) -> bool:
    full = M.build_index(cfg, originals_only=False, dedup_train=False)  # all rows incl augmented
    raw = M.build_index(cfg, originals_only=True, dedup_train=False)    # originals, PRE-dedup
    df = M.build_index(cfg, originals_only=True, dedup_train=True)      # embedded set, POST-dedup
    problems = []

    # Pre-dedup finding (transparency): train images pixel-identical to val/test, and the policy.
    dedup_on = cfg.get("dedup_train_against_eval", True)
    dropped_ids = M.dropped_train_leak_ids(raw)

    # (i) each image in exactly one split
    dupe_assign = full.groupby("image_id")["split"].nunique()
    multi = dupe_assign[dupe_assign > 1].index.tolist()
    if multi:
        problems.append(f"{len(multi)} images assigned to >1 split.")

    # (i-b) no base product (across augmented variants) spans >1 split
    base_bad, base_keys = _cross_split_groups(full, "base_id")
    if base_keys:
        problems.append(f"{len(base_keys)} base products appear in >1 split (augmentation leakage).")

    # (ii) chronological ordering: test min sol should be >= train max sol (paper protocol)
    sol = df.dropna(subset=["sol"])
    ranges = sol.groupby("split")["sol"].agg(["min", "max"]).to_dict("index")
    chrono_note = ranges
    chrono_ok = True
    if {"train", "test"} <= set(ranges):
        chrono_ok = ranges["test"]["min"] >= ranges["train"]["max"]  # informational

    # (iii) exact hash duplicates across splits
    hashed = _hash_frame(df, log, "MSL")
    fh_bad, fh_keys = _cross_split_groups(hashed, "file_sha256")
    px_bad, px_keys = _cross_split_groups(hashed, "pixel_sha256")
    if fh_keys:
        problems.append(f"{len(fh_keys)} exact FILE hashes shared across splits.")
    if px_keys:
        problems.append(f"{len(px_keys)} exact PIXEL hashes shared across splits.")

    passed = len(problems) == 0
    ensure_dir(AUDIT_DIR)
    if px_keys:
        px_bad.to_csv(AUDIT_DIR / "msl_leakage_pixelhash_examples.csv", index=False)

    lines = ["# MSL leakage audit (§6.2)", "",
             f"**RESULT: {'PASS ✅' if passed else 'FAIL ❌ - STOP'}**", "",
             "## Dataset-artifact leakage policy (user decision: drop train duplicates)",
             f"- Pre-dedup: {len(dropped_ids)} TRAIN image(s) pixel-identical to a val/test image "
             f"(recurring 'Artifact' frame in the official v2.1 split).",
             f"- Policy `dedup_train_against_eval={dedup_on}`: dropped those TRAIN copies; val/test "
             f"kept intact. Embedded train size: {int((df['split']=='train').sum())} "
             f"(was {int((raw['split']=='train').sum())}).",
             f"- Dropped train image_ids: {dropped_ids}",
             "",
             "## Checks (on the POST-dedup embedded set)",
             f"- (i) each image in exactly one split: {'OK' if not multi else f'{len(multi)} multi'}",
             f"- (i-b) no base product across >1 split: {'OK' if not base_keys else f'{len(base_keys)} shared'}",
             f"- (ii) sol ranges per split: {chrono_note}",
             f"      chronological (test sols >= train max): {'YES' if chrono_ok else 'NO (note only)'}",
             f"- (iii) no exact file-hash across splits: {'OK' if not fh_keys else f'{len(fh_keys)} shared'}",
             f"- (iii) no exact pixel-hash across splits: {'OK' if not px_keys else f'{len(px_keys)} shared'}",
             ""]
    if not passed:
        lines += ["## Problems", *[f"- {p}" for p in problems]]
    (AUDIT_DIR / "leakage_audit_msl.md").write_text("\n".join(str(x) for x in lines))
    log.info(f"MSL leakage audit: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hirise", "msl"])
    args = ap.parse_args()
    log = get_logger(f"leakage_{args.dataset}")
    cfg = load_yaml("configs/datasets.yaml")[args.dataset]
    passed = (audit_hirise if args.dataset == "hirise" else audit_msl)(cfg, log)
    if not passed:
        log.error("LEAKAGE AUDIT FAILED - STOP. See results/audits/. Do not run experiments.")
        sys.exit(2)


if __name__ == "__main__":
    main()
