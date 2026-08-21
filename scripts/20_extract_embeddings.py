"""Phase C–F (GATE 2): extract frozen embeddings for one dataset+model.

GATED: downloads a checkpoint (first run) and runs heavy MPS compute. Deterministic; no augmentation.

    python scripts/20_extract_embeddings.py --dataset hirise --model resnet50
    python scripts/20_extract_embeddings.py --dataset msl --model dinov2 --batch-size 32

Outputs: embeddings/{dataset}/{model}.h5 (raw + L2) and {model}.metadata.parquet.
"""
import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from src.data import hirise as H
from src.data import msl as M
from src.embeddings.extract import determinism_check, extract_for_index, l2_normalize
from src.embeddings.io import embeddings_exist, save_embeddings
from src.models.registry import build_extractor
from src.utils.config import load_configs
from src.utils.logging_utils import get_logger
from src.utils.seeds import set_global_seed

META_COLUMNS = ["image_id", "source_id", "landmark_id", "class_id", "class_name",
                "split", "instrument", "sol", "filepath"]


def _build_index(dataset: str, dcfg: dict) -> pd.DataFrame:
    if dataset == "hirise":
        idx = H.build_index(dcfg, originals_only=dcfg.get("originals_only", True))
        df = idx.originals.copy()
        df["instrument"] = None
        df["sol"] = None
    else:
        df = M.build_index(dcfg).copy()
        df["source_id"] = None
        df["landmark_id"] = None
    for c in META_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hirise", "msl"])
    ap.add_argument("--model", required=True, choices=["resnet50", "dinov2", "clip"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    log = get_logger(f"extract_{args.dataset}_{args.model}")
    seed = set_global_seed()
    cfg = load_configs()
    dcfg = cfg["datasets"][args.dataset]

    if embeddings_exist(args.dataset, args.model) and not args.overwrite:
        log.info(f"Embeddings already exist for {args.dataset}/{args.model}; use --overwrite. Skipping.")
        return

    index = _build_index(args.dataset, dcfg)
    log.info(f"Index rows: {len(index)} (dataset={args.dataset}, model={args.model})")

    extractor = build_extractor(args.model, cfg["models"])
    log.info(f"Extractor: {extractor.meta()}")

    det = determinism_check(extractor, index)
    log.info(f"Determinism check (byte-identical re-embed): {det}")
    if not det:
        log.warning("Extraction is not byte-identical across runs on this device; results are still "
                    "deterministic in practice but note this in the report.")

    raw = extract_for_index(extractor, index, batch_size=args.batch_size, logger=log)
    l2 = l2_normalize(raw)
    log.info(f"Extracted raw embeddings: {raw.shape}")
    assert raw.shape[1] == extractor.embedding_dim, "embedding dim mismatch"

    meta = index[META_COLUMNS].copy()
    meta["model"] = args.model
    meta["checkpoint"] = extractor.checkpoint
    meta["embedding_dim"] = extractor.embedding_dim
    meta["preprocessing_version"] = extractor.preprocessing_version
    meta["dataset"] = args.dataset
    meta["dataset_version"] = dcfg["full_name"]

    attrs = {**extractor.meta(), "dataset": args.dataset,
             "dataset_version": dcfg["full_name"], "seed": seed,
             "n_rows": raw.shape[0], "determinism_check": det}
    h5_path, meta_path = save_embeddings(args.dataset, args.model, raw, l2, meta, attrs)
    log.info(f"Saved: {h5_path}\n       {meta_path}")


if __name__ == "__main__":
    main()
