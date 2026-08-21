"""Phase B (GATE 1): download + extract a dataset from Zenodo.

GATED: archives are large. Run explicitly per dataset. Verifies the Zenodo MD5 checksum and extracts.

    python scripts/01_download_data.py --dataset hirise
    python scripts/01_download_data.py --dataset msl
    python scripts/01_download_data.py --dataset hirise --list-only   # inspect files, no download
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.download import download_file, extract_archive, list_files
from src.utils.config import ensure_dir, load_yaml, path
from src.utils.logging_utils import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hirise", "msl"])
    ap.add_argument("--list-only", action="store_true",
                    help="Only list the Zenodo files (size/checksum); do not download.")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    log = get_logger(f"download_{args.dataset}")
    cfg = load_yaml("configs/datasets.yaml")[args.dataset]
    record = cfg["zenodo_record"]

    log.info(f"Zenodo record {record} ({cfg['full_name']})")
    files = list_files(record)
    for f in files:
        size_mb = (f["size"] or 0) / 1e6
        log.info(f"  file: {f['key']}  {size_mb:.1f} MB  {f['checksum']}")

    if args.list_only:
        log.info("List-only mode; exiting without download.")
        return

    # Pick the declared archive if present, else the largest .zip.
    declared = cfg.get("archive_filename")
    chosen = next((f for f in files if f["key"] == declared), None)
    if chosen is None:
        zips = [f for f in files if str(f["key"]).lower().endswith(".zip")]
        chosen = max(zips, key=lambda f: f["size"] or 0) if zips else max(files, key=lambda f: f["size"] or 0)
        log.info(f"Declared archive '{declared}' not found; using '{chosen['key']}'.")

    raw_dir = ensure_dir(path(cfg["raw_dir"]))
    dest = raw_dir / chosen["key"]
    log.info(f"Downloading {chosen['key']} -> {dest}")
    download_file(chosen["url"], dest, chosen["checksum"], logger=log)

    if not args.no_extract:
        extract_dir = ensure_dir(path(cfg["extract_dir"]))
        extract_archive(dest, extract_dir, logger=log)
        log.info(f"Extracted to {extract_dir}")
    log.info("Done. Next: run scripts/10_audit.py --dataset " + args.dataset)


if __name__ == "__main__":
    main()
