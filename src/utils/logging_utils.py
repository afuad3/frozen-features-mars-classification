"""Simple logging to both console and a per-run logfile under results/logs/."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import ensure_dir, path


def get_logger(name: str, to_file: bool = True) -> logging.Logger:
    """Return a configured logger. File logs go to results/logs/<name>-<timestamp>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if to_file:
        logdir = ensure_dir(path("results", "logs"))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fh = logging.FileHandler(logdir / f"{name}-{stamp}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
