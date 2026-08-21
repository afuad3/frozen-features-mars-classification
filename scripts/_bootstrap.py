"""Shared import bootstrap: put the project root on sys.path so `import src...` works,
and enable the MPS CPU-fallback for the few ops not yet implemented on Apple Silicon
(e.g. DINOv2's bicubic positional-encoding interpolation). Must be set before torch is used."""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
