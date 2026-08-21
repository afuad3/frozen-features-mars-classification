"""Capture hardware + software environment for reproducibility -> results/environment.md."""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from typing import Dict

from .config import ensure_dir, path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def collect_environment() -> Dict[str, str]:
    """Collect a dict of environment facts (best-effort; missing tools -> 'n/a')."""
    info: Dict[str, str] = {}
    info["timestamp"] = datetime.now().isoformat(timespec="seconds")
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["processor"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    mem = _run(["sysctl", "-n", "hw.memsize"])
    if mem.isdigit():
        info["memory_gb"] = f"{int(mem) / 1024**3:.1f}"
    info["python_version"] = platform.python_version()

    # Library versions
    for mod in ["numpy", "scipy", "torch", "torchvision", "timm", "transformers",
                "sklearn", "umap", "pandas", "h5py", "pyarrow", "matplotlib", "seaborn"]:
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "not-installed"

    # Torch device availability
    try:
        import torch

        info["torch_mps_available"] = str(torch.backends.mps.is_available())
        info["torch_mps_built"] = str(torch.backends.mps.is_built())
        info["torch_cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        info["torch_mps_available"] = "n/a"

    info["disk_free"] = _run(["bash", "-lc", "df -h . | tail -1"])
    return info


def write_environment_md(pip_freeze: str | None = None) -> str:
    """Write results/environment.md and return its path as a string."""
    info = collect_environment()
    if pip_freeze is None:
        pip_freeze = _run(["python", "-m", "pip", "freeze"])

    out = path("results", "environment.md")
    ensure_dir(out.parent)
    lines = ["# Environment", "", f"_Captured: {info['timestamp']}_", "",
             "## Hardware", "",
             f"- Platform: `{info['platform']}`",
             f"- Machine: `{info['machine']}`",
             f"- Processor: `{info['processor']}`",
             f"- Memory: `{info.get('memory_gb', 'n/a')} GB`",
             f"- Disk free: `{info['disk_free']}`",
             "",
             "## Compute device", "",
             f"- Torch MPS available: `{info.get('torch_mps_available')}`",
             f"- Torch MPS built: `{info.get('torch_mps_built')}`",
             f"- Torch CUDA available: `{info.get('torch_cuda_available')}`",
             "",
             "## Software", "",
             f"- Python: `{info['python_version']}`"]
    for mod in ["numpy", "scipy", "torch", "torchvision", "timm", "transformers",
                "sklearn", "umap", "pandas", "h5py", "pyarrow", "matplotlib", "seaborn"]:
        lines.append(f"- {mod}: `{info.get(f'{mod}_version')}`")
    lines += ["", "## pip freeze", "", "```", pip_freeze, "```", ""]
    out.write_text("\n".join(lines))
    return str(out)
