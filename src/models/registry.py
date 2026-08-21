"""Factory that builds a frozen extractor by name from configs/models.yaml."""
from __future__ import annotations

import torch

from .extractors import CLIPExtractor, DINOv2Extractor, FrozenExtractor, ResNet50Extractor

_BUILDERS = {
    "resnet50": lambda c, dev: ResNet50Extractor(weights_enum=c["weights_enum"], device=dev),
    "dinov2": lambda c, dev: DINOv2Extractor(hf_checkpoint=c["hf_checkpoint"], device=dev),
    "clip": lambda c, dev: CLIPExtractor(hf_checkpoint=c["hf_checkpoint"], device=dev),
}

PRIMARY_MODELS = ["resnet50", "dinov2", "clip"]


def build_extractor(name: str, models_cfg: dict,
                    device: torch.device | None = None) -> FrozenExtractor:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Known: {list(_BUILDERS)}")
    return _BUILDERS[name](models_cfg[name], device)
