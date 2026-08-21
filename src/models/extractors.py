"""Frozen feature extractors with a uniform interface (fairness rule §10).

Every extractor exposes ``embed_pil_batch(list[PIL.Image]) -> np.ndarray`` returning RAW features
(N x D, float32). Preprocessing is model-specific and deterministic (no random augmentation), applied
internally with each model's OFFICIAL transform. Grayscale is handled by converting every image to
RGB via channel replication (``PIL.Image.convert('RGB')``) - no artificial colorization.

All models are frozen: eval mode, ``torch.no_grad()``, no gradients, no fine-tuning.
"""
from __future__ import annotations

import os
from typing import List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # DINOv2 bicubic interp not on MPS yet

import numpy as np
import torch
from PIL import Image


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_rgb(img: Image.Image) -> Image.Image:
    """Grayscale/other -> 3-channel RGB by replication (no colorization)."""
    return img if img.mode == "RGB" else img.convert("RGB")


class FrozenExtractor:
    """Base class. Subclasses set: name, checkpoint, embedding_dim, and implement _embed."""

    name: str = "base"
    checkpoint: str = ""
    embedding_dim: int = 0
    preprocessing_version: str = "preproc-v1"

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_device()

    @torch.no_grad()
    def embed_pil_batch(self, imgs: List[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def meta(self) -> dict:
        return {
            "model": self.name,
            "checkpoint": self.checkpoint,
            "embedding_dim": self.embedding_dim,
            "preprocessing_version": self.preprocessing_version,
            "device": str(self.device),
        }


class ResNet50Extractor(FrozenExtractor):
    """torchvision ResNet-50, ImageNet-1k weights, fc removed -> 2048-d global-avg-pool."""

    name = "resnet50"

    def __init__(self, weights_enum: str = "ResNet50_Weights.IMAGENET1K_V1",
                 device: torch.device | None = None):
        super().__init__(device)
        import torchvision
        from torchvision.models import resnet50

        wname = weights_enum.split(".", 1)[1]
        weights = getattr(torchvision.models.ResNet50_Weights, wname)
        self.checkpoint = f"torchvision:{weights_enum}"
        self.embedding_dim = 2048
        self._transform = weights.transforms()  # official: resize256 -> crop224 -> normalize
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()          # penultimate feature
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def embed_pil_batch(self, imgs: List[Image.Image]) -> np.ndarray:
        batch = torch.stack([self._transform(_to_rgb(im)) for im in imgs]).to(self.device)
        feats = self.model(batch)
        return feats.detach().to("cpu").float().numpy()


class DINOv2Extractor(FrozenExtractor):
    """HF facebook/dinov2-base (ViT-B/14). Feature = CLS/pooler_output (768-d)."""

    name = "dinov2"

    def __init__(self, hf_checkpoint: str = "facebook/dinov2-base",
                 device: torch.device | None = None):
        super().__init__(device)
        from transformers import AutoImageProcessor, AutoModel

        self.checkpoint = f"hf:{hf_checkpoint}"
        self.processor = AutoImageProcessor.from_pretrained(hf_checkpoint)
        self.model = AutoModel.from_pretrained(hf_checkpoint).eval().to(self.device)
        self.embedding_dim = int(self.model.config.hidden_size)  # 768

    @torch.no_grad()
    def embed_pil_batch(self, imgs: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=[_to_rgb(im) for im in imgs], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # pooler_output is the CLS token after layernorm; fall back to last_hidden_state[:,0].
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out.last_hidden_state[:, 0]
        return feats.detach().to("cpu").float().numpy()


class CLIPExtractor(FrozenExtractor):
    """HF openai/clip-vit-base-patch16 image tower. Feature = image_embeds (512-d)."""

    name = "clip"

    def __init__(self, hf_checkpoint: str = "openai/clip-vit-base-patch16",
                 device: torch.device | None = None):
        super().__init__(device)
        from transformers import CLIPImageProcessor, CLIPModel

        self.checkpoint = f"hf:{hf_checkpoint}"
        self.processor = CLIPImageProcessor.from_pretrained(hf_checkpoint)
        self.model = CLIPModel.from_pretrained(hf_checkpoint).eval().to(self.device)
        self.embedding_dim = int(self.model.config.projection_dim)  # 512

    @torch.no_grad()
    def embed_pil_batch(self, imgs: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=[_to_rgb(im) for im in imgs], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        return feats.detach().to("cpu").float().numpy()
