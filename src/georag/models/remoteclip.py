"""Small inference adapter for the published RemoteCLIP ViT-B/32 checkpoint."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
import open_clip


REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_ARCHITECTURE = "ViT-B-32"
REMOTECLIP_CHECKPOINT = "RemoteCLIP-ViT-B-32.pt"


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2-normalize a batch of [N,d] image or text features in float32."""

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must have non-empty shape [N, d]")
    features = features.float()
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise ValueError("features must have non-zero norm")
    return features / norms


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a checkpoint using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RemoteCLIPEncoder:
    """Encode RGB images and natural-language queries in one shared space.

    The model is pretrained, frozen, and used only as a retrieval baseline.
    Image inputs must already have passed the model-provided PIL transform.
    Output arrays are float32, L2-normalized, and shaped [N, d].
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
        revision: str = "main",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.revision = revision
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                repo_id=REMOTECLIP_REPO,
                filename=REMOTECLIP_CHECKPOINT,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            REMOTECLIP_ARCHITECTURE
        )
        # The upstream checkpoint is a plain state dict; weights_only avoids
        # unpickling arbitrary Python objects from the downloaded file.
        state_dict = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(REMOTECLIP_ARCHITECTURE)
        self.embedding_dim = int(self.model.text_projection.shape[-1])

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> np.ndarray:
        """Encode preprocessed RGB tensors shaped [N,3,224,224]."""

        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError("images must have shape [N, 3, 224, 224]")
        if images.shape[0] == 0:
            raise ValueError("images must not be empty")
        if not torch.isfinite(images).all():
            raise ValueError("images must be finite")
        with self._autocast():
            features = self.model.encode_image(images.to(self.device, non_blocking=True))
        return normalize_features(features).cpu().numpy()

    @torch.inference_mode()
    def encode_text(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """Encode text strings to normalized [N,d] float32 feature vectors."""

        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must be a non-empty sequence of non-empty strings")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        output: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(list(texts[start : start + batch_size])).to(self.device)
            with self._autocast():
                features = self.model.encode_text(tokens)
            output.append(normalize_features(features).cpu().numpy())
        return np.concatenate(output, axis=0)
