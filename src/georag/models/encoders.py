"""Inspectable CNN and Vision Transformer encoders for small EO tiles."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


class BandStandardizer(nn.Module):
    """Apply fixed, training-split per-band z-score normalization."""

    def __init__(self, mean: Sequence[float], standard_deviation: Sequence[float]) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std_tensor = torch.as_tensor(standard_deviation, dtype=torch.float32).flatten()
        if mean_tensor.numel() == 0 or mean_tensor.shape != std_tensor.shape:
            raise ValueError("mean and standard_deviation must have the same non-empty length")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("band statistics must be finite")
        if torch.any(std_tensor <= 0):
            raise ValueError("band standard deviations must be positive")
        self.register_buffer("mean", mean_tensor.view(1, -1, 1, 1))
        self.register_buffer("standard_deviation", std_tensor.view(1, -1, 1, 1))

    @property
    def channels(self) -> int:
        return self.mean.shape[1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"expected BCHW input, received shape {tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError("encoder inputs must be floating-point tensors")
        if images.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} input bands, received {images.shape[1]}")
        mean = self.mean.to(dtype=images.dtype)
        standard_deviation = self.standard_deviation.to(dtype=images.dtype)
        return (images - mean) / standard_deviation


class CNNEncoder(nn.Module):
    """Three-stage CNN: BCHW imagery -> pooled features -> L2 unit embeddings."""

    def __init__(
        self,
        band_mean: Sequence[float],
        band_standard_deviation: Sequence[float],
        embedding_dim: int = 128,
        channels: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if len(channels) != 3 or any(channel <= 0 or channel % 8 for channel in channels):
            raise ValueError("channels must contain three positive multiples of 8")
        self.normalizer = BandStandardizer(band_mean, band_standard_deviation)
        input_channels = self.normalizer.channels
        blocks: list[nn.Module] = []
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=output_channels),
                    nn.SiLU(),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.projection = nn.Linear(channels[-1], embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map raw scaled tiles `[B,C,H,W]` to normalized embeddings `[B,d]`."""
        normalized = self.normalizer(images)
        feature_map = self.features(normalized)
        pooled = self.pool(feature_map).flatten(start_dim=1)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=1)


class PatchEmbedding(nn.Module):
    """Unfold non-overlapping image patches, then project each patch to a token."""

    def __init__(self, input_channels: int, patch_size: int, width: int) -> None:
        super().__init__()
        if input_channels <= 0 or patch_size <= 0 or width <= 0:
            raise ValueError("input_channels, patch_size, and width must be positive")
        self.input_channels = input_channels
        self.patch_size = patch_size
        patch_values = input_channels * patch_size * patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.projection = nn.Linear(patch_values, width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(
                f"expected BCHW images with {self.input_channels} channels, received {tuple(images.shape)}"
            )
        if images.shape[2] % self.patch_size or images.shape[3] % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        # Unfold returns [B, C*P*P, N]; transpose to one flattened patch per token.
        patches = self.unfold(images).transpose(1, 2)
        return self.projection(patches)


class MultiHeadSelfAttention(nn.Module):
    """Explicit Q/K/V attention over tokens; optionally return attention weights."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output_projection = nn.Linear(width, width)

    def forward(
        self, tokens: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"expected [B,N,{self.width}] tokens, received {tuple(tokens.shape)}")
        batch, token_count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, token_count, 3, self.heads, self.head_width
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        # Q,K,V: [B,H,N,Dh]; scores and weights: [B,H,N,N].
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, token_count, self.width)
        output = self.output_projection(attended)
        if return_attention:
            return output, weights
        return output


class TransformerEncoderBlock(nn.Module):
    """Pre-normalized transformer block with explicit residual paths."""

    def __init__(self, width: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden_width = width * mlp_ratio
        self.attention_norm = nn.LayerNorm(width, eps=1e-6)
        self.attention = MultiHeadSelfAttention(width, heads)
        self.mlp_norm = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens


class ViTEncoder(nn.Module):
    """Small vanilla ViT: patch tokens -> self-attention -> L2 unit embeddings."""

    def __init__(
        self,
        band_mean: Sequence[float],
        band_standard_deviation: Sequence[float],
        image_size: tuple[int, int] = (64, 64),
        patch_size: int = 4,
        width: int = 192,
        depth: int = 4,
        heads: int = 3,
        mlp_ratio: int = 4,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if patch_size <= 0 or any(size % patch_size for size in image_size):
            raise ValueError("patch_size must divide both image dimensions")
        if depth <= 0 or mlp_ratio <= 0 or embedding_dim <= 0:
            raise ValueError("depth, mlp_ratio, and embedding_dim must be positive")
        self.normalizer = BandStandardizer(band_mean, band_standard_deviation)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(self.normalizer.channels, patch_size, width)
        rows, columns = image_size[0] // patch_size, image_size[1] // patch_size
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = nn.Parameter(torch.zeros(1, rows * columns + 1, width))
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(width, heads, mlp_ratio) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(width, eps=1e-6)
        self.projection = nn.Linear(width, embedding_dim)
        self.embedding_dim = embedding_dim
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map raw scaled tiles `[B,C,H,W]` to normalized embeddings `[B,d]`."""
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected image dimensions {self.image_size}, received {tuple(images.shape[-2:])}"
            )
        normalized = self.normalizer(images)
        tokens = self.patch_embedding(normalized)
        batch = tokens.shape[0]
        class_token = self.class_token.expand(batch, -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        class_features = self.final_norm(tokens[:, 0])
        projected = self.projection(class_features)
        return F.normalize(projected, p=2, dim=1)
