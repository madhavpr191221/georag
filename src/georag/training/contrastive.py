"""Explicit 2B NT-Xent objective for paired, normalized image embeddings."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, masked 2B×2B logits, and positive indices.

    For each anchor, its paired view is the target and every other view except
    itself is a negative. The score is cosine similarity divided by temperature.
    """
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same [B,D] shape")
    batch_size, embedding_dim = z1.shape
    if batch_size < 2 or embedding_dim <= 0:
        raise ValueError("NT-Xent requires batch size at least 2 and positive embedding dimension")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(torch.cat((z1, z2), dim=0), p=2, dim=1)
    logits = torch.matmul(z, z.T) / temperature
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, float("-inf"))
    rows = torch.arange(2 * batch_size, device=z.device)
    targets = (rows + batch_size) % (2 * batch_size)
    loss = F.cross_entropy(logits, targets)
    return loss, logits, targets


def positive_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fraction of anchors whose paired view has the highest non-self score."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or targets.shape != (logits.shape[0],):
        raise ValueError("expected square logits and one target index per row")
    return (logits.argmax(dim=1) == targets).float().mean()
