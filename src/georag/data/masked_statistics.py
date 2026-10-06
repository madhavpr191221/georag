"""Streaming per-channel moments over valid pixels in selected training tiles."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from georag.data.core import EOTileDataset


def masked_band_statistics(dataset: EOTileDataset, indices: Sequence[int]) -> tuple[list[float], list[float]]:
    """Compute population mean/std in float64, counting only common-valid pixels."""
    if not indices:
        raise ValueError("statistics require at least one tile index")
    sums: torch.Tensor | None = None
    sums_sq: torch.Tensor | None = None
    count = 0
    for index in indices:
        sample = dataset[index]
        if sample.valid_mask is None:
            raise ValueError(f"tile {sample.record.tile_id} has no validity mask")
        image = sample.image.to(torch.float64)
        mask = sample.valid_mask.to(torch.bool)
        if image.ndim != 3 or mask.shape != image.shape[-2:]:
            raise ValueError("expected image [C,H,W] and mask [H,W]")
        values = image[:, mask]
        if sums is None:
            sums = torch.zeros(image.shape[0], dtype=torch.float64)
            sums_sq = torch.zeros_like(sums)
        sums += values.sum(dim=1)
        sums_sq += values.square().sum(dim=1)
        count += int(mask.sum().item())
    assert sums is not None and sums_sq is not None
    if count == 0:
        raise ValueError("training tiles contain no valid pixels")
    mean = sums / count
    variance = (sums_sq / count - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(1e-12)
    return mean.tolist(), std.tolist()
