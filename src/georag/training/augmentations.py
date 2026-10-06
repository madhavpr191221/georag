"""EO view transforms whose invariance assumptions are explicit."""

from __future__ import annotations

import hashlib
import random

import torch


def transform_d4(image: torch.Tensor, rotation_quadrants: int, horizontal_flip: bool) -> torch.Tensor:
    """Apply one square-grid symmetry without interpolation, identically on all bands."""
    if image.ndim != 3:
        raise ValueError(f"expected CHW image tensor, received {tuple(image.shape)}")
    if image.shape[-2] != image.shape[-1]:
        raise ValueError("D4 transforms require square tiles")
    if rotation_quadrants not in range(4):
        raise ValueError("rotation_quadrants must be 0, 1, 2, or 3")
    transformed = torch.rot90(image, k=rotation_quadrants, dims=(-2, -1))
    if horizontal_flip:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed.contiguous()


def transform_d4_pair(
    image: torch.Tensor,
    valid_mask: torch.Tensor | None,
    rotation_quadrants: int,
    horizontal_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the same exact grid symmetry to an image and its `[H,W]` mask."""
    if valid_mask is not None and (valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(image.shape[-2:])):
        raise ValueError("valid_mask must have [H,W] shape matching the image")
    transformed_image = transform_d4(image, rotation_quadrants, horizontal_flip)
    if valid_mask is None:
        return transformed_image, None
    transformed_mask = torch.rot90(valid_mask, k=rotation_quadrants, dims=(-2, -1))
    if horizontal_flip:
        transformed_mask = torch.flip(transformed_mask, dims=(-1,))
    return transformed_image, transformed_mask.contiguous()


def _view_seed(base_seed: int, epoch: int, tile_id: str, view_index: int) -> int:
    material = f"{base_seed}:{epoch}:{tile_id}:{view_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def make_d4_views(
    image: torch.Tensor,
    tile_id: str,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw two reproducible, independent D4 views for this tile and epoch."""
    views = []
    for view_index in (0, 1):
        generator = random.Random(_view_seed(seed, epoch, tile_id, view_index))
        rotation_quadrants = generator.randrange(4)
        horizontal_flip = bool(generator.randrange(2))
        views.append(transform_d4(image, rotation_quadrants, horizontal_flip))
    return views[0], views[1]


def make_d4_views_with_mask(
    image: torch.Tensor,
    valid_mask: torch.Tensor | None,
    tile_id: str,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Draw reproducible paired D4 views and transform the validity mask in lockstep."""
    views = []
    masks = []
    for view_index in (0, 1):
        generator = random.Random(_view_seed(seed, epoch, tile_id, view_index))
        transformed, transformed_mask = transform_d4_pair(
            image, valid_mask, generator.randrange(4), bool(generator.randrange(2))
        )
        views.append(transformed)
        masks.append(transformed_mask)
    return views[0], views[1], masks[0], masks[1]
