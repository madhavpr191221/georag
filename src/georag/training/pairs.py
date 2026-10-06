"""Dataset and collation for paired, independently transformed EO tiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from georag.data.core import EOTileDataset, TileRecord, TileSample
from georag.training.augmentations import make_d4_views_with_mask


@dataclass(frozen=True)
class PairedTileSample:
    view_a: torch.Tensor
    view_b: torch.Tensor
    record: TileRecord
    mask_a: torch.Tensor | None = None
    mask_b: torch.Tensor | None = None


@dataclass(frozen=True)
class PairedTileBatch:
    view_a: torch.Tensor
    view_b: torch.Tensor
    records: tuple[TileRecord, ...]
    mask_a: torch.Tensor | None = None
    mask_b: torch.Tensor | None = None

    def pin_memory(self) -> "PairedTileBatch":
        """Allow DataLoader pin_memory to pin both image views."""
        return PairedTileBatch(
            self.view_a.pin_memory(), self.view_b.pin_memory(), self.records,
            self.mask_a.pin_memory() if self.mask_a is not None else None,
            self.mask_b.pin_memory() if self.mask_b is not None else None,
        )


class PairedViewDataset(Dataset[PairedTileSample]):
    """Wrap an EO dataset and return two seeded D4 views of each selected tile."""

    def __init__(
        self,
        dataset: EOTileDataset,
        indices: Sequence[int],
        seed: int,
        fixed_views: bool = False,
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)
        self.seed = int(seed)
        self.fixed_views = fixed_views
        self.epoch = 0
        if not self.indices:
            raise ValueError("paired-view dataset requires at least one tile index")
        if any(index < 0 or index >= len(dataset) for index in self.indices):
            raise IndexError("paired-view dataset contains an out-of-range tile index")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = 0 if self.fixed_views else int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> PairedTileSample:
        sample: TileSample = self.dataset[self.indices[index]]
        view_epoch = 0 if self.fixed_views else self.epoch
        view_a, view_b, mask_a, mask_b = make_d4_views_with_mask(
            sample.image, sample.valid_mask, sample.record.tile_id, self.seed, view_epoch
        )
        return PairedTileSample(view_a, view_b, sample.record, mask_a, mask_b)


def collate_paired_views(samples: Sequence[PairedTileSample]) -> PairedTileBatch:
    if not samples:
        raise ValueError("cannot collate an empty paired-view batch")
    shapes = {tuple(sample.view_a.shape) for sample in samples}
    shapes.update(tuple(sample.view_b.shape) for sample in samples)
    if len(shapes) != 1:
        raise ValueError(f"all view tensors must have the same shape; received {sorted(shapes)}")
    has_masks = any(sample.mask_a is not None or sample.mask_b is not None for sample in samples)
    if has_masks and any(sample.mask_a is None or sample.mask_b is None for sample in samples):
        raise ValueError("all paired samples must provide both masks or neither")
    return PairedTileBatch(
        view_a=torch.stack([sample.view_a for sample in samples]),
        view_b=torch.stack([sample.view_b for sample in samples]),
        records=tuple(sample.record for sample in samples),
        mask_a=torch.stack([sample.mask_a for sample in samples]) if has_masks else None,
        mask_b=torch.stack([sample.mask_b for sample in samples]) if has_masks else None,
    )
