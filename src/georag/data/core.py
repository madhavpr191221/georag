"""Core types for channel-agnostic Earth-observation tiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    image_path: Path
    latitude: float | None = None
    longitude: float | None = None
    timestamp: datetime | None = None
    sensor: str | None = None
    bands: tuple[str, ...] = ()
    spatial_resolution_m: tuple[float, float] | None = None
    parent_scene: str | None = None
    labels: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TileSample:
    image: torch.Tensor
    record: TileRecord


@dataclass(frozen=True)
class TileBatch:
    images: torch.Tensor
    records: tuple[TileRecord, ...]


SampleTransform = Callable[[torch.Tensor], torch.Tensor]


class EOTileDataset(Dataset[TileSample], ABC):
    """Minimal interface shared by concrete EO tile collections."""

    @property
    @abstractmethod
    def records(self) -> Sequence[TileRecord]:
        raise NotImplementedError

    @abstractmethod
    def record_for_id(self, tile_id: str) -> TileRecord:
        raise NotImplementedError


def collate_tiles(samples: Sequence[TileSample]) -> TileBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    shapes = {tuple(sample.image.shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"all tile tensors must have the same shape; received {sorted(shapes)}")
    return TileBatch(
        images=torch.stack([sample.image for sample in samples], dim=0),
        records=tuple(sample.record for sample in samples),
    )
