from __future__ import annotations

from pathlib import Path

import torch
import pytest

from georag.data.core import EOTileDataset, TileRecord, TileSample
from georag.data.masked_statistics import masked_band_statistics


class TinyDataset(EOTileDataset):
    def __init__(self) -> None:
        self._records = tuple(TileRecord(f"tile-{i}", Path(f"{i}.tif")) for i in range(2))

    @property
    def records(self):
        return self._records

    def record_for_id(self, tile_id: str) -> TileRecord:
        return self._records[int(tile_id.split("-")[1])]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> TileSample:
        image = torch.tensor([[[1., 100.], [3., 100.]], [[2., 200.], [6., 200.]]]) + index * 4
        mask = torch.tensor([[True, False], [True, False]])
        return TileSample(image, self._records[index], mask)


def test_masked_band_statistics_ignore_invalid_pixels() -> None:
    mean, std = masked_band_statistics(TinyDataset(), [0, 1])
    assert mean == [4.0, 6.0]
    assert std == pytest.approx([5.0**0.5, 8.0**0.5])
