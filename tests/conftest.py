from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
)


@pytest.fixture
def tiny_eurosat_root(tmp_path: Path) -> Path:
    root = tmp_path / "EuroSAT_MS"
    for label_index, label in enumerate(("Forest", "River")):
        class_directory = root / label
        class_directory.mkdir(parents=True)
        for tile_index in range(2):
            data = np.empty((13, 64, 64), dtype=np.uint16)
            for band_index in range(13):
                rows = np.arange(64, dtype=np.uint16)[:, None]
                columns = np.arange(64, dtype=np.uint16)[None, :]
                data[band_index] = (
                    100 * (band_index + 1) + 10 * label_index + tile_index + rows + columns
                )
            with rasterio.open(
                class_directory / f"{label}_{tile_index}.tif",
                "w",
                driver="GTiff",
                width=64,
                height=64,
                count=13,
                dtype="uint16",
                crs="EPSG:32632",
                transform=from_origin(500_000 + tile_index * 640, 5_500_000, 10, 10),
            ) as destination:
                destination.write(data)
    return root
