"""Concrete adapter for the 13-band GeoTIFF distribution of EuroSAT."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates
import torch

from georag.data.core import EOTileDataset, SampleTransform, TileRecord, TileSample
from georag.data.catalog import DatasetInventory, build_tiff_inventory, load_catalog, write_catalog

EUROSAT_STORAGE_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
)


class EuroSATMultispectralDataset(EOTileDataset):
    def __init__(
        self,
        root: str | Path,
        bands: Sequence[str],
        expected_shape: tuple[int, int, int] = (13, 64, 64),
        expected_count: int | None = 27_000,
        value_scale: float = 1.0,
        transform: SampleTransform | None = None,
        catalog_path: str | Path | None = None,
        rebuild_catalog: bool = False,
    ) -> None:
        self.root = Path(root)
        self.bands = tuple(bands)
        self.expected_shape = expected_shape
        self.expected_count = expected_count
        self.value_scale = float(value_scale)
        self.transform = transform
        self.catalog_path = Path(catalog_path) if catalog_path is not None else None
        if self.value_scale <= 0:
            raise ValueError("value_scale must be positive")
        if len(self.bands) != self.expected_shape[0]:
            raise ValueError("band count must match expected_shape[0]")
        if self.bands != EUROSAT_STORAGE_BANDS:
            raise ValueError(
                "EuroSAT_MS GeoTIFF channels must use storage order "
                f"{EUROSAT_STORAGE_BANDS}; received {self.bands}"
            )
        if not self.root.is_dir():
            raise FileNotFoundError(f"EuroSAT root does not exist or is not a directory: {self.root}")

        inventory = build_tiff_inventory(self.root)
        if not inventory.entries:
            raise ValueError(f"no GeoTIFF files found below {self.root}")
        if self.catalog_path is not None and self.catalog_path.is_file() and not rebuild_catalog:
            self._records, self.catalog_metadata = load_catalog(
                self.catalog_path, self.root, inventory, self.bands, self.expected_shape
            )
            self.catalog_source = "cache"
        else:
            self._records = self._scan_records(inventory)
            self.catalog_metadata = {
                "source_fingerprint": inventory.fingerprint,
                "tile_count": len(self._records),
                "total_bytes": inventory.total_bytes,
            }
            self.catalog_source = "headers"
            if self.catalog_path is not None:
                write_catalog(
                    self.catalog_path,
                    self.root,
                    self._records,
                    inventory,
                    self.bands,
                    self.expected_shape,
                )
        if self.expected_count is not None and len(self._records) != self.expected_count:
            raise ValueError(
                f"expected {self.expected_count} EuroSAT tiles under {self.root}, found {len(self._records)}"
            )
        self._index_by_id = {record.tile_id: index for index, record in enumerate(self._records)}
        if len(self._index_by_id) != len(self._records):
            raise ValueError("duplicate EuroSAT tile IDs were generated")

    @property
    def records(self) -> Sequence[TileRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TileSample:
        record = self._records[index]
        with rasterio.open(record.image_path) as source:
            array = source.read()
        if tuple(array.shape) != self.expected_shape:
            raise ValueError(
                f"tile {record.tile_id} has shape {tuple(array.shape)}, expected {self.expected_shape}"
            )
        image = torch.from_numpy(array.astype(np.float32, copy=False)) / self.value_scale
        if not torch.isfinite(image).all():
            raise ValueError(f"tile {record.tile_id} contains non-finite values")
        if self.transform is not None:
            image = self.transform(image)
        return TileSample(image=image, record=record)

    def record_for_id(self, tile_id: str) -> TileRecord:
        try:
            return self._records[self._index_by_id[tile_id]]
        except KeyError as error:
            raise KeyError(f"unknown EuroSAT tile ID: {tile_id}") from error

    def index_for_id(self, tile_id: str) -> int:
        try:
            return self._index_by_id[tile_id]
        except KeyError as error:
            raise KeyError(f"unknown EuroSAT tile ID: {tile_id}") from error

    def _scan_records(self, inventory: DatasetInventory) -> tuple[TileRecord, ...]:
        records: list[TileRecord] = []
        for entry in inventory.entries:
            path = self.root / entry.relative_path
            relative = path.relative_to(self.root)
            label = relative.parent.as_posix()
            tile_id = f"eurosat_ms:{label}/{path.stem}"
            try:
                with rasterio.open(path) as source:
                    shape = (source.count, source.height, source.width)
                    if shape != self.expected_shape:
                        raise ValueError(f"tile {tile_id} has header shape {shape}, expected {self.expected_shape}")
                    latitude, longitude = _wgs84_centroid(source)
                    resolution = _projected_resolution(source)
                    metadata = {
                        "crs": source.crs.to_string() if source.crs is not None else None,
                        "bounds": tuple(float(value) for value in source.bounds),
                        "source_dtype": source.dtypes[0],
                        "nodata": source.nodata,
                        "driver": source.driver,
                    }
            except rasterio.errors.RasterioError as error:
                raise ValueError(f"could not read GeoTIFF header for {path}: {error}") from error
            records.append(
                TileRecord(
                    tile_id=tile_id,
                    image_path=path,
                    latitude=latitude,
                    longitude=longitude,
                    sensor="Sentinel-2 MSI",
                    bands=self.bands,
                    spatial_resolution_m=resolution,
                    labels=(label,),
                    metadata=metadata,
                )
            )
        return tuple(records)


def _wgs84_centroid(source: rasterio.io.DatasetReader) -> tuple[float | None, float | None]:
    if source.crs is None:
        return None, None
    center_x = (source.bounds.left + source.bounds.right) / 2.0
    center_y = (source.bounds.bottom + source.bounds.top) / 2.0
    try:
        longitude, latitude = transform_coordinates(source.crs, "EPSG:4326", [center_x], [center_y])
    except (rasterio.errors.RasterioError, ValueError):
        return None, None
    return float(latitude[0]), float(longitude[0])


def _projected_resolution(source: rasterio.io.DatasetReader) -> tuple[float, float] | None:
    if source.crs is None or not source.crs.is_projected:
        return None
    return abs(float(source.res[0])), abs(float(source.res[1]))
