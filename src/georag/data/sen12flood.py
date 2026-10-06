"""Sentinel-2 multispectral adapter for the STAC-formatted SEN12-FLOOD copy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch

from georag.data.core import EOTileDataset, TileRecord, TileSample

SEN12FLOOD_S2_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12",
)
_SOURCE_ID = re.compile(
    r"^sen12floods_s2_source_(?P<sequence>[^_]+)_(?P<date>\d{4}_\d{2}_\d{2})$"
)
_LABEL_ID = re.compile(
    r"^sen12floods_s2_labels_(?P<sequence>[^_]+)_(?P<date>\d{4}_\d{2}_\d{2})$"
)
_NODATA = np.float32(-9999.0)
_RESAMPLING_BY_NAME = {
    "average": Resampling.average,
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}


class SEN12FloodDataset(EOTileDataset):
    """Lazy Sentinel-2 dataset with all available optical bands on a 20 m grid.

    ``all_records`` includes metadata-only scenes. ``records`` and ``__len__``
    expose only scenes whose twelve configured raster files are present, so
    normal dataset iteration cannot accidentally try to load missing imagery.
    Each image is returned as float32 raw digital numbers in ``[C,H,W]`` order.
    """

    def __init__(
        self,
        root: str | Path,
        bands: Sequence[str] = SEN12FLOOD_S2_BANDS,
        target_resolution_m: float = 20.0,
        value_scale: float = 1.0,
        target_grid_band: str = "B05",
        finer_resampling: str = "average",
        coarser_resampling: str = "bilinear",
        same_resolution_resampling: str = "nearest",
        nodata_fill: float = 0.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.bands = tuple(bands)
        self.target_resolution_m = float(target_resolution_m)
        self.value_scale = float(value_scale)
        self.target_grid_band = target_grid_band
        self.finer_resampling = finer_resampling
        self.coarser_resampling = coarser_resampling
        self.same_resolution_resampling = same_resolution_resampling
        self.nodata_fill = float(nodata_fill)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SEN12-FLOOD root is not a directory: {self.root}")
        if self.bands != SEN12FLOOD_S2_BANDS:
            raise ValueError(f"bands must use the supported order {SEN12FLOOD_S2_BANDS}")
        if self.target_resolution_m <= 0:
            raise ValueError("target_resolution_m must be positive")
        if self.value_scale <= 0:
            raise ValueError("value_scale must be positive")
        if self.target_grid_band != "B05":
            raise ValueError("the current SEN12-FLOOD grid is anchored to B05")
        for setting, value in (
            ("finer_resampling", self.finer_resampling),
            ("coarser_resampling", self.coarser_resampling),
            ("same_resolution_resampling", self.same_resolution_resampling),
        ):
            if value not in _RESAMPLING_BY_NAME:
                raise ValueError(f"{setting} must be one of {tuple(_RESAMPLING_BY_NAME)}")

        self._all_records = self._scan_records()
        self._records = tuple(
            record for record in self._all_records
            if (record.metadata or {}).get("asset_status") == "available"
        )
        self._usable_index_by_id = {
            record.tile_id: index for index, record in enumerate(self._records)
        }
        self._record_by_id = {record.tile_id: record for record in self._all_records}

    @property
    def records(self) -> Sequence[TileRecord]:
        """Records with every selected band file present."""
        return self._records

    @property
    def all_records(self) -> Sequence[TileRecord]:
        """All paired STAC records, including rows missing local imagery."""
        return self._all_records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TileSample:
        record = self._records[index]
        metadata = record.metadata or {}
        band_paths = metadata.get("band_paths")
        if not isinstance(band_paths, Mapping):
            raise ValueError(f"record {record.tile_id} has no band path mapping")
        reference_path = Path(str(band_paths["B05"]))
        with rasterio.open(reference_path) as reference:
            if reference.crs is None or not reference.crs.is_projected:
                raise ValueError(f"B05 target raster for {record.tile_id} needs a projected CRS")
            x_res, y_res = (abs(float(value)) for value in reference.res)
            if not np.isclose(x_res, self.target_resolution_m, atol=0.1) or not np.isclose(
                y_res, self.target_resolution_m, atol=0.1
            ):
                raise ValueError(
                    f"B05 target raster for {record.tile_id} has resolution {reference.res}, "
                    f"expected about {self.target_resolution_m} m"
                )
            target = {
                "crs": reference.crs,
                "transform": reference.transform,
                "width": reference.width,
                "height": reference.height,
            }

        image = np.full(
            (len(self.bands), target["height"], target["width"]),
            self.nodata_fill,
            dtype=np.float32,
        )
        valid_masks = np.zeros_like(image, dtype=bool)
        for band_index, band_name in enumerate(self.bands):
            source_path = Path(str(band_paths[band_name]))
            try:
                with rasterio.open(source_path) as source:
                    if source.crs is None:
                        raise ValueError(f"band {band_name} has no CRS")
                    source_values = source.read(1, masked=True).astype(np.float32)
                    source_array = np.asarray(source_values.filled(_NODATA), dtype=np.float32)
                    destination = np.full(
                        (target["height"], target["width"]), _NODATA, dtype=np.float32
                    )
                    resampling = _resampling_for_resolution(
                        source.res,
                        self.target_resolution_m,
                        self.finer_resampling,
                        self.coarser_resampling,
                        self.same_resolution_resampling,
                    )
                    reproject(
                        source=source_array,
                        destination=destination,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=_NODATA,
                        dst_transform=target["transform"],
                        dst_crs=target["crs"],
                        dst_nodata=_NODATA,
                        resampling=resampling,
                        init_dest_nodata=True,
                    )
            except (rasterio.errors.RasterioError, OSError) as error:
                raise ValueError(f"could not load {band_name} for {record.tile_id}: {error}") from error
            mask = np.isfinite(destination) & (destination != _NODATA)
            valid_masks[band_index] = mask
            image[band_index] = np.where(mask, destination, self.nodata_fill) / self.value_scale

        # A pixel is valid only when all bands in the stack have source support.
        common_valid_mask = valid_masks.all(axis=0)
        return TileSample(
            image=torch.from_numpy(image),
            record=record,
            valid_mask=torch.from_numpy(common_valid_mask),
        )

    def record_for_id(self, tile_id: str) -> TileRecord:
        try:
            return self._record_by_id[tile_id]
        except KeyError as error:
            raise KeyError(f"unknown SEN12-FLOOD tile ID: {tile_id}") from error

    def index_for_id(self, tile_id: str) -> int:
        try:
            return self._usable_index_by_id[tile_id]
        except KeyError as error:
            raise KeyError(f"tile is missing one or more local bands or is unknown: {tile_id}") from error

    def _scan_records(self) -> tuple[TileRecord, ...]:
        source_root = self.root / "sen12floods_s2_source"
        labels_root = self.root / "sen12floods_s2_labels"
        source_items = _read_stac_items(source_root, _SOURCE_ID)
        label_items = _read_stac_items(labels_root, _LABEL_ID)
        source_keys = {key for key, _, _ in source_items}
        label_keys = {key for key, _, _ in label_items}
        if source_keys != label_keys:
            raise ValueError(
                "Sentinel-2 source/label item mismatch; "
                f"missing labels={sorted(source_keys - label_keys)[:5]}, "
                f"missing sources={sorted(label_keys - source_keys)[:5]}"
            )
        labels_by_key = {key: item for key, item, _ in label_items}
        records: list[TileRecord] = []
        for (sequence_id, date_token), source_item, item_directory in source_items:
            label_item = labels_by_key[(sequence_id, date_token)]
            source_id = str(source_item["id"])
            label_properties = label_item.get("properties") or {}
            flood_value = _parse_boolean(label_properties.get("FLOODING"), source_id, "FLOODING")
            coverage_value = _parse_optional_boolean(
                label_properties.get("FULL-DATA-COVERAGE"), source_id, "FULL-DATA-COVERAGE"
            )
            assets = source_item.get("assets") or {}
            band_paths: dict[str, str] = {}
            missing_bands: list[str] = []
            for band_name in self.bands:
                asset = assets.get(band_name)
                href = asset.get("href") if isinstance(asset, Mapping) else None
                if not isinstance(href, str) or not href:
                    missing_bands.append(band_name)
                    band_paths[band_name] = str(item_directory / f"{band_name}.tif")
                    continue
                asset_path = Path(href)
                if asset_path.is_absolute():
                    resolved_path = asset_path.resolve()
                else:
                    resolved_path = (item_directory / asset_path).resolve()
                if not resolved_path.is_relative_to(self.root):
                    raise ValueError(f"asset path escapes dataset root for {source_id}: {href}")
                band_paths[band_name] = str(resolved_path)
                if not resolved_path.is_file():
                    missing_bands.append(band_name)
            asset_status = "available" if not missing_bands else "missing"
            properties = source_item.get("properties") or {}
            bbox = source_item.get("bbox")
            latitude, longitude = _bbox_center(bbox)
            timestamp = _parse_timestamp(properties.get("datetime"), date_token)
            records.append(
                TileRecord(
                    tile_id=source_id,
                    image_path=Path(band_paths["B05"]),
                    latitude=latitude,
                    longitude=longitude,
                    timestamp=timestamp,
                    sensor="Sentinel-2 MSI",
                    bands=self.bands,
                    spatial_resolution_m=(self.target_resolution_m, self.target_resolution_m),
                    labels=("flood" if flood_value else "no_flood",),
                    metadata={
                        "sequence_id": sequence_id,
                        "stac_source_id": source_id,
                        "stac_label_id": str(label_item["id"]),
                        "raw_label_properties": dict(label_properties),
                        "flooding": flood_value,
                        "full_data_coverage": coverage_value,
                        "asset_status": asset_status,
                        "missing_bands": tuple(missing_bands),
                        "band_paths": band_paths,
                        "target_grid_band": self.target_grid_band,
                        "target_resolution_m": self.target_resolution_m,
                        "radiometry": "raw_uint16_dn",
                        "value_scale": self.value_scale,
                        "resampling": {
                            "finer_than_target": self.finer_resampling,
                            "coarser_than_target": self.coarser_resampling,
                            "same_resolution": self.same_resolution_resampling,
                        },
                        "nodata_fill": self.nodata_fill,
                        "source_bbox": tuple(float(value) for value in bbox) if bbox else None,
                    },
                )
            )
        return tuple(sorted(records, key=lambda record: record.tile_id))


def sequence_id_for_record(record: TileRecord) -> str:
    """Return the exact sequence token used to prevent location leakage."""
    value = (record.metadata or {}).get("sequence_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"record {record.tile_id} has no exact sequence_id string")
    return value


def _read_stac_items(
    collection_root: Path, pattern: re.Pattern[str]
) -> list[tuple[tuple[str, str], dict[str, Any], Path]]:
    if not collection_root.is_dir():
        raise FileNotFoundError(f"STAC collection directory not found: {collection_root}")
    result: list[tuple[tuple[str, str], dict[str, Any], Path]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(collection_root.rglob("stac.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read STAC item {path}: {error}") from error
        match = pattern.fullmatch(str(item.get("id", "")))
        if match is None:
            # Collection-level STAC documents are not image items.
            continue
        key = (match.group("sequence"), match.group("date"))
        if key in seen:
            raise ValueError(f"duplicate STAC item for sequence/date {key} below {collection_root}")
        seen.add(key)
        result.append((key, item, path.parent))
    if not result:
        raise ValueError(f"no matching STAC image items found below {collection_root}")
    return result


def _parse_boolean(value: object, item_id: str, name: str) -> bool:
    parsed = _parse_optional_boolean(value, item_id, name)
    if parsed is None:
        raise ValueError(f"required {name} label missing for {item_id}")
    return parsed


def _parse_optional_boolean(value: object, item_id: str, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError(f"{name} must be a boolean or 'True'/'False' string for {item_id}; got {value!r}")


def _parse_timestamp(value: object, date_token: str) -> datetime:
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
    return datetime.strptime(date_token, "%Y_%m_%d")


def _bbox_center(bbox: object) -> tuple[float | None, float | None]:
    if not isinstance(bbox, Sequence) or len(bbox) < 4:
        return None, None
    west, south, east, north = (float(value) for value in bbox[:4])
    return (south + north) / 2.0, (west + east) / 2.0


def _resampling_for_resolution(
    source_resolution: tuple[float, float],
    target_resolution_m: float,
    finer_method: str,
    coarser_method: str,
    same_method: str,
) -> Resampling:
    source_m = max(abs(float(value)) for value in source_resolution)
    if source_m < target_resolution_m * 0.95:
        return _RESAMPLING_BY_NAME[finer_method]
    if source_m > target_resolution_m * 1.05:
        return _RESAMPLING_BY_NAME[coarser_method]
    return _RESAMPLING_BY_NAME[same_method]
