import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from georag.data import SEN12FloodDataset, SEN12FLOOD_S2_BANDS, collate_tiles
from georag.data.sen12flood import _resampling_for_resolution, sequence_id_for_record
from georag.data.sen12flood_cache import SEN12FloodTensorCache
from rasterio.enums import Resampling


def _make_scene(root: Path, sequence_id: str, date: str = "2019_01_02", *, missing_band: str | None = None):
    source_id = f"sen12floods_s2_source_{sequence_id}_{date}"
    label_id = f"sen12floods_s2_labels_{sequence_id}_{date}"
    source_directory = root / "sen12floods_s2_source" / "sen12floods_s2_source" / source_id
    label_directory = root / "sen12floods_s2_labels" / "sen12floods_s2_labels" / label_id
    source_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)

    assets = {}
    for band_index, band in enumerate(SEN12FLOOD_S2_BANDS):
        resolution = 10 if band in {"B02", "B03", "B04", "B08"} else 60 if band in {"B01", "B09"} else 20
        side = {10: 4, 20: 2, 60: 1}[resolution]
        value = band_index + 10
        array = np.full((side, side), value, dtype=np.uint16)
        nodata = None
        if band == "B02":
            array[:2, :2] = 0
            nodata = 0
        band_path = source_directory / f"{band}.tif"
        assets[band] = {"href": f"{band}.tif", "title": band}
        if band == missing_band:
            continue
        with rasterio.open(
            band_path,
            "w",
            driver="GTiff",
            width=side,
            height=side,
            count=1,
            dtype="uint16",
            crs="EPSG:32628",
            transform=from_origin(500_000, 1_000_000, resolution, resolution),
            nodata=nodata,
        ) as raster:
            raster.write(array, 1)

    item = {
        "type": "Feature",
        "id": source_id,
        "bbox": [-6.0, 8.0, -5.0, 9.0],
        "properties": {
            "datetime": date.replace("_", "-") + "T00:00:00+0000",
            "eo:bands": [{"name": band} for band in SEN12FLOOD_S2_BANDS],
        },
        "assets": assets,
    }
    (source_directory / "stac.json").write_text(json.dumps(item), encoding="utf-8")
    label = {
        "type": "Feature",
        "id": label_id,
        "properties": {"FLOODING": "True", "FULL-DATA-COVERAGE": "False"},
        "assets": {},
    }
    (label_directory / "stac.json").write_text(json.dumps(label), encoding="utf-8")
    return source_id


def test_sen12flood_loads_all_bands_on_b05_grid_and_keeps_provenance(tmp_path) -> None:
    root = tmp_path / "sen12flood"
    tile_id = _make_scene(root, "0001")
    dataset = SEN12FloodDataset(root)
    sample = dataset[0]

    assert len(dataset.all_records) == len(dataset) == 1
    assert sample.record.tile_id == tile_id
    assert sample.record.metadata["stac_label_id"] == tile_id.replace("_source_", "_labels_")
    assert sequence_id_for_record(sample.record) == "0001"
    assert sample.image.shape == (12, 2, 2)
    assert sample.image.dtype == torch.float32
    assert sample.image[:, 1, 1].tolist() == [float(index + 10) for index in range(12)]
    assert sample.valid_mask.shape == (2, 2)
    assert not bool(sample.valid_mask[0, 0])
    assert sample.record.labels == ("flood",)
    assert sample.record.metadata["full_data_coverage"] is False

    batch = collate_tiles([sample])
    assert batch.images.shape == (1, 12, 2, 2)
    assert batch.valid_masks.shape == (1, 2, 2)


def test_metadata_only_scene_is_retained_but_not_iterated(tmp_path) -> None:
    root = tmp_path / "sen12flood"
    tile_id = _make_scene(root, "0001", missing_band="B12")
    dataset = SEN12FloodDataset(root)
    assert len(dataset.all_records) == 1
    assert len(dataset.records) == len(dataset) == 0
    record = dataset.record_for_id(tile_id)
    assert record.metadata["asset_status"] == "missing"
    assert record.metadata["missing_bands"] == ("B12",)
    with pytest.raises(KeyError, match="missing one or more local bands"):
        dataset.index_for_id(tile_id)


def test_stac_asset_path_cannot_escape_dataset_root(tmp_path) -> None:
    root = tmp_path / "sen12flood"
    tile_id = _make_scene(root, "0001")
    source_path = next((root / "sen12floods_s2_source").rglob("stac.json"))
    item = json.loads(source_path.read_text(encoding="utf-8"))
    item["assets"]["B12"]["href"] = "../../../../outside.tif"
    source_path.write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes dataset root"):
        SEN12FloodDataset(root)


def test_invalid_boolean_label_is_rejected(tmp_path) -> None:
    root = tmp_path / "sen12flood"
    tile_id = _make_scene(root, "0001")
    label_path = next((root / "sen12floods_s2_labels").rglob("stac.json"))
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["properties"]["FLOODING"] = "unknown"
    label_path.write_text(json.dumps(label), encoding="utf-8")
    with pytest.raises(ValueError, match="FLOODING must be"):
        SEN12FloodDataset(root)


def test_resampling_policy_tracks_source_resolution() -> None:
    assert _resampling_for_resolution((10.0, 10.0), 20.0, "average", "bilinear", "nearest") == Resampling.average
    assert _resampling_for_resolution((60.0, 60.0), 20.0, "average", "bilinear", "nearest") == Resampling.bilinear
    assert _resampling_for_resolution((20.0, 20.0), 20.0, "average", "bilinear", "nearest") == Resampling.nearest


def test_disk_cache_round_trip_and_reuse(tmp_path) -> None:
    root = tmp_path / "sen12flood"
    _make_scene(root, "0001")
    dataset = SEN12FloodDataset(root)
    cache_path = tmp_path / "cache"
    first = SEN12FloodTensorCache(dataset, cache_path)
    sample = first[0]
    assert sample.image.shape == (12, 2, 2)
    assert sample.valid_mask is not None and sample.valid_mask.shape == (2, 2)
    assert sample.image[:, 1, 1].tolist() == [float(i + 10) for i in range(12)]
    first.close()
    second = SEN12FloodTensorCache(dataset, cache_path)
    assert second.fingerprint == first.fingerprint
    assert torch.equal(second[0].image, sample.image)
    second.close()
    band_path = Path(dataset.records[0].metadata["band_paths"]["B02"])
    band_path.touch()
    changed_dataset = SEN12FloodDataset(root)
    rebuilt = SEN12FloodTensorCache(changed_dataset, cache_path)
    assert rebuilt.fingerprint != first.fingerprint
    assert torch.equal(rebuilt[0].image, sample.image)
