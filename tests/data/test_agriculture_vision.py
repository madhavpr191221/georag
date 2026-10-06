from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from georag.data.agriculture_vision import AGRICULTURE_VISION_CLASSES, AgricultureVisionDataset


def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = data.shape
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=count,
                       dtype=data.dtype, transform=from_origin(0, height, 1, 1)) as output:
        output.write(data)


@pytest.fixture
def agriculture_root(tmp_path: Path) -> Path:
    root = tmp_path / "AgricultureVision"
    for split, field in (("train", "field-train"), ("val", "field-val")):
        stem = f"{field}_0-0-512-512"
        rgb = np.stack([np.full((16, 16), value, np.uint8) for value in (20, 40, 60)])
        nir = np.full((1, 16, 16), 80, np.uint8)
        _write(root / split / "images" / "rgb" / f"{stem}.tif", rgb)
        _write(root / split / "images" / "nir" / f"{stem}.tif", nir)
        valid = np.full((1, 16, 16), 255, np.uint8)
        _write(root / split / "masks" / f"{stem}.tif", valid)
        _write(root / split / "boundaries" / f"{stem}.tif", valid)
        mask = np.zeros((1, 16, 16), np.uint8); mask[:, 4:8, 4:8] = 255
        for category in AGRICULTURE_VISION_CLASSES:
            category_mask = mask if category == "water" else np.zeros_like(mask)
            _write(root / split / "labels" / category / f"{stem}.tif", category_mask)
    return root


def test_agriculture_adapter_preserves_canonical_band_order_and_annotations(agriculture_root: Path) -> None:
    dataset = AgricultureVisionDataset(agriculture_root, "train", image_size=8)
    sample = dataset[0]
    assert sample.record.tile_id == "agriculture_vision:train:field-train_0-0-512-512"
    assert sample.record.parent_scene == "field-train"
    assert sample.record.labels == ("water",)
    assert sample.record.bands == ("R", "G", "B", "NIR")
    assert tuple(sample.image.shape) == (4, 8, 8)
    assert sample.image[:, 0, 0].tolist() == pytest.approx([20 / 255, 40 / 255, 60 / 255, 80 / 255])


def test_rgb_mode_skips_nir_and_unlabeled_test_is_rejected(agriculture_root: Path) -> None:
    dataset = AgricultureVisionDataset(agriculture_root, "validation", bands=("R", "G", "B"), image_size=8)
    assert tuple(dataset[0].image.shape) == (3, 8, 8)
    with pytest.raises(ValueError, match="unlabeled test"):
        AgricultureVisionDataset(agriculture_root, "test")


def test_adapter_rejects_missing_nir(agriculture_root: Path) -> None:
    (agriculture_root / "train" / "images" / "nir" / "field-train_0-0-512-512.tif").unlink()
    with pytest.raises(FileNotFoundError, match="no NIR raster"):
        AgricultureVisionDataset(agriculture_root, "train")


def test_adapter_rejects_incomplete_annotation_masks(agriculture_root: Path) -> None:
    mask = agriculture_root / "train" / "labels" / "weed_cluster" / "field-train_0-0-512-512.tif"
    mask.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete labels"):
        AgricultureVisionDataset(agriculture_root, "train")


def test_pattern_outside_valid_field_region_is_not_a_positive_label(agriculture_root: Path) -> None:
    stem = "field-train_0-0-512-512"
    empty_boundary = np.zeros((1, 16, 16), np.uint8)
    _write(agriculture_root / "train" / "boundaries" / f"{stem}.tif", empty_boundary)
    record = AgricultureVisionDataset(agriculture_root, "train").records[0]
    assert "water" not in record.labels


def test_annotation_catalog_reuses_labels_and_invalidates_when_source_changes(
    agriculture_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import georag.data.agriculture_vision as agriculture_vision

    cache_dir = tmp_path / "catalog"
    first = AgricultureVisionDataset(agriculture_root, "train", catalog_cache_dir=cache_dir)
    expected_id = first.records[0].tile_id
    assert first.records[0].labels == ("water",)
    assert (cache_dir / "train.json.gz").is_file()

    def unexpected_mask_read(_item):
        raise AssertionError("warm catalog load should not decode annotation rasters")

    monkeypatch.setattr(agriculture_vision, "_inspect_annotation_paths", unexpected_mask_read)
    warm = AgricultureVisionDataset(agriculture_root, "train", catalog_cache_dir=cache_dir)
    assert warm.records[0].tile_id == expected_id
    assert warm.records[0].labels == ("water",)

    monkeypatch.undo()
    mask_path = agriculture_root / "train" / "labels" / "water" / "field-train_0-0-512-512.tif"
    with rasterio.open(mask_path, "r+") as mask:
        mask.write(np.zeros((1, 16, 16), dtype=np.uint8))
    stat = mask_path.stat()
    import os
    os.utime(mask_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    refreshed = AgricultureVisionDataset(agriculture_root, "train", catalog_cache_dir=cache_dir)
    assert "water" not in refreshed.records[0].labels
