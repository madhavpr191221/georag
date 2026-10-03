import shutil

import pytest
import torch
from torch.utils.data import DataLoader

from georag.data import EuroSATMultispectralDataset, collate_tiles
from georag.data.audit import compute_uint16_band_statistics
from georag.data.catalog import CatalogStaleError
from tests.conftest import BANDS


def test_eurosat_adapter_loads_channel_first_tiles(tiny_eurosat_root) -> None:
    dataset = EuroSATMultispectralDataset(
        tiny_eurosat_root, BANDS, expected_count=4, value_scale=100.0
    )
    sample = dataset[0]
    assert sample.image.shape == (13, 64, 64)
    assert sample.image.dtype == torch.float32
    assert torch.isfinite(sample.image).all()
    assert sample.record.sensor == "Sentinel-2 MSI"
    assert sample.record.latitude is not None
    assert sample.record.longitude is not None
    assert sample.record.spatial_resolution_m == (10.0, 10.0)
    assert dataset.record_for_id(sample.record.tile_id) == sample.record


def test_tile_ids_are_stable_across_root_directories(tiny_eurosat_root, tmp_path) -> None:
    first = EuroSATMultispectralDataset(tiny_eurosat_root, BANDS, expected_count=4)
    copied_root = tmp_path / "elsewhere" / "EuroSAT_MS"
    shutil.copytree(tiny_eurosat_root, copied_root)
    second = EuroSATMultispectralDataset(copied_root, BANDS, expected_count=4)
    assert [record.tile_id for record in first.records] == [record.tile_id for record in second.records]


def test_collated_batch_has_explicit_bchw_shape(tiny_eurosat_root) -> None:
    dataset = EuroSATMultispectralDataset(tiny_eurosat_root, BANDS, expected_count=4)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_tiles)))
    assert batch.images.shape == (3, 13, 64, 64)
    assert len(batch.records) == 3


def test_wrong_expected_count_is_rejected(tiny_eurosat_root) -> None:
    with pytest.raises(ValueError, match="expected 27"):
        EuroSATMultispectralDataset(tiny_eurosat_root, BANDS, expected_count=27)


def test_non_storage_band_order_is_rejected(tiny_eurosat_root) -> None:
    spectral_order = (*BANDS[:8], "B8A", *BANDS[8:12])
    with pytest.raises(ValueError, match="storage order"):
        EuroSATMultispectralDataset(
            tiny_eurosat_root, spectral_order, expected_count=4
        )


def test_valid_catalog_avoids_reopening_headers(tiny_eurosat_root, tmp_path) -> None:
    catalog = tmp_path / "catalogs" / "eurosat.jsonl"
    first = EuroSATMultispectralDataset(
        tiny_eurosat_root, BANDS, expected_count=4, catalog_path=catalog
    )
    assert first.catalog_source == "headers"
    assert catalog.is_file()
    second = EuroSATMultispectralDataset(
        tiny_eurosat_root, BANDS, expected_count=4, catalog_path=catalog
    )
    assert second.catalog_source == "cache"
    assert [record.tile_id for record in second.records] == [record.tile_id for record in first.records]


def test_catalog_rejects_changed_source_inventory(tiny_eurosat_root, tmp_path) -> None:
    catalog = tmp_path / "eurosat.jsonl"
    EuroSATMultispectralDataset(
        tiny_eurosat_root, BANDS, expected_count=4, catalog_path=catalog
    )
    next(tiny_eurosat_root.rglob("*.tif")).touch()
    with pytest.raises(CatalogStaleError, match="files changed"):
        EuroSATMultispectralDataset(
            tiny_eurosat_root, BANDS, expected_count=4, catalog_path=catalog
        )


def test_exact_uint16_audit_uses_all_pixels(tiny_eurosat_root) -> None:
    dataset = EuroSATMultispectralDataset(tiny_eurosat_root, BANDS, expected_count=4)
    statistics, summary = compute_uint16_band_statistics(
        dataset, list(range(len(dataset))), progress_every=0
    )
    assert len(statistics) == 13
    assert statistics[0].count == 4 * 64 * 64
    assert statistics[0].minimum == 100
    assert statistics[-1].maximum == 1437
    assert summary["tiles_audited"] == 4
