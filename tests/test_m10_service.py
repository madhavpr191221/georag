from datetime import datetime, timezone
import gzip
import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from georag.m10_service import M10Service, _bbox_intersects, _eligible_indices, intent_prototype


def test_intent_prototype_uses_only_matching_training_labels_and_normalizes() -> None:
    embeddings = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    labels = ["flood", "no_flood", "flood"]
    actual = intent_prototype(embeddings, labels, "no_flood")
    np.testing.assert_allclose(actual, [0, 1])
    assert np.linalg.norm(actual) == pytest.approx(1.0)


def test_intent_prototype_rejects_misaligned_or_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        intent_prototype(np.ones((2, 3)), ["flood"], "flood")
    with pytest.raises(ValueError, match="unsupported"):
        intent_prototype(np.ones((1, 3)), ["flood"], "fire")


def test_public_runs_exposes_only_selected_cnn_and_vit_presets(tmp_path, monkeypatch) -> None:
    service = object.__new__(M10Service)
    service.selection_path = tmp_path / "selection.json"
    service.selection_path.write_text(json.dumps({"protocol": {"support_tile_ids": ["c"]}, "presets": {
        "cnn": {"representative_run_id": "cnn-run", "mean_macro_precision_at_k": 0.7,
                "std_macro_precision_at_k": 0.02, "seed_count": 3, "representative_macro_precision_at_k": 0.71},
        "vit": {"representative_run_id": "vit-run", "mean_macro_precision_at_k": 0.6,
                "std_macro_precision_at_k": 0.03, "seed_count": 3, "representative_macro_precision_at_k": 0.59},
    }}), encoding="utf-8")
    service.searchable_gallery_ids = {"a", "b"}
    service.prototype_support_ids = {"c"}
    monkeypatch.setattr(service, "available_runs", lambda: [
        {"run_id": "cnn-run", "architecture": "cnn", "modality": "s2_12band", "seed": 17,
         "gallery_count": 3, "test_query_count": 1, "directory": tmp_path},
        {"run_id": "vit-run", "architecture": "vit", "modality": "rgb", "seed": 23,
         "gallery_count": 3, "test_query_count": 1, "directory": tmp_path},
        {"run_id": "unused-run", "architecture": "vit", "modality": "s2_12band", "seed": 42,
         "gallery_count": 3, "test_query_count": 1, "directory": tmp_path},
    ])

    runs = service.public_runs()
    assert [run["architecture"] for run in runs] == ["cnn", "vit"]
    assert [run["display_name"] for run in runs] == ["CNN - 12-band", "ViT - RGB"]
    assert runs[0]["searchable_gallery_count"] == 2
    assert runs[0]["prototype_support_count"] == 1


def test_loaded_run_removes_prototype_support_examples_from_search_gallery(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch_state = {
        "gallery_embeddings": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        "gallery_tile_ids": ["support", "searchable"],
    }
    import torch
    torch.save(torch_state, run_dir / "test_retrieval.pt")
    service = object.__new__(M10Service)
    service.run_cache = OrderedDict()
    service.prototype_support_ids = {"support"}
    service.searchable_gallery_ids = {"searchable"}
    service.record_by_id = {
        "support": SimpleNamespace(tile_id="support", timestamp=None, latitude=None, longitude=None,
                                   sensor="S2", bands=["B04"], labels=["flood"],
                                   metadata={"sequence_id": "a", "source_bbox": None}),
        "searchable": SimpleNamespace(tile_id="searchable", timestamp=None, latitude=None, longitude=None,
                                      sensor="S2", bands=["B04"], labels=["no_flood"],
                                      metadata={"sequence_id": "b", "source_bbox": None}),
    }
    monkeypatch.setattr(service, "available_runs", lambda: [{"run_id": "run-id", "directory": run_dir}])

    run = service._load_run("run-id")
    assert run["prototype_tile_ids"] == ["support"]
    assert run["prototype_labels"] == ["flood"]
    assert run["gallery_tile_ids"] == ["searchable"]
    assert run["gallery_labels"] == ["no_flood"]


def test_prefilter_applies_inclusive_dates_and_place_before_scoring() -> None:
    day = datetime(2019, 2, 1, tzinfo=timezone.utc)
    records = [
        {"tile_id": "inside", "timestamp_dt": day, "bbox": [76, 9, 77, 10]},
        {"tile_id": "outside", "timestamp_dt": day, "bbox": [80, 13, 81, 14]},
        {"tile_id": "missing", "timestamp_dt": day, "bbox": None},
        {"tile_id": "too_early", "timestamp_dt": datetime(2018, 1, 1, tzinfo=timezone.utc),
         "bbox": [76, 9, 77, 10]},
    ]
    eligible = _eligible_indices(records, day, day, box(75, 8, 78, 11))
    assert eligible == [0]


def test_example_tile_is_excluded_from_its_own_gallery_results() -> None:
    records = [
        {"tile_id": "query", "timestamp_dt": None, "bbox": None},
        {"tile_id": "other", "timestamp_dt": None, "bbox": None},
    ]
    assert _eligible_indices(records, None, None, excluded_tile_id="query") == [1]


def test_bbox_intersection_handles_dateline_crossing_bounds() -> None:
    assert _bbox_intersects([170, -5, -170, 5], box(175, -2, 179, 2))
    assert not _bbox_intersects([170, -5, -170, 5], box(0, -2, 5, 2))


def test_spectral_increase_ranks_training_date_pairs_by_delta_mndwi() -> None:
    from datetime import timedelta
    from georag.data.core import TileSample
    from georag.query import EOQuery

    start = datetime(2019, 1, 1, tzinfo=timezone.utc)
    bands = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
    records = []
    images = {}
    for sequence, water_values in (("A", (0.2, 0.8)), ("B", (0.8, 0.2))):
        for day, green in zip((0, 5), water_values, strict=True):
            tile_id = f"{sequence}-{day}"
            image = np.full((12, 2, 2), 0.5, dtype=np.float32)
            image[bands.index("B03")] = green
            image[bands.index("B08")] = 0.5
            image[bands.index("B04")] = 0.2
            image[bands.index("B11")] = 1.0 - green
            record = SimpleNamespace(
                tile_id=tile_id, timestamp=start + timedelta(days=day), latitude=1.0, longitude=2.0,
                labels=("flood",), metadata={"sequence_id": sequence, "source_bbox": None},
            )
            records.append(record)
            images[tile_id] = image

    class FakeSource:
        def __init__(self):
            self.bands = bands
            self.records = records
        def record_for_id(self, tile_id):
            return next(record for record in self.records if record.tile_id == tile_id)

    class FakeDataset:
        def __getitem__(self, index):
            record = records[index]
            return TileSample(torch.from_numpy(images[record.tile_id]), record, torch.ones((2, 2), dtype=torch.bool))

    import torch
    service = object.__new__(M10Service)
    service.source = FakeSource()
    service.spectral_records = tuple(records)
    service.index_by_id = {record.tile_id: i for i, record in enumerate(records)}
    service.dataset = FakeDataset()
    service._spectral_index_cache = OrderedDict()
    service._spectral_scene_features = {}
    service._spectral_pair_features = {}
    service.spectral_bins = 32
    service.spectral_range = (-1.0, 1.0)
    service.change_range = (-2.0, 2.0)
    service.index_epsilon = 1e-6
    service.spectral_max_pair_days = 30
    query = EOQuery(intent="spectral_water_increase", interpretation="Find increasing water-like spectral signal")

    result = service.retrieve_spectral(query, 5)

    assert result["retrieval_method"] == "median_delta_mndwi_rank"
    assert result["results"][0]["sequence_id"] == "A"
    assert result["results"][0]["change_summary"]["mndwi"]["median"] > 0
    from georag.query import SpectralRetrievalResponse
    validated = SpectralRetrievalResponse.model_validate(result)
    assert validated.results[0].before_tile_id == "A-0"
    decrease = service.retrieve_spectral(
        EOQuery(intent="spectral_water_decrease", interpretation="Find decreasing water-like spectral signal"), 5,
    )
    assert decrease["results"][0]["sequence_id"] == "B"
    assert decrease["results"][0]["score"] < 0


def test_place_search_disambiguates_admin1_by_country(tmp_path) -> None:
    path = tmp_path / "gazetteer.json.gz"
    payload = {"places": [
        {"place_id": "us:ga", "level": "admin1", "name": "Georgia", "country": "United States",
         "aliases": [], "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] }},
        {"place_id": "ge:country", "level": "country", "name": "Georgia", "country": "Georgia",
         "aliases": [], "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]] }},
    ]}
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
        compressed.write(json.dumps(payload).encode())
    service = object.__new__(M10Service)
    service.gazetteer_path = path
    service._gazetteer = None
    service._place_geometries = {}
    choices = service.find_places("Georgia", "admin1", country="United States")
    assert [(choice.place_id, choice.country) for choice in choices] == [("us:ga", "United States")]
    assert service.find_places("Georgia", "admin1", country="Georgia") == []
