from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from georag.evaluation.spectral_evidence import (
    INDEX_NAMES,
    change_indices,
    compute_indices,
    cosine_scores,
    index_histogram,
    make_temporal_pairs,
    summarize_indices,
)


BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")


def test_index_maps_match_hand_calculated_values_and_mask_invalid_pixels() -> None:
    image = np.ones((12, 2, 2), dtype=np.float32)
    image[BANDS.index("B03")] = [[3, 3], [0, 1]]
    image[BANDS.index("B08")] = [[1, 1], [0, 3]]
    image[BANDS.index("B04")] = [[1, 1], [0, 1]]
    image[BANDS.index("B11")] = [[1, 3], [0, 1]]
    valid = np.asarray([[True, True], [True, False]])

    indices = compute_indices(image, valid, BANDS)

    assert indices["ndwi"][0, 0] == pytest.approx(0.5)
    assert indices["mndwi"][0, 0] == pytest.approx(0.5)
    assert indices["ndvi"][0, 0] == pytest.approx(0.0)
    assert np.isnan(indices["ndwi"][1, 0])  # zero denominator
    assert np.isnan(indices["mndwi"][1, 0])
    assert np.isnan(indices["ndvi"][1, 0])
    assert np.isnan(indices["ndwi"][1, 1])  # invalid source mask


def test_histograms_have_fixed_bins_and_unit_mass_per_index() -> None:
    indices = {name: np.asarray([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32) for name in INDEX_NAMES}
    descriptor = index_histogram(indices, bins=4)
    assert descriptor.shape == (12,)
    for section in np.split(descriptor, 3):
        assert section.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(cosine_scores(descriptor, descriptor[None]), [1.0])


def test_change_map_is_after_minus_before_and_uses_jointly_finite_pixels() -> None:
    before = {name: np.asarray([[0.1, np.nan]], dtype=np.float32) for name in INDEX_NAMES}
    after = {name: np.asarray([[0.6, 0.9]], dtype=np.float32) for name in INDEX_NAMES}
    changes = change_indices(before, after)
    assert changes["mndwi"][0, 0] == pytest.approx(0.5)
    assert np.isnan(changes["mndwi"][0, 1])
    assert summarize_indices(changes)["mndwi"]["positive_fraction"] == 1.0


def test_temporal_pairs_are_consecutive_same_sequence_and_within_gap() -> None:
    start = datetime(2019, 1, 1, tzinfo=timezone.utc)
    records = [
        SimpleNamespace(tile_id="a", timestamp=start, metadata={"sequence_id": "0001"}),
        SimpleNamespace(tile_id="b", timestamp=start + timedelta(days=10), metadata={"sequence_id": "0001"}),
        SimpleNamespace(tile_id="c", timestamp=start + timedelta(days=50), metadata={"sequence_id": "0001"}),
        SimpleNamespace(tile_id="d", timestamp=start + timedelta(days=3), metadata={"sequence_id": "1"}),
        SimpleNamespace(tile_id="e", timestamp=start + timedelta(days=45), metadata={"sequence_id": "1"}),
    ]
    pairs = make_temporal_pairs(records, max_gap_days=30)
    assert [(pair["before"].tile_id, pair["after"].tile_id, pair["gap_days"]) for pair in pairs] == [("a", "b", 10)]


def test_temporal_pair_builder_rejects_nonpositive_gap_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_temporal_pairs([], max_gap_days=0)
