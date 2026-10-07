from __future__ import annotations

from georag.evaluation.spectral_audit import select_stratified_examples


def test_stratified_examples_are_deterministic_and_cover_label_tails() -> None:
    rows = [
        {"tile_id": f"{label}-{index:02d}", "label": label, "mndwi_median": float(index)}
        for label in ("flood", "no_flood")
        for index in range(16)
    ]
    first = select_stratified_examples(rows, samples_per_stratum=2, seed=17)
    second = select_stratified_examples(list(reversed(rows)), samples_per_stratum=2, seed=17)
    assert first == second
    assert len(first) == 8
    assert {row["stratum"] for row in first} == {
        "flood_low", "flood_high", "no_flood_low", "no_flood_high"
    }
    assert all(row["mndwi_median"] < 4 for row in first if row["stratum"].endswith("low"))
    assert all(row["mndwi_median"] >= 12 for row in first if row["stratum"].endswith("high"))


def test_stratified_examples_gracefully_use_small_groups() -> None:
    rows = [
        {"tile_id": "f-0", "label": "flood", "mndwi_median": 0.2},
        {"tile_id": "n-0", "label": "no_flood", "mndwi_median": -0.1},
    ]
    selected = select_stratified_examples(rows, samples_per_stratum=4, seed=1)
    assert len(selected) == 2
    assert {item["stratum"] for item in selected} == {
        "flood_low", "no_flood_low"
    }


def test_stratified_examples_reject_nonpositive_sample_count() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be positive"):
        select_stratified_examples([], samples_per_stratum=0, seed=17)
