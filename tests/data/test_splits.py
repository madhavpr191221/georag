from pathlib import Path

import pytest

from georag.data.core import TileRecord
from georag.data.splits import (
    create_split_assignments,
    indices_for_split,
    load_split_manifest,
    write_split_manifest,
)


def _records() -> list[TileRecord]:
    return [
        TileRecord(
            tile_id=f"eurosat_ms:{label}/{label}_{index}",
            image_path=Path(f"{label}_{index}.tif"),
            labels=(label,),
        )
        for label in ("Forest", "River")
        for index in range(20)
    ]


def test_stratified_split_is_deterministic_disjoint_and_exhaustive(tmp_path) -> None:
    records = _records()
    first = create_split_assignments(records, (0.8, 0.1, 0.1), seed=17)
    second = create_split_assignments(list(reversed(records)), (0.8, 0.1, 0.1), seed=17)
    assert first == second
    assert set(first) == {record.tile_id for record in records}
    assert set(first.values()) == {"train", "validation", "test"}
    manifest = write_split_manifest(tmp_path / "split.csv", records, first)
    assert load_split_manifest(manifest, records) == first
    split_sets = [
        set(indices_for_split(records, first, name))
        for name in ("train", "validation", "test")
    ]
    assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])
    assert set.union(*split_sets) == set(range(len(records)))


def test_manifest_rejects_unknown_ids(tmp_path) -> None:
    records = _records()
    assignments = create_split_assignments(records, (0.8, 0.1, 0.1), seed=17)
    assignments["unknown"] = "train"
    with pytest.raises(ValueError, match="ID mismatch"):
        write_split_manifest(tmp_path / "split.csv", records, assignments)
