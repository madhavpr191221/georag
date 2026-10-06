from pathlib import Path

import pytest

from georag.data.core import TileRecord
from georag.data.splits import (
    create_grouped_split_assignments,
    create_split_assignments,
    indices_for_split,
    load_group_split_manifest,
    load_split_manifest,
    write_split_manifest,
    write_group_split_manifest,
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


def test_grouped_split_is_deterministic_exhaustive_and_keeps_groups_together(tmp_path) -> None:
    records = [
        TileRecord(
            tile_id=f"tile-{group}-{index}",
            image_path=Path(f"{group}-{index}.tif"),
            labels=(("flood" if index % 2 else "no_flood"),),
            metadata={"sequence_id": group},
        )
        for group, size in (("0001", 5), ("1", 4), ("0002", 3), ("0003", 2), ("0004", 1))
        for index in range(size)
    ]
    get_group = lambda record: str(record.metadata["sequence_id"])
    assignments = create_grouped_split_assignments(records, get_group, (0.8, 0.1, 0.1), seed=17)
    reversed_assignments = create_grouped_split_assignments(
        list(reversed(records)), get_group, (0.8, 0.1, 0.1), seed=17
    )
    assert assignments == reversed_assignments
    assert set(assignments) == {record.tile_id for record in records}
    assert {assignments[record.tile_id] for record in records} == {"train", "validation", "test"}
    for group in {get_group(record) for record in records}:
        assert len({assignments[record.tile_id] for record in records if get_group(record) == group}) == 1

    path = write_group_split_manifest(tmp_path / "groups.csv", records, assignments, get_group)
    assert load_group_split_manifest(path, records, get_group) == assignments
    assert "0001" in path.read_text(encoding="utf-8")
    assert "1," in path.read_text(encoding="utf-8")


def test_grouped_split_rejects_fewer_than_three_groups() -> None:
    records = _records()[:2]
    with pytest.raises(ValueError, match="at least three distinct groups"):
        create_grouped_split_assignments(records, lambda record: record.labels[0], (0.8, 0.1, 0.1), seed=17)


def test_full_catalog_split_can_be_applied_to_loadable_subset() -> None:
    records = [
        TileRecord(
            tile_id=f"tile-{group}-{index}",
            image_path=Path(f"{group}-{index}.tif"),
            metadata={"sequence_id": group},
        )
        for group, size in (("a", 3), ("b", 2), ("c", 2), ("d", 1))
        for index in range(size)
    ]
    get_group = lambda record: str(record.metadata["sequence_id"])
    assignments = create_grouped_split_assignments(records, get_group, (0.5, 0.25, 0.25), seed=8)
    loadable_records = [record for record in records if record.tile_id != "tile-a-0"]
    with pytest.raises(ValueError, match="ID mismatch"):
        indices_for_split(loadable_records, assignments, "train")
    train_indices = indices_for_split(
        loadable_records, assignments, "train", allow_extra_assignments=True
    )
    assert all(assignments[loadable_records[index].tile_id] == "train" for index in train_indices)
    assert len(train_indices) == sum(
        assignments[record.tile_id] == "train" for record in loadable_records
    )
