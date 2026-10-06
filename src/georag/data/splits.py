"""Deterministic, auditable train/validation/test assignment."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
from pathlib import Path

from georag.data.core import TileRecord

SPLIT_NAMES = ("train", "validation", "test")
GroupKey = Callable[[TileRecord], str]


def create_split_assignments(
    records: Sequence[TileRecord],
    fractions: tuple[float, float, float],
    seed: int,
    stratify_by_label: bool = True,
) -> dict[str, str]:
    _validate_split_inputs(records, fractions)

    groups: dict[str, list[TileRecord]] = defaultdict(list)
    for record in records:
        group = record.labels[0] if stratify_by_label and record.labels else "__all__"
        groups[group].append(record)

    assignments: dict[str, str] = {}
    for group_records in groups.values():
        ordered = sorted(group_records, key=lambda record: _stable_key(seed, record.tile_id))
        train_end = round(len(ordered) * fractions[0])
        validation_end = train_end + round(len(ordered) * fractions[1])
        for index, record in enumerate(ordered):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            assignments[record.tile_id] = split
    return assignments


def create_grouped_split_assignments(
    records: Sequence[TileRecord],
    group_for_record: GroupKey,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits while approximately balancing record counts.

    Groups are processed largest-first; ties are broken by a seeded SHA-256 key.
    Each group is assigned to the split with the largest remaining record-count
    deficit. The procedure is deterministic and independent of record order.
    """
    _validate_split_inputs(records, fractions)
    grouped: dict[str, list[TileRecord]] = defaultdict(list)
    for record in records:
        group_id = group_for_record(record)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"group ID must be a non-empty string for tile {record.tile_id}")
        grouped[group_id].append(record)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("at least three distinct groups are required for train/validation/test splits")

    targets = {name: len(records) * fraction for name, fraction in zip(SPLIT_NAMES, fractions, strict=True)}
    counts = {name: 0 for name in SPLIT_NAMES}
    group_assignments: dict[str, str] = {}
    ordered_groups = sorted(
        grouped,
        key=lambda group_id: (-len(grouped[group_id]), _stable_key(seed, group_id)),
    )
    for index, group_id in enumerate(ordered_groups):
        used_splits = set(group_assignments.values())
        empty_splits = [name for name in SPLIT_NAMES if name not in used_splits]
        remaining_groups = len(ordered_groups) - index
        candidates = empty_splits if remaining_groups == len(empty_splits) else list(SPLIT_NAMES)
        group_size = len(grouped[group_id])
        split = max(candidates, key=lambda name: (targets[name] - counts[name], -SPLIT_NAMES.index(name)))
        group_assignments[group_id] = split
        counts[split] += group_size

    return {
        record.tile_id: group_assignments[group_for_record(record)]
        for record in records
    }


def write_group_split_manifest(
    path: str | Path,
    records: Sequence[TileRecord],
    assignments: Mapping[str, str],
    group_for_record: GroupKey,
) -> Path:
    """Write assignments with the exact group ID and asset status for auditing."""
    _validate_assignments(records, assignments)
    groups = {record.tile_id: group_for_record(record) for record in records}
    _validate_group_assignments(assignments, groups)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_by_id = {record.tile_id: record for record in records}
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["tile_id", "group_id", "split", "label", "asset_status"],
        )
        writer.writeheader()
        for tile_id in sorted(assignments):
            record = record_by_id[tile_id]
            writer.writerow({
                "tile_id": tile_id,
                "group_id": groups[tile_id],
                "split": assignments[tile_id],
                "label": record.labels[0] if record.labels else "",
                "asset_status": str((record.metadata or {}).get("asset_status", "available")),
            })
    return destination


def load_group_split_manifest(
    path: str | Path,
    records: Sequence[TileRecord],
    group_for_record: GroupKey,
) -> dict[str, str]:
    """Load a group manifest and verify its IDs and group membership."""
    assignments: dict[str, str] = {}
    manifest_groups: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_columns = ["tile_id", "group_id", "split", "label", "asset_status"]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"group split manifest must have columns: {', '.join(expected_columns)}")
        for row in reader:
            tile_id = row["tile_id"]
            if tile_id in assignments:
                raise ValueError(f"duplicate tile ID in split manifest: {tile_id}")
            assignments[tile_id] = row["split"]
            manifest_groups[tile_id] = row["group_id"]
    _validate_assignments(records, assignments)
    actual_groups = {record.tile_id: group_for_record(record) for record in records}
    if manifest_groups != actual_groups:
        raise ValueError("split manifest group IDs do not match the current records")
    _validate_group_assignments(assignments, manifest_groups)
    return assignments


def write_split_manifest(
    path: str | Path,
    records: Sequence[TileRecord],
    assignments: Mapping[str, str],
) -> Path:
    _validate_assignments(records, assignments)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_by_id = {record.tile_id: record for record in records}
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tile_id", "split", "label"])
        writer.writeheader()
        for tile_id in sorted(assignments):
            record = record_by_id[tile_id]
            writer.writerow({
                "tile_id": tile_id,
                "split": assignments[tile_id],
                "label": record.labels[0] if record.labels else "",
            })
    return destination


def load_split_manifest(path: str | Path, records: Sequence[TileRecord]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["tile_id", "split", "label"]:
            raise ValueError("split manifest must have columns: tile_id, split, label")
        for row in reader:
            tile_id = row["tile_id"]
            if tile_id in assignments:
                raise ValueError(f"duplicate tile ID in split manifest: {tile_id}")
            assignments[tile_id] = row["split"]
    _validate_assignments(records, assignments)
    return assignments


def indices_for_split(
    records: Sequence[TileRecord],
    assignments: Mapping[str, str],
    split: str,
    allow_extra_assignments: bool = False,
) -> list[int]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    if allow_extra_assignments:
        record_ids = {record.tile_id for record in records}
        missing = sorted(record_ids - set(assignments))
        if missing:
            raise ValueError(f"split assignment missing requested record IDs: {missing[:5]}")
        invalid = sorted({value for value in assignments.values() if value not in SPLIT_NAMES})
        if invalid:
            raise ValueError(f"invalid split names: {invalid}")
    else:
        _validate_assignments(records, assignments)
    return [index for index, record in enumerate(records) if assignments[record.tile_id] == split]


def _stable_key(seed: int, tile_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{tile_id}".encode()).digest()


def _validate_split_inputs(records: Sequence[TileRecord], fractions: tuple[float, float, float]) -> None:
    if not records:
        raise ValueError("cannot split an empty record sequence")
    if any(value <= 0 or value >= 1 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must be positive and sum to one")
    tile_ids = [record.tile_id for record in records]
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("tile IDs must be unique before splitting")


def _validate_group_assignments(assignments: Mapping[str, str], groups: Mapping[str, str]) -> None:
    if set(assignments) != set(groups):
        raise ValueError("split assignments and group IDs must contain identical tile IDs")
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for tile_id, group_id in groups.items():
        split_by_group[group_id].add(assignments[tile_id])
    crossing = sorted(group_id for group_id, splits in split_by_group.items() if len(splits) > 1)
    if crossing:
        raise ValueError(f"groups cross split boundaries: {crossing[:5]}")


def _validate_assignments(records: Sequence[TileRecord], assignments: Mapping[str, str]) -> None:
    record_ids = {record.tile_id for record in records}
    assigned_ids = set(assignments)
    if record_ids != assigned_ids:
        missing = sorted(record_ids - assigned_ids)[:5]
        unknown = sorted(assigned_ids - record_ids)[:5]
        raise ValueError(f"split manifest ID mismatch; missing={missing}, unknown={unknown}")
    invalid = sorted({split for split in assignments.values() if split not in SPLIT_NAMES})
    if invalid:
        raise ValueError(f"invalid split names: {invalid}")
