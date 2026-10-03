"""Deterministic, auditable train/validation/test assignment."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path

from georag.data.core import TileRecord

SPLIT_NAMES = ("train", "validation", "test")


def create_split_assignments(
    records: Sequence[TileRecord],
    fractions: tuple[float, float, float],
    seed: int,
    stratify_by_label: bool = True,
) -> dict[str, str]:
    if not records:
        raise ValueError("cannot split an empty record sequence")
    if any(value <= 0 or value >= 1 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must be positive and sum to one")
    tile_ids = [record.tile_id for record in records]
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("tile IDs must be unique before splitting")

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
    records: Sequence[TileRecord], assignments: Mapping[str, str], split: str
) -> list[int]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    _validate_assignments(records, assignments)
    return [index for index, record in enumerate(records) if assignments[record.tile_id] == split]


def _stable_key(seed: int, tile_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{tile_id}".encode()).digest()


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
