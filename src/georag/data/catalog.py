"""Validated, relocatable metadata catalogs for EO tile collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from georag.data.core import TileRecord

CATALOG_SCHEMA_VERSION = 1


class CatalogStaleError(ValueError):
    """Raised when a catalog no longer describes the files under its dataset root."""


@dataclass(frozen=True)
class InventoryEntry:
    relative_path: str
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class DatasetInventory:
    entries: tuple[InventoryEntry, ...]
    fingerprint: str
    total_bytes: int


def build_tiff_inventory(root: Path) -> DatasetInventory:
    """Inventory TIFF paths without opening raster contents or headers."""

    entries: list[InventoryEntry] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() in {".tif", ".tiff"}),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        stat = path.stat()
        relative_path = path.relative_to(root).as_posix()
        entry = InventoryEntry(relative_path, stat.st_size, stat.st_mtime_ns)
        entries.append(entry)
        total_bytes += entry.size_bytes
        digest.update(f"{relative_path}\0{entry.size_bytes}\0{entry.modified_ns}\n".encode("utf-8"))
    return DatasetInventory(tuple(entries), digest.hexdigest(), total_bytes)


def write_catalog(
    path: Path,
    root: Path,
    records: tuple[TileRecord, ...],
    inventory: DatasetInventory,
    bands: tuple[str, ...],
    expected_shape: tuple[int, int, int],
) -> Path:
    """Atomically write a JSONL catalog whose paths are relative to the dataset root."""

    if len(records) != len(inventory.entries):
        raise ValueError("catalog records and source inventory must have equal lengths")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    header = {
        "type": "catalog",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "dataset": "eurosat_ms",
        "source_fingerprint": inventory.fingerprint,
        "tile_count": len(records),
        "total_bytes": inventory.total_bytes,
        "bands": list(bands),
        "expected_shape": list(expected_shape),
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
            for record in records:
                payload = {
                    "type": "tile",
                    "tile_id": record.tile_id,
                    "relative_path": record.image_path.relative_to(root).as_posix(),
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                    "sensor": record.sensor,
                    "bands": list(record.bands),
                    "spatial_resolution_m": list(record.spatial_resolution_m) if record.spatial_resolution_m else None,
                    "parent_scene": record.parent_scene,
                    "labels": list(record.labels),
                    "metadata": dict(record.metadata or {}),
                }
                stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_catalog(
    path: Path,
    root: Path,
    inventory: DatasetInventory,
    bands: tuple[str, ...],
    expected_shape: tuple[int, int, int],
) -> tuple[tuple[TileRecord, ...], dict[str, Any]]:
    """Load a catalog only when its schema and source fingerprint still match."""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        first_line = stream.readline()
        if not first_line:
            raise ValueError(f"catalog is empty: {source}")
        header = json.loads(first_line)
        _validate_header(header, inventory, bands, expected_shape)
        records: list[TileRecord] = []
        for line_number, line in enumerate(stream, start=2):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") != "tile":
                raise ValueError(f"catalog line {line_number} is not a tile record")
            relative_path = Path(payload["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe relative path in catalog line {line_number}")
            metadata = dict(payload.get("metadata") or {})
            if isinstance(metadata.get("bounds"), list):
                metadata["bounds"] = tuple(metadata["bounds"])
            records.append(
                TileRecord(
                    tile_id=str(payload["tile_id"]),
                    image_path=root / relative_path,
                    latitude=_optional_float(payload.get("latitude")),
                    longitude=_optional_float(payload.get("longitude")),
                    timestamp=datetime.fromisoformat(payload["timestamp"]) if payload.get("timestamp") else None,
                    sensor=payload.get("sensor"),
                    bands=tuple(payload.get("bands") or ()),
                    spatial_resolution_m=(
                        tuple(float(value) for value in payload["spatial_resolution_m"])
                        if payload.get("spatial_resolution_m") else None
                    ),
                    parent_scene=payload.get("parent_scene"),
                    labels=tuple(payload.get("labels") or ()),
                    metadata=metadata,
                )
            )
    if len(records) != int(header["tile_count"]):
        raise ValueError(
            f"catalog header declares {header['tile_count']} tiles but contains {len(records)}"
        )
    return tuple(records), header


def _validate_header(
    header: dict[str, Any],
    inventory: DatasetInventory,
    bands: tuple[str, ...],
    expected_shape: tuple[int, int, int],
) -> None:
    if header.get("type") != "catalog" or header.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported or malformed dataset catalog header")
    if header.get("dataset") != "eurosat_ms":
        raise ValueError("catalog does not describe the EuroSAT multispectral dataset")
    if header.get("source_fingerprint") != inventory.fingerprint:
        raise CatalogStaleError("dataset files changed after catalog creation; rebuild the catalog")
    if int(header.get("tile_count", -1)) != len(inventory.entries):
        raise CatalogStaleError("dataset file count changed after catalog creation")
    if tuple(header.get("bands") or ()) != bands:
        raise CatalogStaleError("configured band order differs from the cached catalog")
    if tuple(header.get("expected_shape") or ()) != expected_shape:
        raise CatalogStaleError("configured tile shape differs from the cached catalog")


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
