"""Aligned, portable storage for dense EO embedding corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from georag.data.core import TileRecord
from georag.data.splits import SPLIT_NAMES

EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingArtifact:
    """An embedding matrix and the row-aligned metadata needed to interpret it."""

    embeddings: np.ndarray
    records: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def write_embedding_artifact(
    destination: str | Path,
    embeddings: np.ndarray,
    records: Sequence[TileRecord],
    assignments: Mapping[str, str],
    dataset_root: str | Path,
    manifest: Mapping[str, Any],
    config_text: str,
) -> Path:
    """Atomically write a float32 matrix and JSONL records in identical row order."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"embedding artifact already exists; refusing to overwrite: {output}")
    matrix = np.asarray(embeddings)
    if matrix.dtype != np.float32:
        raise ValueError(f"embeddings must be float32, received {matrix.dtype}")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("embeddings must have shape [N, d] with positive d")
    if len(matrix) != len(records) or len(matrix) == 0:
        raise ValueError("embedding rows and records must have the same positive count")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("embedding rows must be L2-normalized")

    tile_ids = [record.tile_id for record in records]
    if len(set(tile_ids)) != len(tile_ids):
        raise ValueError("tile IDs must be unique")
    if set(assignments) != set(tile_ids):
        raise ValueError("split assignments must cover exactly the artifact tile IDs")
    invalid_splits = sorted(set(assignments.values()) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ValueError(f"invalid split assignments: {invalid_splits}")

    root = Path(dataset_root).resolve()
    record_rows: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        try:
            relative_path = record.image_path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"tile image path is outside the dataset root: {record.image_path}") from error
        record_rows.append(
            {
                "row": row,
                "tile_id": record.tile_id,
                "split": assignments[record.tile_id],
                "relative_path": relative_path,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "timestamp": record.timestamp.isoformat() if record.timestamp else None,
                "sensor": record.sensor,
                "bands": list(record.bands),
                "spatial_resolution_m": list(record.spatial_resolution_m)
                if record.spatial_resolution_m else None,
                "parent_scene": record.parent_scene,
                "labels": list(record.labels),
                "metadata": _json_compatible(record.metadata or {}),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        np.save(temporary / "embeddings.npy", matrix, allow_pickle=False)
        with (temporary / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record_row in record_rows:
                stream.write(json.dumps(record_row, sort_keys=True, allow_nan=False) + "\n")

        artifact_manifest = dict(manifest)
        artifact_manifest.update(
            schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            tile_count=len(matrix),
            embedding_dimension=int(matrix.shape[1]),
            embedding_dtype="float32",
            embedding_shape=list(matrix.shape),
            normalization="L2 unit norm",
            split_counts={
                split: sum(assignment == split for assignment in assignments.values())
                for split in SPLIT_NAMES
            },
            records_file="records.jsonl",
            embeddings_file="embeddings.npy",
        )
        with (temporary / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(artifact_manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        (temporary / "config.toml").write_text(config_text, encoding="utf-8", newline="\n")
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_embedding_artifact(directory: str | Path) -> EmbeddingArtifact:
    """Load and validate vector-to-record alignment without allowing pickle data."""

    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EMBEDDING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported embedding artifact schema version")
    matrix = np.load(root / manifest["embeddings_file"], allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise ValueError("embedding matrix must be a two-dimensional float32 array")
    if list(matrix.shape) != manifest.get("embedding_shape"):
        raise ValueError("embedding matrix shape does not match its manifest")
    if matrix.shape[1] != manifest.get("embedding_dimension"):
        raise ValueError("embedding matrix dimension does not match its manifest")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("embedding matrix contains rows that are not L2-normalized")

    records: list[dict[str, Any]] = []
    with (root / manifest["records_file"]).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row") != len(records):
                raise ValueError(f"record row is out of order at records.jsonl:{line_number}")
            if row.get("split") not in SPLIT_NAMES:
                raise ValueError(f"invalid split at records.jsonl:{line_number}")
            records.append(row)
    if len(records) != len(matrix) or len(records) != manifest.get("tile_count"):
        raise ValueError("embedding matrix, records, and manifest tile counts differ")
    tile_ids = [record.get("tile_id") for record in records]
    if any(not tile_id for tile_id in tile_ids) or len(set(tile_ids)) != len(tile_ids):
        raise ValueError("records must contain unique, non-empty tile IDs")
    actual_split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in SPLIT_NAMES
    }
    if actual_split_counts != manifest.get("split_counts"):
        raise ValueError("record split counts do not match the manifest")
    return EmbeddingArtifact(matrix, tuple(records), manifest)


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"unsupported metadata value for JSON artifact: {type(value).__name__}")
