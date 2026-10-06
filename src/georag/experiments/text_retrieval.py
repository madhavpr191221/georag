"""Build an RGB RemoteCLIP gallery and run exact natural-language retrieval."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from georag.data import AgricultureVisionDataset, TileRecord, TileSample
from georag.evaluation.text_retrieval import evaluate_text_retrieval
from georag.models.remoteclip import (
    REMOTECLIP_ARCHITECTURE,
    REMOTECLIP_CHECKPOINT,
    REMOTECLIP_REPO,
    RemoteCLIPEncoder,
)
from georag.retrieval.flat import SearchResult
from georag.visualization import save_text_retrieval_grid


class _PreprocessedImages(Dataset[tuple[torch.Tensor, int]]):
    """Read the original RGB file and apply RemoteCLIP's official transform."""

    def __init__(self, records: Sequence[TileRecord], preprocess) -> None:
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
            tensor = self.preprocess(image)
        return tensor, index


def run_text_retrieval(
    config_path: str | Path,
    device: str = "auto",
    max_gallery: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run configured text-to-image retrieval and save reproducible evidence."""

    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    query_path = (root / config["queries"]["path"]).resolve()
    with query_path.open("rb") as source:
        query_config = tomllib.load(source)
    queries = query_config.get("queries", [])
    if not queries:
        raise ValueError(f"query file has no [[queries]] entries: {query_path}")

    dataset_cfg = config["dataset"]
    cache_dir = _resolve(root, dataset_cfg["catalog_cache_dir"])
    dataset = AgricultureVisionDataset(
        root=_resolve(root, dataset_cfg["root"]),
        split="train",
        bands=("R", "G", "B"),
        image_size=int(dataset_cfg["display_image_size"]),
        value_scale=float(dataset_cfg.get("value_scale", 255.0)),
        catalog_cache_dir=cache_dir,
        index_workers=int(dataset_cfg.get("index_workers", 8)),
    )
    records = list(dataset.records)
    if max_gallery is not None:
        if max_gallery <= 0:
            raise ValueError("max_gallery must be positive when provided")
        records = records[:max_gallery]
    if not records:
        raise ValueError("the RGB training gallery is empty")

    selected_device = _select_device(device)
    output_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _resolve(root, config["outputs"]["experiment_dir"])
    )
    if max_gallery is not None and output_dir is None:
        output_root = output_root.with_name(f"{output_root.name}_smoke_{max_gallery}")
    output_root.mkdir(parents=True, exist_ok=True)

    model_cfg = config["model"]
    expected_model = {
        "name": "RemoteCLIP",
        "architecture": REMOTECLIP_ARCHITECTURE,
        "repository": REMOTECLIP_REPO,
        "checkpoint": REMOTECLIP_CHECKPOINT,
    }
    mismatches = [
        key for key, expected in expected_model.items()
        if model_cfg.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "this M7 runner supports only the configured RemoteCLIP ViT-B/32 "
            f"checkpoint; mismatched model settings: {mismatches}"
        )
    weights_cache = _resolve(root, model_cfg["cache_dir"])
    encoder = RemoteCLIPEncoder(
        device=selected_device,
        cache_dir=weights_cache,
        revision=str(model_cfg.get("revision", "main")),
    )
    model_metadata = {
        "model": "RemoteCLIP",
        "architecture": REMOTECLIP_ARCHITECTURE,
        "repository": REMOTECLIP_REPO,
        "checkpoint_file": REMOTECLIP_CHECKPOINT,
        "repository_revision": encoder.revision,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "checkpoint_path": str(encoder.checkpoint_path),
        "embedding_dimension": encoder.embedding_dim,
        "bands": ["R", "G", "B"],
        "image_transform": "OpenCLIP ViT-B-32 model-provided transform",
    }

    artifact_root = _resolve(root, config["outputs"]["artifact_dir"])
    if max_gallery is not None:
        artifact_root = artifact_root / f"smoke_{max_gallery}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    embedding_path = artifact_root / "gallery_embeddings.npy"
    records_path = artifact_root / "gallery_records.json"
    manifest_path = artifact_root / "manifest.json"
    fingerprint = _gallery_fingerprint(records, dataset.root)
    embeddings, gallery_records, cache_reused = _load_gallery_cache(
        embedding_path, records_path, manifest_path,
        fingerprint=fingerprint,
        checkpoint_sha256=encoder.checkpoint_sha256,
        expected_count=len(records),
    )
    encoding_seconds = 0.0
    if embeddings is None:
        started = time.perf_counter()
        embeddings = _encode_gallery(
            records,
            encoder,
            batch_size=int(config["inference"]["batch_size"]),
            num_workers=int(config["inference"]["num_workers"]),
            device=selected_device,
        )
        encoding_seconds = time.perf_counter() - started
        gallery_records = [_record_payload(record, dataset.root) for record in records]
        _write_gallery_cache(
            embedding_path,
            records_path,
            manifest_path,
            embeddings,
            gallery_records,
            {
                "schema_version": 1,
                "dataset": "Agriculture-Vision 2021 official train split",
                "dataset_root": str(dataset.root),
                "gallery_fingerprint": fingerprint,
                "gallery_count": len(records),
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": "float32",
                "model": model_metadata,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    assert embeddings is not None and gallery_records is not None

    texts = [str(query["text"]) for query in queries]
    query_embeddings = encoder.encode_text(
        texts, batch_size=int(config["inference"].get("text_batch_size", 64))
    )
    evaluated = evaluate_text_retrieval(
        embeddings,
        gallery_records,
        query_embeddings,
        queries,
        k=min(int(config["retrieval"]["k"]), len(records)),
    )
    ranked = evaluated.pop("ranked_results")

    index_by_id = {record.tile_id: i for i, record in enumerate(dataset.records)}
    query_payload: list[dict[str, Any]] = []
    for query in queries:
        results: list[SearchResult] = ranked[query["id"]]
        result_rows = [
            {
                "rank": result.rank,
                "tile_id": result.tile_id,
                "score": result.score,
                "record": dict(result.record),
            }
            for result in results
        ]
        query_payload.append(
            {
                "query_id": query["id"],
                "text": query["text"],
                "target_label": query["category"],
                "canonical": bool(query.get("audit", False)),
                "metrics": evaluated["queries"][query["id"]],
                "results": result_rows,
            }
        )
        if query.get("audit", False):
            samples = [
                (dataset[index_by_id[result.tile_id]], result.rank, result.score)
                for result in results[: int(config["retrieval"]["audit_k"])]
            ]
            save_text_retrieval_grid(
                str(query["text"]),
                samples,
                output_root / "retrieval_grids" / f"{query['id']}.png",
                target_label=str(query["category"]),
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(selected_device),
        "dataset": "Agriculture-Vision 2021 official train split",
        "gallery_count": len(records),
        "query_count": len(queries),
        "k": evaluated["k"],
        "retrieval_method": "exact_flat_numpy_cosine",
        "cache_reused": cache_reused,
        "gallery_encoding_seconds": encoding_seconds,
        "model": model_metadata,
        "metrics": evaluated,
        "queries": query_payload,
        "artifacts": {
            "gallery_embeddings": str(embedding_path),
            "gallery_records": str(records_path),
            "gallery_manifest": str(manifest_path),
        },
        "human_audit_status": "pending; complete the generated human_audit.csv",
    }
    _write_json_atomic(output_root / "results.json", summary)
    _write_human_audit(output_root / "human_audit.csv", query_payload)
    _write_audit_instructions(output_root / "HUMAN_AUDIT.md")
    print(
        f"RemoteCLIP exact retrieval complete: {len(records):,} RGB gallery tiles, "
        f"{len(queries)} text prompts, macro mAP@{evaluated['k']}="
        f"{evaluated['macro']['map_at_k']}, cache_reused={cache_reused}",
        flush=True,
    )
    return summary


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def _record_payload(record: TileRecord, dataset_root: Path) -> dict[str, Any]:
    return {
        "tile_id": record.tile_id,
        "image_path": record.image_path.relative_to(dataset_root).as_posix(),
        "parent_scene": record.parent_scene,
        "labels": list(record.labels),
        "metadata": {
            "field_id": (record.metadata or {}).get("field_id", record.parent_scene),
            "split": "train",
        },
    }


def _gallery_fingerprint(records: Sequence[TileRecord], dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for record in records:
        path = record.image_path
        stat = path.stat()
        relative_path = path.relative_to(dataset_root).as_posix()
        labels = ",".join(sorted(record.labels))
        digest.update(
            f"{record.tile_id}\0{relative_path}\0{stat.st_size}\0{stat.st_mtime_ns}\0{labels}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _load_gallery_cache(
    embedding_path: Path,
    records_path: Path,
    manifest_path: Path,
    fingerprint: str,
    checkpoint_sha256: str,
    expected_count: int,
) -> tuple[np.ndarray | None, list[dict[str, Any]] | None, bool]:
    if not all(path.is_file() for path in (embedding_path, records_path, manifest_path)):
        return None, None, False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_path, mmap_mode="r")
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None, False
    if (
        manifest.get("gallery_fingerprint") != fingerprint
        or manifest.get("model", {}).get("checkpoint_sha256") != checkpoint_sha256
        or embeddings.shape[0] != expected_count
        or len(records) != expected_count
        or embeddings.ndim != 2
        or not np.isfinite(embeddings).all()
    ):
        return None, None, False
    return np.asarray(embeddings, dtype=np.float32), records, True


def _encode_gallery(
    records: Sequence[TileRecord],
    encoder: RemoteCLIPEncoder,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    loader = DataLoader(
        _PreprocessedImages(records, encoder.preprocess),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    batches: list[np.ndarray] = []
    encoded = 0
    for images, _indices in loader:
        batch = encoder.encode_images(images)
        batches.append(batch)
        encoded += len(batch)
        if encoded % 4096 < len(batch) or encoded == len(records):
            print(f"  encoded {encoded:,}/{len(records):,} RGB gallery tiles", flush=True)
    if encoded != len(records):
        raise RuntimeError(f"gallery row mismatch: encoded {encoded}, expected {len(records)}")
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def _write_gallery_cache(
    embedding_path: Path,
    records_path: Path,
    manifest_path: Path,
    embeddings: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    _write_numpy_atomic(embedding_path, embeddings)
    _write_json_atomic(records_path, list(records))
    _write_json_atomic(manifest_path, dict(manifest))


def _write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            np.save(destination, array, allow_pickle=False)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as destination:
            json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_human_audit(output_path: Path, queries: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "query_id", "query_text", "target_label", "rank", "tile_id", "score",
        "labels", "field_id", "image_path", "human_relevance", "notes",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for query in queries:
            if not query["canonical"]:
                continue
            for result in query["results"][:5]:
                record = result["record"]
                writer.writerow(
                    {
                        "query_id": query["query_id"],
                        "query_text": query["text"],
                        "target_label": query["target_label"],
                        "rank": result["rank"],
                        "tile_id": result["tile_id"],
                        "score": f"{result['score']:.7f}",
                        "labels": ",".join(record.get("labels", [])),
                        "field_id": (record.get("metadata") or {}).get("field_id", ""),
                        "image_path": record["image_path"],
                        "human_relevance": "",
                        "notes": "",
                    }
                )


def _write_audit_instructions(path: Path) -> None:
    path.write_text(
        "# M7 human relevance audit\n\n"
        "Review each retrieved RGB tile against its natural-language query. "
        "In `human_audit.csv`, fill `human_relevance` with `relevant`, `uncertain`, "
        "or `not_relevant`, and optionally add a short note. The target annotation "
        "label is shown for orientation, but judge what the image visibly supports. "
        "This is a small qualitative audit, not a substitute for the repeatable "
        "annotation-proxy metrics. For an automated second opinion, run "
        "`uv run python scripts/run_vlm_audit.py`; those outputs are VLM judgments, "
        "not human labels, and are saved separately.\n",
        encoding="utf-8",
    )
