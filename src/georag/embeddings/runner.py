"""Batch inference and reproducible artifact creation for Milestone 4."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch

from georag.config import ProjectConfig
from georag.data import EuroSATMultispectralDataset
from georag.data.splits import load_split_manifest
from georag.embeddings.corpus import write_embedding_artifact
from georag.models import build_encoder, load_band_statistics
from georag.training.evaluation import embed_indices


def run_embedding_corpus(
    config: ProjectConfig, config_path: str | Path, device: torch.device
) -> dict[str, str]:
    """Encode every catalog tile with the configured completed best checkpoints."""

    if config.model is None or config.embedding is None:
        raise ValueError("Milestone 4 requires both [model] and [embedding] config sections")
    if config.reproducibility.seed != config.split.seed:
        raise ValueError("reproducibility.seed must match split.seed for this corpus run")

    dataset = EuroSATMultispectralDataset(
        root=config.dataset.root,
        bands=config.dataset.bands,
        expected_shape=config.dataset.expected_shape,
        expected_count=config.dataset.expected_count,
        value_scale=config.dataset.value_scale,
        catalog_path=config.dataset.catalog,
    )
    assignments = load_split_manifest(config.split.manifest, dataset.records)
    means, standard_deviations = load_band_statistics(
        config.model.statistics_csv, config.dataset.bands, config.dataset.value_scale
    )
    config_file = Path(config_path).resolve()
    config_text = config_file.read_text(encoding="utf-8")
    config_hash = _sha256_file(config_file)
    split_hash = _sha256_file(config.split.manifest)
    statistics_hash = _sha256_file(config.model.statistics_csv)
    checkpoint_root = config.outputs.experiments_dir / "milestone_3"
    artifact_root = config.outputs.artifacts_dir / "milestone_4"
    destinations = {
        name: artifact_root / f"{name}_seed_{config.reproducibility.seed}"
        for name in config.embedding.models
    }
    existing_outputs = [path for path in destinations.values() if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "embedding outputs already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    output_paths: dict[str, str] = {}
    indices = list(range(len(dataset)))

    for model_name in config.embedding.models:
        run_directory = checkpoint_root / f"{model_name}_seed_{config.reproducibility.seed}"
        checkpoint_path = run_directory / config.embedding.checkpoint_filename
        run_metadata_path = run_directory / "run.json"
        run_config_path = run_directory / "config.toml"
        if not checkpoint_path.is_file() or not run_metadata_path.is_file() or not run_config_path.is_file():
            raise FileNotFoundError(f"completed {model_name} checkpoint run is incomplete: {run_directory}")
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if run_metadata.get("status") != "completed":
            raise ValueError(f"{model_name} training run is not marked completed")
        _validate_training_config(run_config_path, config, model_name)
        if run_metadata.get("dataset_inventory_fingerprint") != dataset.catalog_metadata.get("source_fingerprint"):
            raise ValueError(f"{model_name} checkpoint was trained against a different dataset inventory")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != model_name:
            raise ValueError(f"checkpoint at {checkpoint_path} contains a different model name")
        if checkpoint.get("seed") != config.reproducibility.seed:
            raise ValueError(f"checkpoint at {checkpoint_path} contains a different random seed")
        if checkpoint.get("epoch") != run_metadata.get("best_epoch"):
            raise ValueError(f"{model_name} best checkpoint epoch does not match run metadata")

        model = build_encoder(
            model_name,
            config.model,
            (config.dataset.expected_shape[1], config.dataset.expected_shape[2]),
            means,
            standard_deviations,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        loaded_mean = model.normalizer.mean.detach().cpu().reshape(-1)
        loaded_std = model.normalizer.standard_deviation.detach().cpu().reshape(-1)
        if not torch.allclose(loaded_mean, torch.tensor(means), rtol=1e-6, atol=1e-6):
            raise ValueError(f"{model_name} checkpoint band means differ from current statistics")
        if not torch.allclose(loaded_std, torch.tensor(standard_deviations), rtol=1e-6, atol=1e-6):
            raise ValueError(f"{model_name} checkpoint band standard deviations differ from current statistics")
        model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        embeddings, labels, tile_ids = embed_indices(
            model,
            dataset,
            indices,
            device,
            config.embedding.batch_size,
            config.loader.num_workers,
            progress_every_batches=50,
        )
        expected_ids = [record.tile_id for record in dataset.records]
        if tile_ids != expected_ids:
            raise RuntimeError(f"{model_name} embedding rows do not preserve catalog order")
        expected_labels = [record.labels[0] if record.labels else "" for record in dataset.records]
        if labels != expected_labels:
            raise RuntimeError(f"{model_name} embedding labels do not preserve catalog order")
        if embeddings.shape != (len(dataset.records), config.model.embedding_dim):
            raise RuntimeError(f"{model_name} produced an unexpected embedding shape {tuple(embeddings.shape)}")
        matrix = embeddings.numpy().astype(np.float32, copy=False)

        peak_cuda_bytes: int | None = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
        model_parameters = asdict(config.model)
        model_parameters["statistics_csv"] = str(config.model.statistics_csv.relative_to(config.root))
        manifest: dict[str, Any] = {
            "milestone": 4,
            "model_name": model_name,
            "random_seed": config.reproducibility.seed,
            "model_config": model_parameters,
            "checkpoint": {
                "path": checkpoint_path.relative_to(config.root).as_posix(),
                "sha256": _sha256_file(checkpoint_path),
                "epoch": int(checkpoint["epoch"]),
                "validation_loss": float(checkpoint["validation_loss"]),
            },
            "dataset": {
                "adapter": config.dataset.adapter,
                "tile_count": len(dataset.records),
                "shape_chw": list(config.dataset.expected_shape),
                "bands": list(config.dataset.bands),
                "value_scale": config.dataset.value_scale,
                "inventory_fingerprint": dataset.catalog_metadata.get("source_fingerprint"),
            },
            "split": {
                "path": config.split.manifest.relative_to(config.root).as_posix(),
                "sha256": split_hash,
                "counts": dict(sorted(Counter(assignments.values()).items())),
            },
            "normalization": {
                "method": "training-split per-band z-score inside encoder",
                "statistics_path": config.model.statistics_csv.relative_to(config.root).as_posix(),
                "statistics_sha256": statistics_hash,
                "augmentation": None,
            },
            "run": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": config_file.relative_to(config.root).as_posix(),
                "config_sha256": config_hash,
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "batch_size": config.embedding.batch_size,
                "data_loader_workers": config.loader.num_workers,
            },
        }
        destination = destinations[model_name]
        written_path = write_embedding_artifact(
            destination,
            matrix,
            dataset.records,
            assignments,
            config.dataset.root,
            manifest,
            config_text,
        )
        output_paths[model_name] = str(written_path)
        print(f"Wrote {model_name.upper()} embeddings: {written_path}", flush=True)

    return output_paths


def _validate_training_config(path: Path, config: ProjectConfig, model_name: str) -> None:
    """Ensure the checkpoint's recorded architecture and input scale match M4."""

    import tomllib

    with path.open("rb") as stream:
        training_config = tomllib.load(stream)
    old_dataset = training_config["dataset"]
    current_dataset = config.dataset
    if (
        tuple(old_dataset["expected_shape"]) != current_dataset.expected_shape
        or tuple(old_dataset["bands"]) != current_dataset.bands
        or float(old_dataset["value_scale"]) != current_dataset.value_scale
        or old_dataset["adapter"] != current_dataset.adapter
    ):
        raise ValueError(f"{model_name} checkpoint input configuration differs from M4")
    old_model = training_config["model"]
    if config.model is None:
        raise ValueError("M4 config requires a [model] section")
    model_keys = {
        "embedding_dim", "cnn_channels", "vit_patch_size", "vit_width",
        "vit_depth", "vit_heads", "vit_mlp_ratio",
    }
    current_model = asdict(config.model)
    for key in model_keys:
        old_value = old_model[key]
        current_value = current_model[key]
        if isinstance(old_value, list):
            old_value = tuple(old_value)
        if isinstance(current_value, list):
            current_value = tuple(current_value)
        if old_value != current_value:
            raise ValueError(f"{model_name} checkpoint model setting {key!r} differs from M4")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
