"""Matched self-supervised RGB versus RGB+NIR retrieval experiment (M6)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import tomllib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from torch.utils.data import DataLoader

from georag.data.agriculture_vision import AgricultureVisionDataset
from georag.models.encoders import CNNEncoder
from georag.training.augmentations import make_d4_views
from georag.training.contrastive import nt_xent_loss
from georag.evaluation.spectral_retrieval import evaluate_pattern_retrieval, paired_field_bootstrap_delta
from georag.diagnostics import collect_diagnostics

MODALITIES = {"rgb": ("R", "G", "B"), "rgbnir": ("R", "G", "B", "NIR")}


def run_spectral_retrieval(config_path: str | Path, device: torch.device, smoke_limit: int | None = None, resume: bool = False) -> Path:
    """Run matched CNN/NT-Xent experiments and save checkpoints, metrics, and retrieval evidence."""
    path = Path(config_path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    _validate_config(raw)
    dataset_cfg, train_cfg, model_cfg, experiment_cfg = (raw[key] for key in ("dataset", "training", "model", "experiment"))
    root = _resolve(path, dataset_cfg["root"])
    catalog_cache_dir = _resolve(path, dataset_cfg["catalog_cache_dir"])
    output_root = _resolve(path, experiment_cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    modalities = [name for name in experiment_cfg["modalities"]]
    seeds = [int(value) for value in experiment_cfg["seeds"]]
    all_results: dict[str, Any] = {}
    experiment_metadata: dict[str, Any] = {}
    # Index annotations once. RGB and RGB+NIR comparisons share tile identity,
    # labels, and splits; only the input channel view and its normalization differ.
    base_train = AgricultureVisionDataset(
        root, "train", MODALITIES["rgbnir"], int(dataset_cfg["image_size"]),
        float(dataset_cfg["value_scale"]), max_tiles=smoke_limit,
        catalog_cache_dir=catalog_cache_dir,
    )
    base_query = AgricultureVisionDataset(
        root, "validation", MODALITIES["rgbnir"], int(dataset_cfg["image_size"]),
        float(dataset_cfg["value_scale"]), max_tiles=smoke_limit,
        catalog_cache_dir=catalog_cache_dir,
    )
    statistics_batch_size = min(int(train_cfg["batch_size"]), max(2, len(base_train)))
    base_means, base_stds = _band_statistics(
        base_train, statistics_batch_size, int(train_cfg["num_workers"]), device
    )
    dataset_cache: dict[str, tuple[AgricultureVisionDataset, AgricultureVisionDataset, list[float], list[float]]] = {}

    for seed in seeds:
        for modality in modalities:
            if modality not in MODALITIES:
                raise ValueError(f"unknown modality {modality!r}; choose from {tuple(MODALITIES)}")
            _seed_everything(seed)
            bands = MODALITIES[modality]
            if modality not in dataset_cache:
                train_data = base_train.with_bands(bands)
                query_data = base_query.with_bands(bands)
                channel_indices = [base_train.bands.index(band) for band in bands]
                means = [base_means[index] for index in channel_indices]
                stds = [base_stds[index] for index in channel_indices]
                dataset_cache[modality] = train_data, query_data, means, stds
            train_data, query_data, means, stds = dataset_cache[modality]
            overlapping_fields = {r.parent_scene for r in train_data.records} & {r.parent_scene for r in query_data.records}
            if overlapping_fields:
                examples = sorted(overlapping_fields)[:5]
                raise ValueError(f"train and validation splits share Agriculture-Vision fields: {examples}")
            batch_size = int(train_cfg["batch_size"])
            if smoke_limit is not None:
                batch_size = min(batch_size, (len(train_data) // 2) * 2, (len(query_data) // 2) * 2)
                if batch_size < 2:
                    raise ValueError("smoke_limit must leave at least two tiles in each split")
            model = CNNEncoder(means, stds, int(model_cfg["embedding_dim"]), tuple(model_cfg["channels"])).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
            paired = _PairedViews(train_data, seed)
            validation_pairs = _PairedViews(query_data, seed + 100_000)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(paired, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=int(train_cfg["num_workers"]), pin_memory=device.type == "cuda", generator=generator)
            validation_loader = DataLoader(validation_pairs, batch_size=batch_size, shuffle=False,
                                            drop_last=True, num_workers=int(train_cfg["num_workers"]),
                                            pin_memory=device.type == "cuda")
            run_dir = output_root / f"{modality}_seed_{seed}"
            is_resume = run_dir.exists()
            if is_resume and not resume:
                raise FileExistsError(f"run directory exists; pass --resume to continue or choose another output: {run_dir}")
            if is_resume:
                saved_config = run_dir / "config.toml"
                if not saved_config.is_file():
                    raise ValueError(f"cannot resume because saved config differs: {run_dir}")
                try:
                    saved_settings = tomllib.loads(saved_config.read_text(encoding="utf-8"))
                except tomllib.TOMLDecodeError as error:
                    raise ValueError(f"cannot resume because saved config is invalid: {saved_config}") from error
                if saved_settings != raw:
                    raise ValueError(f"cannot resume because saved config differs: {run_dir}")
            else:
                run_dir.mkdir(parents=True)
                (run_dir / "config.toml").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            history: list[dict[str, float | int]] = []
            start_epoch = 0
            if is_resume and (run_dir / "run.json").is_file():
                existing = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                if existing.get("smoke_limit") != smoke_limit:
                    raise ValueError("cannot resume because --smoke-limit differs from the original run")
                if existing.get("status") == "completed":
                    metrics_path = run_dir / "retrieval_metrics.json"
                    if not metrics_path.is_file():
                        raise FileNotFoundError(f"completed run is missing retrieval metrics: {run_dir}")
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    all_results[f"{modality}_seed_{seed}"] = metrics
                    experiment_metadata[f"{modality}_seed_{seed}"] = existing
                    print(f"Skipping completed run: {run_dir}", flush=True)
                    continue
            if is_resume:
                checkpoint_path = run_dir / "checkpoint_last.pt"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"interrupted run has no latest checkpoint: {run_dir}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                if checkpoint.get("seed") != seed or checkpoint.get("modality") != modality:
                    raise ValueError("checkpoint seed or modality does not match current run")
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                history = checkpoint["history"]
                start_epoch = int(checkpoint["epoch"])
                generator.set_state(checkpoint["loader_generator_state"].cpu())
                _write_history(run_dir / "metrics.csv", history)
            else:
                _write_history(run_dir / "metrics.csv", history)
            run_metadata = {
                "status": "running", "modality": modality, "seed": seed, "bands": bands,
                "smoke_limit": smoke_limit,
                "train_tiles": len(train_data), "validation_tiles": len(query_data),
                "field_disjoint": not bool({r.parent_scene for r in train_data.records} & {r.parent_scene for r in query_data.records}),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "device": str(device), "torch_version": torch.__version__,
                "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "channel_mean": means, "channel_standard_deviation": stds,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epochs = int(train_cfg["epochs"])
            if smoke_limit is not None:
                epochs = min(epochs, 1)
            if start_epoch > epochs:
                raise ValueError("checkpoint exceeds the configured epoch count")
            for epoch in range(start_epoch, epochs):
                paired.epoch = epoch
                model.train()
                epoch_loss, anchors, started = 0.0, 0, time.perf_counter()
                for views_a, views_b in loader:
                    views = torch.cat((views_a, views_b), dim=0).to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    z_a, z_b = model(views).chunk(2, dim=0)
                    loss, _, _ = nt_xent_loss(z_a, z_b, float(train_cfg["temperature"]))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach()) * 2 * len(views_a)
                    anchors += 2 * len(views_a)
                if not anchors:
                    raise ValueError("training loader has no complete batches; increase the train split or lower batch_size")
                model.eval(); validation_loss, validation_anchors = 0.0, 0
                with torch.inference_mode():
                    for views_a, views_b in validation_loader:
                        views = torch.cat((views_a, views_b), dim=0).to(device, non_blocking=True)
                        z_a, z_b = model(views).chunk(2, dim=0)
                        loss, _, _ = nt_xent_loss(z_a, z_b, float(train_cfg["temperature"]))
                        validation_loss += float(loss) * 2 * len(views_a)
                        validation_anchors += 2 * len(views_a)
                if not validation_anchors:
                    raise ValueError("validation split must contain at least two tiles and one complete batch")
                history.append({"epoch": epoch + 1, "train_loss": epoch_loss / anchors,
                                "validation_loss": validation_loss / validation_anchors,
                                "epoch_seconds": time.perf_counter() - started})
                _write_history(run_dir / "metrics.csv", history)
                checkpoint = {"epoch": epoch + 1, "seed": seed, "modality": modality,
                              "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                              "optimizer_state_dict": optimizer.state_dict(), "history": history,
                              "loader_generator_state": generator.get_state(), "smoke_limit": smoke_limit}
                _atomic_torch_save(run_dir / "checkpoint_last.pt", checkpoint)
                run_metadata["last_completed_epoch"] = epoch + 1
                (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
                print(f"{modality.upper()} seed={seed} epoch={epoch + 1}/{epochs}: "
                      f"train={history[-1]['train_loss']:.4f}, val={history[-1]['validation_loss']:.4f}", flush=True)
            _plot_loss(history, run_dir / "loss_curve.png", modality, seed)
            torch.save({"state_dict": model.cpu().state_dict(), "seed": seed, "modality": modality,
                        "bands": bands, "mean": means, "standard_deviation": stds,
                        "embedding_dim": int(model_cfg["embedding_dim"])}, run_dir / "checkpoint.pt")
            model.to(device)
            gallery_x = _embed(model, train_data, int(train_cfg["batch_size"]), int(train_cfg["num_workers"]), device)
            query_x = _embed(model, query_data, int(train_cfg["batch_size"]), int(train_cfg["num_workers"]), device)
            gallery_records = [_record_json(record) for record in train_data.records]
            query_records = [_record_json(record) for record in query_data.records]
            evaluation_started = time.perf_counter()
            metrics = evaluate_pattern_retrieval(
                gallery_x, gallery_records, query_x, query_records,
                k=min(int(experiment_cfg["k"]), len(gallery_records)), minimum_queries=int(experiment_cfg["minimum_queries"]),
                bootstrap_samples=int(experiment_cfg["bootstrap_samples"]), seed=seed,
                query_chunk_size=int(experiment_cfg["query_chunk_size"]),
            )
            metrics["evaluation_seconds_including_bootstrap"] = time.perf_counter() - evaluation_started
            _plot_retrieval_examples(gallery_x, query_x, train_data.records, query_data.records,
                                     run_dir / "retrieval_examples.png", int(experiment_cfg["example_queries"]))
            np.savez_compressed(run_dir / "embeddings.npz", gallery=gallery_x, queries=query_x)
            (run_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            all_results[f"{modality}_seed_{seed}"] = metrics
            run_metadata.update({
                "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "train_tiles": len(train_data), "validation_tiles": len(query_data),
                "field_disjoint": not bool({r.parent_scene for r in train_data.records} & {r.parent_scene for r in query_data.records}),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "channel_mean": means, "channel_standard_deviation": stds,
                "total_training_seconds": sum(float(row["epoch_seconds"]) for row in history),
            })
            (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
            experiment_metadata[f"{modality}_seed_{seed}"] = run_metadata

    paired_deltas: dict[str, Any] = {}
    for seed in seeds:
        first, second = all_results.get(f"rgb_seed_{seed}"), all_results.get(f"rgbnir_seed_{seed}")
        if first is not None and second is not None:
            paired_deltas[str(seed)] = paired_field_bootstrap_delta(
                first, second, int(experiment_cfg["bootstrap_samples"]), seed
            )
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
               "system": collect_diagnostics("cuda" if device.type == "cuda" else "cpu"),
               "dataset": str(root), "modalities": modalities, "seeds": seeds,
               "smoke_limit": smoke_limit, "run_metadata": experiment_metadata,
               "paired_rgbnir_minus_rgb": paired_deltas, "results": all_results}
    destination = output_root / "comparison.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_comparison(all_results, output_root / "retrieval_comparison.png")
    print(f"Wrote M6 experiment: {destination}", flush=True)
    return destination


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent.parent / path


def _validate_config(raw: dict[str, Any]) -> None:
    expected_sections = {"dataset", "model", "training", "experiment"}
    if set(raw) != expected_sections:
        raise ValueError(f"M6 config sections must be exactly {sorted(expected_sections)}")
    expected = {
        "dataset": {"root", "catalog_cache_dir", "image_size", "value_scale"},
        "model": {"embedding_dim", "channels"},
        "training": {"epochs", "batch_size", "num_workers", "learning_rate", "weight_decay", "temperature"},
        "experiment": {"output_dir", "modalities", "seeds", "k", "minimum_queries", "bootstrap_samples", "example_queries", "query_chunk_size"},
    }
    for name, keys in expected.items():
        if set(raw[name]) != keys:
            raise ValueError(f"[{name}] must contain keys {sorted(keys)}")
    dataset, model, training, experiment = (raw[name] for name in ("dataset", "model", "training", "experiment"))
    if int(dataset["image_size"]) <= 0 or float(dataset["value_scale"]) <= 0:
        raise ValueError("dataset.image_size and dataset.value_scale must be positive")
    channels = tuple(int(value) for value in model["channels"])
    if int(model["embedding_dim"]) <= 0 or len(channels) != 3 or any(value <= 0 or value % 8 for value in channels):
        raise ValueError("model needs a positive embedding_dim and three positive channel widths divisible by 8")
    if int(training["epochs"]) <= 0 or int(training["batch_size"]) < 2 or int(training["num_workers"]) < 0:
        raise ValueError("training epochs must be positive, batch_size at least 2, and num_workers non-negative")
    if float(training["learning_rate"]) <= 0 or float(training["weight_decay"]) < 0 or float(training["temperature"]) <= 0:
        raise ValueError("invalid learning rate, weight decay, or contrastive temperature")
    modalities = tuple(experiment["modalities"])
    seeds = tuple(experiment["seeds"])
    if not modalities or len(set(modalities)) != len(modalities) or set(modalities) - MODALITIES.keys():
        raise ValueError(f"experiment.modalities must be unique values from {tuple(MODALITIES)}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment.seeds must be non-empty and unique")
    for key in ("k", "minimum_queries", "example_queries", "query_chunk_size"):
        if int(experiment[key]) <= 0:
            raise ValueError(f"experiment.{key} must be positive")
    if int(experiment["bootstrap_samples"]) < 0:
        raise ValueError("experiment.bootstrap_samples must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _band_statistics(dataset: AgricultureVisionDataset, batch_size: int, workers: int, device: torch.device) -> tuple[list[float], list[float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=_collate_samples)
    total = torch.zeros(len(dataset.bands), dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    pixels = 0
    for batch in loader:
        values = batch.double()
        total += values.sum(dim=(0, 2, 3)); total_sq += (values * values).sum(dim=(0, 2, 3))
        pixels += values.shape[0] * values.shape[2] * values.shape[3]
    mean = total / pixels
    variance = (total_sq / pixels - mean.square()).clamp_min(1e-12)
    return mean.float().tolist(), variance.sqrt().float().tolist()


class _PairedViews(torch.utils.data.Dataset):
    def __init__(self, dataset: AgricultureVisionDataset, seed: int) -> None:
        self.dataset, self.seed, self.epoch = dataset, seed, 0
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        return make_d4_views(sample.image, sample.record.tile_id, self.seed, self.epoch)


@torch.inference_mode()
def _embed(model: torch.nn.Module, dataset: AgricultureVisionDataset, batch_size: int, workers: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=_collate_samples)
    model.eval(); chunks = []
    for batch in loader:
        chunks.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _record_json(record) -> dict[str, Any]:
    return {"tile_id": record.tile_id, "labels": list(record.labels), "parent_scene": record.parent_scene,
            "metadata": {"field_id": record.parent_scene}}


def _collate_samples(samples):
    return torch.stack([sample.image for sample in samples])


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_loss", "validation_loss", "epoch_seconds"))
        writer.writeheader(); writer.writerows(history)
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _plot_loss(history: list[dict[str, float | int]], destination: Path, modality: str, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], marker="o", label="train")
    ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], marker="s", label="validation")
    ax.set(xlabel="Epoch", ylabel="NT-Xent loss", title=f"{modality.upper()} loss (seed {seed})"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def _plot_comparison(results: dict[str, Any], destination: Path) -> None:
    rows = [(name, metrics.get("macro")) for name, metrics in results.items() if metrics.get("macro") is not None]
    if not rows:
        return
    names, recalls = zip(*[(name, value["recall_at_k"]) for name, value in rows])
    fig, ax = plt.subplots(figsize=(max(7, len(names) * .8), 4)); ax.bar(names, recalls)
    ax.set(ylabel="Macro Recall@k", title="Held-out same-pattern retrieval"); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def _plot_retrieval_examples(gallery_x: np.ndarray, query_x: np.ndarray, gallery_records,
                             query_records, destination: Path, query_count: int) -> None:
    """Save query-to-gallery image strips with exact cosine ranks and mask labels."""
    queries = [i for i, record in enumerate(query_records) if record.labels][:query_count]
    if not queries:
        return
    gallery = gallery_x / np.linalg.norm(gallery_x, axis=1, keepdims=True)
    qnorm = query_x / np.linalg.norm(query_x, axis=1, keepdims=True)
    columns = 6
    fig, axes = plt.subplots(len(queries), columns, figsize=(17, 3.2 * len(queries)), squeeze=False)
    for row, query_index in enumerate(queries):
        scores = gallery @ qnorm[query_index]
        order = np.lexsort((np.asarray([r.tile_id for r in gallery_records]), -scores))[:columns - 1]
        selected = [None, *order]
        for col, index in enumerate(selected):
            record = query_records[query_index] if index is None else gallery_records[index]
            with rasterio.open(record.image_path) as source:
                image = source.read((1, 2, 3), out_shape=(3, 128, 128), resampling=Resampling.bilinear)
            image = np.moveaxis(image, 0, -1).astype(np.float32)
            lo, hi = np.percentile(image, (2, 98))
            image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
            axes[row, col].imshow(image)
            score_text = "query" if index is None else f"#{col} cosine={scores[index]:.3f}"
            labels = ", ".join(record.labels) or "no pattern label"
            axes[row, col].set_title(f"{score_text}\n{labels}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Validation queries and exact training-gallery neighbors (RGB display)")
    fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)
