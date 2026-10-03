"""Reproducible contrastive runs for the small CNN and ViT encoders."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from georag.config import ProjectConfig
from georag.data import EuroSATMultispectralDataset
from georag.data.splits import indices_for_split, load_split_manifest
from georag.diagnostics import collect_diagnostics
from georag.models import build_encoder, load_band_statistics
from georag.training.contrastive import nt_xent_loss, positive_top1_accuracy
from georag.training.evaluation import embed_indices, knn_label_agreement
from georag.training.pairs import PairedTileBatch, PairedViewDataset, collate_paired_views

METRIC_FIELDS = (
    "epoch", "train_loss", "validation_loss", "train_positive_top1",
    "validation_positive_top1", "learning_rate", "epoch_seconds", "tiles_per_second",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _move_views(batch: PairedTileBatch, device: torch.device) -> torch.Tensor:
    return torch.cat((batch.view_a, batch.view_b), dim=0).to(device, non_blocking=True)


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader[PairedTileBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_anchors = 0
    total_tiles = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = _move_views(batch, device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(images)
        z1, z2 = embeddings.chunk(2, dim=0)
        loss, logits, targets = nt_xent_loss(z1, z2, temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite NT-Xent loss during training")
        loss.backward()
        gradient_flags = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradient_flags or not bool(torch.stack(gradient_flags).all().item()):
            raise FloatingPointError("missing or non-finite model gradients")
        optimizer.step()

        anchors = logits.shape[0]
        total_loss += float(loss.detach().item()) * anchors
        total_correct += float((logits.argmax(dim=1) == targets).sum().item())
        total_anchors += anchors
        total_tiles += len(batch.records)
    if total_anchors == 0:
        raise ValueError("training epoch produced no batches; check train split and batch size")
    return total_loss / total_anchors, total_correct / total_anchors, total_tiles


@torch.inference_mode()
def _run_validation_epoch(
    model: nn.Module,
    loader: DataLoader[PairedTileBatch],
    device: torch.device,
    temperature: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_anchors = 0
    for batch in loader:
        embeddings = model(_move_views(batch, device))
        z1, z2 = embeddings.chunk(2, dim=0)
        loss, logits, targets = nt_xent_loss(z1, z2, temperature)
        anchors = logits.shape[0]
        total_loss += float(loss.item()) * anchors
        total_correct += float((logits.argmax(dim=1) == targets).sum().item())
        total_anchors += anchors
    if total_anchors == 0:
        raise ValueError("validation produced no batches; check validation split and batch size")
    return total_loss / total_anchors, total_correct / total_anchors


def _save_checkpoint(
    path: Path,
    epoch: int,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    validation_loss: float,
    seed: int,
    sampler_generator: torch.Generator,
) -> None:
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    payload = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "seed": seed,
        "sampler_generator_state": sampler_generator.get_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _write_learning_curves(metrics_path: Path, output_path: Path, title: str) -> None:
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    validation_loss = [float(row["validation_loss"]) for row in rows]
    train_accuracy = [float(row["train_positive_top1"]) for row in rows]
    validation_accuracy = [float(row["validation_positive_top1"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(epochs, train_loss, label="train", color="#147d72", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="validation", color="#d47c2c", linewidth=2)
    axes[0].set(title="NT-Xent loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, train_accuracy, label="train", color="#147d72", linewidth=2)
    axes[1].plot(epochs, validation_accuracy, label="validation", color="#d47c2c", linewidth=2)
    axes[1].set(
        title="Positive-pair top-1 (contrastive batch)",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.0, 1.0),
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    figure.suptitle(title, fontsize=12)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_comparison_curves(run_directories: dict[str, Path], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    colors = {"cnn": "#147d72", "vit": "#7855a5"}
    for name, run_directory in run_directories.items():
        with (run_directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], color=colors[name],
                     linestyle="-", label=f"{name.upper()} train")
        axes[0].plot(epochs, [float(row["validation_loss"]) for row in rows], color=colors[name],
                     linestyle="--", label=f"{name.upper()} validation")
        axes[1].plot(epochs, [float(row["train_positive_top1"]) for row in rows], color=colors[name],
                     linestyle="-", label=f"{name.upper()} train")
        axes[1].plot(epochs, [float(row["validation_positive_top1"]) for row in rows], color=colors[name],
                     linestyle="--", label=f"{name.upper()} validation")
    axes[0].set(title="NT-Xent loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Positive-pair top-1 (contrastive batch)", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _evaluate_final_knn(
    model: nn.Module,
    dataset: EuroSATMultispectralDataset,
    train_indices: list[int],
    validation_indices: list[int],
    retrieval_ks: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    train_embeddings, train_labels, train_ids = embed_indices(
        model, dataset, train_indices, device, batch_size, num_workers
    )
    validation_embeddings, validation_labels, validation_ids = embed_indices(
        model, dataset, validation_indices, device, batch_size, num_workers
    )
    metrics = knn_label_agreement(
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
        retrieval_ks,
        device=device,
    )
    metrics["training_tile_id_sha256"] = hashlib.sha256("\n".join(train_ids).encode()).hexdigest()
    metrics["validation_tile_id_sha256"] = hashlib.sha256("\n".join(validation_ids).encode()).hexdigest()
    return metrics


def _run_one_model(
    model_name: str,
    config: ProjectConfig,
    dataset: EuroSATMultispectralDataset,
    train_indices: list[int],
    validation_indices: list[int],
    means: list[float],
    standard_deviations: list[float],
    device: torch.device,
    config_path: Path,
    experiment_root: Path,
    resume: bool,
) -> tuple[Path, dict[str, object]]:
    if config.model is None or config.training is None:
        raise ValueError("Milestone 3 requires both [model] and [training] config sections")
    train_config = config.training
    if len(train_indices) < train_config.batch_size or len(validation_indices) < train_config.batch_size:
        raise ValueError("train and validation splits must each contain at least one full training batch")

    run_directory = experiment_root / f"{model_name}_seed_{config.reproducibility.seed}"
    is_resume = run_directory.exists()
    if is_resume and not resume:
        raise FileExistsError(
            f"run directory already exists: {run_directory}. Use --resume to continue an interrupted run."
        )
    if not is_resume:
        run_directory.mkdir(parents=True)
        shutil.copyfile(config_path, run_directory / "config.toml")
    elif (run_directory / "config.toml").read_bytes() != config_path.read_bytes():
        raise ValueError("cannot resume because the run's saved configuration differs from the current config")

    model_seed = config.reproducibility.seed + (0 if model_name == "cnn" else 1)
    _seed_everything(model_seed)
    if config.model is None:
        raise ValueError("training config requires a [model] section")
    model = build_encoder(
        model_name,
        config.model,
        (config.dataset.expected_shape[1], config.dataset.expected_shape[2]),
        means,
        standard_deviations,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_pairs = PairedViewDataset(dataset, train_indices, config.reproducibility.seed)
    validation_pairs = PairedViewDataset(
        dataset, validation_indices, config.reproducibility.seed + 10_000, fixed_views=True
    )
    sampler_generator = torch.Generator().manual_seed(config.reproducibility.seed)
    loader_options = {
        "batch_size": train_config.batch_size,
        "num_workers": config.loader.num_workers,
        "pin_memory": config.loader.pin_memory and device.type == "cuda",
        "collate_fn": collate_paired_views,
        "persistent_workers": config.loader.num_workers > 0,
    }
    train_loader = DataLoader(
        train_pairs,
        shuffle=True,
        generator=sampler_generator,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_pairs,
        shuffle=False,
        drop_last=True,
        **loader_options,
    )

    metrics_path = run_directory / "metrics.csv"
    run_metadata_path = run_directory / "run.json"
    start_epoch = 0
    if is_resume:
        if not (run_directory / "checkpoint_last.pt").is_file() or not run_metadata_path.is_file():
            raise FileNotFoundError("interrupted run is missing its latest checkpoint or run metadata")
        run_metadata: dict[str, Any] = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if run_metadata.get("status") == "completed":
            raise ValueError("run is already completed; choose another model or omit it")
        checkpoint = torch.load(
            run_directory / "checkpoint_last.pt", map_location=device, weights_only=True
        )
        if checkpoint.get("model_name") != model_name or checkpoint.get("seed") != config.reproducibility.seed:
            raise ValueError("checkpoint model name or seed does not match the current run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        if start_epoch >= train_config.epochs:
            raise ValueError("checkpoint already reached the configured final epoch")
        if "sampler_generator_state" in checkpoint:
            sampler_generator.set_state(checkpoint["sampler_generator_state"].cpu())
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < start_epoch:
            raise ValueError("metrics.csv ends before the latest complete checkpoint")
        if len(rows) > start_epoch:
            rows = rows[:start_epoch]
            with metrics_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        best_checkpoint_path = run_directory / "checkpoint_best.pt"
        if not best_checkpoint_path.is_file():
            raise FileNotFoundError("interrupted run is missing checkpoint_best.pt")
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
        best_epoch = int(best_checkpoint["epoch"])
        best_validation_loss = float(best_checkpoint["validation_loss"])
        run_metadata.update(status="running", resumed_at_utc=datetime.now(timezone.utc).isoformat())
        run_metadata["resume_count"] = int(run_metadata.get("resume_count", 0)) + 1
        run_metadata["resume_shuffle_policy"] = "deterministic seed per epoch after restart"
    else:
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()
        run_metadata = {
        "status": "running",
        "model": model_name,
        "seed": config.reproducibility.seed,
        "model_initialization_seed": model_seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_inventory_fingerprint": dataset.catalog_metadata.get("source_fingerprint"),
        "training_tile_count": len(train_indices),
        "validation_tile_count": len(validation_indices),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": train_config.batch_size,
        "epochs": train_config.epochs,
        "data_loader_workers": config.loader.num_workers,
        "sampler_policy": "deterministic seed per epoch",
        "temperature": train_config.temperature,
        "learning_rate": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
        "optimizer": "AdamW",
        "augmentation": "independent uniform D4 transformations, synchronized across bands",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        best_validation_loss = math.inf
        best_epoch = 0
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    for epoch in range(start_epoch, train_config.epochs):
        sampler_generator.manual_seed(config.reproducibility.seed + epoch)
        train_pairs.set_epoch(epoch)
        started = time.perf_counter()
        try:
            train_loss, train_accuracy, train_tiles = _run_train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train_config.temperature,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                f"CUDA ran out of memory for {model_name} at batch_size={train_config.batch_size}; "
                "lower training.batch_size explicitly and rerun with a new experiment directory."
            ) from error
        validation_loss, validation_accuracy = _run_validation_epoch(
            model, validation_loader, device, train_config.temperature
        )
        elapsed = time.perf_counter() - started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_positive_top1": train_accuracy,
            "validation_positive_top1": validation_accuracy,
            "learning_rate": learning_rate,
            "epoch_seconds": elapsed,
            "tiles_per_second": train_tiles / elapsed,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writerow(metrics)
            stream.flush()

        _save_checkpoint(
            run_directory / "checkpoint_last.pt", epoch + 1, model_name, model,
            optimizer, validation_loss, config.reproducibility.seed, sampler_generator,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            _save_checkpoint(
                run_directory / "checkpoint_best.pt", epoch + 1, model_name, model,
                optimizer, validation_loss, config.reproducibility.seed, sampler_generator,
            )
        _write_learning_curves(metrics_path, run_directory / "learning_curves.png", model_name.upper())
        run_metadata.update(
            last_completed_epoch=epoch + 1,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )
        run_metadata_path.write_text(
            json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{model_name.upper()} epoch {epoch + 1}/{train_config.epochs}: "
            f"train loss={train_loss:.4f}, val loss={validation_loss:.4f}, "
            f"train pos@1={train_accuracy:.3f}, val pos@1={validation_accuracy:.3f}, "
            f"{elapsed:.1f}s",
            flush=True,
        )

    best_checkpoint = torch.load(
        run_directory / "checkpoint_best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    retrieval_metrics = _evaluate_final_knn(
        model, dataset, train_indices, validation_indices, train_config.retrieval_ks,
        device, train_config.batch_size, config.loader.num_workers,
    )
    (run_directory / "retrieval_diagnostics.json").write_text(
        json.dumps(retrieval_metrics, indent=2) + "\n", encoding="utf-8"
    )
    peak_cuda_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    run_metadata.update(
        status="completed",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        peak_cuda_memory_allocated_bytes=peak_cuda_bytes,
        retrieval_diagnostics="retrieval_diagnostics.json",
    )
    run_metadata_path.write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return run_directory, retrieval_metrics


def run_training(
    config: ProjectConfig,
    config_path: str | Path,
    device: torch.device,
    model_names: tuple[str, ...] = ("cnn", "vit"),
    resume: bool = False,
) -> dict[str, object]:
    if config.model is None or config.training is None:
        raise ValueError("Milestone 3 requires both [model] and [training] config sections")
    if not model_names or any(name not in {"cnn", "vit"} for name in model_names):
        raise ValueError("model_names must select 'cnn', 'vit', or both")
    means, standard_deviations = load_band_statistics(
        config.model.statistics_csv, config.dataset.bands, config.dataset.value_scale
    )
    dataset = EuroSATMultispectralDataset(
        root=config.dataset.root,
        bands=config.dataset.bands,
        expected_shape=config.dataset.expected_shape,
        expected_count=config.dataset.expected_count,
        value_scale=config.dataset.value_scale,
        catalog_path=config.dataset.catalog,
    )
    assignments = load_split_manifest(config.split.manifest, dataset.records)
    train_indices = indices_for_split(dataset.records, assignments, "train")
    validation_indices = indices_for_split(dataset.records, assignments, "validation")
    experiment_root = config.outputs.experiments_dir / "milestone_3"
    experiment_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    run_directories: dict[str, Path] = {}
    for model_name in model_names:
        expected_run_directory = experiment_root / f"{model_name}_seed_{config.reproducibility.seed}"
        if resume and expected_run_directory.is_dir():
            metadata_path = expected_run_directory / "run.json"
            if metadata_path.is_file():
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if existing_metadata.get("status") == "completed":
                    diagnostics_path = expected_run_directory / "retrieval_diagnostics.json"
                    if not diagnostics_path.is_file():
                        raise FileNotFoundError("completed run is missing retrieval diagnostics")
                    run_directories[model_name] = expected_run_directory
                    results[model_name] = {
                        "run_directory": str(expected_run_directory),
                        "retrieval_diagnostics": json.loads(diagnostics_path.read_text(encoding="utf-8")),
                    }
                    print(f"Skipping completed {model_name.upper()} run: {expected_run_directory}")
                    continue
        run_directory, retrieval_metrics = _run_one_model(
            model_name,
            config,
            dataset,
            train_indices,
            validation_indices,
            means,
            standard_deviations,
            device,
            Path(config_path).resolve(),
            experiment_root,
            resume,
        )
        run_directories[model_name] = run_directory
        results[model_name] = {
            "run_directory": str(run_directory),
            "retrieval_diagnostics": retrieval_metrics,
        }

    if len(run_directories) == 2:
        _write_comparison_curves(run_directories, experiment_root / "comparison_learning_curves.png")
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "system": collect_diagnostics("cuda" if device.type == "cuda" else "cpu"),
        "config": str(Path(config_path).resolve()),
        "dataset_inventory_fingerprint": dataset.catalog_metadata.get("source_fingerprint"),
        "models": results,
    }
    comparison_path = experiment_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison summary: {comparison_path}")
    return comparison
