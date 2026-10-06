"""Matched from-scratch CNN/ViT and RGB/12-band NT-Xent experiment."""

from __future__ import annotations

import csv
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
import tomllib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from georag.data.core import EOTileDataset, TileSample, collate_tiles
from georag.data.masked_statistics import masked_band_statistics
from georag.data.sen12flood import SEN12FloodDataset, sequence_id_for_record
from georag.data.sen12flood_cache import SEN12FloodTensorCache
from georag.data.splits import indices_for_split, load_group_split_manifest
from georag.models.encoders import CNNEncoder, ViTEncoder
from georag.training.contrastive import nt_xent_loss, positive_top1_accuracy
from georag.training.evaluation import knn_label_agreement
from georag.training.pairs import PairedViewDataset, collate_paired_views

RGB_CHANNELS = (3, 2, 1)  # Sentinel-2 B04/B03/B02, conventional visible RGB.


class ChannelSelectionDataset(Dataset[TileSample]):
    """Expose selected spectral channels while preserving masks and metadata."""

    def __init__(self, dataset: EOTileDataset, channels: Sequence[int]) -> None:
        self.dataset = dataset
        self.channels = tuple(channels)
        self.records = dataset.records

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TileSample:
        sample = self.dataset[index]
        return TileSample(sample.image[list(self.channels)], sample.record, sample.valid_mask)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_model(architecture: str, mean: Sequence[float], std: Sequence[float], config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    if architecture == "cnn":
        return CNNEncoder(mean, std, model_cfg["embedding_dim"], tuple(model_cfg["cnn_channels"]))
    if architecture == "vit":
        return ViTEncoder(
            mean, std, tuple(model_cfg["vit_image_size"]), model_cfg["vit_patch_size"],
            model_cfg["vit_width"], model_cfg["vit_depth"], model_cfg["vit_heads"],
            model_cfg["vit_mlp_ratio"], model_cfg["embedding_dim"],
        )
    raise ValueError(f"unsupported architecture: {architecture}")


def _model_batch(model: torch.nn.Module, images: torch.Tensor, masks: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True) if masks is not None else None
    return model(images, masks)


@torch.no_grad()
def _contrastive_validation(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, temperature: float
) -> tuple[float, float]:
    model.eval()
    losses, accuracies, count = 0.0, 0.0, 0
    for batch in loader:
        batch_size = len(batch.records)
        images = torch.cat((batch.view_a, batch.view_b), dim=0)
        masks = torch.cat((batch.mask_a, batch.mask_b), dim=0) if batch.mask_a is not None else None
        embeddings = _model_batch(model, images, masks, device)
        loss, logits, targets = nt_xent_loss(embeddings[:batch_size], embeddings[batch_size:], temperature)
        losses += float(loss.item()) * batch_size
        accuracies += float(positive_top1_accuracy(logits, targets).item()) * batch_size
        count += batch_size
    return losses / count, accuracies / count


@torch.inference_mode()
def _embed(
    model: torch.nn.Module, dataset: Dataset[TileSample], indices: Sequence[int],
    device: torch.device, batch_size: int,
) -> tuple[torch.Tensor, list[str], list[str]]:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_tiles)
    model.eval()
    embeddings, labels, ids = [], [], []
    for batch in loader:
        embeddings.append(_model_batch(model, batch.images, batch.valid_masks, device).cpu())
        labels.extend(record.labels[0] if record.labels else "" for record in batch.records)
        ids.extend(record.tile_id for record in batch.records)
    return torch.cat(embeddings), labels, ids


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _one_run(
    config: dict, dataset: EOTileDataset, split_indices: dict[str, list[int]], full_statistics: tuple[list[float], list[float]],
    architecture: str, modality: str, seed: int, device: torch.device,
    run_dir: Path, epochs: int, smoke: bool,
) -> dict[str, object]:
    _seed_everything(seed)
    channels = RGB_CHANNELS if modality == "rgb" else tuple(range(12))
    selected = ChannelSelectionDataset(dataset, channels)
    mean_all, std_all = full_statistics
    mean = [mean_all[index] for index in channels]
    std = [std_all[index] for index in channels]
    stats = {"band_indices": list(channels), "mean": mean, "std": std, "computed_on": "valid pixels from train split only"}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "band_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    train_indices = split_indices["train"]
    val_indices = split_indices["validation"]
    test_indices = split_indices["test"]
    if smoke:
        train_indices = train_indices[:32]
        val_indices = val_indices[:16]
        test_indices = test_indices[:16]
    batch_size = int(config["training"]["batch_size"])
    train_pairs = PairedViewDataset(selected, train_indices, seed=seed)
    val_pairs = PairedViewDataset(selected, val_indices, seed=seed, fixed_views=True)
    train_loader = DataLoader(
        train_pairs, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_paired_views,
    )
    val_loader = DataLoader(
        val_pairs, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_paired_views,
    )
    model = _build_model(architecture, mean, std, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    temperature = float(config["training"]["temperature"])
    best_val = float("inf")
    best_epoch = 0
    curve_path = run_dir / "metrics.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "train_positive_top1", "validation_loss", "validation_positive_top1", "epoch_seconds"])
        writer.writeheader()
        for epoch in range(epochs):
            start = time.perf_counter()
            train_pairs.set_epoch(epoch)
            model.train()
            sum_loss = sum_acc = 0.0
            seen = 0
            for batch in train_loader:
                original_batch = len(batch.records)
                images = torch.cat((batch.view_a, batch.view_b), dim=0)
                masks = torch.cat((batch.mask_a, batch.mask_b), dim=0) if batch.mask_a is not None else None
                embeddings = _model_batch(model, images, masks, device)
                loss, logits, targets = nt_xent_loss(embeddings[:original_batch], embeddings[original_batch:], temperature)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sum_loss += float(loss.detach().item()) * original_batch
                sum_acc += float(positive_top1_accuracy(logits.detach(), targets).item()) * original_batch
                seen += original_batch
            if not seen:
                raise ValueError("training split is smaller than one full batch")
            val_loss, val_acc = _contrastive_validation(model, val_loader, device, temperature)
            seconds = time.perf_counter() - start
            writer.writerow({"epoch": epoch + 1, "train_loss": sum_loss / seen, "train_positive_top1": sum_acc / seen,
                             "validation_loss": val_loss, "validation_positive_top1": val_acc, "epoch_seconds": seconds})
            stream.flush()
            print(f"{run_dir.name} epoch {epoch + 1}/{epochs}: train={sum_loss/seen:.4f} val={val_loss:.4f} ({seconds:.1f}s)", flush=True)
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": best_epoch,
                                    "seed": seed, "config": config, "band_statistics": stats}, run_dir / "checkpoint_best.pt")
            _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                                "seed": seed, "config": config, "band_statistics": stats}, run_dir / "checkpoint_last.pt")
    checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    gallery_indices = split_indices["train"]
    query_indices = split_indices["test"]
    if smoke:
        gallery_indices, query_indices = gallery_indices[:32], query_indices[:16]
    gallery, gallery_labels, gallery_ids = _embed(model, selected, gallery_indices, device, batch_size)
    queries, query_labels, query_ids = _embed(model, selected, query_indices, device, batch_size)
    retrieval = knn_label_agreement(
        gallery, gallery_labels, queries, query_labels, ks=(1, 5, 10), device=device,
    )
    torch.save({"gallery_embeddings": gallery, "gallery_tile_ids": gallery_ids,
                "query_embeddings": queries, "query_tile_ids": query_ids}, run_dir / "test_retrieval.pt")
    result = {
        "run_id": run_dir.name, "architecture": architecture, "modality": modality, "seed": seed,
        "best_epoch": best_epoch, "best_validation_loss": best_val,
        "test_flood_label_neighbor_agreement": retrieval,
        "gallery_count": len(gallery_ids), "test_query_count": len(query_ids), "smoke": smoke,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_milestone_9(config_path: str | Path, smoke: bool = False, device_override: str | None = None) -> Path:
    config_path = Path(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    m8_path = Path(config["dataset"]["config"])
    m8 = tomllib.loads(m8_path.read_text(encoding="utf-8"))
    d, res = m8["dataset"], m8["resampling"]
    source = SEN12FloodDataset(
        d["root"], d["bands"], d["target_resolution_m"], d["value_scale"], d["target_grid_band"],
        res["finer_than_target"], res["coarser_than_target"], res["same_resolution"], res["nodata_fill"],
    )
    manifest = Path(config["split"]["manifest"])
    assignments = load_group_split_manifest(manifest, source.all_records, sequence_id_for_record)
    root = Path(config["dataset"]["cache"])
    dataset = SEN12FloodTensorCache(source, root)
    split_indices = {
        name: indices_for_split(dataset.records, assignments, name, allow_extra_assignments=True)
        for name in ("train", "validation", "test")
    }
    if any(not split_indices[name] for name in split_indices):
        raise ValueError(f"empty available-scene split: { {key: len(value) for key, value in split_indices.items()} }")
    requested_device = device_override or config["runtime"]["device"]
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else
                          "cpu" if requested_device == "auto" else requested_device)
    output_root = Path(config["runtime"]["output_dir"])
    run_stamp = time.strftime("%Y%m%d_%H%M%S") + ("_smoke" if smoke else "")
    experiment_dir = output_root / run_stamp
    experiment_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.toml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    run_info = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "dataset_records": len(dataset),
        "split_counts": {key: len(value) for key, value in split_indices.items()},
        "cache_fingerprint": dataset.fingerprint, "smoke": smoke,
    }
    (experiment_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    full_statistics = masked_band_statistics(dataset, split_indices["train"])
    (experiment_dir / "training_band_statistics_all_12.json").write_text(
        json.dumps({"mean": full_statistics[0], "std": full_statistics[1], "valid_pixels_only": True}, indent=2),
        encoding="utf-8",
    )
    seeds = config["training"]["seeds"][:1] if smoke else config["training"]["seeds"]
    epochs = 1 if smoke else int(config["training"]["epochs"])
    results = []
    for architecture in config["training"]["architectures"]:
        for modality in config["training"]["modalities"]:
            for seed in seeds:
                name = f"{architecture}_{modality}_seed_{seed}"
                result = _one_run(config, dataset, split_indices, full_statistics, architecture, modality, seed,
                                  device, experiment_dir / name, epochs, smoke)
                results.append(result)
                (experiment_dir / "completed_runs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (experiment_dir / "comparison.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"Completed {len(results)} runs in {experiment_dir}", flush=True)
    return experiment_dir
