from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import math
import json
from pathlib import Path
import time
from typing import Any

import torch

from georag.config import ProjectConfig, load_config
from georag.data import EuroSATMultispectralDataset
from georag.data.splits import indices_for_split, load_split_manifest
from georag.diagnostics import select_device
from georag.models import CNNEncoder, ViTEncoder


def _load_band_statistics(
    path: Path, expected_bands: tuple[str, ...], value_scale: float
) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"training statistics not found: {path}. Run "
            "`uv run python scripts/audit_eurosat.py --config configs/milestone_1.toml` first."
        )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [row.get("band") for row in rows] != list(expected_bands):
        raise ValueError("training statistics bands or order do not match the configured dataset bands")
    means = [float(row["mean"]) / value_scale for row in rows]
    standard_deviations = [float(row["standard_deviation"]) / value_scale for row in rows]
    if any(not math.isfinite(value) for value in (*means, *standard_deviations)):
        raise ValueError("training statistics contain non-finite values")
    if any(value <= 0 for value in standard_deviations):
        raise ValueError("training statistics must have positive per-band standard deviations")
    return means, standard_deviations


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _verify_model(
    name: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    embedding_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(images)
        _sync(device)
        started = time.perf_counter()
        for _ in range(10):
            embeddings = model(images)
        _sync(device)
        latency_ms = (time.perf_counter() - started) * 100.0

    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        _sync(device)
        memory_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory_before = None

    embeddings = model(images)
    target = torch.linspace(
        -1.0, 1.0, embeddings.numel(), device=device, dtype=embeddings.dtype
    ).reshape_as(embeddings)
    probe_loss = (embeddings * target).sum()
    probe_loss.backward()
    _sync(device)

    gradient_squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradient_squares:
        raise RuntimeError(f"{name} produced no parameter gradients")
    gradient_norm = torch.stack(gradient_squares).sum().sqrt().item()
    norms = torch.linalg.vector_norm(embeddings.detach().float(), ord=2, dim=1)
    output_finite = bool(torch.isfinite(embeddings).all().item())
    gradients_finite = all(
        bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if embeddings.shape != (images.shape[0], embedding_dim):
        raise RuntimeError(f"{name} returned unexpected embedding shape {tuple(embeddings.shape)}")
    if not output_finite or not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"{name} outputs must be finite unit-length embeddings")
    if not gradients_finite or not torch.isfinite(torch.tensor(gradient_norm)) or gradient_norm <= 0:
        raise RuntimeError(f"{name} must produce finite, nonzero parameter gradients")

    peak_delta = None
    peak_total = None
    if device.type == "cuda" and memory_before is not None:
        peak_total = torch.cuda.max_memory_allocated(device)
        peak_delta = max(0, peak_total - memory_before)

    result: dict[str, Any] = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_shape": list(embeddings.shape),
        "output_finite": output_finite,
        "embedding_norm_min": norms.min().item(),
        "embedding_norm_max": norms.max().item(),
        "gradient_l2_norm": gradient_norm,
        "gradients_finite": gradients_finite,
        "forward_latency_ms_mean": latency_ms,
        "optimizer_steps": 0,
        "weights_trained": False,
    }
    if peak_delta is not None:
        result["forward_backward_peak_cuda_bytes_above_baseline"] = peak_delta
        result["forward_backward_peak_cuda_bytes_total"] = peak_total
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    return result


def run_verification(config: ProjectConfig, device: torch.device) -> dict[str, Any]:
    if config.model is None:
        raise ValueError("the selected config must include a [model] section")
    model_config = config.model
    means, standard_deviations = _load_band_statistics(
        model_config.statistics_csv, config.dataset.bands, config.dataset.value_scale
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
    batch_size = config.loader.batch_size
    if len(train_indices) < batch_size:
        raise ValueError(f"need {batch_size} training tiles, found {len(train_indices)}")
    images = torch.stack(
        [dataset[index].image for index in train_indices[:batch_size]], dim=0
    ).to(device)

    torch.manual_seed(config.reproducibility.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.reproducibility.seed)
    cnn = CNNEncoder(
        means,
        standard_deviations,
        embedding_dim=model_config.embedding_dim,
        channels=model_config.cnn_channels,
    )
    torch.manual_seed(config.reproducibility.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.reproducibility.seed)
    vit = ViTEncoder(
        means,
        standard_deviations,
        image_size=(config.dataset.expected_shape[1], config.dataset.expected_shape[2]),
        patch_size=model_config.vit_patch_size,
        width=model_config.vit_width,
        depth=model_config.vit_depth,
        heads=model_config.vit_heads,
        mlp_ratio=model_config.vit_mlp_ratio,
        embedding_dim=model_config.embedding_dim,
    )
    results = {
        "cnn": _verify_model("cnn", cnn, images, model_config.embedding_dim, device),
        "vit": _verify_model("vit", vit, images, model_config.embedding_dim, device),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split": "train",
        "statistics_csv": str(model_config.statistics_csv),
        "statistics_csv_sha256": hashlib.sha256(
            model_config.statistics_csv.read_bytes()
        ).hexdigest(),
        "seed": config.reproducibility.seed,
        "split_manifest": str(config.split.manifest),
        "dataset_inventory_fingerprint": dataset.catalog_metadata.get("source_fingerprint"),
        "sample_tile_ids": [dataset.records[index].tile_id for index in train_indices[:batch_size]],
        "input_shape": list(images.shape),
        "input_dtype": str(images.dtype),
        "bands": list(config.dataset.bands),
        "normalization": "training-split per-band z-score",
        "normalization_statistics": [
            {"band": band, "mean": mean, "standard_deviation": standard_deviation}
            for band, mean, standard_deviation in zip(
                config.dataset.bands, means, standard_deviations, strict=True
            )
        ],
        "embedding_dim": model_config.embedding_dim,
        "architecture": {
            "cnn": {
                "channels": list(model_config.cnn_channels),
                "stride_two_stages": 3,
                "pooling": "global_average",
                "normalization": "GroupNorm(groups=8)",
            },
            "vit": {
                "patch_size": model_config.vit_patch_size,
                "image_patch_tokens": (
                    config.dataset.expected_shape[1] // model_config.vit_patch_size
                ) * (config.dataset.expected_shape[2] // model_config.vit_patch_size),
                "class_token_included": True,
                "width": model_config.vit_width,
                "depth": model_config.vit_depth,
                "heads": model_config.vit_heads,
                "mlp_ratio": model_config.vit_mlp_ratio,
                "attention": "softmax(QK^T / sqrt(head_width)) V",
            },
        },
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify random-init CNN and ViT encoders on one EuroSAT batch."
    )
    parser.add_argument("--config", default="configs/milestone_2.toml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    device = select_device(arguments.device or config.device.mode)
    results = run_verification(config, device)
    output = config.outputs.artifacts_dir / "milestone_2" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
