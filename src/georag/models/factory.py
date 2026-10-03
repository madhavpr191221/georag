"""Shared construction and normalization-statistics loading for EO encoders."""

from __future__ import annotations

from collections.abc import Sequence
import csv
import math
from pathlib import Path

from torch import nn

from georag.config import ModelConfig
from georag.models.encoders import CNNEncoder, ViTEncoder


def load_band_statistics(
    path: Path, bands: tuple[str, ...], value_scale: float
) -> tuple[list[float], list[float]]:
    """Load the M1 per-band statistics using the same scaling as tile loading."""

    if not path.is_file():
        raise FileNotFoundError(
            f"training statistics not found: {path}. Run the Milestone 1 EuroSAT audit first."
        )
    if value_scale <= 0:
        raise ValueError("value_scale must be positive")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [row.get("band") for row in rows] != list(bands):
        raise ValueError("training statistics band order does not match the configured dataset")
    means = [float(row["mean"]) / value_scale for row in rows]
    standard_deviations = [float(row["standard_deviation"]) / value_scale for row in rows]
    if any(not math.isfinite(value) for value in (*means, *standard_deviations)):
        raise ValueError("training statistics contain non-finite values")
    if any(value <= 0 for value in standard_deviations):
        raise ValueError("training statistics must have positive standard deviations")
    return means, standard_deviations


def build_encoder(
    name: str,
    model_config: ModelConfig,
    image_size: tuple[int, int],
    band_mean: Sequence[float],
    band_standard_deviation: Sequence[float],
) -> nn.Module:
    """Construct one configured encoder without loading weights."""

    if name == "cnn":
        return CNNEncoder(
            band_mean, band_standard_deviation,
            model_config.embedding_dim, model_config.cnn_channels,
        )
    if name == "vit":
        return ViTEncoder(
            band_mean,
            band_standard_deviation,
            image_size=image_size,
            patch_size=model_config.vit_patch_size,
            width=model_config.vit_width,
            depth=model_config.vit_depth,
            heads=model_config.vit_heads,
            mlp_ratio=model_config.vit_mlp_ratio,
            embedding_dim=model_config.embedding_dim,
        )
    raise ValueError(f"unknown encoder {name!r}; expected 'cnn' or 'vit'")
