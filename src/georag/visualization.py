"""Technical displays for multispectral tile inspection."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from georag.data.core import TileSample


def make_composite(
    image: torch.Tensor,
    bands: Sequence[str],
    composite_bands: Sequence[str],
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Select three named bands and apply one joint display-only percentile stretch.

    A shared low/high range preserves the relative scale between the selected
    bands. Stretching each channel independently would improve contrast but
    silently alter the composite's color balance.
    """

    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W]")
    if image.shape[0] != len(bands):
        raise ValueError("band names must match the tensor channel count")
    if len(composite_bands) != 3:
        raise ValueError("a display composite must contain exactly three band names")
    try:
        indices = [bands.index(name) for name in composite_bands]
    except ValueError as error:
        raise ValueError(f"unknown band in composite {tuple(composite_bands)}") from error

    selected = image[indices].detach().cpu().numpy().astype(np.float32, copy=False)
    low, high = np.percentile(selected, [lower_percentile, upper_percentile])
    display = (
        np.zeros_like(selected)
        if high <= low
        else np.clip((selected - low) / (high - low), 0.0, 1.0)
    )
    return np.moveaxis(display, 0, -1)


def save_sample_grid(
    samples: Sequence[TileSample],
    composite_bands: Sequence[str],
    output_path: str | Path,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    columns: int = 4,
) -> Path:
    if not samples:
        raise ValueError("at least one sample is required")
    rows = math.ceil(len(samples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 3.8 * rows), squeeze=False)
    for axis, sample in zip(axes.flat, samples, strict=False):
        axis.imshow(make_composite(
            sample.image, sample.record.bands, composite_bands,
            lower_percentile, upper_percentile,
        ))
        label = sample.record.labels[0] if sample.record.labels else "unlabeled"
        coordinates = ""
        if sample.record.latitude is not None and sample.record.longitude is not None:
            coordinates = f"\n{sample.record.latitude:.4f}, {sample.record.longitude:.4f}"
        axis.set_title(f"{sample.record.tile_id}\n{label}{coordinates}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(samples):]:
        axis.axis("off")
    figure.suptitle(
        f"Bands {tuple(composite_bands)}; joint per-tile {lower_percentile:g}-{upper_percentile:g}% display stretch",
        fontsize=11,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), pad=1.2)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_retrieval_grid(
    query: TileSample,
    retrieved: Sequence[tuple[TileSample, int, float]],
    metric: str,
    output_path: str | Path,
    bands: Sequence[str],
    composite_bands: Sequence[str],
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    columns: int = 3,
) -> Path:
    """Render a query and ranked neighbors with metric scores and tile metadata."""

    if metric not in {"cosine", "inner_product", "euclidean"}:
        raise ValueError("unsupported retrieval metric")
    if not retrieved:
        raise ValueError("at least one retrieved sample is required")
    if columns <= 0:
        raise ValueError("columns must be positive")
    low, high = lower_percentile, upper_percentile
    if not 0 <= low < high <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    items = [(query, None, None), *retrieved]
    rows = math.ceil(len(items) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    metric_label = {
        "cosine": "cosine",
        "inner_product": "inner product",
        "euclidean": "L2 distance",
    }[metric]
    for item_index, (axis, (sample, rank, score)) in enumerate(
        zip(axes.flat, items, strict=False)
    ):
        axis.imshow(
            make_composite(
                sample.image,
                bands,
                composite_bands,
                lower_percentile,
                upper_percentile,
            ),
            interpolation="nearest",
        )
        label = sample.record.labels[0] if sample.record.labels else "unlabeled"
        location = ""
        if sample.record.latitude is not None and sample.record.longitude is not None:
            location = f"\n{sample.record.latitude:.4f}, {sample.record.longitude:.4f}"
        if item_index == 0:
            title = f"QUERY | {label}\n{sample.record.tile_id}{location}"
        else:
            title = (
                f"Rank {rank} | {metric_label}={score:.5f} | {label}\n"
                f"{sample.record.tile_id}{location}"
            )
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(items):]:
        axis.axis("off")
    figure.suptitle(
        f"Exact Flat retrieval | display bands {tuple(composite_bands)}",
        fontsize=11,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_text_retrieval_grid(
    query_text: str,
    retrieved: Sequence[tuple[TileSample, int, float]],
    output_path: str | Path,
    target_label: str,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    columns: int = 3,
) -> Path:
    """Render a text query with ranked RGB tiles, scores, and annotation labels."""

    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if not target_label.strip():
        raise ValueError("target_label must not be empty")
    if not retrieved:
        raise ValueError("at least one retrieved sample is required")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    rows = math.ceil(len(retrieved) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (sample, rank, score) in zip(axes.flat, retrieved, strict=False):
        axis.imshow(
            make_composite(
                sample.image,
                sample.record.bands,
                ("R", "G", "B"),
                lower_percentile,
                upper_percentile,
            ),
            interpolation="nearest",
        )
        labels = ", ".join(sample.record.labels) if sample.record.labels else "no annotation"
        axis.set_title(
            f"Rank {rank} | cosine={score:.4f}\n{sample.record.tile_id}\n{labels}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.flat[len(retrieved):]:
        axis.axis("off")
    figure.suptitle(
        f'Text query for "{target_label}": {query_text}',
        fontsize=10,
        wrap=True,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
