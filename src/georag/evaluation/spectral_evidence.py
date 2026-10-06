"""Inspectable Sentinel-2 spectral indices and exact descriptor ranking."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np


INDEX_NAMES = ("ndwi", "mndwi", "ndvi")
INDEX_BANDS = {
    "ndwi": ("B03", "B08"),
    "mndwi": ("B03", "B11"),
    "ndvi": ("B08", "B04"),
}
DEFAULT_MAX_PAIR_DAYS = 30
DEFAULT_HISTOGRAM_BINS = 32


def compute_indices(
    image: np.ndarray, valid_mask: np.ndarray, bands: tuple[str, ...], denominator_epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute normalized-difference index maps; invalid cells are NaN.

    ``image`` is a channel-first [C,H,W] stored-value array. Ratios remove a
    common multiplicative scale, but these values are not asserted to be
    calibrated surface reflectance.
    """
    values = np.asarray(image, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if denominator_epsilon < 0:
        raise ValueError("denominator_epsilon cannot be negative")
    if values.ndim != 3 or values.shape[0] != len(bands):
        raise ValueError("image must have shape [len(bands), H, W]")
    if mask.shape != values.shape[1:]:
        raise ValueError("valid_mask must have shape [H, W]")
    band_to_index = {name: index for index, name in enumerate(bands)}
    result: dict[str, np.ndarray] = {}
    for name, (left_band, right_band) in INDEX_BANDS.items():
        if left_band not in band_to_index or right_band not in band_to_index:
            raise ValueError(f"bands required for {name}: {left_band}, {right_band}")
        left = values[band_to_index[left_band]]
        right = values[band_to_index[right_band]]
        denominator = left + right
        usable = mask & np.isfinite(left) & np.isfinite(right) & (np.abs(denominator) > denominator_epsilon)
        output = np.full(mask.shape, np.nan, dtype=np.float32)
        np.divide(left - right, denominator, out=output, where=usable)
        result[name] = output
    return result


def index_histogram(
    indices: dict[str, np.ndarray], bins: int = DEFAULT_HISTOGRAM_BINS,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Return concatenated, unit-mass fixed-range histograms."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    chunks = []
    for name in INDEX_NAMES:
        values = np.asarray(indices[name], dtype=np.float32)
        values = values[np.isfinite(values)]
        histogram, _ = np.histogram(values, bins=bins, range=value_range)
        total = int(histogram.sum())
        if total == 0:
            chunks.append(np.zeros(bins, dtype=np.float32))
            continue
        chunks.append(histogram.astype(np.float32) / total)
    return np.concatenate(chunks)


def summarize_indices(indices: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Summarize each map with valid-pixel count and robust/simple statistics."""
    result: dict[str, dict[str, float]] = {}
    for name in INDEX_NAMES:
        values = np.asarray(indices[name], dtype=np.float32)
        values = values[np.isfinite(values)]
        if not values.size:
            result[name] = {"valid_pixels": 0, "mean": 0.0, "median": 0.0, "positive_fraction": 0.0}
            continue
        result[name] = {
            "valid_pixels": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "positive_fraction": float(np.mean(values > 0.0)),
        }
    return result


def make_temporal_pairs(records: list[Any], max_gap_days: int = DEFAULT_MAX_PAIR_DAYS) -> list[dict[str, Any]]:
    """Pair consecutive dated scenes inside each exact sequence token."""
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be positive")
    groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        metadata = record.metadata or {}
        sequence = metadata.get("sequence_id")
        if sequence and record.timestamp:
            groups[str(sequence)].append(record)
    pairs: list[dict[str, Any]] = []
    for sequence, sequence_records in groups.items():
        ordered = sorted(sequence_records, key=lambda item: (item.timestamp, item.tile_id))
        for before, after in zip(ordered, ordered[1:]):
            gap = (after.timestamp.date() - before.timestamp.date()).days
            if 0 < gap <= max_gap_days:
                pairs.append({"before": before, "after": after, "sequence_id": sequence, "gap_days": gap})
    return sorted(pairs, key=lambda pair: (pair["sequence_id"], pair["after"].timestamp))


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Exact cosine scores between one descriptor and [N,D] candidates."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(candidates, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != q.size:
        raise ValueError("query and candidates must have compatible descriptor dimensions")
    q_norm = float(np.linalg.norm(q))
    norms = np.linalg.norm(matrix, axis=1)
    if q_norm <= 1e-12 or not np.isfinite(q_norm) or np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("descriptors must have finite non-zero norms")
    return np.clip((matrix @ q) / (norms * q_norm), -1.0, 1.0)


def change_indices(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute after-minus-before changes over pixels valid at both dates."""
    if set(before) != set(after):
        raise ValueError("before and after index maps must have the same names")
    changed: dict[str, np.ndarray] = {}
    for name in INDEX_NAMES:
        left = np.asarray(before[name], dtype=np.float32)
        right = np.asarray(after[name], dtype=np.float32)
        if left.shape != right.shape:
            raise ValueError("before and after grids must have the same shape")
        valid = np.isfinite(left) & np.isfinite(right)
        delta = np.full(left.shape, np.nan, dtype=np.float32)
        delta[valid] = right[valid] - left[valid]
        changed[name] = delta
    return changed
