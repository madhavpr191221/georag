"""Exact, streaming diagnostics for the EuroSAT multispectral corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from georag.data.core import TileRecord
from georag.data.eurosat import EuroSATMultispectralDataset

INK = "#171717"
SECONDARY = "#5E6462"
ACCENT = "#1F6F5F"
STRUCTURE = "#E4E7E5"


@dataclass(frozen=True)
class BandStatistics:
    band: str
    count: int
    minimum: int
    p01: int
    p02: int
    median: int
    p98: int
    p99: int
    maximum: int
    mean: float
    standard_deviation: float
    zero_fraction: float
    declared_nodata_fraction: float | None


def compute_uint16_band_statistics(
    dataset: EuroSATMultispectralDataset,
    indices: Sequence[int],
    progress_every: int = 1_000,
) -> tuple[list[BandStatistics], dict[str, Any]]:
    """Compute exact statistics using one 65,536-bin histogram per source band.

    EuroSAT stores unsigned 16-bit digital numbers. A complete histogram is
    small enough to retain exact extrema, moments, and empirical percentiles
    without keeping the corpus in memory or using a sampling approximation.
    """

    if not indices:
        raise ValueError("at least one tile index is required for pixel statistics")
    channel_count = len(dataset.bands)
    histograms = np.zeros((channel_count, 65_536), dtype=np.uint64)
    declared_nodata_counts = np.zeros(channel_count, dtype=np.uint64)
    declared_nodata_available = np.zeros(channel_count, dtype=bool)
    started = time.perf_counter()

    for position, index in enumerate(indices, start=1):
        record = dataset.records[index]
        with rasterio.open(record.image_path) as source:
            array = source.read()
            nodata = source.nodata
        if array.dtype != np.uint16:
            raise ValueError(f"{record.tile_id} has dtype {array.dtype}; exact audit expects uint16")
        if tuple(array.shape) != dataset.expected_shape:
            raise ValueError(f"{record.tile_id} changed shape after catalog validation")
        for channel in range(channel_count):
            histograms[channel] += np.bincount(
                array[channel].reshape(-1), minlength=65_536
            ).astype(np.uint64, copy=False)
            if nodata is not None:
                declared_nodata_available[channel] = True
                declared_nodata_counts[channel] += np.count_nonzero(array[channel] == nodata)
        if progress_every and (position % progress_every == 0 or position == len(indices)):
            elapsed = time.perf_counter() - started
            print(f"Audited {position:,}/{len(indices):,} tiles ({elapsed:.1f} s)", flush=True)

    values = np.arange(65_536, dtype=np.float64)
    statistics: list[BandStatistics] = []
    for channel, band in enumerate(dataset.bands):
        histogram = histograms[channel]
        count = int(histogram.sum())
        present = np.flatnonzero(histogram)
        if not count or not len(present):
            raise ValueError(f"band {band} contains no pixels")
        weights = histogram.astype(np.float64)
        mean = float(np.dot(weights, values) / count)
        variance = max(float(np.dot(weights, values * values) / count - mean * mean), 0.0)
        statistics.append(
            BandStatistics(
                band=band,
                count=count,
                minimum=int(present[0]),
                p01=_histogram_quantile(histogram, 0.01),
                p02=_histogram_quantile(histogram, 0.02),
                median=_histogram_quantile(histogram, 0.50),
                p98=_histogram_quantile(histogram, 0.98),
                p99=_histogram_quantile(histogram, 0.99),
                maximum=int(present[-1]),
                mean=mean,
                standard_deviation=variance**0.5,
                zero_fraction=float(histogram[0] / count),
                declared_nodata_fraction=(
                    float(declared_nodata_counts[channel] / count)
                    if declared_nodata_available[channel] else None
                ),
            )
        )
    summary = {
        "tiles_audited": len(indices),
        "pixels_per_band": statistics[0].count,
        "source_dtype": "uint16",
        "histogram_bins_per_band": 65_536,
        "duration_seconds": time.perf_counter() - started,
        "percentiles": "exact empirical percentiles from complete uint16 histograms",
    }
    return statistics, summary


def summarize_metadata(
    records: Sequence[TileRecord], assignments: Mapping[str, str]
) -> dict[str, Any]:
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    crs_counts: Counter[str] = Counter()
    missing_coordinates = 0
    latitudes: list[float] = []
    longitudes: list[float] = []
    x_resolutions: list[float] = []
    y_resolutions: list[float] = []
    declared_nodata_tiles = 0
    for record in records:
        label = record.labels[0] if record.labels else "unlabeled"
        split = assignments[record.tile_id]
        class_counts[label]["all"] += 1
        class_counts[label][split] += 1
        metadata = record.metadata or {}
        crs_counts[str(metadata.get("crs") or "missing")] += 1
        if metadata.get("nodata") is not None:
            declared_nodata_tiles += 1
        if record.latitude is None or record.longitude is None:
            missing_coordinates += 1
        else:
            latitudes.append(record.latitude)
            longitudes.append(record.longitude)
        if record.spatial_resolution_m is not None:
            x_resolutions.append(record.spatial_resolution_m[0])
            y_resolutions.append(record.spatial_resolution_m[1])
    return {
        "tile_count": len(records),
        "class_counts": {label: dict(counts) for label, counts in sorted(class_counts.items())},
        "split_counts": dict(sorted(Counter(assignments.values()).items())),
        "distinct_crs_count": len(crs_counts),
        "crs_counts": dict(crs_counts.most_common()),
        "missing_coordinate_tiles": missing_coordinates,
        "declared_nodata_tiles": declared_nodata_tiles,
        "latitude_range": [min(latitudes), max(latitudes)] if latitudes else None,
        "longitude_range": [min(longitudes), max(longitudes)] if longitudes else None,
        "x_resolution_m_range": [min(x_resolutions), max(x_resolutions)] if x_resolutions else None,
        "y_resolution_m_range": [min(y_resolutions), max(y_resolutions)] if y_resolutions else None,
    }


def select_balanced_indices(
    records: Sequence[TileRecord],
    assignments: Mapping[str, str],
    split: str,
    count: int,
    seed: int,
) -> list[int]:
    """Select deterministic examples by cycling across labels."""

    by_label: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if assignments[record.tile_id] == split:
            label = record.labels[0] if record.labels else "unlabeled"
            by_label[label].append(index)
    for label, label_indices in by_label.items():
        label_indices.sort(key=lambda index: _stable_digest(seed, records[index].tile_id))
    selected: list[int] = []
    labels = sorted(by_label)
    depth = 0
    while len(selected) < count:
        added = False
        for label in labels:
            if depth < len(by_label[label]):
                selected.append(by_label[label][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def write_band_statistics(path: Path, statistics: Sequence[BandStatistics]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(BandStatistics.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(statistic) for statistic in statistics)
    return path


def write_class_counts(path: Path, metadata: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["class", "all", "train", "validation", "test"]
        )
        writer.writeheader()
        for label, counts in metadata["class_counts"].items():
            writer.writerow({"class": label, **{name: counts.get(name, 0) for name in ("all", "train", "validation", "test")}})
    return path


def write_audit_json(
    path: Path,
    metadata: Mapping[str, Any],
    statistics: Sequence[BandStatistics],
    pixel_summary: Mapping[str, Any],
    catalog_metadata: Mapping[str, Any],
    catalog_load_seconds: float,
    assignments: Mapping[str, str],
) -> Path:
    payload = {
        "audit_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "EuroSAT_MS",
        "metadata": metadata,
        "bands": [asdict(statistic) for statistic in statistics],
        "pixel_audit": dict(pixel_summary),
        "catalog": dict(catalog_metadata),
        "catalog_load_seconds": catalog_load_seconds,
        "split_fingerprint": split_fingerprint(assignments),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_reusable_audit(
    path: Path,
    dataset: EuroSATMultispectralDataset,
    assignments: Mapping[str, str],
) -> tuple[dict[str, Any], list[BandStatistics], dict[str, Any]] | None:
    """Load completed pixel evidence when it still matches catalog and splits."""

    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    catalog = payload.get("catalog") or {}
    if catalog.get("source_fingerprint") != dataset.catalog_metadata.get("source_fingerprint"):
        return None
    if int((payload.get("metadata") or {}).get("tile_count", -1)) != len(dataset):
        return None
    expected_split_counts = dict(sorted(Counter(assignments.values()).items()))
    if (payload.get("metadata") or {}).get("split_counts") != expected_split_counts:
        return None
    stored_split_fingerprint = payload.get("split_fingerprint")
    if stored_split_fingerprint is not None and stored_split_fingerprint != split_fingerprint(assignments):
        return None
    raw_bands = payload.get("bands") or []
    if [item.get("band") for item in raw_bands] != list(dataset.bands):
        return None
    return (
        dict(payload["metadata"]),
        [BandStatistics(**item) for item in raw_bands],
        dict(payload["pixel_audit"]),
    )


def split_fingerprint(assignments: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for tile_id, split in sorted(assignments.items()):
        digest.update(f"{tile_id}\0{split}\n".encode("utf-8"))
    return digest.hexdigest()


def plot_class_distribution(metadata: Mapping[str, Any], output_stem: Path) -> dict[str, Any]:
    counts = [(label, values["all"]) for label, values in metadata["class_counts"].items()]
    counts.sort(key=lambda item: (-item[1], item[0]))
    labels = [label for label, _ in counts]
    values = np.asarray([value for _, value in counts])
    _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=ACCENT, height=0.62)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 3_350)
    axis.set_xticks(np.arange(0, 3_001, 500))
    axis.set_xlabel("Tiles (count)")
    axis.set_title("EuroSAT class counts vary from 2,000 to 3,000 tiles", loc="left", pad=10)
    for position, value in zip(positions, values, strict=True):
        axis.text(value + 45, position, f"{value:,}", va="center", fontsize=8.5, color=SECONDARY)
    _style_axis(axis, x_grid=True)
    outputs, qa = _save_figure(figure, output_stem)
    plt.close(figure)
    return {"outputs": outputs, "qa": qa}


def plot_band_intervals(
    statistics: Sequence[BandStatistics], output_stem: Path
) -> dict[str, Any]:
    _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    positions = np.arange(len(statistics))
    lower = np.asarray([item.p02 for item in statistics])
    median = np.asarray([item.median for item in statistics])
    upper = np.asarray([item.p98 for item in statistics])
    for position, low, middle, high in zip(positions, lower, median, upper, strict=True):
        axis.plot([low, high], [position, position], color=SECONDARY, linewidth=1.2, zorder=2)
        axis.scatter([middle], [position], color=ACCENT, s=24, zorder=3)
    axis.set_yticks(positions, [item.band for item in statistics])
    axis.invert_yaxis()
    percentile_ceiling = int(np.ceil(max(upper) / 1_000.0) * 1_000)
    axis.set_xlim(0, percentile_ceiling * 1.04)
    axis.set_xticks(np.arange(0, percentile_ceiling + 1, 1_000))
    axis.set_xlabel("Raw digital number (2nd–98th percentile; dot = median)")
    axis.set_title("Raw-value distributions differ across Sentinel-2 bands", loc="left", pad=10)
    _style_axis(axis, x_grid=True)
    outputs, qa = _save_figure(figure, output_stem)
    plt.close(figure)
    return {"outputs": outputs, "qa": qa}


def write_visual_spec(path: Path, source_root: Path, tile_count: int) -> Path:
    spec = {
        "source": str(source_root),
        "evidence_lock": {
            "tiles": tile_count,
            "pixel_population": "all pixels in training-split uint16 GeoTIFFs",
            "uncertainty": "not applicable; descriptive population summaries",
        },
        "figures": [
            {
                "claim": "EuroSAT class counts vary from 2,000 to 3,000 tiles",
                "visual_form": "horizontal bar chart",
                "metric": "tile count",
                "unit": "tiles",
                "dimensions_inches": [7.0, 4.6],
                "outputs": ["svg", "pdf", "png"],
            },
            {
                "claim": "Raw-value distributions differ across Sentinel-2 bands",
                "visual_form": "horizontal percentile intervals with median dots",
                "metric": "empirical raw-value distribution",
                "unit": "digital number",
                "dimensions_inches": [7.0, 5.0],
                "outputs": ["svg", "pdf", "png"],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def write_markdown_report(
    path: Path,
    dataset_root: Path,
    metadata: Mapping[str, Any],
    statistics: Sequence[BandStatistics],
    pixel_summary: Mapping[str, Any],
    catalog_load_seconds: float,
) -> Path:
    split_counts = metadata["split_counts"]
    zero_peak = max(statistics, key=lambda statistic: statistic.zero_fraction)
    report = [
        "# EuroSAT multispectral dataset audit",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Findings",
        "",
        f"- The catalog validates **{metadata['tile_count']:,}** GeoTIFF headers with 13 bands and 64×64 pixels across 10 land-cover classes; all training-split tile payloads were read during the exact pixel audit.",
        f"- The deterministic split contains **{split_counts['train']:,} train**, **{split_counts['validation']:,} validation**, and **{split_counts['test']:,} test** tiles.",
        f"- Exact training-pixel statistics cover **{pixel_summary['pixels_per_band']:,} pixels per band** and were computed from complete uint16 histograms, without pixel sampling.",
        f"- **{metadata['declared_nodata_tiles']:,}** tiles declare a nodata value; zeros are reported separately. The largest zero fraction is {zero_peak.zero_fraction * 100:.8f}% in {zero_peak.band}.",
        f"- All but **{metadata['missing_coordinate_tiles']:,}** tiles expose usable centroid coordinates. The catalog loaded in **{catalog_load_seconds:.2f} s** after source-fingerprint validation.",
        "",
        "## Dataset contract",
        "",
        "| Property | Observed value |",
        "|---|---:|",
        f"| Dataset root | `{dataset_root}` |",
        f"| Tiles | {metadata['tile_count']:,} |",
        "| Tensor shape | `[13, 64, 64]` |",
        "| Source dtype | `uint16` |",
        f"| Distinct CRS values | {metadata['distinct_crs_count']:,} |",
        f"| Latitude range | {_format_range(metadata['latitude_range'], 4)} |",
        f"| Longitude range | {_format_range(metadata['longitude_range'], 4)} |",
        f"| X pixel-size range (m) | {_format_range(metadata['x_resolution_m_range'], 3)} |",
        f"| Y pixel-size range (m) | {_format_range(metadata['y_resolution_m_range'], 3)} |",
        "",
        "## Class and split composition",
        "",
        "| Class | All | Train | Validation | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, counts in metadata["class_counts"].items():
        report.append(
            f"| {label} | {counts['all']:,} | {counts['train']:,} | {counts['validation']:,} | {counts['test']:,} |"
        )
    report.extend([
        "",
        "![Class distribution](figures/class_distribution.png)",
        "",
        "## Training-split spectral statistics",
        "",
        "All percentiles are exact empirical percentiles over every training pixel. Values remain in the source digital-number scale.",
        "",
        "| Band | Min | P02 | Median | P98 | Max | Mean | SD | Zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in statistics:
        report.append(
            f"| {item.band} | {item.minimum:,} | {item.p02:,} | {item.median:,} | {item.p98:,} | "
            f"{item.maximum:,} | {item.mean:,.1f} | {item.standard_deviation:,.1f} | {item.zero_fraction * 100:.8f}% |"
        )
    report.extend([
        "",
        "![Band distributions](figures/band_intervals.png)",
        "",
        "## Visual inspection",
        "",
        "Each grid uses one joint 2nd–98th percentile stretch across the selected three bands for each tile. This preserves within-tile relative band scale, but it must not be interpreted as preserving radiometric comparability between tiles.",
        "",
        "| Split | RGB | Color infrared | SWIR |",
        "|---|---|---|---|",
        "| Train | [grid](grids/train_rgb.png) | [grid](grids/train_color_infrared.png) | [grid](grids/train_swir.png) |",
        "| Validation | [grid](grids/validation_rgb.png) | [grid](grids/validation_color_infrared.png) | [grid](grids/validation_swir.png) |",
        "| Test | [grid](grids/test_rgb.png) | [grid](grids/test_color_infrared.png) | [grid](grids/test_swir.png) |",
        "",
        "## Interpretation boundaries",
        "",
        "- The class-stratified split is suitable for plumbing and initial representation experiments, but it does not test geographic generalization or prevent nearby tiles from crossing split boundaries.",
        "- EuroSAT class labels are useful diagnostic labels; they are not a complete relevance judgment for image retrieval.",
        "- Acquisition timestamps and parent Sentinel-2 scene identifiers are unavailable in this adapter, so temporal leakage cannot yet be audited.",
        "- Per-band scaling or normalization for the encoder must be chosen from the recorded training statistics and then frozen; validation and test pixels must not influence it.",
        "",
        "## Machine-readable evidence",
        "",
        "- [`audit.json`](audit.json)",
        "- [`band_statistics.csv`](band_statistics.csv)",
        "- [`class_distribution.csv`](class_distribution.csv)",
        "- [`visual_spec.json`](visual_spec.json)",
        "- [`visual_qa.json`](visual_qa.json)",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def _histogram_quantile(histogram: np.ndarray, probability: float) -> int:
    count = int(histogram.sum())
    target = int(round(probability * (count - 1)))
    return int(np.searchsorted(np.cumsum(histogram, dtype=np.uint64), target + 1))


def _stable_digest(seed: int, tile_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{tile_id}".encode("utf-8")).digest()


def _configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _style_axis(axis: mpl.axes.Axes, x_grid: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(STRUCTURE)
    axis.spines["bottom"].set_color(STRUCTURE)
    axis.tick_params(length=3, width=0.7, color=STRUCTURE)
    if x_grid:
        axis.grid(axis="x", color=STRUCTURE, linewidth=0.7)
        axis.set_axisbelow(True)


def _save_figure(figure: mpl.figure.Figure, output_stem: Path) -> tuple[list[str], dict[str, Any]]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    qa = _audit_figure(figure)
    if qa["errors"]:
        raise ValueError(f"figure QA failed: {qa['errors']}")
    outputs: list[str] = []
    for suffix in ("svg", "pdf", "png"):
        destination = output_stem.with_suffix(f".{suffix}")
        figure.savefig(destination, dpi=300 if suffix == "png" else None)
        outputs.append(str(destination))
    return outputs, qa


def _audit_figure(figure: mpl.figure.Figure) -> dict[str, Any]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    errors: list[str] = []
    warnings: list[str] = []
    for text in figure.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if text.get_fontsize() < 8.0:
            errors.append(f"text below 8 pt: {text.get_text()[:40]!r}")
        bounds = text.get_window_extent(renderer=renderer)
        canvas = figure.bbox
        if bounds.x0 < canvas.x0 - 1 or bounds.y0 < canvas.y0 - 1 or bounds.x1 > canvas.x1 + 1 or bounds.y1 > canvas.y1 + 1:
            errors.append(f"text outside canvas: {text.get_text()[:40]!r}")
    for axis in figure.axes:
        if axis.spines["top"].get_visible() or axis.spines["right"].get_visible():
            warnings.append("top or right spine is visible")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "dimensions_inches": [float(value) for value in figure.get_size_inches()],
        "minimum_text_pt": 8.0,
    }


def _format_range(value: object, precision: int) -> str:
    if not value:
        return "—"
    lower, upper = value
    return f"{lower:.{precision}f}–{upper:.{precision}f}"
