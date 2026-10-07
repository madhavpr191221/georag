"""Data and index-input audit for the SEN12-FLOOD spectral evidence path."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from georag.data.sen12flood import SEN12FloodDataset, sequence_id_for_record
from georag.data.splits import load_group_split_manifest
from georag.evaluation.spectral_evidence import INDEX_BANDS, compute_indices, summarize_indices


def select_stratified_examples(
    rows: list[dict[str, Any]], samples_per_stratum: int, seed: int
) -> list[dict[str, Any]]:
    """Select low/high index examples within each scene label deterministically.

    Low/high means the bottom/top quartile of scene median MNDWI within a
    label. This is a visualization sample, not an evaluation set.
    """
    if samples_per_stratum < 1:
        raise ValueError("samples_per_stratum must be positive")
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for label in ("flood", "no_flood"):
        group = sorted(
            (row for row in rows if row["label"] == label and row["mndwi_median"] is not None),
            key=lambda row: (row["mndwi_median"], row["tile_id"]),
        )
        if not group:
            continue
        quarter = max(1, len(group) // 4)
        tails = [("low", group[:quarter])]
        if len(group) > 1:
            tails.append(("high", group[-quarter:]))
        for tail, subset in tails:
            pool = list(subset)
            rng.shuffle(pool)
            selected.extend({**row, "stratum": f"{label}_{tail}"} for row in pool[:samples_per_stratum])
    return sorted(selected, key=lambda row: (row["stratum"], row["tile_id"]))


def run_spectral_input_audit(
    dataset: SEN12FloodDataset,
    split_manifest: str | Path,
    split_name: str,
    output_dir: str | Path,
    samples_per_stratum: int = 4,
    seed: int = 17,
    denominator_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Measure configured split scenes and write local audit artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assignments = load_group_split_manifest(
        split_manifest, dataset.all_records, sequence_id_for_record
    )
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    split_records = [
        record for record in dataset.records if assignments[record.tile_id] == split_name
    ]
    if not split_records:
        raise ValueError(f"no available image records in split {split_name}")

    scene_rows: list[dict[str, Any]] = []
    for index, record in enumerate(split_records, start=1):
        sample = dataset[dataset.index_for_id(record.tile_id)]
        image = sample.image.numpy()
        mask = sample.valid_mask.numpy().astype(bool)
        maps = compute_indices(image, mask, dataset.bands, denominator_epsilon)
        summary = summarize_indices(maps)
        all_zero_fraction = (
            float(np.mean(np.all(image[:, mask] == 0.0, axis=0))) if mask.any() else None
        )
        finite_mndwi = maps["mndwi"][np.isfinite(maps["mndwi"])]
        scene_rows.append({
            "tile_id": record.tile_id,
            "sequence_id": sequence_id_for_record(record),
            "label": record.labels[0],
            "timestamp": record.timestamp.isoformat() if record.timestamp else "",
            "valid_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "valid_fraction": float(mask.mean()),
            "all_bands_zero_fraction_within_mask": all_zero_fraction,
            "mndwi_median": float(np.median(finite_mndwi)) if finite_mndwi.size else None,
            "mndwi_mean": summary["mndwi"]["mean"],
            "ndwi_median": float(np.nanmedian(maps["ndwi"])) if np.isfinite(maps["ndwi"]).any() else None,
            "ndvi_median": float(np.nanmedian(maps["ndvi"])) if np.isfinite(maps["ndvi"]).any() else None,
            "index_valid_pixels": {name: value["valid_pixels"] for name, value in summary.items()},
            "index_ranges": {
                name: {
                    "min": float(np.nanmin(values)) if np.isfinite(values).any() else None,
                    "max": float(np.nanmax(values)) if np.isfinite(values).any() else None,
                    "outside_expected_range": int(np.sum(np.isfinite(values) & ((values < -1.00001) | (values > 1.00001)))),
                }
                for name, values in maps.items()
            },
        })
        if index % 25 == 0 or index == len(split_records):
            print(f"Audited indices: {index:,}/{len(split_records):,} {split_name} scenes", flush=True)

    examples = select_stratified_examples(scene_rows, samples_per_stratum, seed)
    example_ids = {row["tile_id"] for row in examples}
    examples = [row for row in examples if row["tile_id"] in example_ids]
    record_by_id = {record.tile_id: record for record in split_records}
    pixel_rows: list[dict[str, Any]] = []
    pixel_formula_differences: list[float] = []
    selected_raster_headers: list[dict[str, Any]] = []
    plot_items: list[tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    for item in examples:
        record = record_by_id[item["tile_id"]]
        sample = dataset[dataset.index_for_id(record.tile_id)]
        image = sample.image.numpy()
        mask = sample.valid_mask.numpy().astype(bool)
        maps = compute_indices(image, mask, dataset.bands, denominator_epsilon)
        valid_positions = np.argwhere(mask)
        if valid_positions.size:
            row_i, col_i = valid_positions[len(valid_positions) // 2]
            values = {band: float(image[dataset.bands.index(band), row_i, col_i]) for band in dataset.bands}
            pixel = {"tile_id": record.tile_id, "row": int(row_i), "col": int(col_i), "valid": True, **values}
            for index_name, (left_name, right_name) in INDEX_BANDS.items():
                left, right = values[left_name], values[right_name]
                pixel[index_name] = float((left - right) / (left + right)) if abs(left + right) > denominator_epsilon else None
                pixel[f"{index_name}_map_value"] = float(maps[index_name][row_i, col_i])
                pixel_formula_differences.append(abs(pixel[index_name] - pixel[f"{index_name}_map_value"]))
            pixel_rows.append(pixel)
        else:
            pixel_rows.append({"tile_id": record.tile_id, "valid": False})
        for band in dataset.bands:
            path = Path(str((record.metadata or {})["band_paths"][band]))
            with rasterio.open(path) as src:
                selected_raster_headers.append({
                    "tile_id": record.tile_id, "band": band, "path": str(path),
                    "width": src.width, "height": src.height, "crs": str(src.crs),
                    "resolution_x": abs(float(src.res[0])), "resolution_y": abs(float(src.res[1])),
                    "transform": tuple(src.transform)[:6], "dtype": src.dtypes[0],
                    "nodata": src.nodata, "scale": src.scales[0], "offset": src.offsets[0],
                    "units": src.units[0], "mask_valid_fraction": float(np.mean(src.read_masks(1) > 0)),
                    "area_or_point": src.tags().get("AREA_OR_POINT", ""),
                })
        plot_items.append((item, image, mask, maps))

    inventory = _inventory_stac(dataset.root)
    header_scales = sorted({float(row["scale"]) for row in selected_raster_headers})
    header_offsets = sorted({float(row["offset"]) for row in selected_raster_headers})
    index_ranges = {
        name: {
            "minimum": min((row["index_ranges"][name]["min"] for row in scene_rows if row["index_ranges"][name]["min"] is not None), default=None),
            "maximum": max((row["index_ranges"][name]["max"] for row in scene_rows if row["index_ranges"][name]["max"] is not None), default=None),
            "out_of_range_pixels": sum(row["index_ranges"][name]["outside_expected_range"] for row in scene_rows),
            "valid_pixel_measurements": sum(row["index_valid_pixels"][name] for row in scene_rows),
        }
        for name in INDEX_BANDS
    }
    summary = {
        "dataset_root": str(dataset.root), "split_manifest": str(Path(split_manifest).resolve()),
        "split_manifest_sha256": _sha256(Path(split_manifest)), "split": split_name,
        "seed": seed, "samples_per_stratum": samples_per_stratum,
        "bands": list(dataset.bands), "target_grid_band": dataset.target_grid_band,
        "target_resolution_m": dataset.target_resolution_m, "value_scale": dataset.value_scale,
        "radiometry": "raw_uint16_dn; selected raster headers report scale/offset values listed below; these headers alone do not establish reflectance calibration",
        "selected_raster_header_scales": header_scales,
        "selected_raster_header_offsets": header_offsets,
        "stac_radiometry_metadata": inventory["radiometry_metadata"],
        "resampling": {"finer": dataset.finer_resampling, "coarser": dataset.coarser_resampling, "same_resolution": dataset.same_resolution_resampling},
        "records_in_split": len(split_records), "strata_counts": _count_strata(examples),
        "quality_assets": inventory["quality_asset_keys"], "stac_asset_key_counts": inventory["asset_key_counts"],
        "scene_valid_fraction_min": min(row["valid_fraction"] for row in scene_rows),
        "scene_valid_fraction_median": float(np.median([row["valid_fraction"] for row in scene_rows])),
        "scene_valid_fraction_max": max(row["valid_fraction"] for row in scene_rows),
        "scenes_with_zero_common_valid_pixels": sum(row["valid_pixels"] == 0 for row in scene_rows),
        "scenes_with_no_valid_index_pixels": {
            name: sum(row["index_valid_pixels"][name] == 0 for row in scene_rows) for name in INDEX_BANDS
        },
        "scenes_at_least_99_percent_all_band_zero_within_mask": [
            row["tile_id"] for row in scene_rows
            if row["all_bands_zero_fraction_within_mask"] is not None
            and row["all_bands_zero_fraction_within_mask"] >= 0.99
        ],
        "index_ranges_and_valid_counts": index_ranges,
        "hand_checked_pixel_formula_max_absolute_difference": max(pixel_formula_differences, default=None),
        "indices": {name: list(bands) for name, bands in INDEX_BANDS.items()},
    }
    _write_json(output / "audit_summary.json", summary)
    _write_json(output / "scene_measurements.json", scene_rows)
    _write_csv(output / "selected_examples.csv", examples)
    _write_csv(output / "selected_raster_metadata.csv", selected_raster_headers)
    _write_csv(output / "hand_checked_pixels.csv", pixel_rows)
    _plot_examples(output / "spectral_input_contact_sheet.png", plot_items, dataset.bands)
    return summary


def _inventory_stac(root: Path) -> dict[str, Any]:
    files = sorted((root / "sen12floods_s2_source").rglob("stac.json"))
    key_counts: dict[str, int] = {}
    radiometry_counts = {field: 0 for field in ("eo:bands", "raster:bands", "scale", "offset", "unit", "nodata")}
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        assets = item.get("assets") or {}
        for key in assets:
            key_counts[key] = key_counts.get(key, 0) + 1
            asset = assets[key]
            if not isinstance(asset, dict):
                continue
            for field in ("eo:bands", "raster:bands"):
                if asset.get(field):
                    radiometry_counts[field] += 1
            raster_bands = asset.get("raster:bands") or []
            if isinstance(raster_bands, list):
                for band_metadata in raster_bands:
                    if isinstance(band_metadata, dict):
                        for field in ("scale", "offset", "unit", "nodata"):
                            if field in band_metadata:
                                radiometry_counts[field] += 1
    quality_terms = ("cloud", "shadow", "scl", "mask", "quality", "qa")
    return {
        "source_items": len(files), "asset_key_counts": dict(sorted(key_counts.items())),
        "quality_asset_keys": sorted(key for key in key_counts if any(term in key.lower() for term in quality_terms)),
        "radiometry_metadata": radiometry_counts,
    }


def _plot_examples(path: Path, items: list[tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]], bands: tuple[str, ...]) -> None:
    if not items:
        return
    fig, axes = plt.subplots(len(items), 5, figsize=(16, max(3.2 * len(items), 7)), squeeze=False)
    channel = {name: bands.index(name) for name in bands}
    for row_index, (record, image, mask, maps) in enumerate(items):
        rgb = np.stack([image[channel[name]] for name in ("B04", "B03", "B02")], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)
        low, high = np.percentile(rgb[mask], (2, 98)) if mask.any() else (0.0, 1.0)
        rgb = np.clip((rgb - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].set_title(f"{record['stratum']}\n{record['tile_id'][-28:]}")
        for col, index_name in enumerate(("ndwi", "mndwi", "ndvi"), start=1):
            axes[row_index, col].imshow(maps[index_name], cmap="RdYlBu", vmin=-1, vmax=1)
            axes[row_index, col].set_title(index_name.upper())
        axes[row_index, 4].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 4].set_title(f"common valid\n{mask.mean():.1%}")
        for ax in axes[row_index]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Validation examples: RGB display, normalized-difference indices, and common valid pixels", y=1.002)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _count_strata(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
