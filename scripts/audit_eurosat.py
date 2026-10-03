from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from georag.config import load_config
from georag.data import EuroSATMultispectralDataset
from georag.data.audit import (
    compute_uint16_band_statistics,
    load_reusable_audit,
    plot_band_intervals,
    plot_class_distribution,
    select_balanced_indices,
    summarize_metadata,
    write_audit_json,
    write_band_statistics,
    write_class_counts,
    write_markdown_report,
    write_visual_spec,
)
from georag.data.splits import indices_for_split, load_split_manifest
from georag.visualization import save_sample_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the real EuroSAT multispectral corpus.")
    parser.add_argument("--config", default="configs/milestone_1.toml")
    parser.add_argument("--output", default="artifacts/milestone_1")
    parser.add_argument(
        "--recompute-pixels",
        action="store_true",
        help="Ignore compatible audit evidence and reread every training pixel.",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    output = Path(arguments.output).resolve()

    started = time.perf_counter()
    dataset = EuroSATMultispectralDataset(
        root=config.dataset.root,
        bands=config.dataset.bands,
        expected_shape=config.dataset.expected_shape,
        expected_count=config.dataset.expected_count,
        value_scale=config.dataset.value_scale,
        catalog_path=config.dataset.catalog,
    )
    catalog_load_seconds = time.perf_counter() - started
    if dataset.catalog_source != "cache":
        print("Created catalog while loading the dataset.", flush=True)
    assignments = load_split_manifest(config.split.manifest, dataset.records)
    train_indices = indices_for_split(dataset.records, assignments, "train")

    reusable = None if arguments.recompute_pixels else load_reusable_audit(
        output / "audit.json", dataset, assignments
    )
    if reusable is None:
        metadata = summarize_metadata(dataset.records, assignments)
        statistics, pixel_summary = compute_uint16_band_statistics(dataset, train_indices)
    else:
        metadata, statistics, pixel_summary = reusable
        print("Reusing compatible completed pixel audit.", flush=True)
    write_band_statistics(output / "band_statistics.csv", statistics)
    write_class_counts(output / "class_distribution.csv", metadata)
    write_audit_json(
        output / "audit.json", metadata, statistics, pixel_summary,
        dataset.catalog_metadata, catalog_load_seconds, assignments,
    )
    write_visual_spec(output / "visual_spec.json", config.dataset.root, len(dataset))

    visual_qa = {
        "class_distribution": plot_class_distribution(
            metadata, output / "figures" / "class_distribution"
        ),
        "band_intervals": plot_band_intervals(
            statistics, output / "figures" / "band_intervals"
        ),
    }
    for split in ("train", "validation", "test"):
        selected = select_balanced_indices(
            dataset.records, assignments, split, config.visualization.samples,
            config.reproducibility.seed,
        )
        samples = [dataset[index] for index in selected]
        for composite_name, composite_bands in config.visualization.composites.items():
            save_sample_grid(
                samples,
                composite_bands,
                output / "grids" / f"{split}_{composite_name}.png",
                config.visualization.lower_percentile,
                config.visualization.upper_percentile,
            )
    (output / "visual_qa.json").write_text(
        json.dumps(visual_qa, indent=2) + "\n", encoding="utf-8"
    )
    report = write_markdown_report(
        output / "report.md", config.dataset.root, metadata, statistics,
        pixel_summary, catalog_load_seconds,
    )
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
