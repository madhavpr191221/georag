from __future__ import annotations

import argparse
from collections import Counter

from georag.config import load_config
from georag.data.eurosat import EuroSATMultispectralDataset
from georag.data.splits import create_split_assignments, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EuroSAT and create its split manifest.")
    parser.add_argument("--config", default="configs/milestone_1.toml")
    parser.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Reopen every GeoTIFF header even when a valid catalog exists.",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if config.dataset.adapter != "eurosat_ms":
        raise ValueError(f"unsupported dataset adapter: {config.dataset.adapter}")

    dataset = EuroSATMultispectralDataset(
        root=config.dataset.root,
        bands=config.dataset.bands,
        expected_shape=config.dataset.expected_shape,
        expected_count=config.dataset.expected_count,
        value_scale=config.dataset.value_scale,
        catalog_path=config.dataset.catalog,
        rebuild_catalog=arguments.rebuild_catalog,
    )
    assignments = create_split_assignments(
        dataset.records,
        (config.split.train_fraction, config.split.validation_fraction, config.split.test_fraction),
        config.split.seed,
        config.split.stratify_by_label,
    )
    destination = write_split_manifest(config.split.manifest, dataset.records, assignments)
    print(f"Validated {len(dataset):,} tiles")
    print(f"Metadata source: {dataset.catalog_source}")
    print(f"Catalog: {config.dataset.catalog}")
    print(f"Labels: {dict(sorted(Counter(r.labels[0] for r in dataset.records).items()))}")
    print(f"Splits: {dict(sorted(Counter(assignments.values()).items()))}")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
