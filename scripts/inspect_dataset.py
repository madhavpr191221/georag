from __future__ import annotations

import argparse

from torch.utils.data import DataLoader, Subset

from georag.config import load_config
from georag.data import EuroSATMultispectralDataset, collate_tiles
from georag.data.splits import indices_for_split, load_split_manifest
from georag.visualization import save_sample_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a EuroSAT batch and save a composite grid.")
    parser.add_argument("--config", default="configs/milestone_1.toml")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--composite", default="rgb")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.composite not in config.visualization.composites:
        raise ValueError(
            f"unknown composite {arguments.composite!r}; choose from "
            f"{sorted(config.visualization.composites)}"
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
    indices = indices_for_split(dataset.records, assignments, arguments.split)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        collate_fn=collate_tiles,
    )
    batch = next(iter(loader))
    print(f"Batch tensor shape: {tuple(batch.images.shape)}")
    print(f"Batch value range: [{batch.images.min().item()}, {batch.images.max().item()}]")

    samples = [dataset[index] for index in indices[: config.visualization.samples]]
    destination = config.outputs.artifacts_dir / "dataset" / f"{arguments.split}_{arguments.composite}.png"
    save_sample_grid(
        samples,
        config.visualization.composites[arguments.composite],
        destination,
        config.visualization.lower_percentile,
        config.visualization.upper_percentile,
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
