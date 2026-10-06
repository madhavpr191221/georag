"""Build the local SEN12-FLOOD catalog and sequence-group split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

from georag.data.sen12flood import SEN12FloodDataset, sequence_id_for_record
from georag.data.splits import create_grouped_split_assignments, write_group_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/milestone_8.toml"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    split_config = config["split"]
    resampling_config = config["resampling"]
    if split_config["group_field"] != "sequence_id":
        raise ValueError("M8 requires split.group_field = 'sequence_id'")

    dataset = SEN12FloodDataset(
        root=dataset_config["root"],
        bands=dataset_config["bands"],
        target_resolution_m=dataset_config["target_resolution_m"],
        value_scale=dataset_config["value_scale"],
        target_grid_band=dataset_config["target_grid_band"],
        finer_resampling=resampling_config["finer_than_target"],
        coarser_resampling=resampling_config["coarser_than_target"],
        same_resolution_resampling=resampling_config["same_resolution"],
        nodata_fill=resampling_config["nodata_fill"],
    )
    fractions = (
        split_config["train_fraction"],
        split_config["validation_fraction"],
        split_config["test_fraction"],
    )
    seed = split_config["seed"]
    assignments = create_grouped_split_assignments(
        dataset.all_records, sequence_id_for_record, fractions, seed
    )
    manifest_path = write_group_split_manifest(
        split_config["manifest"], dataset.all_records, assignments, sequence_id_for_record
    )
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    split_summary = {}
    for split_name in ("train", "validation", "test"):
        rows = [
            record for record in dataset.all_records
            if assignments[record.tile_id] == split_name
        ]
        usable = [record for record in rows if record.metadata["asset_status"] == "available"]
        split_summary[split_name] = {
            "records": len(rows),
            "available_scenes": len(usable),
            "missing_scenes": len(rows) - len(usable),
            "flood": sum(bool(record.metadata["flooding"]) for record in rows),
            "no_flood": sum(not bool(record.metadata["flooding"]) for record in rows),
            "sequence_groups": len({sequence_id_for_record(record) for record in rows}),
        }

    summary = {
        "dataset": "SEN12-FLOOD Sentinel-2",
        "root": str(Path(dataset_config["root"]).resolve()),
        "catalog_records": len(dataset.all_records),
        "available_scenes": len(dataset.records),
        "missing_scenes": len(dataset.all_records) - len(dataset.records),
        "sequence_groups": len({sequence_id_for_record(record) for record in dataset.all_records}),
        "split_seed": seed,
        "split_fractions": dict(zip(("train", "validation", "test"), fractions, strict=True)),
        "split_summary": split_summary,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "bands": list(dataset.bands),
        "target_resolution_m": dataset.target_resolution_m,
        "target_grid_band": dataset_config["target_grid_band"],
        "value_scale": dataset.value_scale,
        "resampling": resampling_config,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
