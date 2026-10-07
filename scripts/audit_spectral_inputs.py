"""Run the M11 SEN12-FLOOD spectral input and map-alignment audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from georag.data.sen12flood import SEN12FloodDataset
from georag.evaluation.spectral_audit import run_spectral_input_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone_8.toml")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--samples-per-stratum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="artifacts/milestone_11/spectral_input_audit")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    root = Path(dataset_config["root"])
    if not root.is_absolute():
        root = (config_path.parent.parent / root).resolve()
    split_path = Path(args.split_manifest or config["split"]["manifest"])
    if not split_path.is_absolute():
        split_path = (config_path.parent.parent / split_path).resolve()
    dataset = SEN12FloodDataset(
        root=root,
        bands=dataset_config["bands"],
        target_resolution_m=dataset_config["target_resolution_m"],
        value_scale=dataset_config["value_scale"],
        target_grid_band=dataset_config["target_grid_band"],
        finer_resampling=config["resampling"]["finer_than_target"],
        coarser_resampling=config["resampling"]["coarser_than_target"],
        same_resolution_resampling=config["resampling"]["same_resolution"],
        nodata_fill=config["resampling"]["nodata_fill"],
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    result = run_spectral_input_audit(
        dataset=dataset,
        split_manifest=split_path,
        split_name=args.split,
        output_dir=output_dir,
        samples_per_stratum=args.samples_per_stratum,
        seed=args.seed,
        denominator_epsilon=float(config.get("spectral", {}).get("denominator_epsilon", 1e-6)),
    )
    print(json.dumps(result, indent=2))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
