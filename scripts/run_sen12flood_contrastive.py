"""Run the M9 matched CNN/ViT and RGB/12-band contrastive experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from georag.experiments.sen12flood_contrastive import run_milestone_9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/milestone_9.toml"))
    parser.add_argument("--smoke", action="store_true", help="run one epoch per architecture/modality on tiny split subsets")
    parser.add_argument("--device", help="override config device, e.g. cuda or cpu")
    args = parser.parse_args()
    run_milestone_9(args.config, smoke=args.smoke, device_override=args.device)


if __name__ == "__main__":
    main()
