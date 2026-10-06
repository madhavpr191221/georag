from __future__ import annotations

import argparse

from georag.diagnostics import select_device
from georag.experiments.spectral_retrieval import run_spectral_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RGB and RGB+NIR image retrieval on Agriculture-Vision.")
    parser.add_argument("--config", default="configs/milestone_6.toml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-limit", type=int, default=None,
                        help="train and evaluate on a small prefix of each split for a pipeline check")
    parser.add_argument("--resume", action="store_true", help="resume interrupted runs and skip completed ones")
    args = parser.parse_args()
    output = run_spectral_retrieval(args.config, select_device(args.device), args.smoke_limit, args.resume)
    print(output)


if __name__ == "__main__":
    main()
