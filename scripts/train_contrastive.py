from __future__ import annotations

import argparse
import json

from georag.config import load_config
from georag.diagnostics import select_device
from georag.training.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GeoRAG's explicit CNN and ViT NT-Xent baselines."
    )
    parser.add_argument("--config", default="configs/milestone_3.toml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--model", choices=("both", "cnn", "vit"), default="both")
    parser.add_argument("--resume", action="store_true", help="resume incomplete runs and skip completed ones")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    device = select_device(arguments.device or config.device.mode)
    model_names = ("cnn", "vit") if arguments.model == "both" else (arguments.model,)
    result = run_training(config, arguments.config, device, model_names, resume=arguments.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
