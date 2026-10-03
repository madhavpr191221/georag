from __future__ import annotations

import argparse
import json

from georag.config import load_config
from georag.diagnostics import select_device
from georag.embeddings.runner import run_embedding_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned EuroSAT-MS embedding corpora from trained GeoRAG encoders."
    )
    parser.add_argument("--config", default="configs/milestone_4.toml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    device = select_device(arguments.device or config.device.mode)
    result = run_embedding_corpus(config, arguments.config, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
