from __future__ import annotations

import argparse
import json

from georag.config import load_config
from georag.diagnostics import collect_diagnostics, write_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoRAG device diagnostics.")
    parser.add_argument("--config", default="configs/milestone_1.toml")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    diagnostics = collect_diagnostics(config.device.mode)
    destination = write_diagnostics(config.outputs.artifacts_dir / "system" / "device.json", diagnostics)
    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
