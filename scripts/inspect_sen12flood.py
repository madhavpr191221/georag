"""Save deterministic true-colour, false-colour, and nodata-mask examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np

from georag.data.sen12flood import SEN12FloodDataset


def _stretch(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        values = rgb[channel][valid]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            output[channel] = np.clip((rgb[channel] - low) / max(float(high - low), 1e-6), 0, 1)
    return np.moveaxis(output, 0, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/milestone_8.toml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestone_9/data_review.png"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    d, res = config["dataset"], config["resampling"]
    dataset = SEN12FloodDataset(
        d["root"], d["bands"], d["target_resolution_m"], d["value_scale"], d["target_grid_band"],
        res["finer_than_target"], res["coarser_than_target"], res["same_resolution"], res["nodata_fill"],
    )
    wanted = []
    for label in ("no_flood", "flood"):
        matches = [i for i, record in enumerate(dataset.records) if label in record.labels]
        if not matches:
            raise ValueError(f"no available examples for label {label}")
        wanted.append(matches[len(matches) // 2])
    fig, axes = plt.subplots(len(wanted), 3, figsize=(10, 7), constrained_layout=True)
    for row, index in enumerate(wanted):
        sample = dataset[index]
        image = sample.image.numpy()
        valid = sample.valid_mask.numpy()
        true_colour = image[[3, 2, 1]]
        false_colour = image[[7, 3, 2]]  # NIR -> red, red -> green, green -> blue.
        axes[row, 0].imshow(_stretch(true_colour, valid))
        axes[row, 1].imshow(_stretch(false_colour, valid))
        axes[row, 2].imshow(valid, cmap="gray", vmin=0, vmax=1)
        label = sample.record.labels[0]
        for col, title in enumerate(("True colour (B04/B03/B02)", "False colour (B08/B04/B03)", "Common valid-data mask")):
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].set_axis_off()
        axes[row, 0].set_ylabel(label, rotation=90, size=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("SEN12-FLOOD Sentinel-2 visual and nodata inspection")
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output} ({len(wanted)} scenes)")


if __name__ == "__main__":
    main()
