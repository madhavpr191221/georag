"""Plot M9 learning curves and exact held-out label-proxy comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="an M9 experiment directory containing run.json")
    args = parser.parse_args()
    root = args.run_dir
    completed_path = root / "comparison.json"
    results = json.loads(completed_path.read_text(encoding="utf-8"))["results"] if completed_path.exists() else []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = {"cnn_rgb": "#457b9d", "cnn_s2_12band": "#2a9d8f", "vit_rgb": "#e9c46a", "vit_s2_12band": "#e76f51"}
    labelled: set[str] = set()
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            continue
        values = np.genfromtxt(metrics_path, delimiter=",", names=True)
        if values.size == 0:
            continue
        values = np.atleast_1d(values)
        condition = run_dir.name.rsplit("_seed_", 1)[0]
        color = colors.get(condition, "#777777")
        label = condition if condition not in labelled else None
        labelled.add(condition)
        axes[0].plot(values["epoch"], values["train_loss"], alpha=0.55, linewidth=1, marker="o", markersize=3, color=color, label=label)
        axes[0].plot(values["epoch"], values["validation_loss"], alpha=0.85, linewidth=1, linestyle="--", marker="o", markersize=3, color=color)
    axes[0].set(title="NT-Xent optimization curves", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].set_xlim(0.5, max(1.5, float(max((line.get_xdata().max() for line in axes[0].lines), default=1)) + 0.5))
    if labelled:
        axes[0].legend(fontsize=8)

    if results:
        names, means, stds = [], [], []
        groups: dict[tuple[str, str], list[float]] = {}
        for result in results:
            key = (result["architecture"], result["modality"])
            per_class = result["test_flood_label_neighbor_agreement"]["per_class"]
            score = float(np.mean([metrics["5"] for metrics in per_class.values()]))
            groups.setdefault(key, []).append(float(score))
        for key in sorted(groups):
            names.append(f"{key[0]}\n{key[1]}")
            means.append(float(np.mean(groups[key])))
            stds.append(float(np.std(groups[key], ddof=1)) if len(groups[key]) > 1 else 0.0)
        x = np.arange(len(names))
        bar_colors = [colors.get(f"{key[0]}_{key[1]}", "#777777") for key in sorted(groups)]
        axes[1].bar(x, means, yerr=stds, capsize=4, color=bar_colors)
        axes[1].set_xticks(x, names)
        axes[1].set_ylim(0, 1)
        axes[1].set(title="Macro test label agreement, exact Top-5", ylabel="Mean per-class fraction matching query label")
    else:
        axes[1].text(0.5, 0.5, "No completed test evaluations yet", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    fig.suptitle("GeoRAG M9: SEN12-FLOOD representation experiment")
    destination = root / "milestone_9_summary.png"
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
