"""Rebuild the tracked M6 report figures from local experiment artifacts.

Visual specification: two matched-seed dot/slope panels for retrieval scores,
and two learning-curve panels with descriptive min/max envelopes across seeds.
The output is 178 x 86 mm; retrieval values and loss CSVs are source of truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


SEEDS = (17, 23, 42)
INK = "#171717"
MUTED = "#666C69"
HAIRLINE = "#D8DCD9"
BASELINE = "#8D9390"
FOCAL = "#1F6F5F"
PALETTE = {"rgb": BASELINE, "rgbnir": FOCAL}
MARKERS = {17: "o", 23: "s", 42: "^"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiments/milestone_6")
    parser.add_argument("--output-dir", default="reports/assets")
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (experiment_dir / "comparison.json").open(encoding="utf-8") as source:
        comparison = json.load(source)
    _plot_retrieval(comparison, output_dir)
    _plot_learning_curves(experiment_dir, output_dir)


def _plot_retrieval(comparison: dict, output_dir: Path) -> None:
    metrics = (
        ("map_at_k", "mAP@10", "Higher is better"),
        ("recall_at_k", "Recall@10", "Fraction of all relevant gallery tiles found"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.45), constrained_layout=True)
    x_positions = {"rgb": 0.0, "rgbnir": 1.0}
    seed_jitter = {17: -0.055, 23: 0.0, 42: 0.055}

    for axis, (key, label, note) in zip(axes, metrics, strict=True):
        for seed in SEEDS:
            rgb_value = comparison["results"][f"rgb_seed_{seed}"]["macro"][key]
            rgbnir_value = comparison["results"][f"rgbnir_seed_{seed}"]["macro"][key]
            x0 = x_positions["rgb"] + seed_jitter[seed]
            x1 = x_positions["rgbnir"] + seed_jitter[seed]
            axis.plot([x0, x1], [rgb_value, rgbnir_value], color=HAIRLINE,
                      linewidth=1.0, zorder=1)
            axis.scatter(x0, rgb_value, marker=MARKERS[seed], s=42,
                         color=PALETTE["rgb"], edgecolor="white", linewidth=0.65, zorder=3)
            axis.scatter(x1, rgbnir_value, marker=MARKERS[seed], s=48,
                         color=PALETTE["rgbnir"], edgecolor="white", linewidth=0.65, zorder=3)
        axis.set_xlim(-0.28, 1.40)
        axis.set_xticks([0, 1], ["RGB", "RGB + NIR"])
        axis.set_ylabel(label)
        axis.set_title(note, loc="left", fontsize=8.3, color=MUTED, pad=8)
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.grid(axis="y", color=HAIRLINE, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(HAIRLINE)
        axis.tick_params(axis="both", colors=MUTED, labelsize=8.2, length=0)

    seed_handles = [
        Line2D([0], [0], marker=MARKERS[seed], linestyle="none", markerfacecolor=INK,
               markeredgecolor="white", markersize=6, label=f"Seed {seed}")
        for seed in SEEDS
    ]
    modality_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor="white", markersize=6, label=label)
        for color, label in ((PALETTE["rgb"], "RGB"), (PALETTE["rgbnir"], "RGB + NIR"))
    ]
    figure.legend(handles=modality_handles + seed_handles, loc="lower center", ncol=5,
                  frameon=False, fontsize=8.0, bbox_to_anchor=(0.5, -0.08),
                  columnspacing=1.4, handletextpad=0.45)
    figure.suptitle(
        "RGB + NIR raises macro mAP@10 in all three seeds",
        x=0.04, ha="left", fontsize=11, fontweight="semibold", color=INK,
    )
    figure.text(
        0.04, -0.12,
        "Exact cosine Top-10; official train gallery and field-disjoint validation queries. "
        "Points are individual seeds; no seed-level confidence interval is implied.",
        fontsize=7.4, color=MUTED,
    )
    _save_figure(figure, output_dir, "milestone_6_rgb_vs_rgbnir")


def _plot_learning_curves(experiment_dir: Path, output_dir: Path) -> None:
    rows: dict[str, dict[int, list[dict[str, float]]]] = {"rgb": {}, "rgbnir": {}}
    for modality in rows:
        for seed in SEEDS:
            path = experiment_dir / f"{modality}_seed_{seed}" / "metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(f"missing completed-run metrics: {path}")
            with path.open(newline="", encoding="utf-8") as source:
                parsed = [
                    {key: float(value) for key, value in row.items() if value}
                    for row in csv.DictReader(source)
                ]
            if not parsed or len(parsed) != 20:
                raise ValueError(f"expected 20 epoch rows in {path}, found {len(parsed)}")
            rows[modality][seed] = parsed

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "train_loss", "Training NT-Xent"),
        (axes[1], "validation_loss", "Validation NT-Xent"),
    ):
        for modality, display in (("rgb", "RGB"), ("rgbnir", "RGB + NIR")):
            trajectories = np.asarray(
                [[epoch[key] for epoch in rows[modality][seed]] for seed in SEEDS],
                dtype=np.float64,
            )
            epochs = np.asarray([epoch["epoch"] for epoch in rows[modality][SEEDS[0]]])
            mean = trajectories.mean(axis=0)
            low = trajectories.min(axis=0)
            high = trajectories.max(axis=0)
            axis.plot(epochs, mean, color=PALETTE[modality], linewidth=1.8, label=display)
            axis.fill_between(epochs, low, high, color=PALETTE[modality], alpha=0.13, linewidth=0)
        axis.set_title(title, loc="left", fontsize=9, color=INK, pad=8)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("NT-Xent loss · lower is better")
        axis.set_xlim(1, 20)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axis.grid(axis="y", color=HAIRLINE, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(HAIRLINE)
        axis.tick_params(axis="both", colors=MUTED, labelsize=8.2, length=0)
    figure.suptitle("Both encoders reduce contrastive loss during training",
                    x=0.04, ha="left", fontsize=11, fontweight="semibold", color=INK)
    figure.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["rgb"], linewidth=1.8, label="RGB"),
            Line2D([0], [0], color=PALETTE["rgbnir"], linewidth=1.8, label="RGB + NIR"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=8.2,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.text(
        0.04, -0.09,
        "Lines show the mean across three seeds; shaded areas show the min–max range, "
        "not confidence intervals. Loss measures view alignment, not retrieval quality.",
        fontsize=7.4, color=MUTED,
    )
    _save_figure(figure, output_dir, "milestone_6_learning_curves")


def _save_figure(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
