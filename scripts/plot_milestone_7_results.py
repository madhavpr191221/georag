"""Plot canonical/paraphrase mAP@10 from the saved M7 retrieval results.

Visual spec: paired horizontal dot plot over the full [0,1] mAP@10 scale,
ordered by the official Agriculture-Vision class list. The output is 178 mm
wide, with exact values retained in the report table; annotations are a proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


CLASS_ORDER = (
    "double_plant", "drydown", "endrow", "nutrient_deficiency",
    "planter_skip", "storm_damage", "water", "waterway", "weed_cluster",
)
LABELS = {
    "double_plant": "Double plant",
    "drydown": "Drydown",
    "endrow": "Endrow",
    "nutrient_deficiency": "Nutrient deficiency",
    "planter_skip": "Planter skip",
    "storm_damage": "Storm damage",
    "water": "Water",
    "waterway": "Waterway",
    "weed_cluster": "Weed cluster",
}
INK = "#171717"
MUTED = "#666C69"
HAIRLINE = "#D8DCD9"
BASELINE = "#8D9390"
FOCAL = "#1F6F5F"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/milestone_7/results.json")
    parser.add_argument("--output-dir", default="reports/assets")
    args = parser.parse_args()
    result_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(result_path.read_text(encoding="utf-8"))
    queries = {query["query_id"]: query for query in results["queries"]}
    categories = results["metrics"]["categories"]

    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    for row, category in enumerate(CLASS_ORDER):
        canonical = queries[f"{category}_canonical"]["metrics"]["map_at_k"]
        paraphrase = queries[f"{category}_paraphrase"]["metrics"]["map_at_k"]
        support = categories[category]["relevant_gallery_tiles"]
        y = row
        axis.plot([canonical, paraphrase], [y, y], color=HAIRLINE,
                  linewidth=1.1, zorder=1)
        axis.scatter(canonical, y, marker="o", s=42, facecolor="white",
                     edgecolor=BASELINE, linewidth=1.3, zorder=3)
        axis.scatter(paraphrase, y, marker="D", s=38, color=FOCAL,
                     edgecolor="white", linewidth=0.55, zorder=3)
        axis.text(-0.53, y, f"{LABELS[category]}  ·  {support:,} positives",
                  ha="right", va="center", fontsize=8.0, color=INK,
                  transform=axis.get_yaxis_transform())

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(len(CLASS_ORDER) - 0.5, -0.5)
    axis.set_yticks([])
    axis.set_xlabel("mAP@10 against pattern annotations · higher is better")
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.grid(axis="x", color=HAIRLINE, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(HAIRLINE)
    axis.tick_params(axis="x", colors=MUTED, labelsize=8.2, length=0, pad=5)
    figure.suptitle(
        "Text retrieval performance varies sharply by agricultural pattern",
        x=0.43, y=0.97, ha="left", fontsize=10.5,
        fontweight="bold", color=INK,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=BASELINE, markersize=6, label="Canonical wording"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=FOCAL,
               markeredgecolor="white", markersize=6, label="Paraphrase"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                  fontsize=8.0, bbox_to_anchor=(0.53, -0.055))
    figure.text(
        0.01, -0.075,
        "Two fixed prompts per pattern; annotation labels are relevance proxies. "
        "Human review of the retrieved examples is pending.",
        fontsize=7.4, color=MUTED,
    )
    figure.subplots_adjust(left=0.43, right=0.98, top=0.87, bottom=0.18)
    for suffix, options in (
        ("svg", {}),
        ("png", {"dpi": 320}),
    ):
        figure.savefig(output_dir / f"milestone_7_pattern_retrieval.{suffix}",
                       bbox_inches="tight", **options)
    plt.close(figure)


if __name__ == "__main__":
    main()
