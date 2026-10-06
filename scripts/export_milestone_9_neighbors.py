"""Export exact test-to-train Top-10 neighbors and local RGB inspection sheets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from georag.data.sen12flood import SEN12FloodDataset
from georag.data.sen12flood_cache import SEN12FloodTensorCache


def _display(sample) -> np.ndarray:
    image = sample.image[[3, 2, 1]].numpy()
    mask = sample.valid_mask.numpy() if sample.valid_mask is not None else np.ones(image.shape[-2:], dtype=bool)
    rgb = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        values = image[channel][mask]
        low, high = np.percentile(values, [2, 98])
        rgb[channel] = np.clip((image[channel] - low) / max(float(high - low), 1e-6), 0, 1)
    return np.moveaxis(rgb, 0, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/milestone_9.toml"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    m8 = tomllib.loads(Path(config["dataset"]["config"]).read_text(encoding="utf-8"))
    d, res = m8["dataset"], m8["resampling"]
    base = SEN12FloodDataset(
        d["root"], d["bands"], d["target_resolution_m"], d["value_scale"], d["target_grid_band"],
        res["finer_than_target"], res["coarser_than_target"], res["same_resolution"], res["nodata_fill"],
    )
    cache = SEN12FloodTensorCache(base, config["dataset"]["cache"])
    index_by_id = {record.tile_id: index for index, record in enumerate(cache.records)}
    label_by_id = {record.tile_id: (record.labels[0] if record.labels else "") for record in cache.records}
    records_by_id = {record.tile_id: record for record in cache.records}
    for run_dir in sorted(path for path in args.experiment_dir.iterdir() if path.is_dir()):
        embedding_path = run_dir / "test_retrieval.pt"
        if not embedding_path.exists():
            continue
        payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
        gallery = F.normalize(payload["gallery_embeddings"].float(), dim=1)
        queries = F.normalize(payload["query_embeddings"].float(), dim=1)
        gallery_ids = payload["gallery_tile_ids"]
        query_ids = payload["query_tile_ids"]
        top_k = min(args.k, len(gallery_ids))
        scores, neighbors = (queries @ gallery.T).topk(top_k, dim=1)
        csv_path = run_dir / "exact_neighbors.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["query_tile_id", "query_label", "rank", "neighbor_tile_id", "neighbor_label", "cosine_score", "query_timestamp", "neighbor_timestamp", "query_sequence", "neighbor_sequence"])
            writer.writeheader()
            for query_row, query_id in enumerate(query_ids):
                query_record = records_by_id[query_id]
                for rank, neighbor_position in enumerate(neighbors[query_row].tolist(), start=1):
                    neighbor_id = gallery_ids[neighbor_position]
                    neighbor_record = records_by_id[neighbor_id]
                    writer.writerow({
                        "query_tile_id": query_id, "query_label": label_by_id[query_id], "rank": rank,
                        "neighbor_tile_id": neighbor_id, "neighbor_label": label_by_id[neighbor_id],
                        "cosine_score": f"{float(scores[query_row, rank - 1]):.8f}",
                        "query_timestamp": query_record.timestamp.isoformat() if query_record.timestamp else "",
                        "neighbor_timestamp": neighbor_record.timestamp.isoformat() if neighbor_record.timestamp else "",
                        "query_sequence": (query_record.metadata or {}).get("sequence_id", ""),
                        "neighbor_sequence": (neighbor_record.metadata or {}).get("sequence_id", ""),
                    })
        print(f"Wrote {csv_path}")

        if run_dir.name.endswith("seed_17"):
            examples = []
            for label in ("flood", "no_flood"):
                row = next((i for i, query_id in enumerate(query_ids) if label_by_id[query_id] == label), None)
                if row is not None:
                    examples.append((label, row))
            fig, axes = plt.subplots(len(examples), top_k + 1, figsize=(3 * (top_k + 1), 5 * len(examples)), constrained_layout=True)
            if len(examples) == 1:
                axes = np.expand_dims(axes, axis=0)
            for plot_row, (label, query_row) in enumerate(examples):
                query_id = query_ids[query_row]
                query_record = records_by_id[query_id]
                for col in range(top_k + 1):
                    ax = axes[plot_row, col]
                    if col == 0:
                        tile_id = query_id
                        title = f"QUERY\n{label}"
                    else:
                        neighbor_id = gallery_ids[int(neighbors[query_row, col - 1])]
                        record = records_by_id[neighbor_id]
                        tile_id = neighbor_id
                        title = f"Rank {col} | {float(scores[query_row, col - 1]):.3f}\n{label_by_id[tile_id]}"
                        title += f"\n{record.timestamp.date() if record.timestamp else 'date n/a'}"
                    ax.imshow(_display(cache[index_by_id[tile_id]]))
                    ax.set_title(title, fontsize=9)
                    ax.set_axis_off()
                axes[plot_row, 0].set_ylabel(f"{label}\n{query_record.tile_id}", fontsize=9)
            fig.suptitle(f"{run_dir.name}: exact cosine neighbors (displayed as Sentinel-2 RGB)")
            sheet = run_dir / "retrieval_examples.png"
            fig.savefig(sheet, dpi=130)
            plt.close(fig)
            print(f"Saved {sheet}")


if __name__ == "__main__":
    main()
