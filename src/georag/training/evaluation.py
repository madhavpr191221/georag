"""Exact, chunked embedding and weak label-neighbor diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from georag.data.core import EOTileDataset, collate_tiles


@torch.inference_mode()
def embed_indices(
    model: nn.Module,
    dataset: EOTileDataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    progress_every_batches: int = 0,
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Encode selected tiles without augmentation and return CPU vectors/labels/IDs."""
    if not indices or batch_size <= 0:
        raise ValueError("indices must be non-empty and batch_size positive")
    if progress_every_batches < 0:
        raise ValueError("progress_every_batches must be non-negative")
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_tiles,
    )
    previous_mode = model.training
    model.eval()
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[str] = []
    all_tile_ids: list[str] = []
    try:
        total_batches = len(loader)
        for batch_number, batch in enumerate(loader, start=1):
            embeddings = model(batch.images.to(device, non_blocking=True))
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.extend(record.labels[0] if record.labels else "" for record in batch.records)
            all_tile_ids.extend(record.tile_id for record in batch.records)
            if progress_every_batches and (
                batch_number % progress_every_batches == 0 or batch_number == total_batches
            ):
                print(
                    f"Embedded {min(batch_number * batch_size, len(indices)):,}/"
                    f"{len(indices):,} tiles",
                    flush=True,
                )
    finally:
        model.train(previous_mode)
    return torch.cat(all_embeddings), all_labels, all_tile_ids


def knn_label_agreement(
    database_embeddings: torch.Tensor,
    database_labels: Sequence[str],
    query_embeddings: torch.Tensor,
    query_labels: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
    device: torch.device | None = None,
    query_chunk_size: int = 128,
) -> dict[str, object]:
    """Measure same-class nearest-neighbor fractions; labels are weak proxies only."""
    if database_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("embedding matrices must be two-dimensional")
    if len(database_embeddings) == 0 or len(query_embeddings) == 0:
        raise ValueError("database and query embeddings must both be non-empty")
    if database_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("database and query embedding dimensions must match")
    if len(database_labels) != len(database_embeddings) or len(query_labels) != len(query_embeddings):
        raise ValueError("each embedding must have exactly one label")
    if not ks or any(k <= 0 or k > len(database_labels) for k in ks):
        raise ValueError("each k must lie between 1 and the database size")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must not contain duplicates")

    device = device or torch.device("cpu")
    database = torch.nn.functional.normalize(database_embeddings.to(device), dim=1)
    queries = torch.nn.functional.normalize(query_embeddings.to(device), dim=1)
    label_names = sorted(set(database_labels) | set(query_labels))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    database_codes = torch.tensor([label_to_index[label] for label in database_labels], device=device)
    query_codes = torch.tensor([label_to_index[label] for label in query_labels], device=device)
    max_k = max(ks)
    matches: list[torch.Tensor] = []
    for start in range(0, len(queries), query_chunk_size):
        stop = min(start + query_chunk_size, len(queries))
        similarities = queries[start:stop] @ database.T
        neighbor_indices = similarities.topk(max_k, dim=1).indices
        matches.append(database_codes[neighbor_indices] == query_codes[start:stop, None])
    all_matches = torch.cat(matches, dim=0).float().cpu()

    overall = {
        str(k): float(all_matches[:, :k].mean().item())
        for k in ks
    }
    per_class: dict[str, dict[str, float]] = {}
    for label in sorted(set(query_labels)):
        query_rows = torch.tensor([item == label for item in query_labels], dtype=torch.bool)
        class_matches = all_matches[query_rows]
        per_class[label] = {
            str(k): float(class_matches[:, :k].mean().item())
            for k in ks
        }
    return {
        "metric": "fraction_of_top_k_neighbors_with_same_EuroSAT_label",
        "interpretation": "weak diagnostic only; class labels are not retrieval relevance judgments",
        "overall": overall,
        "per_class": per_class,
        "query_count": len(query_labels),
        "database_count": len(database_labels),
    }
