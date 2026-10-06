"""Exact held-out anomaly-pattern retrieval metrics for RGB/RGB-NIR comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np


def evaluate_pattern_retrieval(
    gallery_embeddings: np.ndarray,
    gallery_records: Sequence[Mapping[str, Any]],
    query_embeddings: np.ndarray,
    query_records: Sequence[Mapping[str, Any]],
    k: int = 10,
    minimum_queries: int = 30,
    bootstrap_samples: int = 1000,
    seed: int = 17,
    query_chunk_size: int = 128,
) -> dict[str, Any]:
    """Evaluate whether same annotated patterns are retrieved from another field.

    Each annotated query is evaluated separately for every pattern it contains.
    Relevant gallery rows have that same pattern. Recall@k divides retrieved
    relevant items by all relevant gallery items; AP@k uses the truncated AP
    denominator ``min(k, number_of_relevant_gallery_items)``. Confidence
    intervals resample query field IDs, preserving clusters of tiles per field.
    """
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    queries = np.asarray(query_embeddings, dtype=np.float32)
    if gallery.ndim != 2 or queries.ndim != 2 or gallery.shape[1] != queries.shape[1]:
        raise ValueError("gallery and query embeddings must be [N,d] and [Q,d] with matching d")
    if len(gallery) != len(gallery_records) or len(queries) != len(query_records):
        raise ValueError("embedding rows must align with their records")
    if not len(gallery) or not len(queries) or not 1 <= k <= len(gallery):
        raise ValueError("gallery/query must be non-empty and k must not exceed gallery size")
    if minimum_queries <= 0 or bootstrap_samples < 0 or query_chunk_size <= 0:
        raise ValueError("minimum_queries and query_chunk_size must be positive; bootstrap_samples non-negative")
    if not np.isfinite(gallery).all() or not np.isfinite(queries).all():
        raise ValueError("embeddings must be finite")
    gallery_norm = np.linalg.norm(gallery, axis=1)
    query_norm = np.linalg.norm(queries, axis=1)
    if np.any(gallery_norm == 0) or np.any(query_norm == 0):
        raise ValueError("cosine retrieval requires non-zero embeddings")
    gallery = gallery / gallery_norm[:, None]
    queries = queries / query_norm[:, None]
    gallery_labels = [set(row.get("labels", ())) for row in gallery_records]
    query_labels = [set(row.get("labels", ())) for row in query_records]
    categories = sorted(set().union(*gallery_labels, *query_labels))
    gallery_ids = np.asarray([str(row["tile_id"]) for row in gallery_records])
    category_masks = {
        category: np.fromiter((category in labels for labels in gallery_labels), dtype=bool, count=len(gallery_labels))
        for category in categories
    }
    category_counts = {category: int(mask.sum()) for category, mask in category_masks.items()}
    query_rows: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)

    retrieval_started = time.perf_counter()
    for chunk_start in range(0, len(queries), query_chunk_size):
        chunk_end = min(chunk_start + query_chunk_size, len(queries))
        # Bound the working score matrix to [query_chunk_size, gallery_size].
        score_chunk = queries[chunk_start:chunk_end] @ gallery.T
        for local_index, score_row in enumerate(score_chunk):
            qi = chunk_start + local_index
            labels = query_labels[qi]
            if not labels:
                continue
            order = _top_k_order(score_row, gallery_ids, k)
            metadata = query_records[qi].get("metadata", {}) or {}
            query_field = str(metadata.get("field_id", query_records[qi].get("parent_scene", "unknown")))
            for category in labels:
                relevant = category_masks[category]
                total_relevant = category_counts[category]
                if total_relevant == 0:
                    continue
                hits = relevant[order]
                hit_positions = np.flatnonzero(hits)
                recall = float(hits.sum() / total_relevant)
                precision_at_hits = [(rank + 1) / (position + 1) for rank, position in enumerate(hit_positions)]
                average_precision = float(sum(precision_at_hits) / min(k, total_relevant))
                query_rows[category].append((recall, average_precision, query_field, str(query_records[qi]["tile_id"])))
    retrieval_seconds = time.perf_counter() - retrieval_started

    result: dict[str, Any] = {
        "k": k, "minimum_queries": minimum_queries, "categories": {}, "excluded_categories": {},
        "retrieval_seconds": retrieval_seconds,
        "milliseconds_per_query": retrieval_seconds * 1000 / len(query_records),
    }
    retained = []
    for category in categories:
        rows = query_rows.get(category, [])
        if len(rows) < minimum_queries:
            result["excluded_categories"][category] = {"query_count": len(rows), "reason": "below minimum_queries"}
            continue
        metrics = np.asarray([(row[0], row[1]) for row in rows], dtype=np.float64)
        retained.append(category)
        summary: dict[str, Any] = {
            "query_count": len(rows),
            "recall_at_k": float(metrics[:, 0].mean()),
            "map_at_k": float(metrics[:, 1].mean()),
        }
        result.setdefault("query_metrics", {})[category] = [
            {"tile_id": tile_id, "field_id": field, "recall_at_k": recall, "map_at_k": average_precision}
            for recall, average_precision, field, tile_id in rows
        ]
        if bootstrap_samples:
            summary["confidence_95"] = _field_bootstrap(rows, bootstrap_samples, seed)
        result["categories"][category] = summary
    if retained:
        result["macro"] = {
            key: float(np.mean([result["categories"][category][key] for category in retained]))
            for key in ("recall_at_k", "map_at_k")
        }
    else:
        result["macro"] = None
    result["query_count"] = len(query_records)
    result["gallery_count"] = len(gallery_records)
    return result


def paired_field_bootstrap_delta(
    first: Mapping[str, Any], second: Mapping[str, Any], samples: int = 1000, seed: int = 17
) -> dict[str, Any]:
    """Return second-minus-first paired deltas, resampling held-out fields."""
    categories = sorted(set(first.get("query_metrics", {})) & set(second.get("query_metrics", {})))
    category_deltas: dict[str, Any] = {}
    for category in categories:
        a = {row["tile_id"]: row for row in first["query_metrics"][category]}
        b = {row["tile_id"]: row for row in second["query_metrics"][category]}
        common = sorted(set(a) & set(b))
        rows = [(b[key]["recall_at_k"] - a[key]["recall_at_k"],
                 b[key]["map_at_k"] - a[key]["map_at_k"], a[key]["field_id"]) for key in common]
        if not rows:
            continue
        item: dict[str, Any] = {"paired_query_count": len(rows),
                               "recall_at_k_delta": float(np.mean([r[0] for r in rows])),
                               "map_at_k_delta": float(np.mean([r[1] for r in rows]))}
        if samples:
            item["confidence_95"] = _field_bootstrap(rows, samples, seed)
        category_deltas[category] = item
    return {"direction": "second_minus_first", "categories": category_deltas,
            "macro": {key: float(np.mean([item[key] for item in category_deltas.values()]))
                      for key in ("recall_at_k_delta", "map_at_k_delta")} if category_deltas else None}


def _field_bootstrap(rows: Sequence[tuple[float, float, str]], samples: int, seed: int) -> dict[str, list[float]]:
    accumulators: dict[str, list[float]] = {}
    for recall, average_precision, field, *_ in rows:
        values = accumulators.setdefault(field, [0.0, 0.0, 0])
        values[0] += recall
        values[1] += average_precision
        values[2] += 1
    fields = sorted(accumulators)
    rng = np.random.default_rng(seed)
    field_sums = np.asarray([accumulators[field][:2] for field in fields], dtype=np.float64)
    field_counts = np.asarray([accumulators[field][2] for field in fields], dtype=np.int64)
    sampled = rng.integers(0, len(fields), size=(samples, len(fields)))
    draw_sums = field_sums[sampled].sum(axis=1)
    draw_counts = field_counts[sampled].sum(axis=1)
    draws = draw_sums / draw_counts[:, None]
    return {
        "recall_at_k": [float(np.quantile(draws[:, 0], 0.025)), float(np.quantile(draws[:, 0], 0.975))],
        "map_at_k": [float(np.quantile(draws[:, 1], 0.025)), float(np.quantile(draws[:, 1], 0.975))],
    }


def _top_k_order(scores: np.ndarray, tile_ids: np.ndarray, k: int) -> np.ndarray:
    """Exact Top-k with stable tile-ID ties, avoiding a full-gallery sort."""
    if k == len(scores):
        candidates = np.arange(len(scores))
    else:
        partition = np.argpartition(scores, len(scores) - k)[len(scores) - k:]
        threshold = float(scores[partition].min())
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = k - len(above)
        if remaining < len(tied):
            tied = tied[np.argsort(tile_ids[tied], kind="stable")[:remaining]]
        candidates = np.concatenate((above, tied))
    ranked = np.lexsort((tile_ids[candidates], -scores[candidates]))
    return candidates[ranked]
