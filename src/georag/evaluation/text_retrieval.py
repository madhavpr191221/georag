"""Label-proxy metrics for natural-language-to-image retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from georag.retrieval.flat import FlatIndex, SearchResult


def evaluate_text_retrieval(
    gallery_embeddings: np.ndarray,
    gallery_records: Sequence[Mapping[str, Any]],
    text_embeddings: np.ndarray,
    queries: Sequence[Mapping[str, Any]],
    k: int = 10,
) -> dict[str, Any]:
    """Retrieve exact cosine neighbors and score against annotation labels.

    One query is mapped to one Agriculture-Vision pattern label. Relevant
    gallery tiles are those carrying that annotation. Recall divides hits at
    ``k`` by all positive gallery tiles; AP is truncated at ``k``. Metrics
    are first averaged over paraphrases within a category and then macro-
    averaged across categories, so common classes do not dominate the score.

    These metrics are annotation proxies, not human judgments of semantic
    relevance. The returned ranked results remain available for inspection.
    """

    matrix = np.asarray(gallery_embeddings, dtype=np.float32)
    text_matrix = np.asarray(text_embeddings, dtype=np.float32)
    if matrix.ndim != 2 or text_matrix.ndim != 2 or matrix.shape[1] != text_matrix.shape[1]:
        raise ValueError("gallery and text embeddings must be [N,d] and [Q,d] with matching d")
    if len(matrix) != len(gallery_records):
        raise ValueError("gallery embedding rows must align with gallery records")
    if len(text_matrix) != len(queries) or len(queries) == 0:
        raise ValueError("one text embedding is required for every non-empty query list")
    if not 1 <= k <= len(matrix):
        raise ValueError("k must be between 1 and the gallery size")
    if not np.isfinite(matrix).all() or not np.isfinite(text_matrix).all():
        raise ValueError("embeddings must be finite")

    categories = [query.get("category") for query in queries]
    if any(not isinstance(category, str) or not category for category in categories):
        raise ValueError("every query must name a non-empty target category")
    query_ids = [query.get("id") for query in queries]
    if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
        raise ValueError("every query must have a non-empty string id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be unique")

    normalized_texts = _normalize_numpy(text_matrix)
    index = FlatIndex(matrix, gallery_records)
    category_membership = {
        category: np.fromiter(
            (category in set(record.get("labels", ())) for record in gallery_records),
            dtype=bool,
            count=len(gallery_records),
        )
        for category in sorted(set(categories))
    }

    ranked: dict[str, list[SearchResult]] = {}
    per_query: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row, query in enumerate(queries):
        category = str(query["category"])
        results = index.search(normalized_texts[row], k, metric="cosine")
        ranked[str(query["id"])] = results
        relevant_count = int(category_membership[category].sum())
        relevant_ranks = [
            rank for rank, result in enumerate(results, start=1)
            if category in set(result.record.get("labels", ()))
        ]
        hit_count = len(relevant_ranks)
        precision = hit_count / k
        recall = hit_count / relevant_count if relevant_count else 0.0
        ap = (
            sum((hit_index + 1) / rank for hit_index, rank in enumerate(relevant_ranks))
            / min(k, relevant_count)
            if relevant_count else 0.0
        )
        per_query[str(query["id"])] = {
            "category": category,
            "relevant_gallery_tiles": relevant_count,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "map_at_k": ap,
            "retrieved_relevant": hit_count,
        }
        if relevant_count:
            grouped[category].append((precision, recall, ap))

    per_category: dict[str, Any] = {}
    for category in sorted(set(categories)):
        query_rows = grouped.get(category, [])
        relevant_count = int(category_membership[category].sum())
        if not relevant_count:
            per_category[category] = {
                "status": "excluded_no_relevant_gallery_tiles",
                "prompt_count": sum(value == category for value in categories),
                "relevant_gallery_tiles": 0,
            }
            continue
        values = np.asarray(query_rows, dtype=np.float64)
        per_category[category] = {
            "status": "included",
            "prompt_count": len(query_rows),
            "relevant_gallery_tiles": relevant_count,
            "precision_at_k": float(values[:, 0].mean()),
            "recall_at_k": float(values[:, 1].mean()),
            "map_at_k": float(values[:, 2].mean()),
        }

    included = [value for value in per_category.values() if value["status"] == "included"]
    macro = {
        metric: float(np.mean([value[metric] for value in included])) if included else None
        for metric in ("precision_at_k", "recall_at_k", "map_at_k")
    }
    return {
        "k": k,
        "gallery_count": len(gallery_records),
        "query_count": len(queries),
        "relevance_definition": "gallery tile contains the query's target annotation label",
        "metrics_are_annotation_proxies": True,
        "macro": macro,
        "categories": per_category,
        "queries": per_query,
        "ranked_results": ranked,
    }


def _normalize_numpy(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("cosine retrieval requires non-zero text embeddings")
    return vectors / norms
