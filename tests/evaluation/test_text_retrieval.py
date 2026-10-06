from __future__ import annotations

import numpy as np
import pytest

from georag.evaluation.text_retrieval import evaluate_text_retrieval


def test_text_retrieval_uses_exact_cosine_and_macro_averages_categories() -> None:
    gallery = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    records = [
        {"tile_id": "a", "labels": ["water"]},
        {"tile_id": "b", "labels": ["water"]},
        {"tile_id": "c", "labels": ["waterway"]},
    ]
    text = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = [
        {"id": "water_a", "category": "water", "text": "standing water"},
        {"id": "water_b", "category": "water", "text": "water in farmland"},
        {"id": "waterway", "category": "waterway", "text": "a waterway"},
    ]

    result = evaluate_text_retrieval(gallery, records, text, queries, k=1)

    assert result["metrics_are_annotation_proxies"] is True
    assert result["macro"]["map_at_k"] == pytest.approx(0.75)
    assert result["macro"]["recall_at_k"] == pytest.approx(0.625)
    assert result["categories"]["water"]["prompt_count"] == 2
    assert result["categories"]["water"]["precision_at_k"] == pytest.approx(0.5)
    assert result["ranked_results"]["waterway"][0].tile_id == "c"


def test_text_retrieval_excludes_label_without_gallery_positives() -> None:
    result = evaluate_text_retrieval(
        np.eye(2, dtype=np.float32),
        [{"tile_id": "a", "labels": ["water"]}, {"tile_id": "b", "labels": []}],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        [{"id": "waterway", "category": "waterway", "text": "waterway"}],
        k=1,
    )

    assert result["categories"]["waterway"]["status"] == "excluded_no_relevant_gallery_tiles"
    assert result["macro"]["map_at_k"] is None


def test_text_retrieval_rejects_misaligned_and_invalid_inputs() -> None:
    records = [{"tile_id": "a", "labels": ["water"]}]
    query = [{"id": "water", "category": "water", "text": "water"}]
    with pytest.raises(ValueError, match="align"):
        evaluate_text_retrieval(
            np.eye(2, dtype=np.float32), records,
            np.eye(1, 2, dtype=np.float32), query, k=1,
        )
    with pytest.raises(ValueError, match="unique"):
        evaluate_text_retrieval(
            np.eye(2, dtype=np.float32),
            [{"tile_id": "a", "labels": ["water"]}, {"tile_id": "b", "labels": ["water"]}],
            np.eye(2, dtype=np.float32),
            [query[0], query[0]], k=1,
        )
