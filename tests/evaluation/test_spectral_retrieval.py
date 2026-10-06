import numpy as np

from georag.evaluation.spectral_retrieval import evaluate_pattern_retrieval


def test_pattern_retrieval_exact_metrics_and_low_support_exclusion() -> None:
    gallery = np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    gallery_records = [
        {"tile_id": "g1", "labels": ["water"], "metadata": {"field_id": "a"}},
        {"tile_id": "g2", "labels": ["water"], "metadata": {"field_id": "b"}},
        {"tile_id": "g3", "labels": ["vegetation"], "metadata": {"field_id": "c"}},
    ]
    query_records = [
        {"tile_id": "q1", "labels": ["water"], "metadata": {"field_id": "q-a"}},
        {"tile_id": "q2", "labels": ["vegetation"], "metadata": {"field_id": "q-b"}},
    ]
    result = evaluate_pattern_retrieval(gallery, gallery_records, queries, query_records, k=1, minimum_queries=1, bootstrap_samples=20)
    assert result["categories"]["water"]["recall_at_k"] == 0.5
    assert result["categories"]["water"]["map_at_k"] == 1.0
    assert result["categories"]["vegetation"]["recall_at_k"] == 1.0
    assert result["macro"]["recall_at_k"] == 0.75
    assert len(result["categories"]["water"]["confidence_95"]["recall_at_k"]) == 2


def test_category_below_minimum_query_count_is_not_in_macro() -> None:
    result = evaluate_pattern_retrieval(
        np.eye(2, dtype=np.float32),
        [{"tile_id": "a", "labels": ["rare"]}, {"tile_id": "b", "labels": []}],
        np.eye(2, dtype=np.float32),
        [{"tile_id": "q", "labels": ["rare"]}, {"tile_id": "q2", "labels": []}],
        k=1, minimum_queries=2, bootstrap_samples=0,
    )
    assert result["macro"] is None
    assert result["excluded_categories"]["rare"]["query_count"] == 1
