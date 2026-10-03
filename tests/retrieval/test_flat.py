from __future__ import annotations

import numpy as np
import pytest

from georag.retrieval import FlatIndex


def _records(*tile_ids: str) -> list[dict[str, str]]:
    return [{"tile_id": tile_id, "label": f"label_{tile_id}"} for tile_id in tile_ids]


def test_flat_search_returns_hand_computed_order_and_metadata() -> None:
    vectors = np.asarray(
        [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32
    )
    records = _records("a", "b", "c", "d")
    index = FlatIndex(vectors, records)

    cosine = index.search(np.asarray([1, 0], dtype=np.float32), 4, "cosine")
    inner_product = index.search(np.asarray([1, 0], dtype=np.float32), 4, "inner_product")
    euclidean = index.search(np.asarray([1, 0], dtype=np.float32), 4, "euclidean")

    expected_order = ["a", "b", "c", "d"]
    assert [result.tile_id for result in cosine] == expected_order
    assert [result.tile_id for result in inner_product] == expected_order
    assert [result.tile_id for result in euclidean] == expected_order
    assert [result.rank for result in cosine] == [1, 2, 3, 4]
    assert cosine[1].score == pytest.approx(0.8)
    assert euclidean[1].score == pytest.approx(np.sqrt(0.4))
    assert cosine[1].record["label"] == "label_b"


def test_cosine_and_inner_product_differ_for_unnormalized_vectors() -> None:
    vectors = np.asarray([[0.7, 0], [0.9, 0.5]], dtype=np.float32)
    index = FlatIndex(vectors, _records("aligned", "large_norm"))
    query = np.asarray([1, 0], dtype=np.float32)

    cosine = index.search(query, 1, "cosine")
    inner_product = index.search(query, 1, "inner_product")

    assert cosine[0].tile_id == "aligned"
    assert inner_product[0].tile_id == "large_norm"


def test_unit_vectors_have_identical_rankings_for_all_metrics() -> None:
    generator = np.random.default_rng(7)
    vectors = generator.normal(size=(100, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = generator.normal(size=16).astype(np.float32)
    query /= np.linalg.norm(query)
    index = FlatIndex(vectors, _records(*(f"tile_{i:03}" for i in range(len(vectors)))))

    rankings = [
        [result.tile_id for result in index.search(query, 20, metric)]
        for metric in ("cosine", "inner_product", "euclidean")
    ]

    assert rankings[0] == rankings[1] == rankings[2]


def test_ties_are_broken_by_tile_id() -> None:
    index = FlatIndex(
        np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        _records("z_duplicate", "a_duplicate", "other"),
    )

    results = index.search(np.asarray([1, 0], dtype=np.float32), 2)

    assert [result.tile_id for result in results] == ["a_duplicate", "z_duplicate"]


@pytest.mark.parametrize(
    ("vectors", "records", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), [], "non-empty"),
        (np.asarray([[1, 0]], dtype=np.float32), [], "one metadata record"),
        (np.asarray([[np.nan, 0]], dtype=np.float32), _records("bad"), "finite"),
    ],
)
def test_flat_index_rejects_invalid_database(
    vectors: np.ndarray, records: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FlatIndex(vectors, records)


def test_flat_search_rejects_invalid_query_metric_and_k() -> None:
    index = FlatIndex(np.eye(2, dtype=np.float32), _records("a", "b"))
    with pytest.raises(ValueError, match="shape"):
        index.search(np.asarray([[1, 0]], dtype=np.float32), 1)
    with pytest.raises(ValueError, match="finite"):
        index.search(np.asarray([np.inf, 0], dtype=np.float32), 1)
    with pytest.raises(ValueError, match="zero-norm"):
        index.search(np.zeros(2, dtype=np.float32), 1, "cosine")
    with pytest.raises(ValueError, match="metric"):
        index.search(np.asarray([1, 0], dtype=np.float32), 1, "manhattan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 1 and database size"):
        index.search(np.asarray([1, 0], dtype=np.float32), 3)
