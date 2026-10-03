"""Readable exhaustive nearest-neighbor search over dense vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Metric = Literal["cosine", "inner_product", "euclidean"]


@dataclass(frozen=True)
class SearchResult:
    """One ranked vector result and its row-aligned record."""

    rank: int
    tile_id: str
    score: float
    record: Mapping[str, Any]


class FlatIndex:
    """Score a query against every row in a dense database matrix."""

    def __init__(
        self,
        embeddings: NDArray[np.floating[Any]],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("embeddings must have non-empty shape [N, d]")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite values")
        if len(records) != len(matrix):
            raise ValueError("one metadata record is required for every embedding row")
        tile_ids = [record.get("tile_id") for record in records]
        if any(not isinstance(tile_id, str) or not tile_id for tile_id in tile_ids):
            raise ValueError("each record must contain a non-empty string tile_id")
        if len(set(tile_ids)) != len(tile_ids):
            raise ValueError("tile IDs must be unique")
        self.embeddings = np.ascontiguousarray(matrix)
        self.records = tuple(records)
        self.tile_ids = np.asarray(tile_ids, dtype=str)
        self._database_norms = np.linalg.norm(self.embeddings, axis=1)

    @property
    def size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def dimension(self) -> int:
        return self.embeddings.shape[1]

    def search(self, query: NDArray[np.floating[Any]], k: int, metric: Metric = "cosine") -> list[SearchResult]:
        """Return exact Top-k rows; similarity sorts down, distance sorts up.

        Ties are ordered by tile ID to make repeated searches deterministic.
        """

        if metric not in {"cosine", "inner_product", "euclidean"}:
            raise ValueError("metric must be 'cosine', 'inner_product', or 'euclidean'")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= self.size:
            raise ValueError(f"k must be an integer between 1 and database size ({self.size})")
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError(f"query must have shape ({self.dimension},)")
        if not np.isfinite(vector).all():
            raise ValueError("query must contain only finite values")

        if metric == "inner_product":
            scores = self.embeddings @ vector
            order = self._similarity_order(scores)
        elif metric == "cosine":
            query_norm = float(np.linalg.norm(vector))
            if query_norm == 0 or np.any(self._database_norms == 0):
                raise ValueError("cosine similarity is undefined for zero-norm vectors")
            scores = (self.embeddings @ vector) / (self._database_norms * query_norm)
            scores = np.clip(scores, -1.0, 1.0)
            order = self._similarity_order(scores)
        else:
            scores = np.linalg.norm(self.embeddings - vector[None, :], axis=1)
            order = np.lexsort((self.tile_ids, scores))

        top_indices = order[:k]
        return [
            SearchResult(
                rank=rank,
                tile_id=str(self.tile_ids[index]),
                score=float(scores[index]),
                record=self.records[index],
            )
            for rank, index in enumerate(top_indices, start=1)
        ]

    def _similarity_order(self, scores: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        return np.lexsort((self.tile_ids, -scores))
