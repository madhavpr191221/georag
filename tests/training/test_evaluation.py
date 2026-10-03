import pytest
import torch

from georag.training.evaluation import knn_label_agreement


def test_exact_chunked_knn_label_agreement_on_separated_vectors() -> None:
    database = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    queries = torch.tensor([[0.95, 0.02], [0.02, 0.95]])

    result = knn_label_agreement(
        database,
        ["a", "a", "b", "b"],
        queries,
        ["a", "b"],
        ks=(1, 2),
        query_chunk_size=1,
    )

    assert result["overall"] == {"1": 1.0, "2": 1.0}


def test_knn_diagnostic_rejects_empty_query_set() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        knn_label_agreement(
            torch.ones(2, 3), ["a", "b"], torch.empty(0, 3), [], ks=(1,)
        )
