from __future__ import annotations

import numpy as np
import pytest
import torch

from georag.models.remoteclip import RemoteCLIPEncoder, normalize_features


class _FakeModel:
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.float()


def _fake_encoder() -> RemoteCLIPEncoder:
    encoder = object.__new__(RemoteCLIPEncoder)
    encoder.device = torch.device("cpu")
    encoder.model = _FakeModel()
    encoder.tokenizer = lambda texts: torch.tensor(
        [[float(len(text)), 1.0, 2.0] for text in texts]
    )
    return encoder


def test_normalize_features_returns_unit_float32_rows() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float16)

    normalized = normalize_features(values)

    assert normalized.dtype == torch.float32
    assert torch.linalg.vector_norm(normalized, dim=1).tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "values",
    [torch.zeros((1, 2)), torch.tensor([[float("nan"), 1.0]])],
)
def test_normalize_features_rejects_zero_or_nonfinite_rows(values: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        normalize_features(values)


def test_remoteclip_encoder_methods_return_normalized_numpy_vectors() -> None:
    encoder = _fake_encoder()
    images = torch.stack(
        [torch.ones((3, 224, 224)), torch.arange(1, 4).view(3, 1, 1).expand(3, 224, 224)]
    )

    image_vectors = encoder.encode_images(images)
    text_vectors = encoder.encode_text(["water in fields", "crop rows"], batch_size=1)

    assert image_vectors.shape == (2, 3)
    assert text_vectors.shape == (2, 3)
    assert image_vectors.dtype == text_vectors.dtype == np.float32
    assert np.linalg.norm(image_vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert np.linalg.norm(text_vectors, axis=1) == pytest.approx([1.0, 1.0])


def test_remoteclip_encoder_rejects_wrong_image_shape_and_empty_text() -> None:
    encoder = _fake_encoder()

    with pytest.raises(ValueError, match="224"):
        encoder.encode_images(torch.zeros((2, 3, 128, 128)))
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_text(["  "])
