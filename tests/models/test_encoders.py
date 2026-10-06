import pytest
import torch

from georag.models import CNNEncoder, MultiHeadSelfAttention, PatchEmbedding, ViTEncoder


def _encoders(device: torch.device):
    mean = [100.0] * 13
    standard_deviation = [10.0] * 13
    return (
        CNNEncoder(mean, standard_deviation).to(device),
        ViTEncoder(mean, standard_deviation).to(device),
    )


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_both_encoders_return_unit_embeddings_and_gradients(device: torch.device) -> None:
    images = (torch.randn(2, 13, 64, 64) * 10 + 100).to(device)
    target = torch.linspace(-1, 1, 2 * 128, device=device).reshape(2, 128)
    for model in _encoders(device):
        output = model(images)
        assert output.shape == (2, 128)
        assert torch.isfinite(output).all()
        assert torch.allclose(torch.linalg.vector_norm(output, dim=1), torch.ones(2), atol=1e-5)
        (output * target).sum().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.float().square().sum() for gradient in gradients).sqrt() > 0


def test_patch_embedding_and_attention_shapes() -> None:
    images = torch.randn(2, 13, 64, 64)
    patch_embedding = PatchEmbedding(input_channels=13, patch_size=4, width=192)
    tokens = patch_embedding(images)
    assert tokens.shape == (2, 256, 192)

    attention = MultiHeadSelfAttention(width=192, heads=3)
    output, weights = attention(torch.randn(2, 257, 192), return_attention=True)
    assert output.shape == (2, 257, 192)
    assert weights.shape == (2, 3, 257, 257)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 257), atol=1e-6)


def test_encoder_rejects_wrong_band_count() -> None:
    model = CNNEncoder([0.0] * 13, [1.0] * 13)
    with pytest.raises(ValueError, match="input bands"):
        model(torch.randn(2, 12, 64, 64))


def test_invalid_pixels_are_zero_after_standardization() -> None:
    model = CNNEncoder([10.0, 20.0, 30.0], [2.0, 4.0, 5.0])
    image = torch.tensor([[[[12.0, -999.0], [14.0, -999.0]],
                           [[24.0, -999.0], [28.0, -999.0]],
                           [[35.0, -999.0], [40.0, -999.0]]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    normalized = model.normalizer(image, mask)
    assert torch.equal(normalized[0, :, :, 1], torch.zeros(3, 2))
    assert torch.allclose(normalized[0, :, :, 0], torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_both_encoders_run_forward_and_backward_on_cuda() -> None:
    device = torch.device("cuda")
    images = torch.randn(2, 13, 64, 64, device=device)
    target = torch.randn(2, 128, device=device)
    for model in _encoders(device):
        output = model(images)
        (output * target).sum().backward()
        assert output.is_cuda and torch.isfinite(output).all()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
