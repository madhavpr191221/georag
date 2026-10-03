import torch

from georag.training.contrastive import nt_xent_loss, positive_top1_accuracy


def test_nt_xent_matrix_targets_and_diagonal_mask() -> None:
    z1 = torch.eye(4)
    z2 = torch.eye(4)

    loss, logits, targets = nt_xent_loss(z1, z2, temperature=0.1)

    assert logits.shape == (8, 8)
    assert targets.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert torch.isneginf(logits.diagonal()).all()
    assert torch.isfinite(loss)
    assert positive_top1_accuracy(logits, targets) == 1.0


def test_nt_xent_backpropagates_and_aligned_pairs_are_easier() -> None:
    z1 = torch.eye(4, requires_grad=True)
    aligned = torch.eye(4)
    permuted = aligned.roll(1, dims=0)

    aligned_loss, _, _ = nt_xent_loss(z1, aligned, temperature=0.2)
    permuted_loss, _, _ = nt_xent_loss(z1, permuted, temperature=0.2)
    aligned_loss.backward()

    assert aligned_loss < permuted_loss
    assert z1.grad is not None
    assert torch.isfinite(z1.grad).all()
