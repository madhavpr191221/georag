import torch

from georag.training.augmentations import make_d4_views, transform_d4


def test_d4_is_eight_distinct_band_synchronous_grid_symmetries() -> None:
    plane = torch.arange(64).reshape(8, 8)
    image = torch.stack((plane, plane + 100, plane + 200))
    outputs = [
        transform_d4(image, rotation, flip)
        for rotation in range(4)
        for flip in (False, True)
    ]

    assert len({output[0].numpy().tobytes() for output in outputs}) == 8
    for output in outputs:
        assert torch.equal(output[1] - output[0], torch.full((8, 8), 100))
        assert torch.equal(output[2] - output[0], torch.full((8, 8), 200))
        assert torch.equal(output[0].flatten().sort().values, plane.flatten().sort().values)


def test_tile_views_are_deterministic_for_seed_epoch_and_id() -> None:
    image = torch.arange(3 * 8 * 8).reshape(3, 8, 8)
    first = make_d4_views(image, "tile-17", seed=9, epoch=2)
    second = make_d4_views(image, "tile-17", seed=9, epoch=2)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
