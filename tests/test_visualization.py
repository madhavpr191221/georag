import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from georag.data.core import TileRecord, TileSample
from georag.visualization import make_composite, save_retrieval_grid, save_sample_grid
from tests.conftest import BANDS


def test_named_band_composite_has_display_shape_and_range() -> None:
    image = torch.arange(13 * 8 * 8, dtype=torch.float32).reshape(13, 8, 8)
    composite = make_composite(image, BANDS, ("B04", "B03", "B02"))
    assert composite.shape == (8, 8, 3)
    assert np.isfinite(composite).all()
    assert 0.0 <= composite.min() <= composite.max() <= 1.0


def test_unknown_composite_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown band"):
        make_composite(torch.zeros(13, 8, 8), BANDS, ("B04", "B03", "thermal"))


def test_sample_grid_is_written(tmp_path) -> None:
    sample = TileSample(
        image=torch.arange(13 * 8 * 8, dtype=torch.float32).reshape(13, 8, 8),
        record=TileRecord(
            tile_id="eurosat_ms:Forest/Forest_1",
            image_path=tmp_path / "Forest_1.tif",
            latitude=49.5,
            longitude=9.0,
            sensor="Sentinel-2 MSI",
            bands=BANDS,
            labels=("Forest",),
        ),
    )
    output = save_sample_grid(
        [sample], ("B04", "B03", "B02"), tmp_path / "inspection.png"
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_retrieval_grid_shows_query_and_ranked_results(tmp_path) -> None:
    query = TileSample(
        image=torch.rand(13, 8, 8),
        record=TileRecord(
            tile_id="query", image_path=tmp_path / "query.tif", bands=BANDS, labels=("Forest",)
        ),
    )
    retrieved = TileSample(
        image=torch.rand(13, 8, 8),
        record=TileRecord(
            tile_id="result", image_path=tmp_path / "result.tif", bands=BANDS, labels=("River",)
        ),
    )

    output = save_retrieval_grid(
        query,
        [(retrieved, 1, 0.875)],
        "cosine",
        tmp_path / "retrieval.png",
        BANDS,
        ("B04", "B03", "B02"),
    )

    assert output.is_file()
    assert output.stat().st_size > 0
