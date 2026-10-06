import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

import georag.experiments.spectral_retrieval as spectral_runner
from georag.experiments.spectral_retrieval import run_spectral_retrieval
from georag.data.agriculture_vision import AGRICULTURE_VISION_CLASSES


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[-1], height=array.shape[-2],
                       count=array.shape[0], dtype=array.dtype, transform=from_origin(0, 16, 1, 1)) as target:
        target.write(array)


def test_complete_spectral_pipeline_on_tiny_local_fixture(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data" / "AgricultureVision"
    for split, field_prefix in (("train", "tr"), ("val", "va")):
        for index in range(2):
            stem = f"{field_prefix}{index}_0-0-16-16"
            rgb = np.random.default_rng(index).integers(5, 250, size=(3, 16, 16), dtype=np.uint8)
            nir = np.random.default_rng(index + 4).integers(5, 250, size=(1, 16, 16), dtype=np.uint8)
            _write(data_root / split / "images" / "rgb" / f"{stem}.tif", rgb)
            _write(data_root / split / "images" / "nir" / f"{stem}.tif", nir)
            valid = np.full((1, 16, 16), 255, dtype=np.uint8)
            _write(data_root / split / "masks" / f"{stem}.tif", valid)
            _write(data_root / split / "boundaries" / f"{stem}.tif", valid)
            mask = np.zeros((1, 16, 16), dtype=np.uint8); mask[:, 2:5, 2:5] = 255
            for category in AGRICULTURE_VISION_CLASSES:
                category_mask = mask if category == "water" else np.zeros_like(mask)
                _write(data_root / split / "labels" / category / f"{stem}.tif", category_mask)

    config_dir = tmp_path / "configs"; config_dir.mkdir()
    config = config_dir / "milestone_6.toml"
    config.write_text('''[dataset]
root = "data/AgricultureVision"
catalog_cache_dir = "artifacts/milestone_6/catalog"
image_size = 16
value_scale = 255.0
[model]
embedding_dim = 8
channels = [8, 8, 8]
[training]
epochs = 1
batch_size = 2
num_workers = 0
learning_rate = 0.0003
weight_decay = 0.0001
temperature = 0.1
[experiment]
output_dir = "experiments/milestone_6"
modalities = ["rgb", "rgbnir"]
seeds = [17]
k = 1
minimum_queries = 1
bootstrap_samples = 5
example_queries = 1
query_chunk_size = 2
''', encoding="utf-8")
    def interrupt_after_training(*args, **kwargs):
        raise RuntimeError("simulated interruption after checkpoint")

    monkeypatch.setattr(spectral_runner, "_embed", interrupt_after_training)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_spectral_retrieval(config, torch.device("cpu"), smoke_limit=2)
    monkeypatch.undo()
    saved_config = tmp_path / "experiments" / "milestone_6" / "rgb_seed_17" / "config.toml"
    saved_config.write_bytes(saved_config.read_bytes().replace(b"\r\n", b"\n"))
    summary_path = run_spectral_retrieval(config, torch.device("cpu"), smoke_limit=2, resume=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_metadata"]["rgb_seed_17"]["field_disjoint"] is True
    assert "17" in summary["paired_rgbnir_minus_rgb"]
    for modality in ("rgb", "rgbnir"):
        run = tmp_path / "experiments" / "milestone_6" / f"{modality}_seed_17"
        assert (run / "checkpoint.pt").is_file()
        assert (run / "metrics.csv").is_file()
        assert (run / "loss_curve.png").is_file()
        assert (run / "retrieval_examples.png").is_file()
        assert (run / "retrieval_metrics.json").is_file()
    resumed_path = run_spectral_retrieval(config, torch.device("cpu"), smoke_limit=2, resume=True)
    resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resumed["run_metadata"]["rgb_seed_17"]["status"] == "completed"
