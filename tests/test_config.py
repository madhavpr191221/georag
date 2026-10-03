from pathlib import Path

import pytest

from georag.config import load_config


def test_milestone_config_loads() -> None:
    config = load_config("configs/milestone_1.toml")
    assert config.dataset.expected_shape == (13, 64, 64)
    assert len(config.dataset.bands) == 13
    assert config.dataset.bands[-1] == "B8A"
    assert config.visualization.composites["rgb"] == ("B04", "B03", "B02")
    assert config.dataset.root.is_absolute()
    assert config.dataset.catalog.is_absolute()


def test_milestone_2_encoder_config_loads() -> None:
    config = load_config("configs/milestone_2.toml")
    assert config.model is not None
    assert config.model.embedding_dim == 128
    assert config.model.cnn_channels == (32, 64, 128)
    assert config.model.vit_patch_size == 4
    assert config.model.vit_width % config.model.vit_heads == 0
    assert config.model.statistics_csv.is_absolute()


def test_milestone_3_training_config_loads() -> None:
    config = load_config("configs/milestone_3.toml")
    assert config.training is not None
    assert config.training.epochs == 30
    assert config.training.batch_size == 64
    assert config.training.temperature == 0.1
    assert config.training.retrieval_ks == (1, 5, 10)


def test_milestone_4_embedding_config_loads() -> None:
    config = load_config("configs/milestone_4.toml")
    assert config.model is not None
    assert config.embedding is not None
    assert config.embedding.models == ("cnn", "vit")
    assert config.embedding.checkpoint_filename == "checkpoint_best.pt"
    assert config.embedding.batch_size == 64


def test_invalid_embedding_model_selection_fails_early(tmp_path: Path) -> None:
    source = Path("configs/milestone_4.toml").read_text(encoding="utf-8")
    invalid = source.replace('models = ["cnn", "vit"]', 'models = ["cnn", "cnn"]')
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    path = config_directory / "invalid.toml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_config(path)


def test_invalid_vit_head_dimension_fails_early(tmp_path: Path) -> None:
    source = Path("configs/milestone_2.toml").read_text(encoding="utf-8")
    invalid = source.replace("vit_heads = 3", "vit_heads = 5")
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    path = config_directory / "invalid.toml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="divisible by model.vit_heads"):
        load_config(path)


def test_invalid_split_fractions_fail_early(tmp_path: Path) -> None:
    source = Path("configs/milestone_1.toml").read_text(encoding="utf-8")
    invalid = source.replace("test_fraction = 0.10", "test_fraction = 0.20")
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    path = config_directory / "invalid.toml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="sum to one"):
        load_config(path)
