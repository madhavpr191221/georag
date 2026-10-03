"""Typed configuration loading for GeoRAG's deliberately small config surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, cast


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int


@dataclass(frozen=True)
class DeviceConfig:
    mode: str


@dataclass(frozen=True)
class OutputsConfig:
    artifacts_dir: Path
    experiments_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    adapter: str
    root: Path
    catalog: Path
    expected_count: int
    expected_shape: tuple[int, int, int]
    bands: tuple[str, ...]
    value_scale: float


@dataclass(frozen=True)
class SplitConfig:
    manifest: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    seed: int
    stratify_by_label: bool


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class VisualizationConfig:
    samples: int
    lower_percentile: float
    upper_percentile: float
    composites: dict[str, tuple[str, str, str]]


@dataclass(frozen=True)
class ModelConfig:
    statistics_csv: Path
    embedding_dim: int
    cnn_channels: tuple[int, int, int]
    vit_patch_size: int
    vit_width: int
    vit_depth: int
    vit_heads: int
    vit_mlp_ratio: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    temperature: float
    retrieval_ks: tuple[int, ...]


@dataclass(frozen=True)
class EmbeddingConfig:
    models: tuple[str, ...]
    checkpoint_filename: str
    batch_size: int


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    reproducibility: ReproducibilityConfig
    device: DeviceConfig
    outputs: OutputsConfig
    dataset: DatasetConfig
    split: SplitConfig
    loader: LoaderConfig
    visualization: VisualizationConfig
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    embedding: EmbeddingConfig | None = None


def _require_keys(section: str, data: dict[str, Any], expected: set[str]) -> None:
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing:
        raise ValueError(f"[{section}] is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"[{section}] has unknown keys: {sorted(unknown)}")


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(path: str | Path) -> ProjectConfig:
    """Load TOML and reject misspelled or internally inconsistent settings."""

    config_path = Path(path).resolve()
    project_root = config_path.parent.parent
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    required_root_keys = {
        "reproducibility", "device", "outputs", "dataset", "split", "loader", "visualization"
    }
    missing_root_keys = required_root_keys - raw.keys()
    unknown_root_keys = raw.keys() - required_root_keys - {"model", "training", "embedding"}
    if missing_root_keys:
        raise ValueError(f"[root] is missing keys: {sorted(missing_root_keys)}")
    if unknown_root_keys:
        raise ValueError(f"[root] has unknown keys: {sorted(unknown_root_keys)}")
    _require_keys("reproducibility", raw["reproducibility"], {"seed"})
    _require_keys("device", raw["device"], {"mode"})
    _require_keys("outputs", raw["outputs"], {"artifacts_dir", "experiments_dir"})
    _require_keys(
        "dataset", raw["dataset"],
        {"adapter", "root", "catalog", "expected_count", "expected_shape", "bands", "value_scale"},
    )
    _require_keys(
        "split", raw["split"],
        {"manifest", "train_fraction", "validation_fraction", "test_fraction", "seed", "stratify_by_label"},
    )
    _require_keys("loader", raw["loader"], {"batch_size", "num_workers", "pin_memory"})
    _require_keys(
        "visualization", raw["visualization"],
        {"samples", "lower_percentile", "upper_percentile", "composites"},
    )

    device_mode = str(raw["device"]["mode"])
    if device_mode not in {"auto", "cpu", "cuda"}:
        raise ValueError("device.mode must be one of: auto, cpu, cuda")

    raw_shape = tuple(int(value) for value in raw["dataset"]["expected_shape"])
    if len(raw_shape) != 3 or any(value <= 0 for value in raw_shape):
        raise ValueError("dataset.expected_shape must be [C, H, W] with positive values")
    shape = cast(tuple[int, int, int], raw_shape)
    bands = tuple(str(value) for value in raw["dataset"]["bands"])
    if len(bands) != shape[0] or len(set(bands)) != len(bands):
        raise ValueError("dataset.bands must be unique and match expected_shape[0]")
    value_scale = float(raw["dataset"]["value_scale"])
    if value_scale <= 0:
        raise ValueError("dataset.value_scale must be positive")
    expected_count = int(raw["dataset"]["expected_count"])
    if expected_count <= 0:
        raise ValueError("dataset.expected_count must be positive")

    fractions = (
        float(raw["split"]["train_fraction"]),
        float(raw["split"]["validation_fraction"]),
        float(raw["split"]["test_fraction"]),
    )
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("split fractions must each lie strictly between zero and one")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")

    loader = raw["loader"]
    if int(loader["batch_size"]) <= 0 or int(loader["num_workers"]) < 0:
        raise ValueError("loader.batch_size must be positive and num_workers non-negative")

    visualization = raw["visualization"]
    if int(visualization["samples"]) <= 0:
        raise ValueError("visualization.samples must be positive")
    lower = float(visualization["lower_percentile"])
    upper = float(visualization["upper_percentile"])
    if not 0 <= lower < upper <= 100:
        raise ValueError("visualization percentiles must satisfy 0 <= lower < upper <= 100")
    composites: dict[str, tuple[str, str, str]] = {}
    for name, selected_bands in visualization["composites"].items():
        selected = tuple(str(value) for value in selected_bands)
        if len(selected) != 3 or any(value not in bands for value in selected):
            raise ValueError(f"visualization composite {name!r} must name three configured bands")
        composites[str(name)] = cast(tuple[str, str, str], selected)

    model_config = None
    if "model" in raw:
        model_raw = raw["model"]
        _require_keys(
            "model",
            model_raw,
            {
                "statistics_csv", "embedding_dim", "cnn_channels", "vit_patch_size",
                "vit_width", "vit_depth", "vit_heads", "vit_mlp_ratio",
            },
        )
        cnn_channels = tuple(int(value) for value in model_raw["cnn_channels"])
        if len(cnn_channels) != 3 or any(value <= 0 or value % 8 for value in cnn_channels):
            raise ValueError("model.cnn_channels must contain three positive multiples of 8")
        embedding_dim = int(model_raw["embedding_dim"])
        patch_size = int(model_raw["vit_patch_size"])
        vit_width = int(model_raw["vit_width"])
        vit_depth = int(model_raw["vit_depth"])
        vit_heads = int(model_raw["vit_heads"])
        vit_mlp_ratio = int(model_raw["vit_mlp_ratio"])
        if embedding_dim <= 0:
            raise ValueError("model.embedding_dim must be positive")
        if patch_size <= 0 or shape[1] % patch_size or shape[2] % patch_size:
            raise ValueError("model.vit_patch_size must divide both configured image dimensions")
        if vit_width <= 0 or vit_heads <= 0 or vit_width % vit_heads:
            raise ValueError("model.vit_width must be positive and divisible by model.vit_heads")
        if vit_depth <= 0 or vit_mlp_ratio <= 0:
            raise ValueError("model.vit_depth and model.vit_mlp_ratio must be positive")
        model_config = ModelConfig(
            statistics_csv=_project_path(project_root, str(model_raw["statistics_csv"])),
            embedding_dim=embedding_dim,
            cnn_channels=cast(tuple[int, int, int], cnn_channels),
            vit_patch_size=patch_size,
            vit_width=vit_width,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            vit_mlp_ratio=vit_mlp_ratio,
        )

    training_config = None
    if "training" in raw:
        training_raw = raw["training"]
        _require_keys(
            "training",
            training_raw,
            {"epochs", "batch_size", "learning_rate", "weight_decay", "temperature", "retrieval_ks"},
        )
        epochs = int(training_raw["epochs"])
        batch_size = int(training_raw["batch_size"])
        learning_rate = float(training_raw["learning_rate"])
        weight_decay = float(training_raw["weight_decay"])
        temperature = float(training_raw["temperature"])
        retrieval_ks = tuple(int(value) for value in training_raw["retrieval_ks"])
        if epochs <= 0 or batch_size < 2:
            raise ValueError("training.epochs must be positive and training.batch_size at least 2")
        if learning_rate <= 0 or weight_decay < 0 or temperature <= 0:
            raise ValueError("training.learning_rate and temperature must be positive; weight_decay non-negative")
        if not retrieval_ks or any(value <= 0 for value in retrieval_ks):
            raise ValueError("training.retrieval_ks must contain positive integers")
        if tuple(sorted(set(retrieval_ks))) != retrieval_ks:
            raise ValueError("training.retrieval_ks must be unique and increasing")
        training_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            temperature=temperature,
            retrieval_ks=retrieval_ks,
        )

    embedding_config = None
    if "embedding" in raw:
        embedding_raw = raw["embedding"]
        _require_keys(
            "embedding", embedding_raw, {"models", "checkpoint_filename", "batch_size"}
        )
        models = tuple(str(name) for name in embedding_raw["models"])
        if not models or any(name not in {"cnn", "vit"} for name in models):
            raise ValueError("embedding.models must select 'cnn', 'vit', or both")
        if len(set(models)) != len(models):
            raise ValueError("embedding.models must not contain duplicates")
        checkpoint_filename = str(embedding_raw["checkpoint_filename"])
        if not checkpoint_filename or Path(checkpoint_filename).name != checkpoint_filename:
            raise ValueError("embedding.checkpoint_filename must be a filename, not a path")
        embedding_batch_size = int(embedding_raw["batch_size"])
        if embedding_batch_size <= 0:
            raise ValueError("embedding.batch_size must be positive")
        embedding_config = EmbeddingConfig(models, checkpoint_filename, embedding_batch_size)

    return ProjectConfig(
        root=project_root,
        reproducibility=ReproducibilityConfig(int(raw["reproducibility"]["seed"])),
        device=DeviceConfig(device_mode),
        outputs=OutputsConfig(
            _project_path(project_root, raw["outputs"]["artifacts_dir"]),
            _project_path(project_root, raw["outputs"]["experiments_dir"]),
        ),
        dataset=DatasetConfig(
            adapter=str(raw["dataset"]["adapter"]),
            root=_project_path(project_root, raw["dataset"]["root"]),
            catalog=_project_path(project_root, raw["dataset"]["catalog"]),
            expected_count=expected_count,
            expected_shape=shape,
            bands=bands,
            value_scale=value_scale,
        ),
        split=SplitConfig(
            manifest=_project_path(project_root, raw["split"]["manifest"]),
            train_fraction=fractions[0], validation_fraction=fractions[1], test_fraction=fractions[2],
            seed=int(raw["split"]["seed"]),
            stratify_by_label=bool(raw["split"]["stratify_by_label"]),
        ),
        loader=LoaderConfig(
            int(loader["batch_size"]), int(loader["num_workers"]), bool(loader["pin_memory"])
        ),
        visualization=VisualizationConfig(
            int(visualization["samples"]), lower, upper, composites
        ),
        model=model_config,
        training=training_config,
        embedding=embedding_config,
    )
