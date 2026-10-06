"""Adapter for the labeled Agriculture-Vision challenge directory layout.

Expected root layout (as distributed by the challenge):
``train|val/{images/{rgb,nir},labels/<pattern>,masks,boundaries}``.
RGB and NIR are read as separate rasters and returned in canonical RGBNIR order.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
import torch

from georag.data.core import EOTileDataset, TileRecord, TileSample

AGRICULTURE_VISION_CLASSES = (
    "double_plant", "drydown", "endrow", "nutrient_deficiency", "planter_skip",
    "storm_damage", "water", "waterway", "weed_cluster",
)
SPLIT_DIRECTORIES = {"train": "train", "validation": "val"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
CATALOG_CACHE_VERSION = 1
logger = logging.getLogger(__name__)


class AgricultureVisionDataset(EOTileDataset):
    """Labeled Agriculture-Vision tiles with configurable RGB or RGB+NIR input.

    ``split`` is either ``train`` or ``validation`` (mapped to disk's ``train``
    or ``val``). The test subset is deliberately excluded because the challenge
    distribution does not provide its anomaly labels.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        bands: Sequence[str] = ("R", "G", "B", "NIR"),
        image_size: int = 128,
        value_scale: float = 255.0,
        transform=None,
        max_tiles: int | None = None,
        index_workers: int = 8,
        catalog_cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if split not in SPLIT_DIRECTORIES:
            raise ValueError("split must be 'train' or 'validation'; unlabeled test is not supported")
        self.split = split
        self.split_directory = SPLIT_DIRECTORIES[split]
        self.bands = tuple(bands)
        if self.bands not in (("R", "G", "B"), ("R", "G", "B", "NIR")):
            raise ValueError("bands must be ('R','G','B') or ('R','G','B','NIR')")
        if image_size <= 0 or value_scale <= 0:
            raise ValueError("image_size and value_scale must be positive")
        self.image_size = int(image_size)
        self.value_scale = float(value_scale)
        self.transform = transform
        if max_tiles is not None and max_tiles <= 0:
            raise ValueError("max_tiles must be positive when provided")
        if index_workers <= 0:
            raise ValueError("index_workers must be positive")
        self.max_tiles = max_tiles
        self.index_workers = int(index_workers)
        self.catalog_cache_path = (
            Path(catalog_cache_dir).resolve() / f"{self.split}.json.gz"
            if catalog_cache_dir is not None and max_tiles is None else None
        )
        split_root = self.root / self.split_directory
        rgb_root = split_root / "images" / "rgb"
        nir_root = split_root / "images" / "nir"
        if not rgb_root.is_dir():
            raise FileNotFoundError(f"Agriculture-Vision RGB directory not found: {rgb_root}")
        if "NIR" in self.bands and not nir_root.is_dir():
            raise FileNotFoundError(f"Agriculture-Vision NIR directory not found: {nir_root}")

        self._records: tuple[TileRecord, ...] = self._scan_records(split_root, rgb_root, nir_root)
        if not self._records:
            raise ValueError(f"no RGB tile images found under {rgb_root}")
        self._index = {record.tile_id: i for i, record in enumerate(self._records)}
        if len(self._index) != len(self._records):
            raise ValueError("duplicate tile IDs found in Agriculture-Vision")

    @property
    def records(self) -> Sequence[TileRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TileSample:
        record = self._records[index]
        rgb_path = Path(str(record.metadata["rgb_path"]))
        bands: list[np.ndarray] = []
        with rasterio.open(rgb_path) as src:
            rgb = src.read(out_shape=(3, self.image_size, self.image_size), resampling=Resampling.bilinear)
        if rgb.shape[0] != 3:
            raise ValueError(f"expected three RGB bands in {rgb_path}, found {rgb.shape[0]}")
        bands.extend(rgb.astype(np.float32, copy=False))
        if "NIR" in self.bands:
            nir_path = Path(str(record.metadata["nir_path"]))
            with rasterio.open(nir_path) as src:
                nir = src.read(out_shape=(1, self.image_size, self.image_size), resampling=Resampling.bilinear)
            bands.append(nir[0].astype(np.float32, copy=False))
        image = torch.from_numpy(np.stack(bands)) / self.value_scale
        if not torch.isfinite(image).all():
            raise ValueError(f"tile {record.tile_id} contains non-finite values")
        if self.transform is not None:
            image = self.transform(image)
        return TileSample(image, record)

    def record_for_id(self, tile_id: str) -> TileRecord:
        try:
            return self._records[self._index[tile_id]]
        except KeyError as error:
            raise KeyError(f"unknown Agriculture-Vision tile ID: {tile_id}") from error

    def with_bands(self, bands: Sequence[str]) -> "AgricultureVisionDataset":
        """Return a cheap view over the same indexed tiles and labels, selecting channels."""
        selected = tuple(bands)
        if selected not in (("R", "G", "B"), ("R", "G", "B", "NIR")):
            raise ValueError("bands must be ('R','G','B') or ('R','G','B','NIR')")
        view = object.__new__(type(self))
        view.__dict__ = self.__dict__.copy()
        view.bands = selected
        view._records = tuple(
            TileRecord(
                tile_id=record.tile_id, image_path=record.image_path,
                latitude=record.latitude, longitude=record.longitude, timestamp=record.timestamp,
                sensor=record.sensor, bands=selected, spatial_resolution_m=record.spatial_resolution_m,
                parent_scene=record.parent_scene, labels=record.labels, metadata=record.metadata,
            ) for record in self._records
        )
        return view

    def _scan_records(self, split_root: Path, rgb_root: Path, nir_root: Path) -> tuple[TileRecord, ...]:
        print(f"Discovering {self.split} tile and annotation files...", flush=True)
        paths = sorted(path for path in rgb_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if self.max_tiles is not None:
            paths = paths[:self.max_tiles]
        print(f"Found {len(paths):,} {self.split} RGB tiles.", flush=True)
        stems = {path.stem for path in paths}
        nir_by_stem = _index_paths(nir_root, stems) if "NIR" in self.bands else {}
        validity_by_directory = {
            name: _index_paths(split_root / name, stems) for name in ("masks", "boundaries")
        }
        labels_by_category = {
            category: _index_paths(split_root / "labels" / category, stems)
            for category in AGRICULTURE_VISION_CLASSES
        }
        print(f"Indexed {self.split} RGB/NIR and annotation paths.", flush=True)
        pending: list[tuple[Path, Path | None, dict[str, str], tuple[tuple[str, str], ...]]] = []
        for rgb_path in paths:
            stem = rgb_path.stem
            nir_path = nir_by_stem.get(stem)
            if "NIR" in self.bands and nir_path is None:
                raise FileNotFoundError(f"no NIR raster matches RGB tile {rgb_path.name}")
            validity = {name: str(index.get(stem) or _missing_path(split_root / name, stem))
                        for name, index in validity_by_directory.items()}
            masks: list[tuple[str, str]] = []
            for category, index in labels_by_category.items():
                mask = index.get(stem)
                if mask is None:
                    raise FileNotFoundError(
                        f"missing {category} annotation mask for tile {stem}; "
                        "incomplete labels cannot safely be interpreted as a negative example"
                    )
                masks.append((category, str(mask)))
            pending.append((rgb_path, nir_path, validity, tuple(masks)))

        fingerprint = None
        cached_annotations = None
        if self.catalog_cache_path is not None:
            catalog_paths = [*paths, *nir_by_stem.values(), *validity_by_directory["masks"].values(),
                             *validity_by_directory["boundaries"].values()]
            catalog_paths.extend(path for index in labels_by_category.values() for path in index.values())
            print(f"Checking {self.split} annotation catalog cache...", flush=True)
            fingerprint = _file_inventory_fingerprint(self.root, catalog_paths)
            cached_annotations = _read_annotation_catalog(
                self.catalog_cache_path, self.root, self.split, fingerprint, stems
            )

        if cached_annotations is not None:
            print(f"Loaded {self.split} annotation catalog ({len(pending):,} tiles).", flush=True)
            annotations = [cached_annotations[path.stem] for path in paths]
        else:
            if self.catalog_cache_path is not None:
                print(f"Building {self.split} annotation catalog ({len(pending):,} tiles)...", flush=True)
            annotations = []
            if len(pending) < 128 or self.index_workers == 1:
                annotations = list(map(_inspect_annotation_paths, pending))
            else:
                with ThreadPoolExecutor(max_workers=self.index_workers) as pool:
                    for index, result in enumerate(pool.map(_inspect_annotation_paths, pending, chunksize=16), start=1):
                        annotations.append(result)
                        if index % 2048 == 0:
                            print(f"  inspected {index:,}/{len(pending):,} {self.split} tiles", flush=True)
            if self.catalog_cache_path is not None:
                _write_annotation_catalog(
                    self.catalog_cache_path, self.root, self.split, fingerprint,
                    [(path.stem, parent_scene, labels) for path, (parent_scene, labels) in zip(paths, annotations)],
                )
                print(f"Saved {self.split} annotation catalog: {self.catalog_cache_path}", flush=True)

        records: list[TileRecord] = []
        for (rgb_path, nir_path, validity_paths, mask_paths), (parent_scene, labels) in zip(pending, annotations):
            tile_id = rgb_path.stem
            metadata: dict[str, object] = {
                "split": self.split,
                "field_id": parent_scene,
                "rgb_path": str(rgb_path),
                "nir_path": str(nir_path) if nir_path else None,
                "mask_paths": dict(mask_paths),
                "validity_paths": validity_paths,
            }
            records.append(TileRecord(
                tile_id=f"agriculture_vision:{self.split}:{tile_id}",
                image_path=rgb_path,
                sensor="Agriculture-Vision aerial RGB-NIR",
                bands=self.bands,
                parent_scene=parent_scene,
                labels=labels,
                metadata=metadata,
            ))
        return tuple(records)


def _file_inventory_fingerprint(root: Path, paths: Sequence[Path]) -> str:
    """Hash relative paths and stat metadata without reading source image bytes."""
    digest = hashlib.sha256()
    unique_paths = sorted(set(paths), key=lambda value: value.as_posix())
    for index, path in enumerate(unique_paths, start=1):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        if index % 100_000 == 0:
            print(f"  fingerprinted {index:,}/{len(unique_paths):,} source files", flush=True)
    return digest.hexdigest()


def _read_annotation_catalog(
    cache_path: Path,
    root: Path,
    split: str,
    fingerprint: str,
    expected_stems: set[str],
) -> dict[str, tuple[str, tuple[str, ...]]] | None:
    if not cache_path.is_file():
        return None
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if (payload.get("version") != CATALOG_CACHE_VERSION
                or payload.get("root") != str(root)
                or payload.get("split") != split
                or payload.get("fingerprint") != fingerprint):
            logger.info("Ignoring stale Agriculture-Vision annotation catalog: %s", cache_path)
            return None
        catalog = {
            item["stem"]: (item["parent_scene"], tuple(item["labels"]))
            for item in payload["records"]
        }
        if len(catalog) != len(payload["records"]) or catalog.keys() != expected_stems:
            logger.warning("Ignoring incomplete Agriculture-Vision annotation catalog: %s", cache_path)
            return None
        return catalog
    except (OSError, EOFError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        logger.warning("Ignoring unreadable Agriculture-Vision annotation catalog %s: %s", cache_path, error)
        return None


def _write_annotation_catalog(
    cache_path: Path,
    root: Path,
    split: str,
    fingerprint: str,
    records: Sequence[tuple[str, str, tuple[str, ...]]],
) -> None:
    """Atomically persist labels so an interrupted full run can resume cheaply."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "version": CATALOG_CACHE_VERSION,
        "root": str(root),
        "split": split,
        "fingerprint": fingerprint,
        "records": [
            {"stem": stem, "parent_scene": parent_scene, "labels": list(labels)}
            for stem, parent_scene, labels in records
        ],
    }
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
        temporary.replace(cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _match_stem(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _index_paths(directory: Path, expected_stems: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required Agriculture-Vision directory not found: {directory}")
    paths: dict[str, Path] = {}
    if len(expected_stems) < 4096:
        # Smoke runs should touch only the requested tiles, not enumerate every
        # file in a 57k-image directory for each of the eleven annotation folders.
        for stem in expected_stems:
            path = _match_stem(directory, stem)
            if path is not None:
                paths[stem] = path
        return paths
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                path = Path(entry.path)
                if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem in expected_stems:
                    if path.stem in paths:
                        raise ValueError(f"duplicate files with stem {path.stem!r} under {directory}")
                    paths[path.stem] = path
    return paths


def _missing_path(directory: Path, stem: str) -> str:
    raise FileNotFoundError(f"missing required {directory.name} mask for labeled tile {stem}")


def _inspect_annotation_paths(
    item: tuple[Path, Path | None, dict[str, str], tuple[tuple[str, str], ...]],
) -> tuple[str, tuple[str, ...]]:
    """Read one tile's annotation rasters and verify they align with its imagery."""
    rgb_path, nir_path, validity_paths, mask_paths = item
    with Image.open(rgb_path) as source:
        image_shape = (source.height, source.width)
    if nir_path is not None:
        with Image.open(nir_path) as source:
            if (source.height, source.width) != image_shape:
                raise ValueError(f"RGB/NIR raster shape mismatch for {rgb_path.stem}")
    valid_arrays = []
    for name in ("masks", "boundaries"):
        with Image.open(validity_paths[name]) as source:
            valid = np.asarray(source)
            if valid.shape != image_shape:
                raise ValueError(f"{name} raster shape mismatch for {rgb_path.stem}")
            valid_arrays.append(valid > 0)
    valid_region = valid_arrays[0] & valid_arrays[1]
    labels = []
    for category, path in mask_paths:
        with Image.open(path) as source:
            annotation = np.asarray(source)
            if annotation.shape != image_shape:
                raise ValueError(f"{category} mask shape mismatch for {rgb_path.stem}")
            if np.any((annotation > 0) & valid_region):
                labels.append(category)
    return rgb_path.stem.split("_", 1)[0], tuple(labels)
