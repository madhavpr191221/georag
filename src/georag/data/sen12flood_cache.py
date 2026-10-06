"""Fingerprint-checked, disk-backed SEN12-FLOOD raster cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from georag.data.core import EOTileDataset, TileRecord, TileSample
from georag.data.sen12flood import SEN12FloodDataset


class SEN12FloodTensorCache(EOTileDataset):
    """Store warped imagery as float32 memmap and a common validity mask as bool.

    The image array is `[N,C,H,W]` (about 6.4 GiB for this corpus), while the
    mask is `[N,H,W]`. Arrays are published atomically after a complete build.
    """

    def __init__(self, dataset: SEN12FloodDataset, cache_dir: str | Path) -> None:
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = self.cache_dir / "images.npy"
        self.mask_path = self.cache_dir / "valid_masks.npy"
        self.metadata_path = self.cache_dir / "cache.json"
        self.fingerprint = _fingerprint(dataset)
        if not self._is_valid():
            self._build()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.shape = tuple(metadata["image_shape"])
        self._images: np.memmap | None = None
        self._masks: np.memmap | None = None
        self._record_to_index = {record.tile_id: index for index, record in enumerate(dataset.records)}

    @property
    def records(self) -> tuple[TileRecord, ...]:
        return tuple(self.dataset.records)

    def __len__(self) -> int:
        return len(self.dataset.records)

    def __getitem__(self, index: int) -> TileSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._images is None:
            self._images = np.load(self.image_path, mmap_mode="r")
            self._masks = np.load(self.mask_path, mmap_mode="r")
        assert self._masks is not None
        # Copies make tensor storage writable and independent from the memmap.
        image = torch.from_numpy(np.array(self._images[index], dtype=np.float32, copy=True))
        mask = torch.from_numpy(np.array(self._masks[index], dtype=np.bool_, copy=True))
        return TileSample(image, self.dataset.records[index], mask)

    def record_for_id(self, tile_id: str) -> TileRecord:
        try:
            return self.dataset.record_for_id(tile_id)
        except KeyError:
            raise

    def close(self) -> None:
        self._images = None
        self._masks = None

    def _is_valid(self) -> bool:
        if not (self.metadata_path.is_file() and self.image_path.is_file() and self.mask_path.is_file()):
            return False
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            images = np.load(self.image_path, mmap_mode="r")
            masks = np.load(self.mask_path, mmap_mode="r")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        height, width = self._spatial_shape()
        expected_image_shape = (len(self.dataset), len(self.dataset.bands), height, width)
        expected_mask_shape = (len(self.dataset), height, width)
        return (
            metadata.get("fingerprint") == self.fingerprint
            and tuple(images.shape) == expected_image_shape
            and tuple(masks.shape) == expected_mask_shape
            and images.dtype == np.float32
            and masks.dtype == np.bool_
        )

    def _build(self) -> None:
        if not len(self.dataset):
            raise ValueError("cannot cache an empty SEN12-FLOOD dataset")
        height, width = self._spatial_shape()
        image_shape = (len(self.dataset), len(self.dataset.bands), height, width)
        mask_shape = (len(self.dataset), height, width)
        image_tmp = self.cache_dir / "images.build.npy"
        mask_tmp = self.cache_dir / "valid_masks.build.npy"
        image_map = np.lib.format.open_memmap(image_tmp, mode="w+", dtype=np.float32, shape=image_shape)
        mask_map = np.lib.format.open_memmap(mask_tmp, mode="w+", dtype=np.bool_, shape=mask_shape)
        try:
            for index in range(len(self.dataset)):
                sample = self.dataset[index]
                if tuple(sample.image.shape) != image_shape[1:] or sample.valid_mask is None:
                    raise ValueError(f"unexpected image or mask shape for {sample.record.tile_id}")
                image_map[index] = sample.image.numpy()
                mask_map[index] = sample.valid_mask.numpy()
                if (index + 1) % 100 == 0 or index + 1 == len(self.dataset):
                    print(f"Raster cache: {index + 1:,}/{len(self.dataset):,} scenes", flush=True)
            image_map.flush()
            mask_map.flush()
        except BaseException:
            del image_map, mask_map
            image_tmp.unlink(missing_ok=True)
            mask_tmp.unlink(missing_ok=True)
            raise
        del image_map, mask_map
        os.replace(image_tmp, self.image_path)
        os.replace(mask_tmp, self.mask_path)
        metadata: dict[str, Any] = {
            "format_version": 1,
            "fingerprint": self.fingerprint,
            "tile_ids": [record.tile_id for record in self.dataset.records],
            "image_shape": list(image_shape),
            "image_dtype": "float32",
            "mask_shape": list(mask_shape),
            "mask_dtype": "bool",
            "estimated_bytes": int(np.prod(image_shape) * 4 + np.prod(mask_shape)),
        }
        temporary_metadata = self.metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_metadata, self.metadata_path)

    def _spatial_shape(self) -> tuple[int, int]:
        record = self.dataset.records[0]
        path = Path(str((record.metadata or {})["band_paths"]["B05"]))
        with rasterio.open(path) as source:
            return source.height, source.width


def _fingerprint(dataset: SEN12FloodDataset) -> str:
    material: dict[str, Any] = {
        "root": str(dataset.root),
        "bands": dataset.bands,
        "target_resolution_m": dataset.target_resolution_m,
        "target_grid_band": dataset.target_grid_band,
        "value_scale": dataset.value_scale,
        "resampling": [dataset.finer_resampling, dataset.coarser_resampling, dataset.same_resolution_resampling],
        "nodata_fill": dataset.nodata_fill,
        "records": [],
    }
    for record in dataset.records:
        paths = (record.metadata or {}).get("band_paths", {})
        files = []
        for band in dataset.bands:
            path = Path(str(paths[band]))
            stat = path.stat()
            files.append((band, str(path), stat.st_size, stat.st_mtime_ns))
        material["records"].append((record.tile_id, files))
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
