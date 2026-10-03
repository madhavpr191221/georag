from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from georag.data.core import EOTileDataset, TileRecord, TileSample
from georag.embeddings.corpus import load_embedding_artifact, write_embedding_artifact
from georag.training.evaluation import embed_indices


class TinyTileDataset(EOTileDataset):
    def __init__(self, root: Path) -> None:
        self._records = tuple(
            TileRecord(
                tile_id=f"tile:{index}",
                image_path=root / f"tile_{index}.tif",
                labels=("forest" if index == 0 else "water",),
                metadata={"bounds": (1.0, 2.0, 3.0, 4.0)},
            )
            for index in range(3)
        )

    @property
    def records(self) -> tuple[TileRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TileSample:
        # Each row has a unique direction after flattening and normalization.
        image = torch.zeros((1, 1, 3), dtype=torch.float32)
        image.view(-1)[index] = 1.0
        return TileSample(image, self._records[index])

    def record_for_id(self, tile_id: str) -> TileRecord:
        return self._records[int(tile_id.split(":")[1])]


def test_embedding_inference_preserves_dataset_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tiles"
    dataset_root.mkdir()
    dataset = TinyTileDataset(dataset_root)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.eye(3))

    embeddings, labels, tile_ids = embed_indices(
        model, dataset, [0, 1, 2], torch.device("cpu"), batch_size=2
    )

    assert embeddings.shape == (3, 3)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3))
    assert labels == ["forest", "water", "water"]
    assert tile_ids == ["tile:0", "tile:1", "tile:2"]


def test_embedding_artifact_round_trip_keeps_aligned_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tiles"
    dataset_root.mkdir()
    records = TinyTileDataset(dataset_root).records
    assignments = {"tile:0": "train", "tile:1": "validation", "tile:2": "test"}
    vectors = np.eye(3, dtype=np.float32)
    destination = tmp_path / "cnn_seed_17"

    write_embedding_artifact(
        destination,
        vectors,
        records,
        assignments,
        dataset_root,
        {"model_name": "cnn", "random_seed": 17},
        "[embedding]\nmodels = [\"cnn\"]\n",
    )
    artifact = load_embedding_artifact(destination)

    assert artifact.embeddings.shape == (3, 3)
    assert artifact.embeddings.dtype == np.float32
    assert [row["tile_id"] for row in artifact.records] == ["tile:0", "tile:1", "tile:2"]
    assert [row["split"] for row in artifact.records] == ["train", "validation", "test"]
    assert artifact.records[0]["relative_path"] == "tile_0.tif"
    assert artifact.records[0]["metadata"]["bounds"] == [1.0, 2.0, 3.0, 4.0]
    assert artifact.manifest["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert (destination / "config.toml").is_file()


@pytest.mark.parametrize(
    ("vectors", "assignments", "error"),
    [
        (np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
         {"tile:0": "train", "tile:1": "train", "tile:2": "train"}, "L2-normalized"),
        (np.eye(3, dtype=np.float32), {"tile:0": "train"}, "cover exactly"),
    ],
)
def test_artifact_writer_rejects_invalid_vectors_or_assignments(
    tmp_path: Path, vectors: np.ndarray, assignments: dict[str, str], error: str
) -> None:
    dataset_root = tmp_path / "tiles"
    dataset_root.mkdir()
    records = TinyTileDataset(dataset_root).records
    with pytest.raises(ValueError, match=error):
        write_embedding_artifact(
            tmp_path / "artifact", vectors, records, assignments, dataset_root, {}, ""
        )


def test_artifact_loader_rejects_misaligned_record_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tiles"
    dataset_root.mkdir()
    records = TinyTileDataset(dataset_root).records
    destination = tmp_path / "artifact"
    write_embedding_artifact(
        destination,
        np.eye(3, dtype=np.float32),
        records,
        {record.tile_id: "train" for record in records},
        dataset_root,
        {},
        "",
    )
    rows = (destination / "records.jsonl").read_text(encoding="utf-8").splitlines()
    first, second = json.loads(rows[0]), json.loads(rows[1])
    first["row"], second["row"] = 1, 0
    rows[0], rows[1] = json.dumps(first), json.dumps(second)
    (destination / "records.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out of order"):
        load_embedding_artifact(destination)


def test_artifact_loader_rejects_manifest_split_count_mismatch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tiles"
    dataset_root.mkdir()
    records = TinyTileDataset(dataset_root).records
    destination = tmp_path / "artifact"
    write_embedding_artifact(
        destination,
        np.eye(3, dtype=np.float32),
        records,
        {record.tile_id: "train" for record in records},
        dataset_root,
        {},
        "",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_counts"]["train"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="split counts"):
        load_embedding_artifact(destination)


def test_artifact_writer_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    destination = tmp_path / "exists"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_embedding_artifact(destination, np.eye(1, dtype=np.float32), [], {}, tmp_path, {}, "")
