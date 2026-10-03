"""Image-query orchestration and inspectable result artifacts for M5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Literal

import numpy as np
import torch

from georag.config import load_config
from georag.data import EuroSATMultispectralDataset
from georag.embeddings import EmbeddingArtifact, load_embedding_artifact
from georag.retrieval.flat import FlatIndex, Metric
from georag.visualization import save_retrieval_grid


def run_flat_search(
    config_path: str | Path,
    model_selection: Literal["both", "cnn", "vit"] = "both",
    query_tile_id: str | None = None,
    k: int = 5,
    metric: Metric = "cosine",
) -> dict[str, Any]:
    """Search train embeddings for one held-out validation tile and save evidence."""

    if model_selection not in {"both", "cnn", "vit"}:
        raise ValueError("model_selection must be 'both', 'cnn', or 'vit'")
    if k <= 0:
        raise ValueError("k must be positive")
    config = load_config(config_path)
    model_names = ("cnn", "vit") if model_selection == "both" else (model_selection,)
    artifacts: dict[str, EmbeddingArtifact] = {}
    for model_name in model_names:
        artifact_path = (
            config.outputs.artifacts_dir
            / "milestone_4"
            / f"{model_name}_seed_{config.reproducibility.seed}"
        )
        artifacts[model_name] = load_embedding_artifact(artifact_path)

    reference_records = artifacts[model_names[0]].records
    reference_rows = [(record["tile_id"], record["split"]) for record in reference_records]
    for model_name in model_names[1:]:
        current_rows = [
            (record["tile_id"], record["split"])
            for record in artifacts[model_name].records
        ]
        if current_rows != reference_rows:
            raise ValueError("selected embedding artifacts do not have matching tile/split row order")

    if query_tile_id is None:
        query_record = next(
            (record for record in reference_records if record["split"] == "validation"), None
        )
        if query_record is None:
            raise ValueError("embedding artifact contains no validation query tiles")
        query_tile_id = str(query_record["tile_id"])
    query_rows = [
        index for index, record in enumerate(reference_records)
        if record["tile_id"] == query_tile_id
    ]
    if len(query_rows) != 1:
        raise KeyError(f"query tile ID is not in the embedding corpus: {query_tile_id}")
    query_row = query_rows[0]
    if reference_records[query_row]["split"] != "validation":
        raise ValueError("M5 queries must come from the validation split; candidates are training tiles")
    candidate_rows = [
        index for index, record in enumerate(reference_records)
        if record["split"] == "train"
    ]
    if not candidate_rows:
        raise ValueError("embedding artifact contains no training candidates")

    output_name = f"{hashlib.sha256(query_tile_id.encode('utf-8')).hexdigest()[:12]}_{metric}_top{k}"
    destination = config.outputs.artifacts_dir / "milestone_5" / output_name
    if destination.exists():
        raise FileExistsError(f"query result already exists; refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.tmp-", dir=destination.parent))
    try:
        dataset = EuroSATMultispectralDataset(
            root=config.dataset.root,
            bands=config.dataset.bands,
            expected_shape=config.dataset.expected_shape,
            expected_count=config.dataset.expected_count,
            value_scale=config.dataset.value_scale,
            catalog_path=config.dataset.catalog,
        )
        if dataset.records[query_row].tile_id != query_tile_id:
            raise RuntimeError("embedding row order does not match the current dataset catalog")
        query_sample = dataset[dataset.index_for_id(query_tile_id)]
        query_metadata = dict(reference_records[query_row])
        all_results: dict[str, Any] = {}

        for model_name in model_names:
            artifact = artifacts[model_name]
            matrix = artifact.embeddings
            database = matrix[candidate_rows]
            database_records = [reference_records[index] for index in candidate_rows]
            index = FlatIndex(database, database_records)
            query = matrix[query_row]
            started = time.perf_counter_ns()
            results = index.search(query, k=min(k, len(candidate_rows)), metric=metric)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000

            result_payload = {
                "query": query_metadata,
                "model_name": model_name,
                "embedding_artifact": str(
                    (config.outputs.artifacts_dir / "milestone_4" / f"{model_name}_seed_{config.reproducibility.seed}")
                    .relative_to(config.root)
                ),
                "retrieval": {
                    "method": "exact_flat_numpy",
                    "metric": metric,
                    "top_k_requested": k,
                    "results_returned": len(results),
                    "query_split": "validation",
                    "candidate_split": "train",
                    "candidate_count": len(candidate_rows),
                    "embedding_dimension": int(matrix.shape[1]),
                    "search_latency_ms": latency_ms,
                },
                "results": [
                    {
                        "rank": result.rank,
                        "tile_id": result.tile_id,
                        "score": result.score,
                        "record": dict(result.record),
                    }
                    for result in results
                ],
            }
            (temporary / f"{model_name}_results.json").write_text(
                json.dumps(result_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            result_samples = [
                (
                    dataset[dataset.index_for_id(result.tile_id)],
                    result.rank,
                    result.score,
                )
                for result in results
            ]
            save_retrieval_grid(
                query_sample,
                result_samples,
                metric,
                temporary / f"{model_name}_results.png",
                config.dataset.bands,
                config.visualization.composites["rgb"],
                config.visualization.lower_percentile,
                config.visualization.upper_percentile,
            )
            all_results[model_name] = result_payload
            print(
                f"{model_name.upper()} exact {metric} Top-{len(results)} "
                f"for {query_tile_id}: {latency_ms:.3f} ms",
                flush=True,
            )

        summary = {
            "query_tile_id": query_tile_id,
            "query_split": "validation",
            "candidate_split": "train",
            "metric": metric,
            "top_k": k,
            "models": list(model_names),
            "result_files": {
                name: {
                    "json": f"{name}_results.json",
                    "visualization": f"{name}_results.png",
                }
                for name in model_names
            },
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Wrote exact retrieval results: {destination}", flush=True)
    return {"output_directory": str(destination), "results": all_results}
