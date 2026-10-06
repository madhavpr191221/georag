"""Evaluate M10 flood/no-flood prototypes on separate validation scenes."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import torch

from georag.data.sen12flood import sequence_id_for_record
from georag.data.splits import indices_for_split, load_group_split_manifest
from georag.evaluation.prototype_retrieval import (
    choose_family_presets,
    evaluate_class_prototypes,
    split_prototype_support,
)
from georag.experiments.sen12flood_contrastive import (
    ChannelSelectionDataset,
    RGB_CHANNELS,
    _build_model,
    _embed,
)
from georag.m10_service import DEFAULT_M10_CONFIG, M10Service


def run_evaluation(config_path: str | Path = DEFAULT_M10_CONFIG, device_name: str = "auto") -> Path:
    config_path = Path(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    m9_path = Path(config["dataset"]["milestone_9_config"])
    m9 = tomllib.loads(m9_path.read_text(encoding="utf-8"))
    service = M10Service(m9_config_path=m9_path, m10_config_path=config_path)
    manifest = Path(m9["split"]["manifest"])
    assignments = load_group_split_manifest(manifest, service.source.all_records, sequence_id_for_record)
    train_records = [record for record in service.source.records if assignments[record.tile_id] == "train"]
    support_ids, gallery_ids = split_prototype_support(
        train_records, float(config["prototype"]["support_fraction"]), int(config["prototype"]["split_seed"])
    )
    validation_ids = {
        record.tile_id for record in service.source.records if assignments[record.tile_id] == "validation"
    }
    if not validation_ids:
        raise ValueError("the SEN12-FLOOD validation split is empty")
    requested = device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    batch_size = int(config["evaluation"]["batch_size"])
    k = int(config["evaluation"]["k"])
    evaluations: list[dict] = []
    support_sequence_ids = sorted({
        str((record.metadata or {}).get("sequence_id"))
        for record in train_records if record.tile_id in support_ids
    })

    for run in service.available_runs():
        if run["architecture"] not in {"cnn", "vit"} or run["modality"] not in {"rgb", "s2_12band"}:
            continue
        state = torch.load(run["directory"] / "test_retrieval.pt", map_location="cpu", weights_only=False)
        training_ids = [str(value) for value in state["gallery_tile_ids"]]
        training_embeddings = torch.as_tensor(state["gallery_embeddings"], dtype=torch.float32).numpy()
        by_id = {tile_id: training_embeddings[index] for index, tile_id in enumerate(training_ids)}
        missing_support = sorted(support_ids - set(by_id))
        if missing_support:
            raise ValueError(f"run {run['run_id']} lacks support embeddings: {missing_support[:5]}")
        support_order = sorted(support_ids)
        support_embeddings = np.stack([by_id[tile_id] for tile_id in support_order])
        record_by_id = service.record_by_id
        support_labels = [record_by_id[tile_id].labels[0] for tile_id in support_order]

        checkpoint = torch.load(run["directory"] / "checkpoint_best.pt", map_location=device, weights_only=False)
        stats = json.loads((run["directory"] / "band_statistics.json").read_text(encoding="utf-8"))
        channels = RGB_CHANNELS if run["modality"] == "rgb" else tuple(range(12))
        selected = ChannelSelectionDataset(service.dataset, channels)
        model = _build_model(run["architecture"], stats["mean"], stats["std"], checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        validation_indices = sorted(service.index_by_id[tile_id] for tile_id in validation_ids)
        val_embeddings, val_labels, val_ids = _embed(model, selected, validation_indices, device, batch_size)
        metrics = evaluate_class_prototypes(
            support_embeddings, support_labels, val_embeddings.numpy(), val_labels, k=k,
        )
        evaluations.append({
            "run_id": run["run_id"], "architecture": run["architecture"],
            "modality": run["modality"], "seed": int(run["seed"]), "metrics": metrics,
            "support_count": len(support_ids), "searchable_training_count": len(gallery_ids),
            "validation_count": len(val_ids),
        })
        del model, checkpoint, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{run['run_id']}: macro P@{k}={metrics['macro_precision_at_k']:.4f}", flush=True)

    presets = choose_family_presets(evaluations)
    payload = {
        "protocol": {
            "name": "training class prototypes ranked against held-out validation scenes",
            "support_fraction": float(config["prototype"]["support_fraction"]),
            "split_seed": int(config["prototype"]["split_seed"]),
            "support_tile_ids": sorted(support_ids),
            "support_sequence_ids": support_sequence_ids,
            "searchable_training_count": len(gallery_ids),
            "validation_count": len(validation_ids),
            "k": k,
            "metric": "macro mean of flood and no_flood label precision@k",
            "limitation": "scene-label proxy, not human relevance or pixel-level flood detection",
        },
        "evaluations": evaluations,
        "presets": presets,
    }
    destination = Path(config["selection"]["artifact"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote model selection to {destination}", flush=True)
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_M10_CONFIG))
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    arguments = parser.parse_args()
    run_evaluation(arguments.config, arguments.device)
