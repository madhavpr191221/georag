"""Prototype-query evaluation helpers for the SEN12-FLOOD M10 interface."""

from __future__ import annotations

from collections import defaultdict
import random
from collections.abc import Sequence
from typing import Any

import numpy as np


INTENTS = ("flood", "no_flood")


def split_prototype_support(
    records: Sequence[Any], support_fraction: float = 0.2, seed: int = 17,
) -> tuple[set[str], set[str]]:
    """Split tiles by whole sequence, reserving some tiles as prototype examples."""
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must be between zero and one")
    groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        metadata = record.metadata or {}
        group_id = metadata.get("sequence_id")
        if not group_id:
            raise ValueError(f"tile {record.tile_id} has no sequence_id for grouped split")
        groups[str(group_id)].append(record)
    if len(groups) < 2:
        raise ValueError("at least two distinct sequences are required")

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    target = round(len(records) * support_fraction)
    support_groups: set[str] = set()
    support_count = 0
    for group_id in group_ids:
        group_count = len(groups[group_id])
        if support_count == 0 or abs((support_count + group_count) - target) < abs(support_count - target):
            support_groups.add(group_id)
            support_count += group_count

    # Keep at least one whole sequence on each side.
    if not support_groups:
        support_groups.add(group_ids[0])
    support_ids = {r.tile_id for group_id in support_groups for r in groups[group_id]}
    all_ids = {record.tile_id for record in records}
    gallery_ids = all_ids - support_ids
    if not support_ids or not gallery_ids:
        raise ValueError("support split must leave both prototype examples and searchable scenes")
    support_labels = {label for record in records if record.tile_id in support_ids for label in record.labels}
    gallery_labels = {label for record in records if record.tile_id in gallery_ids for label in record.labels}
    if not set(INTENTS).issubset(support_labels) or not set(INTENTS).issubset(gallery_labels):
        # Deterministically try alternate seeds instead of silently building a one-class bank.
        if seed < 117:
            return split_prototype_support(records, support_fraction, seed + 1)
        raise ValueError("support/gallery split must contain both flood and no_flood labels")
    return support_ids, gallery_ids


def class_prototype(embeddings: np.ndarray, labels: Sequence[str], intent: str) -> np.ndarray:
    """Return a unit class-mean vector built from prototype-support examples."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(labels) != len(matrix):
        raise ValueError("embeddings and labels must be aligned [N,d]")
    if intent not in INTENTS:
        raise ValueError(f"unsupported intent: {intent}")
    selected = matrix[np.asarray(labels, dtype=str) == intent]
    if len(selected) == 0:
        raise ValueError(f"prototype examples contain no {intent} tiles")
    prototype = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    norm = float(np.linalg.norm(prototype))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("class prototype has zero or non-finite norm")
    return prototype / norm


def evaluate_class_prototypes(
    support_embeddings: np.ndarray,
    support_labels: Sequence[str],
    candidate_embeddings: np.ndarray,
    candidate_labels: Sequence[str],
    k: int = 5,
) -> dict[str, Any]:
    """Rank a separate candidate set for both intents and report label precision@k."""
    support = np.asarray(support_embeddings, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    if support.ndim != 2 or candidates.ndim != 2 or support.shape[1] != candidates.shape[1]:
        raise ValueError("support and candidate embeddings must be compatible [N,d] matrices")
    if len(support) != len(support_labels) or len(candidates) != len(candidate_labels):
        raise ValueError("embedding rows and labels must be aligned")
    if len(candidates) == 0:
        raise ValueError("candidate set cannot be empty")
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("candidate embeddings must have finite non-zero norms")
    candidate_matrix = candidates / norms
    labels = np.asarray(candidate_labels, dtype=str)
    depth = min(max(1, int(k)), len(candidates))
    details: dict[str, Any] = {}
    for intent in INTENTS:
        prototype = class_prototype(support, support_labels, intent)
        scores = candidate_matrix @ prototype
        top = np.argsort(-scores, kind="stable")[:depth]
        details[intent] = {
            "precision_at_k": float(np.mean(labels[top] == intent)),
            "candidate_count": len(candidates),
            "k": depth,
        }
    details["macro_precision_at_k"] = float(np.mean([
        details[intent]["precision_at_k"] for intent in INTENTS
    ]))
    return details


def choose_family_presets(evaluations: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose one modality and representative seed for CNN and ViT deterministically."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in evaluations:
        grouped[str(item["architecture"])][str(item["modality"])].append(item)
    selected: dict[str, dict[str, Any]] = {}
    for architecture in ("cnn", "vit"):
        modalities = grouped.get(architecture, {})
        if not modalities:
            raise ValueError(f"no validation evaluations available for {architecture}")
        summaries = []
        for modality, runs in modalities.items():
            scores = np.asarray([run["metrics"]["macro_precision_at_k"] for run in runs], dtype=np.float64)
            summaries.append({
                "modality": modality,
                "mean": float(scores.mean()),
                "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "runs": runs,
            })
        # Ties prefer the smaller RGB input, then a stable lexical modality order.
        best = sorted(summaries, key=lambda row: (-row["mean"], row["modality"] != "rgb", row["modality"]))[0]
        representative = min(
            best["runs"],
            key=lambda run: (
                abs(run["metrics"]["macro_precision_at_k"] - best["mean"]),
                int(run["seed"]),
            ),
        )
        selected[architecture] = {
            "architecture": architecture,
            "modality": best["modality"],
            "mean_macro_precision_at_k": best["mean"],
            "std_macro_precision_at_k": best["std"],
            "seed_count": len(best["runs"]),
            "representative_run_id": representative["run_id"],
            "representative_seed": representative["seed"],
            "representative_macro_precision_at_k": representative["metrics"]["macro_precision_at_k"],
        }
    return selected
