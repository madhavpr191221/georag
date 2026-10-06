from types import SimpleNamespace

import numpy as np
import pytest

from georag.evaluation.prototype_retrieval import (
    choose_family_presets,
    evaluate_class_prototypes,
    split_prototype_support,
)


def _record(tile_id: str, sequence: str, label: str):
    return SimpleNamespace(tile_id=tile_id, labels=(label,), metadata={"sequence_id": sequence})


def test_support_gallery_split_is_repeatable_sequence_disjoint_and_has_both_labels() -> None:
    records = [
        _record(f"{sequence}-{observation}", sequence, "flood" if sequence in {"a", "b", "e", "f"} else "no_flood")
        for sequence in "abcdefghij" for observation in range(2)
    ]
    support, gallery = split_prototype_support(records, support_fraction=0.3, seed=17)
    support_again, gallery_again = split_prototype_support(records, support_fraction=0.3, seed=17)
    assert support == support_again
    assert gallery == gallery_again
    assert support.isdisjoint(gallery)
    groups = {record.tile_id: record.metadata["sequence_id"] for record in records}
    assert {groups[tile_id] for tile_id in support}.isdisjoint({groups[tile_id] for tile_id in gallery})
    assert {next(record.labels[0] for record in records if record.tile_id == tile_id) for tile_id in support} == {"flood", "no_flood"}
    assert {next(record.labels[0] for record in records if record.tile_id == tile_id) for tile_id in gallery} == {"flood", "no_flood"}


def test_class_prototypes_rank_a_separate_validation_gallery_by_macro_precision() -> None:
    support = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    support_labels = ["flood", "flood", "no_flood", "no_flood"]
    validation = np.asarray([[1, 0], [0.95, 0.05], [0, 1], [0.05, 0.95]], dtype=np.float32)
    validation_labels = ["flood", "flood", "no_flood", "no_flood"]
    result = evaluate_class_prototypes(support, support_labels, validation, validation_labels, k=2)
    assert result["flood"]["precision_at_k"] == pytest.approx(1.0)
    assert result["no_flood"]["precision_at_k"] == pytest.approx(1.0)
    assert result["macro_precision_at_k"] == pytest.approx(1.0)


def test_family_selection_uses_seed_mean_then_picks_nearest_seed() -> None:
    evaluations = []
    for architecture in ("cnn", "vit"):
        for modality, scores in (("rgb", [0.55, 0.60, 0.65]), ("s2_12band", [0.70, 0.71, 0.72])):
            if architecture == "vit":
                scores = [score - 0.2 for score in scores]
            for seed, score in zip((17, 23, 42), scores, strict=True):
                evaluations.append({"run_id": f"{architecture}_{modality}_{seed}", "architecture": architecture,
                                    "modality": modality, "seed": seed,
                                    "metrics": {"macro_precision_at_k": score}})
    presets = choose_family_presets(evaluations)
    assert presets["cnn"]["modality"] == "s2_12band"
    assert presets["cnn"]["representative_seed"] == 23
    assert presets["vit"]["modality"] == "s2_12band"
    assert presets["vit"]["seed_count"] == 3
