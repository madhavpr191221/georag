from __future__ import annotations

import numpy as np

from georag.experiments.text_retrieval import _load_gallery_cache, _write_gallery_cache


def test_gallery_cache_round_trips_only_for_matching_provenance(tmp_path) -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    records = [
        {"tile_id": "tile-a", "labels": ["water"]},
        {"tile_id": "tile-b", "labels": ["drydown"]},
    ]
    manifest = {
        "gallery_fingerprint": "gallery-v1",
        "model": {"checkpoint_sha256": "weights-v1"},
    }
    paths = [tmp_path / name for name in ("embeddings.npy", "records.json", "manifest.json")]
    _write_gallery_cache(*paths, embeddings, records, manifest)

    cached_embeddings, cached_records, reused = _load_gallery_cache(
        *paths,
        fingerprint="gallery-v1",
        checkpoint_sha256="weights-v1",
        expected_count=2,
    )

    assert reused is True
    np.testing.assert_array_equal(cached_embeddings, embeddings)
    assert cached_records == records

    stale_embeddings, stale_records, reused = _load_gallery_cache(
        *paths,
        fingerprint="gallery-v2",
        checkpoint_sha256="weights-v1",
        expected_count=2,
    )
    assert (stale_embeddings, stale_records, reused) == (None, None, False)


def test_gallery_cache_rejects_wrong_checkpoint_and_corrupt_artifacts(tmp_path) -> None:
    paths = [tmp_path / name for name in ("embeddings.npy", "records.json", "manifest.json")]
    _write_gallery_cache(
        *paths,
        np.eye(1, 2, dtype=np.float32),
        [{"tile_id": "tile-a", "labels": []}],
        {"gallery_fingerprint": "gallery", "model": {"checkpoint_sha256": "weights-a"}},
    )

    assert _load_gallery_cache(
        *paths, fingerprint="gallery", checkpoint_sha256="weights-b", expected_count=1
    ) == (None, None, False)

    paths[0].write_text("not a NumPy artifact", encoding="utf-8")
    assert _load_gallery_cache(
        *paths, fingerprint="gallery", checkpoint_sha256="weights-a", expected_count=1
    ) == (None, None, False)
