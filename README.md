# GeoRAG

GeoRAG is a systems-level project for retrieving Earth-observation imagery, eventually from natural-language descriptions of visual content. Its path is:

```text
EO imagery -> learned embedding -> index -> retrieval -> structured evidence
```

The EuroSAT-MS data layer, two inspectable from-scratch encoders, a contrastive training experiment, and the first embedding corpora are complete. Exact Flat image retrieval is next. Natural-language image retrieval will follow using a separate pretrained image-text model; it complements rather than replaces the from-scratch encoders. GeoRAG returns ranked evidence before attempting answer generation.

See the [GeoRAG roadmap](docs/ROADMAP.md) for the staged milestones, deliverables, and completion criteria.

## Setup and data

```powershell
uv sync
uv run python scripts/check_device.py
uv run pytest -q
```

Extract EuroSAT-MS at `data/EuroSAT_MS/EuroSAT_MS/` (see [data/README.md](data/README.md)), then prepare the catalog and split:

```powershell
uv run python scripts/prepare_eurosat.py --config configs/milestone_1.toml
uv run python scripts/audit_eurosat.py --config configs/milestone_1.toml
uv run python scripts/inspect_dataset.py --config configs/milestone_1.toml --split train --composite rgb
```

Preparation validates the local TIFF corpus and caches metadata. The audit computes training-split per-band statistics used by the encoder and creates inspection reports. GeoTIFF values are divided by the configured `value_scale` (1.0 by default); normalization is explicit per-band z-scoring from the training split. EuroSAT's TIFF band order is checked, including `B8A` as the last channel. Tensors are channel-first `[13, 64, 64]`, but dataset and model interfaces take configurable channel counts.

The deterministic label-stratified split is useful for plumbing and weak retrieval diagnostics. It does not establish geographic generalization or define semantic relevance.

## Encoders and contrastive training

Milestone 2 defines a small CNN and a small ViT directly in PyTorch. Each accepts `[B, C, 64, 64]` and returns L2-normalized `[B, 128]`. The ViT patchifies each tile into 256 tokens (patch size 4), then applies learned positional embeddings, a class token, and explicit self-attention. The architecture and per-band normalization are inspectable in `src/georag/models/`.

Milestone 3 forms two independently sampled square-grid D4 views of each tile. The same rotation/reflection is applied to every spectral band without interpolation. This assumes orientation is irrelevant to the scene representation; it preserves the full tile and its spectrum. No crop, color-jitter, or cross-tile positive pairs are introduced.

For batch size `B`, the model produces two sets of embeddings `z1` and `z2`, each `[B, d]`. The loss concatenates and normalizes them as `Z = [z1; z2]`, computes the visible score matrix `S = Z Zᵀ / τ` of shape `[2B, 2B]`, masks each self-comparison, and uses the other view of the same tile as the cross-entropy target. For row `i`, the positive column is `i+B` when `i < B`, and `i-B` otherwise. All other non-self views in the batch are negatives. This is the symmetric `2B` NT-Xent objective.

Train both models using the reproducible 30-epoch configuration:

```powershell
uv run python scripts/train_contrastive.py --config configs/milestone_3.toml
```

Use `--model cnn` or `--model vit` to run one baseline, and `--device cpu` / `--device cuda` to override automatic device selection. The default uses batch 64, AdamW (`lr=3e-4`, `weight_decay=1e-4`), temperature `0.1`, and FP32. Configured views and dataloader order are deterministic per seed and epoch. Existing run directories are never overwritten; move a completed run before reusing its name.

If training is interrupted, rerun with `--resume`. It restores the latest model and optimizer checkpoint, preserves the completed metric rows, and skips models already marked complete. A resumed run deterministically reseeds each subsequent epoch's shuffle; this is reproducible after restart, though its sample order can differ from a hypothetical uninterrupted run if the interruption preceded per-epoch seeding.

Runs are written under `experiments/milestone_3/`. Each model directory contains the copied configuration, per-epoch `metrics.csv`, `learning_curves.png`, `checkpoint_last.pt`, `checkpoint_best.pt`, `run.json`, and `retrieval_diagnostics.json`. The root contains cross-model curves and `comparison.json`. Outputs are local and ignored by Git. Loss and positive-pair top-1 track optimization; the final exact validation-to-training cosine kNN agreement at `k=1,5,10` is only a weak label proxy, not a human relevance evaluation. The test split remains untouched.

The first full CNN/ViT comparison and its limitations are documented in [the Milestone 3 report](reports/milestone_3_training.md).

## Embedding corpus

Milestone 4 encodes every EuroSAT-MS tile with each best-validation-loss checkpoint. Build the corpus with:

```powershell
uv run python scripts/embed_corpus.py --config configs/milestone_4.toml
```

Each model writes a separate local artifact under `artifacts/milestone_4/`, containing a float32 `[27000, 128]` matrix, row-aligned tile records with split assignments, a manifest, and a copy of the run configuration. Artifacts are ignored by Git and existing output directories are not overwritten.

## Exact image retrieval

Milestone 5 searches validation tiles against training embeddings with exact NumPy Flat search. Search both scratch encoders with cosine similarity and display the query beside its Top-5 results:

```powershell
uv run python scripts/search_flat.py --config configs/milestone_4.toml --model both --metric cosine --k 5
```

Use `--query-tile-id` to select a particular validation tile, or choose `--metric inner_product` / `--metric euclidean` for equivalence checks. Results and visualizations are saved under `artifacts/milestone_5/`. The example run and its limitations are documented in [the Milestone 5 report](reports/milestone_5_exact_retrieval.md).

## Current boundary

The immediate next milestone is a natural-language-to-image baseline using exact search in a separate image-text aligned embedding space. ANN indexing, metadata-aware evidence, diversification, temporal change retrieval, and optional answer generation are later staged milestones. Nearest-neighbor agreement and ANN recall are measurements of different properties; neither alone establishes that retrieved evidence is relevant.
