# GeoRAG

GeoRAG retrieves Earth-observation scenes as inspectable evidence for questions about physical surface conditions. Its first research question is whether multispectral Sentinel-2 representations can improve flood-evidence retrieval over matched RGB on unseen locations. The system returns ranked imagery, scores, metadata, and provenance; M9's label-agreement diagnostic is only an initial proxy for retrieval quality.

Natural language expresses a supported retrieval request. GeoRAG does not answer open-ended questions or generate explanations; retrieval quality comes first. The system path is:

```text
EO imagery -> learned embedding -> index -> retrieval -> structured evidence
```

**Current status:** M0-M11 are implemented as staged foundations. M6 and M7 are exploratory Agriculture-Vision studies; M8 established the SEN12-FLOOD data contract; M9 compared scratch CNN and ViT representations trained on RGB or twelve Sentinel-2 optical bands. M10 adds narrow natural-language query parsing, explicit date/place filters, and exact evidence retrieval. M11 adds a separate index-based multispectral search for water-like signals and same-location temporal changes. Neither scene-label agreement nor spectral indices are pixel flood maps or human relevance ground truth.

The repository already contains EuroSAT-MS tooling, scratch encoders and contrastive training, exact image retrieval, an Agriculture-Vision RGB-versus-RGB+NIR study (M6), and an exploratory RGB RemoteCLIP text-retrieval baseline (M7). Those are useful foundations and historical results, but they do not answer the new Sentinel-2 flood-retrieval question. See the [project specification](docs/PROJECT_SPEC.md) and [roadmap](docs/ROADMAP.md).

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

## M6: RGB versus RGB+NIR retrieval

M6 isolates one question: does NIR improve example-image retrieval for annotated agricultural surface patterns? It uses Agriculture-Vision challenge imagery as separate RGB and NIR rasters. Two matched from-scratch CNNs are trained with the same D4-view NT-Xent objective. Labels are not used for training; they define retrieval relevance after training. Since the challenge test split has no released anomaly labels, the official labeled validation split supplies queries and the official train split supplies the gallery. Official splits are field-disjoint.

Place the labeled challenge subset under `data/AgricultureVision/Agriculture-Vision-2021/` (see [local data notes](data/README.md)). The run is configured for 128-pixel inputs, 20 epochs, and three seeds. It builds a local annotation catalog under `artifacts/milestone_6/catalog/`, validates it against source file paths, sizes, and modification times, and reuses it on restart:

```powershell
uv run python scripts/run_spectral_retrieval.py --config configs/milestone_6.toml
```

The default compares exact cosine Top-10 retrieval for RGB and RGB+NIR across nine annotated patterns, reports category-level Recall@10 and mAP@10, excludes categories with fewer than 30 positive queries from macro scores, and bootstraps confidence intervals by query field. Results include training loss curves, checkpoints, embeddings, exact-neighbor image strips, run metadata, paired RGB+NIR-minus-RGB deltas, and a comparison plot under `experiments/milestone_6/`. `--smoke-limit 8` exercises the pipeline on a small prefix of both splits; it is a plumbing check, not a quality result. Output run directories are never overwritten. Each epoch saves an atomic latest checkpoint; resume an interrupted comparison with `--resume`.

This experiment answers only whether spectral input changes example-image retrieval under these annotations and splits. It does not establish general semantic relevance or geospatial generalization beyond this dataset. See the [M6 implementation report](reports/milestone_6_implementation.md) for the protocol, results, and limitations.

## M7 historical baseline: RGB natural-language image retrieval

M7 uses the pretrained RemoteCLIP ViT-B/32 model as an RGB text-to-image retrieval baseline. It embeds the Agriculture-Vision train gallery and a fixed set of natural-language pattern descriptions; the existing exact NumPy Flat index ranks each gallery tile by cosine similarity. RemoteCLIP weights are downloaded on first run into the ignored local artifact directory and their SHA-256 is recorded with the embedding manifest. An optional OpenAI vision audit assesses the returned RGB tiles against each query. It is a model-assisted evaluation, not answer generation or human ground truth. RemoteCLIP is RGB-only, so this is not text-to-multispectral retrieval.

Run a small execution check first (this downloads the pretrained checkpoint once):

```powershell
uv run python scripts/run_text_retrieval.py --config configs/milestone_7.toml --max-gallery 64
```

Build or reuse the complete gallery and query it with all 18 fixed descriptions:

```powershell
uv run python scripts/run_text_retrieval.py --config configs/milestone_7.toml
```

Full embeddings, ranked results, nine Top-5 image grids, and a 45-row review sheet are saved locally under `artifacts/milestone_7/` and `experiments/milestone_7/`. The full run completed: macro mAP@10 is 0.0984 and macro precision@10 is 0.15 against annotation labels, with performance concentrated in drydown and water. These are proxies, not natural-language ground truth. To run the optional VLM audit, set `OPENAI_API_KEY` in the shell and run:

```powershell
uv run python scripts/run_vlm_audit.py --limit 1  # one-pair API smoke check
uv run python scripts/run_vlm_audit.py            # all retrieved examples; resumes saved judgments
```

The audit writes separate files under `experiments/milestone_7/vlm_audit/`: resumable `judgments.jsonl`, row-level CSV, JSON summary, Markdown report, and an HTML gallery with images and model decisions. It does not modify `human_audit.csv`. Review the [M7 retrieval report](reports/milestone_7_text_retrieval.md) for the retrieval baseline and keep VLM judgments distinct from human relevance labels.

See the [GeoRAG roadmap](docs/ROADMAP.md), [M6 results report](reports/milestone_6_implementation.md), and [M7 query set](configs/milestone_7_queries.toml).

## M8-M9: SEN12-FLOOD data and representation study

The first focused task uses the [SEN12-FLOOD Kaggle dataset](https://www.kaggle.com/datasets/virajkadam/sen12flood), downloaded manually and extracted under `data/sen12flood/`. The inspected STAC collections are nested under `data/sen12flood/sen12flood/`. This directory is ignored by Git. Do not commit the archive, imagery, extracted data, or result galleries containing source imagery. Kaggle currently reports the dataset license as unknown; record that status and do not redistribute the data.

M8 now includes a STAC-backed Sentinel-2 adapter, a 20 m B05-anchored twelve-band loader, and a deterministic location-group split. Prepare the local catalog and manifest with `uv run python scripts/prepare_sen12flood.py --config configs/milestone_8.toml`. The verified catalog has 2,236 records, 2,138 available scenes, 98 missing-image records, and 335 exact sequence tokens. Keep tokens as strings: `0001` and `1` refer to different locations. The upstream documentation reports 337 locations; that discrepancy remains documented, while the local split uses only observed IDs. The [M8 audit report](reports/milestone_8_sen12flood_data_audit.md) records the adapter contract, split fingerprint, and data limitations. See [local data notes](data/README.md).

M9 compares four from-scratch representation conditions: CNN or small ViT, each trained on Sentinel-2 RGB (B04/B03/B02) or all twelve optical bands. Paired views use exact 90-degree rotations/reflections; the validity mask undergoes the same transform. Training-only valid-pixel statistics prevent nodata fill values from affecting normalization. A disk-backed cache under ignored `artifacts/milestone_9/cache/` stores the aligned image stack and masks. Create an inspection figure and run a one-epoch smoke test with:

```powershell
uv run python scripts/inspect_sen12flood.py
uv run python scripts/run_sen12flood_contrastive.py --smoke
```

The full configuration uses NT-Xent, AdamW, batch size 16 original tiles, 20 epochs, and seeds 17/23/42. Run with `uv run python scripts/run_sen12flood_contrastive.py`. Each run saves checkpoints, training/validation curves, exact test-to-train cosine neighbors, and a flood-label neighbor-agreement diagnostic. That label agreement is only a weak proxy: it does not measure human relevance, spatial flood extent, or generalized natural-language understanding. Details and final results belong in the [M9 report](reports/milestone_9_sen12flood_representation.md).

After a completed experiment, export inspectable exact neighbors and representative result sheets with `uv run python scripts/export_milestone_9_neighbors.py artifacts/milestone_9/runs/<run-directory>`. The CSV includes query/gallery IDs, rank, score, date, sequence token, and flood label. The image sheet displays the query beside its Top-10 neighbors as true-color RGB for visual review; it does not stand in for ground-truth relevance judgments.

M10 adds narrow natural-language query parsing and geographic/date filtering while keeping image ranking explicit. M11 adds a separate index-based multispectral path for single scenes and same-location temporal change; it does not perform learned text-to-multispectral alignment or answer generation. ANN indexing and generated answers remain later stages.

## M10: Natural-language query and exact evidence interface

M10 adds a local React interface and FastAPI backend for supported SEN12-FLOOD requests. The OpenAI Responses API parser converts query text into a typed request for flood/no-flood intent, dates, and a country or state/province. The UI shows the interpretation before retrieval. The parser receives text only; it does not see imagery, score candidates, or generate an answer.

The UI exposes two M9 choices: one CNN and one ViT, each selected using held-out validation prototype Precision@5 across three seeds. Each is one exact checkpoint and has its own embedding space. For flood/no-flood requests, GeoRAG reserves 20% of training sequences as labeled examples that define the class prototype; those examples are excluded from the searchable training gallery. All remaining scenes are ranked by exact Flat cosine similarity, regardless of their label. This is a label-informed image-space baseline, not learned text-image alignment. An optional example tile is encoded by the selected checkpoint.

The validation selector evaluates each class prototype against separate validation scenes, chooses the strongest RGB/12-band configuration within each encoder family by mean macro Precision@5, and exposes the representative seed nearest that mean. For the completed local M9 runs, both families select RGB; CNN and ViT each score 0.667 mean macro P@5, with different across-seed variation. These coarse scene-label proxy scores are selection diagnostics, not human relevance judgments or proof that RGB is generally superior.

Generate the local selection artifact before starting the API (and again if M9 runs or configuration change):

```powershell
uv run python scripts/evaluate_m10_prototypes.py
```

The local data has dates from 2018-12-13 through 2019-05-20. The inspected STAC items contain footprints and timestamps but no administrative place names. Resolve a country or first-order state/province using a pinned offline Natural Earth 1:10m Admin-0/Admin-1 gazetteer and scene-bbox intersection. District coverage is not included. The Natural Earth map data is public domain; SEN12-FLOOD stays local.

Prepare the gazetteer once:

```powershell
uv run python scripts/prepare_natural_earth.py
```

Keep the API and UI in separate PowerShell terminals:

```powershell
uv run uvicorn georag.api:app --host 127.0.0.1 --port 8000
```

```powershell
cd web
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. The API reads `OPENAI_API_KEY` from the project `.env` and never sends it to the browser. `GEORAG_QUERY_MODEL` can override the default parser model. The existing M9 raster cache, completed M9 checkpoints, and M10 selection artifact must be present. The app keeps the best selected CNN and ViT as RGB embedding baselines. Spectral requests use band-derived NDWI, MNDWI, and NDVI measurements; they do not use the RGB checkpoint score.

## M11: Multispectral spectral evidence retrieval

M11 adds text-directed searches for water-like spectral response and before/after change. It computes NDWI from B03/B08, MNDWI from B03/B11, and NDVI from B08/B04 with the common valid-pixel mask. For change searches, it pairs consecutive available Sentinel-2 observations from the same exact location sequence when the dates are no more than 30 days apart, then calculates after-minus-before index maps. Text requests can rank stronger/increasing/decreasing MNDWI evidence, while selected example scenes or pairs use exact cosine similarity over fixed index histograms.

RGB thumbnails remain a visual reference. Spectral result cards show MNDWI maps and summary statistics for all three indices. The scores are measured index values or histogram similarities, not flood probabilities. The source copy has scene/date flood labels but no pixel flood masks, and its stored values are not asserted to be calibrated surface reflectance. Read the [M11 implementation report](reports/milestone_11_spectral_evidence.md) and [configuration](configs/milestone_11_spectral_evidence.toml) for equations, pairing rules, scoring, and limits. Synthetic tests validate the formulas and ranking mechanics; inspect real results before making retrieval-quality claims.
