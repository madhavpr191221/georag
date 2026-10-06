# GeoRAG Roadmap

GeoRAG is an Earth-observation retrieval system. Its research focus is whether multispectral measurements help retrieve useful evidence about physical surface conditions, beyond what RGB appearance alone supports. Natural language is the query interface; the project is not a generic vision RAG demo.

The first task is flood-evidence retrieval on the user's local Kaggle copy of SEN12-FLOOD. The first controlled comparison is Sentinel-2 RGB versus its available optical bands, with identical small encoders, training protocol, data partitions, and seeds. Sentinel-1 SAR is a later modality experiment. Retrieval quality and exact-search results come before ANN engineering; generated answers come much later.

## Current state

M0–M7 established the repository, EuroSAT-MS adapter, from-scratch image encoders, contrastive training, embedding artifacts, exact image retrieval, an RGB-versus-RGB+NIR Agriculture-Vision experiment, and an RGB RemoteCLIP text-retrieval experiment. M6 and M7 are useful exploratory results on their stated datasets and proxies; neither establishes general natural-language multispectral retrieval. Preserve their reports as historical evidence.

M8's data path and M9's initial representation experiment are implemented. The extracted STAC collections, metadata counts, missing Sentinel-2 assets, sequence grouping, 20 m tensor contract, and split manifest are recorded in the [M8 audit report](../reports/milestone_8_sen12flood_data_audit.md). All 12 M9 training runs and exact test retrieval evaluations completed; see the [M9 report](../reports/milestone_9_sen12flood_representation.md) for the bounded findings and capacity caveat.

M11 now has an implemented spectral evidence path: NDWI/MNDWI/NDVI scene summaries, water-like scene ranking, exact index-distribution search, and same-sequence before/after change ranking. See the [M11 report](../reports/milestone_11_spectral_evidence.md). The report and UI explicitly treat these measurements as evidence clues, not pixel flood labels.

## Milestones

### M0–M5 — EO foundations (complete)

- Package/configuration setup; EuroSAT-MS loading and inspection; explicit CNN and ViT encoders; NT-Xent training; persisted embedding corpora; exact Flat retrieval.
- Existing reports and tests document behavior and limits. These milestones provide reusable mechanics, not proof of task relevance.

### M6 — RGB versus RGB+NIR on Agriculture-Vision (complete; exploratory)

- Matched from-scratch encoders and self-supervised training compared RGB with RGB+NIR for retrieval against Agriculture-Vision anomaly labels.
- The reported gain is specific to that dataset, task, and annotation-proxy protocol. Do not generalize it to Sentinel-2 or flood retrieval.

### M7 — RGB text-to-image baseline (complete; exploratory)

- RemoteCLIP ranked RGB Agriculture-Vision tiles for fixed text descriptions, with annotation-proxy metrics and optional model-assisted review.
- This is an RGB baseline only. Its language/image space is separate from the scratch EO embeddings; VLM judgments are not human ground truth.

### M8 — SEN12-FLOOD data and evaluation contract (data layer implemented)

- User downloads the Kaggle archive manually into `data/sen12flood/`; the observed collections are nested under `data/sen12flood/sen12flood/`. Raw archives and extracted data remain untracked.
- Initial audit: four STAC collections; 2,236 Sentinel-2 and 3,331 Sentinel-1 source/label pairs across 335 sequence IDs. Source and label item IDs have collection-specific prefixes and pair by exact sequence/date suffix. A 98-item subset has STAC metadata and labels but none of the 12 linked Sentinel-2 TIFFs present; preserve and flag these rows, and exclude them from optical model inputs.
- Treat the sequence token as an exact string: 0001 and 1 refer to different bboxes. The local split uses the 335 observed string IDs; the upstream 337-location discrepancy remains documented, with no IDs merged or inferred.
- The adapter loads the twelve available optical bands onto a per-scene 20 m grid anchored to B05. It area-averages finer bands, bilinearly interpolates coarser bands, uses nearest-neighbor for matching-resolution bands, and exposes a common valid-pixel mask.
- A deterministic 80/10/10 split with seed 17 groups by exact sequence and includes metadata-only rows in the manifest. No group crosses partitions. Keep Sentinel-1/Sentinel-2 cross-sensor temporal matching as a separate concern.
- Record labels from STAC properties, dates, sequence/location identifiers, partial coverage, and Kaggle provenance/license status against SEN12-FLOOD documentation.
- Freeze supported query intents and evaluation: flood/no-flood relevance comes from the dataset's image/date labels; geographic and temporal constraints are evaluated separately. State that labels are image-level and that post-event dates may remain labeled flooded.
- **Data-layer acceptance:** tests pass; local catalog counts and four representative tensor loads are verified; the report records the split fingerprint, band/grid contract, missing-file handling, label limitations, and reproduction command without copying data into Git. Full raster-corruption validation remains unperformed.

### M9 — Matched RGB and multispectral flood representations (first experiment complete)

- Cache the 2,138 available Sentinel-2 scenes on the M8 20 m grid as disk-backed float32 imagery plus common valid-data masks. Fingerprint source files and preprocessing so stale caches rebuild.
- Compute per-band population mean/std using valid pixels from training sequences only. RGB uses Sentinel-2 B04/B03/B02 and the same common mask as the 12-band condition; invalid pixels are zeroed after normalization.
- Compare from-scratch CNN and small ViT encoders under RGB and all 12 optical bands (four conditions), using paired D4 transforms and explicit NT-Xent. D4 means right-angle rotation/reflection invariance; the same transformation is applied to every band and its mask.
- Use AdamW, batch 16 original tiles (32 augmented views, 30 in-batch negatives per anchor), temperature 0.1, 20 epochs, and seeds 17/23/42. Select checkpoints by fixed-view validation NT-Xent loss. Start in full precision.
- The train split is the retrieval gallery; the sequence-disjoint test split supplies query tiles. Exact cosine neighbors are summarized by flood/no-flood label agreement as a weak task proxy only. This is not human relevance or evidence of pixel-level flood detection.
- A smoke run completed all four conditions (one seed, one epoch, small subsets) and validates plumbing only. Its perfect small-subset label agreement is not interpretable as retrieval quality.
- Deferred from M9: supervised contrastive learning, spectral-index baselines, ANN, language parsing, and generated answers. Each requires a separate controlled comparison.
- **Complete:** all 12 runs have checkpoints, learning curves, exact test neighbors, per-seed metrics, local inspection sheets, run/split/cache provenance, and a limitations-aware report.

### M10 — EO query parsing and exact evidence retrieval (implemented)

- A local React/TypeScript interface and FastAPI backend expose typed query parsing, manual review, run selection, optional example-tile search, geographic disambiguation, and ranked evidence cards. The OpenAI parser receives query text only and is not used to rank or judge imagery.
- A pinned Natural Earth 1:10m Admin-0/Admin-1 gazetteer resolves countries and state/province names because those names are absent from the inspected STAC items. Date and scene-footprint constraints are prefilters; exact cosine Flat ranks the remaining training gallery.
- Text-only flood/no-flood requests use normalized class prototypes made from 20% of training sequences; these support examples are excluded from the searchable training gallery. All eligible gallery labels remain rankable, and the UI clearly shows each scene/date label. Example-tile queries use the selected run's checkpoint and normalization statistics.
- `scripts/evaluate_m10_prototypes.py` ranks support-derived flood/no-flood prototypes against the separate validation split, reports macro label Precision@5 across seeds, and chooses one CNN plus one ViT preset. The current local selection chooses RGB for both families (0.667 mean macro P@5 for each; distinct uncertainty). This is only a coarse scene-label selection proxy.
- Results expose the selected run, query-vector method, validation selection summary, cosine score, RGB preview, source tile ID, date, coordinates, label, bands, and provenance. Natural-language text still does not directly embed or judge the imagery.
- **Implemented checks:** validation selection artifact generated from all 12 M9 runs; support/search sequences are disjoint; API model list exposes exactly two presets; query results are ranked only from the remaining training sequences. Unit tests cover grouped support splits, prototype scoring, deterministic preset selection, result labels, and existing query/filter behavior. M11 still owns broader retrieval-quality and failure analysis.

### M11 — Retrieval analysis and evidence review

- **Implemented baseline:** multispectral index maps and summaries, text-directed water-like ranking, histogram-based scene similarity, and same-sequence temporal index-change retrieval. The configured gallery is train-sequence-only, with a 30-day maximum gap between consecutive observations.
- Next, inspect ranked examples and report label-defined retrieval metrics only as scene-level proxies, along with query-constraint satisfaction, latency, provenance, and failure cases.
- Compare RGB learned retrieval against spectral evidence under the same sequence/location protocol; do not infer that image-level flood labels identify flooded pixels.
- Inspect false neighbors and failure categories: representation, query parsing, metadata filtering, label limitations, or missing relevant examples. Treat label agreement as task-specific evidence, not universal human relevance.
- **Complete when:** results support a bounded conclusion about whether multispectral input helps this task, including negative or inconclusive outcomes.

### M12 — Educational IVF, then PQ/IVF-PQ

- Implement IVF against exact Flat ground truth only after M11 establishes a useful exact-retrieval baseline. Then add educational PQ and IVF-PQ.
- Measure Recall@k, latency, memory, build time, and distance evaluations; keep ANN recall distinct from label-defined semantic retrieval quality.
- **Complete when:** correctness tests pass and recall/latency/memory tradeoffs are reported on the fixed flood query set.

### M13 — HNSW systems comparison

- Add a mature HNSW implementation as a systems baseline, not a from-scratch production graph index. Compare with Flat, IVF, and IVF-PQ using the same embeddings and queries.
- **Complete when:** quality, latency, memory, and build-time measurements are reproducible with parameters and machine details recorded.

### M14 — Broader EO query tasks and modalities

- Add vegetation condition/agricultural anomaly retrieval and Sentinel-1 SAR as separately scoped experiments with their own data, relevance definitions, and matched baselines.
- Do not combine tasks or modalities until each has a clear evaluation protocol; temporal change retrieval requires paired observations and a separate change-focused objective.

### M15 — Evidence-grounded answers (optional, last)

- Only after retrieval is useful, allow an optional multimodal model to summarize a structured evidence object and cite supporting tile IDs. Preserve retrieved imagery, scores, metadata, and provenance independently of generated prose.
- **Complete when:** evidence support and unsupported claims are evaluated; GeoRAG remains useful without answer generation.

## Project-wide rules

- Data stays local and out of Git. Record source, license status, archive/file fingerprints, preprocessing, configuration, seeds, code revision, and system details for each experiment.
- Correctness and task relevance precede ANN speed. Follow reference implementation → tests → benchmark → optimization.
- A pretty projection, training loss, VLM judgment, class-label agreement, and ANN recall each measure different things; none alone proves useful EO evidence retrieval.
- Keep experiments feasible for an 8 GB VRAM laptop GPU and 16 GB system RAM; smoke tests validate plumbing only.
