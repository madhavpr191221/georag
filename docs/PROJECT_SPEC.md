# GeoRAG Project Specification

## Purpose

GeoRAG retrieves Earth-observation evidence for physical-surface questions. Its first focused task is retrieval of flood-related Sentinel-2 observations from the user's local Kaggle copy of SEN12-FLOOD.

The initial research question is:

> For natural-language requests about flooding, does a multispectral Sentinel-2 representation retrieve relevant flood evidence from unseen sequences/locations better than an otherwise matched RGB-only representation?

This is a task-specific retrieval study. It is not generic image RAG, a flood-mapping product, a VLM evaluation, or an answer-generation project. The system should remain useful as a transparent retrieval/evidence tool without a generator.

## Dataset and provenance

- The user downloads the Kaggle SEN12-FLOOD dataset manually. The inspected local root is `data/sen12flood/`; the extracted STAC collections are nested under `data/sen12flood/sen12flood/`. The adapter must follow the observed layout rather than an assumed archive layout.
- Raw downloads, extracted imagery, and derived tiles are not committed to Git. Document the Kaggle URL, dataset owner/version, download date, archive/file manifest, upstream citation, and license status.
- Kaggle currently reports the dataset license as `Unknown`. Do not claim a specific Kaggle license or redistribute the downloaded data. Re-check the record before publishing derived artifacts that could contain source imagery.
- The local audit found four STAC collections, 2,236 Sentinel-2 and 3,331 Sentinel-1 source/label item pairs across 335 sequence IDs. Source/label IDs have different collection prefixes and are paired by exact sequence/date suffix. There are 98 Sentinel-2 STAC items whose 12 band files are all absent; the adapter retains them in `all_records` with unavailable-assets status and excludes them from normal image iteration.
- Preserve sequence IDs as exact text tokens: 0001 and 1 resolve to different Sentinel-2 bounding boxes. Do not strip zero padding or convert the token to an integer. The local deterministic split uses the 335 observed IDs; upstream documentation describes 337 locations, and the unresolved count difference is recorded without inventing or merging groups.
- The adapter stacks bands in the order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 on each scene's B05 20 m grid. It area-averages finer bands, bilinearly interpolates coarser bands, applies nearest-neighbor to matching-resolution bands, fills invalid tensor cells with zero, and returns a common validity mask. Raw uint16 DN values are converted to float32 with `value_scale = 1.0`; model normalization is separate.
- Any later normalization or spatial augmentation must account for the returned validity mask; fill zeros are not valid surface observations and must not contaminate training-only band statistics.
- No official split key was found in the local JSON metadata. The checked-in configuration generates a deterministic 80/10/10 split grouped by exact sequence using seed 17; the manifest includes all metadata records and is fingerprinted in the M8 report. Source/label pairing does not imply Sentinel-1 and Sentinel-2 acquisitions are simultaneous.
- Validate contents against the SEN12-FLOOD documentation: Sentinel-2 optical time series, Sentinel-1 SAR time series, per-image/date flood labels, acquisition dates, sequence/location metadata, and georeferencing where available. Preserve missing/partial coverage flags.
- SEN12-FLOOD labels are image-level flood/no-flood labels. The documentation notes that subsequent observations after a flood may remain labeled flooded; results must state this limitation. The task is not pixel-accurate flood delineation.

## Retrieval question and query contract

Natural language is a way to express an EO retrieval request, not the representation-learning objective itself. A parser may be foundation-model-backed, but its only role is to map text into a validated, inspectable structure. It does not embed imagery, rank candidates, or judge result relevance. M10 uses a local Natural Earth Admin-0/Admin-1 gazetteer because the inspected SEN12-FLOOD STAC items contain footprints and timestamps but no administrative place names. The boundary layer resolves names and spatial predicates; it is not source imagery metadata.

The initial conceptual `EOQuery` contains:

- `intent`: one supported flood-state intent (flood / no-flood) for the first task;
- optional `start_time` and `end_time`;
- optional country or first-order administrative-region constraint resolved against a pinned offline boundary gazetteer and intersected with the scene footprint;
- `sensor` / modality request (Sentinel-2 RGB or Sentinel-2 multispectral initially; SAR deferred);
- optional `example_tile_id` for image-conditioned retrieval;
- original query text and parser provenance for auditability.

Unsupported or ambiguous conditions must be surfaced explicitly instead of silently converted into filters. M10 resolves only country and first-order state/province names using Natural Earth; it does not claim to provide district coverage or general natural-language geospatial reasoning.

For text-only condition requests, derive an intent prototype from a fixed, sequence-grouped support subset of training embeddings and rank eligible items from the remaining training sequences. The support examples are not searchable results, preventing a prototype from ranking the examples that defined it. For a query with an example tile, use that tile's embedding as the visual query and apply the parsed temporal/geographic constraints to the candidate set. Prototype and example-tile ranking are distinct query modes and must be reported separately. The flood/no-flood labels used for quantitative relevance are never read from held-out validation/test scenes to construct the query vector.

A retrieval result should expose rank, stable tile ID, similarity score, source image reference, date, location/sequence, sensor/band description, and retrieval/model/index provenance where present.

### M11 spectral evidence path

M11 adds a separate retrieval path whose scores come from named Sentinel-2 bands, not from the M9 RGB checkpoints. It computes NDWI `(B03-B08)/(B03+B08)`, MNDWI `(B03-B11)/(B03+B11)`, and NDVI `(B08-B04)/(B08+B04)` with the common valid-pixel mask. Scene summaries expose valid-pixel count, mean, median, and positive fraction. Text-only water-signal requests rank by median MNDWI; scene examples compare fixed 32-bin per-index histograms with exact cosine similarity.

Temporal evidence uses consecutive available observations in one exact sequence, ordered by date and at most 30 days apart. Change is `after - before`; text-directed increase/decrease requests rank by the signed median MNDWI change. Selected before/after examples compare fixed 32-bin histograms for each index change with exact cosine similarity. Searchable scenes and pairs come from training sequences, and the complete example sequence is excluded from example-conditioned results. Date and geography constraints apply to the later scene in a pair.

The result UI may show RGB as display context, but RGB pixels/checkpoint scores are not part of the spectral score. Index values are clues, not flood probabilities. SEN12-FLOOD labels remain scene/date-level, and the source values are not asserted to be calibrated surface reflectance. M11 does not perform pixel flood classification.

## Representation and baselines

- Implement or reuse a small explicit PyTorch encoder with configurable input channels and inspectable intermediate shapes. Train from scratch; avoid giant pretrained vision transformers for the first multispectral comparison.
- Compare Sentinel-2 RGB channels (B04/B03/B02 in red/green/blue order) with the available optical-band stack. The twelve bands and resampling policy are fixed by M8; model-side normalization is selected, fitted on training groups only, and recorded in the run manifest.
- Match architecture capacity, training data, valid augmentation assumptions, optimizer, training budget, and seeds across RGB and multispectral conditions.
- M9 first isolates the self-supervised NT-Xent condition. Do not combine label-aware supervised contrastive learning or spectral-index ranking with this experiment; they are follow-up baselines with distinct assumptions.
- The M9 model matrix is CNN/ViT crossed with RGB/all-12 optical bands. Use three fixed seeds, the sequence-group split, training-only valid-pixel band statistics, paired synchronized D4 transforms, and validation loss for checkpoint selection.
- A training batch has 16 source tiles and 32 augmented views; each anchor has 30 other views as negatives after excluding itself and its paired positive. Temperature is 0.1. Start in FP32 unless measured device behavior motivates a documented mixed-precision follow-up.
- RGB selects B04/B03/B02 from the 12-band grid and keeps the common validity mask, ensuring an equivalent observed-pixel comparison. Inputs are `[B,C,256,256]`; outputs are L2-normalized `[B,128]`.
- Compute each channel's population mean and standard deviation over valid pixels from training sequences only. After standardization, invalid pixels are set to zero. Apply identical D4 choices across channels and masks.
- Retrieval uses training sequences as gallery and held-out test sequences as queries. Report exact cosine neighbors and flood/no-flood label-neighbor fraction only as a weak dataset-label proxy; it is not human relevance and cannot substantiate pixel-level flood detection.
- Do not mix embeddings from distinct model checkpoints or embedding spaces.

## Evaluation contract

- Preserve the dataset's official sequence-level split when present. All dates and sensor observations from one sequence/location must remain on one side of a split. If the Kaggle layout does not preserve a usable official split, define one deterministic group split by sequence and publish its manifest/fingerprint.
- Build retrieval gallery and query/evaluation sets from disjoint sequences/locations. All model fitting, normalization statistics, and intent prototypes use training sequences only; prototype support sequences are also disjoint from the searchable training gallery.
- Primary quantitative relevance is the dataset's flood/no-flood label. Report category-specific Precision/Recall and mAP@k, class balance, query counts, and uncertainty across seeds. Treat spatial/temporal constraint satisfaction as separate system correctness metrics.
- Compare all encoders and index methods on the same query set and candidate pool. Exact Flat is the ranking ground truth for later ANN evaluation; ANN recall is not semantic relevance.
- Inspect representative true positives, false positives, false negatives, duplicate/near-duplicate results, and failures caused by query parsing, metadata, representations, labels, or absent evidence.
- A positive multispectral result is limited to this dataset, label protocol, region/time distribution, and model/training setup. An inconclusive or negative result is valid and must be reported.

## Staged delivery

1. **M8 — Data audit and adapter:** STAC pairing, metadata catalog, missing-asset handling, 20 m multispectral loading, validity masks, and deterministic sequence-group split are implemented. The unresolved 337-versus-335 location-count discrepancy remains an explicit dataset limitation.
2. **M9 — Representation comparison:** matched CNN/ViT encoders for RGB and 12-band Sentinel-2, self-supervised NT-Xent, mask-aware normalization/transforms, test-time exact cosine diagnostics, and reproducible artifacts. Supervised contrastive learning and spectral-index baselines are deferred follow-ups.
3. **M10 — Query and exact retrieval:** typed OpenAI query parsing, a local React/FastAPI interface, offline country/state/province resolution, date/footprint constraints, support-derived class prototypes, validation-selected CNN/ViT presets, image-conditioned queries, RGB preview cards, and exact Flat evidence results. This is a retrieval interface, not open-ended answering or text-image alignment.
4. **M11 — Spectral evidence and evaluation:** index maps and summaries, water-signal ranking, same-sequence temporal index-change retrieval, then held-out retrieval metrics and failure inspection. The implementation is in place; scientific interpretation still requires review of real ranked results.
5. **M12+ — Indexing and expansion:** IVF, PQ/IVF-PQ, HNSW, pixel-mask-based segmentation on a separately labeled dataset, additional EO phenomena/modalities, and only then optional answer generation.

The detailed deliverables and completion criteria are maintained in [the roadmap](ROADMAP.md). Historical M6 and M7 reports remain unchanged and must not be presented as results for the SEN12-FLOOD experiment.
