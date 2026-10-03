# GeoRAG Roadmap

GeoRAG is a staged project for retrieving Earth-observation imagery from images and, later, natural-language descriptions. The intended result is inspectable evidence: retrieved tiles, similarity scores, and available metadata. A generated answer is a much later, optional layer.

The project develops its own image encoders and retrieval/indexing mechanisms. A pretrained image-text model is introduced as a baseline because the current from-scratch CNN and ViT were trained to compare images; they were not trained to align words with images. Keep these model spaces separate and compare them only on tasks each can perform.

Implementation proceeds one milestone at a time. A milestone is complete when its deliverables and checks are reviewed; later stages do not need to be implemented in advance.

## Current state

Milestones 0–4 are complete: repository foundation, EuroSAT-MS data layer, small from-scratch CNN and ViT, contrastive training, and persisted full-corpus embeddings for both encoders. The Milestone 3 report documents the training experiment and its limits. Same-label neighbors are only a weak proxy for relevance, and the existing split does not establish geographic generalization.

The immediate next milestone is M6: a natural-language-to-image baseline. Exact image-to-image retrieval now works as the Flat reference before any approximate index.

## Milestones

### M0 — Repository foundation (complete)

- Establish the Python package, `uv` environment, configuration convention, tests, local experiment outputs, and device diagnostics.
- **Complete when:** setup and basic checks run from the documented commands.

### M1 — EO dataset and inspection (complete)

- Load EuroSAT-MS tiles through the dataset adapter, retain stable IDs and available metadata, create deterministic splits, and inspect multispectral composites and per-band statistics.
- **Complete when:** dataset, split, shape, metadata, and visualization checks pass on the local corpus.

### M2 — From-scratch image encoders (complete)

- Implement the small CNN and ViT in PyTorch with configurable input channels and an explicit embedding dimension.
- **Complete when:** output shape, normalization, finite values, gradients, parameter counts, and device execution are verified.

### M3 — Contrastive image training (complete)

- Train the encoders with two geometric views of the same tile and an explicit symmetric NT-Xent objective. Save configurations, checkpoints, metrics, and loss curves.
- **Complete when:** runs are reproducible and recoverable, and the report distinguishes optimization diagnostics from semantic retrieval quality.

### M4 — Embedding corpus (complete)

- Encode the catalog with each selected scratch-model checkpoint in batches.
- Persist the embedding matrix with ordered stable tile IDs, metadata, preprocessing/model identity, and a reproducible run configuration.
- The completed run stores separate `[27000, 128]` float32 corpora for CNN and ViT, with all train/validation/test rows and their split assignments.
- **Complete when:** reloaded artifacts preserve vector-to-tile alignment and produce finite embeddings with the expected dimensions. Both local artifacts passed this check.

### M5 — Exact image-to-image retrieval (complete)

- Implement dense Flat search for cosine similarity, inner product, and Euclidean distance. Use scratch-model embeddings and return ranked tiles with scores and metadata.
- Add query/result image-grid inspection and small synthetic correctness cases.
- The NumPy CPU reference was tested on a held-out validation query against training candidates in both model spaces. For this normalized corpus, cosine, inner product, and Euclidean distance produced identical Top-5 rankings.
- **Complete when:** exact Top-k agrees with hand-computed results on synthetic vectors and a real EuroSAT query can be traced to its source tile and metadata. Both checks pass; see the [M5 report](../reports/milestone_5_exact_retrieval.md).

### M6 — Natural-language-to-image baseline

- Add a local pretrained image-text model as a separate retrieval path. Use IBM [MS-CLIP](https://github.com/IBM/MS-CLIP) as the initial multispectral candidate; verify the released EuroSAT-compatible band mapping, normalization, and inference path before embedding the corpus.
- Precompute corpus image embeddings, encode a visual-concept text query, and retrieve with exact Flat cosine search in that model's shared image-text space.
- Initial queries describe visible content, such as “dense forest beside fields.” Do not add geographic/date parsing or generated answers in this milestone.
- **Complete when:** a local query returns reproducible ranked tiles and all scores and IDs belong to the MS-CLIP space; no vectors are mixed with scratch-model embeddings.

### M7 — Relevance evaluation

- Evaluate a fixed set of EuroSAT class-name queries against held-out labels and report per-class and macro retrieval metrics such as Recall@k and mAP or nDCG.
- Review a small, fixed set of free-form/paraphrased queries by inspecting their Top-5 results. Record relevance judgments and failure examples separately from class-label metrics.
- **Complete when:** results clearly state that EuroSAT labels and a small manual review set are limited relevance evidence, not a general semantic benchmark.

### M8 — Educational IVF

- Implement coarse-centroid training, vector assignment, posting lists, nearest-cell probing, candidate scoring, and Top-k search.
- Compare against the same embedding space's exact Flat ground truth while varying cluster count and `nprobe`.
- **Complete when:** assignments and probing pass correctness checks, and recall/latency tradeoffs are measured against Flat.

### M9 — PQ and IVF-PQ

- Implement vector subdivision, subspace codebooks, encoding, reconstruction, and approximate distance computation; combine PQ with IVF after standalone PQ is checked.
- Measure quantization error, compression, index memory, Recall@k, and latency against Flat.
- **Complete when:** encoding dimensions and reconstruction are validated and controlled tradeoff results are recorded.

### M10 — HNSW systems comparison

- Integrate a mature HNSW implementation as a benchmark, rather than writing a production graph index from scratch.
- Compare search quality, latency, memory, and build time with Flat and the educational indexes on the same corpus and queries.
- **Complete when:** parameters and machine details are recorded and exact-vs-approximate retrieval quality is reported.

### M11 — Metadata, diversity, and evidence

- Add explicit metadata predicates, MMR-style diversification, and a structured evidence result that records query, ranked tiles, scores, metadata, retrieval method, index parameters, and provenance.
- Evaluate pre-filtering and retrieval-then-filtering as distinct strategies; report when filtering reduces candidate recall. Measure whether diversification reduces near-duplicate results while preserving relevance.
- **Complete when:** filter behavior and evidence provenance are inspectable and tests cover filtering and diversification on known examples.

### M12 — Retrieved-image captions (optional evidence aid)

- Add on-demand local captions for retrieved tiles only, displayed as generated descriptions alongside the source imagery. For EuroSAT, render the B04/B03/B02 RGB composite for the captioner.
- Keep captions out of embedding generation, retrieval scoring, and relevance labels. Record caption model/version and prompt; select and hardware-check the local model when this stage begins.
- **Complete when:** captions are clearly labeled as generated, traceable to model/prompt, and never presented as metadata or verified event claims.

### M13 — From-scratch image-text model (research extension)

- After the pretrained text-image baseline is evaluated, select a suitable paired EO image-text dataset and train an inspectable dual encoder with an explicit contrastive objective.
- Compare its text-to-image retrieval against the pretrained baseline using the same query set and relevance protocol. Dataset selection and licensing must be resolved before this stage.
- **Complete when:** the training data, objective, and evaluation are documented and the custom model produces its own consistently versioned shared embedding space.

### M14 — Temporal change retrieval (later research stage)

- Introduce timestamped, paired observations and learn embeddings for changes between acquisitions, rather than static scene appearance.
- Evaluate retrieval of analogous changes with temporal/geographic provenance and explicit relevance criteria.
- **Complete when:** a held-out evaluation demonstrates retrieval of change patterns and separates change similarity from appearance similarity.

### M15 — Evidence-grounded answer generation (optional, last)

- Only after retrieval is useful and evaluated, add a local multimodal reasoning layer that receives the query and structured retrieved evidence.
- Require answers to distinguish retrieved observations from interpretation and to identify supporting tiles. Retrieval remains independently usable without generation.
- **Complete when:** answer quality is evaluated against evidence-grounding criteria and unsupported claims are analyzed.

## Project-wide working rules

- Preserve small, reproducible runs suited to the RTX 5060 Laptop GPU (8 GB VRAM) and 16 GB system RAM; use local artifacts and record configurations, seeds, model identity, and relevant system details.
- For each algorithm, follow reference implementation, correctness checks, benchmark, then optimization.
- Keep representation quality, semantic relevance, ANN recall, metadata filtering, and evidence diversity as separate evaluation questions.
- Do not treat a visualization, label agreement, or nearest-neighbor score alone as proof of retrieval quality.
