# M11: Spectral Evidence Evaluation Plan

## Why this matters

A color map can look convincing even when its bands are mixed up, shifted, cloudy, or showing permanent water instead of a flood. This evaluation checks that GeoRAG is measuring what it says it measures, then asks whether those measurements are useful for the flood question. It helps us make claims that match the evidence.

## What GeoRAG does today

The SEN12-FLOOD adapter places twelve Sentinel-2 optical bands on a shared 20 m grid. The current spectral search uses four of them:

- NDWI uses green B03 and near-infrared B08.
- MNDWI uses green B03 and short-wave infrared B11.
- NDVI uses near-infrared B08 and red B04.

The water-like scene search ranks by median MNDWI. A higher value means a stronger water-like index response; it is not a flood probability. The current valid-pixel mask removes missing or invalid data, but it is not a cloud or shadow mask. SEN12-FLOOD provides scene/date flood labels, not maps of which pixels flooded.

## Questions this evaluation will answer

1. Are the correct bands, pixels, and measurements being used?
2. Do the index measurements relate to the dataset's flood/no-flood scene labels on locations held out from training?
3. Does MNDWI change differently for no-flood-to-flood pairs than for stable no-flood pairs?
4. Do reviewed maps look like plausible water-related signals, or are clouds, shadows, soil, and permanent water causing confusing matches?

This first evaluation will not claim pixel-level flood detection or prove that spectral indices outperform RGB. Comparing the index path with the learned RGB models is a separate matched experiment.

## Evaluation steps and current status

### 1. Check the data and index maps

**Status: first-pass audit complete on the validation split.** The executable
audit, saved outputs, and findings are documented in
[`reports/milestone_11_spectral_input_audit.md`](../reports/milestone_11_spectral_input_audit.md).
The test split was not read or used.

- **Done:** confirmed the configured B03/B08, B03/B11, and B08/B04 channel mappings and recomputed all three maps for 210 available validation scenes.
- **Done:** checked every output index value against the expected [-1, 1] range and hand-calculated one valid pixel per selected example. The report records the maximum formula difference.
- **Done:** inspected source-raster headers and a deterministic sample of 16 validation scenes: four each from flood/high-MNDWI, flood/low-MNDWI, no-flood/high-MNDWI, and no-flood/low-MNDWI. The sample is for input QA only, not model selection or final evaluation.
- **Done:** the source asset catalog contains the twelve band assets, but no cloud, shadow, SCL, QA, or mask asset keys. No cloud/shadow filtering was performed; these remain confounders.
- **Finding requiring follow-up:** 12 validation scenes have valid-pixel masks but zero valid pixels for all three indices. At least 99% of their masked pixels have zero values across all 12 bands; one inspected scene is entirely zero. Their source mask does not identify those zeros as invalid. Investigate the source files and decide an explicit no-signal policy before interpreting scene-level spectral metrics. Do not silently label zeros as valid signal.
- **Done:** sample raster headers report uint16 values, scale 1, offset 0, and no nodata value; the inspected STAC assets carry no per-band `eo:bands` or `raster:bands` scale/unit metadata. The data adapter therefore treats values as raw stored DNs. Do not call them calibrated reflectance without independent provenance.
- **Done:** the 12 bands have different native grids/resolutions, as expected; the adapter resamples them onto the B05-anchored 20 m grid. The resulting RGB and index maps share a pixel grid. The contact sheet supports visual alignment inspection but is not a geometric accuracy test.

### 2. Test scene-level label association

- Use the existing sequence-grouped test split only for final metrics. Do not fit thresholds or choose an index from test results.
- The primary score is scene median MNDWI against the scene/date flood label. Report ROC-AUC, average precision, class counts, and the flood-label prevalence as the average-precision reference.
- ROC-AUC measures how often a flood-labeled scene gets a higher score than a no-flood scene. Average precision measures whether flood-labeled scenes appear near the top; compare it with the fraction of scenes labeled flood.
- Report median and positive-pixel fraction for NDWI, MNDWI, and NDVI as secondary measurements. These help show whether results are specific to one index or shared across them.
- Estimate uncertainty by resampling whole sequence IDs 1,000 times with a fixed seed of 17. Scenes from one location must stay together in each resample because nearby dates are related.
- If there are too few held-out flood or no-flood sequences to estimate a metric reliably, report that limitation instead of changing the split after seeing the results.

### 3. Test temporal change

- Build consecutive pairs only within the existing test sequences and keep the current 1-to-30-day gap rule.
- Compare after-minus-before median MNDWI for no-flood-to-flood pairs against no-flood-to-no-flood control pairs. Report pair counts, date-gap distributions, score distributions, ROC-AUC, and average precision.
- Estimate uncertainty by resampling sequence IDs, not individual pairs. Report flood-to-no-flood pairs separately as descriptive results rather than mixing them into the primary comparison.
- Treat the result as a scene-label change proxy. Clouds, shadows, season, soil moisture, and acquisition conditions can also change between dates; labels may not mark the precise start of flooding.

### 4. Review examples without label or score hints

- Select up to 24 test scenes across the four label/high-low-MNDWI groups and up to 12 pairs across no-flood-to-flood and stable no-flood cases.
- Hide labels, rank, and score during review. Show RGB, index maps, band summaries, dates, and the before/after change map.
- Record whether the water-like pattern looks plausible, whether clouds or shadows are visible, whether permanent water or land cover could explain it, whether the map aligns with the imagery, and whether change appears spatially credible. Allow an "uncertain" rating.
- Treat this as a structured visual audit, not independent ground truth or a population-level accuracy estimate.

## Outputs and success criteria

The implementation produces a data-quality audit, metric tables and plots, a saved list of reviewed examples and their notes, and a results report. Keep dataset files and generated artifacts local and out of Git. Record the split-manifest fingerprint, configuration, random seed, code revision, sample counts, and any available quality-mask information. The first-pass audit artifacts are local under `artifacts/milestone_11/spectral_input_audit/`.

**Next gate:** explain and handle the zero-valued, mask-valid scenes before running the scene-label association or temporal-change metrics below. Keep the current index equations and mask policy unchanged until the source-data cause is understood and an explicit policy is reviewed.

The evaluation is complete when the report answers the four questions above, separates software/data checks from label-proxy metrics and visual review, and gives a bounded conclusion that may be positive, negative, or inconclusive. Any claim about flooded pixels requires a separate dataset with independent pixel-level flood annotations.
