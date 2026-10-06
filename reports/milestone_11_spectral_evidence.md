# M11: Multispectral spectral evidence retrieval

## Goal and status

M11 adds an inspectable spectral search path to the local GeoRAG interface. The ranking in this mode is calculated from Sentinel-2 band measurements and index maps. RGB CNN/ViT retrieval remains available as a separate comparison path; it does not contribute to spectral scores.

The current SEN12-FLOOD copy contains scene/date flood labels, not flood extent masks. M11 therefore reports spectral evidence and temporal index change. It does not classify flooded pixels or claim that positive MNDWI proves flooding.

## Measurements

The adapter provides twelve aligned optical bands on each scene's 20 m B05 grid. M11 calculates three normalized-difference maps from named channels:

$$
\mathrm{NDWI} = \frac{B03-B08}{B03+B08},\qquad
\mathrm{MNDWI} = \frac{B03-B11}{B03+B11},\qquad
\mathrm{NDVI} = \frac{B08-B04}{B08+B04}.
$$

NDWI and MNDWI expose water-related spectral response; NDVI provides vegetation context. Computation respects the common valid-pixel mask. A denominator whose absolute value is at most `denominator_epsilon` is marked invalid and shown as missing. Scene summaries report valid-pixel count, mean, median, and fraction above zero for each index.

The source tensors are stored as raw uint16-derived values with `value_scale = 1`. A common multiplicative scale cancels in these ratios, but M11 does not claim that the source is calibrated surface reflectance. Atmospheric effects, mixed pixels, terrain, shadows, cloud contamination, and permanent water can all affect the indices.

## Retrieval behavior

- **Water-like scene request:** exact ranking by scene median MNDWI, highest first. The score is an index value, not a probability or a calibrated water fraction.
- **Scene example request:** compute 32-bin histograms over fixed `[-1, 1]` ranges for NDWI, MNDWI, and NDVI, concatenate them, and rank by exact cosine similarity. This compares index distributions, not pixel arrangement.
- **Water-like signal increase/decrease:** construct consecutive, dated observation pairs within each exact SEN12-FLOOD sequence. Keep only gaps from 1 through 30 days. Compute each index change as `after - before`; rank by median MNDWI change (or its negation for a decrease request).
- **Change example request:** validate that the selected before/after scenes form one of those consecutive pairs. Compare the concatenated 32-bin index-change histograms over fixed `[-2, 2]` ranges by exact cosine similarity.

Temporal candidates are formed only from training sequences. A selected example's whole sequence is excluded from the ranked gallery. Date and geographic filters apply to the scene for single-scene requests and to the after-date scene for pair requests. The displayed flood/no-flood badge is metadata only; it does not affect the spectral rank.

## Evidence shown

Result cards include conventional B04/B03/B02 RGB previews for scene recognition, an MNDWI color map, index summaries, score meaning, dates, sequence ID, and the scene/date label. Pair cards show both RGB dates, both MNDWI maps, and an after-minus-before MNDWI map. Color maps use a fixed scale and include a colorbar; they do not indicate flood probability.

Use text requests such as:

- `Find scenes with a strong water-like spectral signal in Manicaland, Zimbabwe during 2019`
- `Find locations where the water-like spectral signal increased in 2019`
- `Find locations where the water-like spectral signal decreased`
- `Find scenes with spectral patterns similar to an example tile`
- `Find before-and-after pairs with change similar to my selected example pair`

For the last two requests, select the example tile or both ordered example tiles in the interface. Change-pair examples must belong to the same sequence and be no more than 30 days apart.

## Configuration and tests

`configs/milestone_11_spectral_evidence.toml` records index names, fixed histogram ranges and bin count, denominator tolerance, maximum pair gap, and gallery split. The backend rejects a configuration that changes the supported index set without a code change.

Synthetic tests verify index values by hand, nodata and zero-denominator handling, histogram normalization, cosine self-similarity, after-minus-before direction, sequence isolation, maximum gap, and spectral pair ranking. These tests establish algorithm correctness; they do not establish that the archive contains useful human-relevant retrieval results.

## Local archive smoke check

The implementation was exercised against the existing disk-backed SEN12-FLOOD cache. A text-directed high-water-like-signal request ranked 1,230 eligible training scenes with usable MNDWI. Its first result was `sen12floods_s2_source_0232_2019_04_23` with median MNDWI 0.8735; the service generated a valid PNG index map for that scene. A water-signal-increase request ranked 950 usable training date pairs. The first pair was `sen12floods_s2_source_0231_2019_03_27` to `sen12floods_s2_source_0231_2019_04_08` (12 days), with median MNDWI change +1.1405, and its change-map endpoint generated a valid PNG.

These are plumbing and inspectability checks only. The high scores reflect the chosen ranking rule, not verified flood relevance. The top scenes/pairs still need visual review against the source imagery and independent evidence before a quality claim is made.

## Limits and next evaluation

- Positive MNDWI is only a water-like spectral clue. No threshold in M11 is presented as a flood detector.
- Scene/date labels are not pixel masks and may persist after a flood event.
- Histogram retrieval forgets where index values occur spatially.
- Consecutive dates can still differ due to clouds, illumination, atmosphere, season, or acquisition conditions.
- This milestone does not use all twelve bands in every score: it uses the relevant named bands for three transparent indices. The RGB preview is display-only in spectral mode.
- Exact retrieval is used; no ANN index is involved.

The next scientific check should inspect a small set of ranked single scenes and temporal pairs, compare the maps with RGB composites and labels, and record false matches before making a claim that the spectral ranking retrieves flood evidence well.
