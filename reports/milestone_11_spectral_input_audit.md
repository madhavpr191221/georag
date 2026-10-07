# M11 spectral-input audit

## Purpose

Before treating a spectral index as evidence, check that GeoRAG reads the intended bands, places them on one map grid, handles invalid pixels consistently, and computes the displayed values correctly. This is a data and software audit. It does **not** test flood-detection accuracy or prove that high MNDWI means flooding.

## What was checked

The audit used the local SEN12-FLOOD Sentinel-2 archive and the sequence-grouped seed-17 validation split. It processed all **210 available validation scenes** (55 labeled flood, 155 labeled no-flood) and selected 16 scenes for visual inspection: four in each flood/no-flood × high/low median-MNDWI group. The test split was not read. The selected sample was chosen for visual coverage, not to tune the index or retrieve results.

The implementation recomputed the existing indices from the configured 12-band tensor:

| Index | Bands used | Formula |
|---|---|---|
| NDWI | B03 green, B08 NIR | $(B03-B08)/(B03+B08)$ |
| MNDWI | B03 green, B11 SWIR | $(B03-B11)/(B03+B11)$ |
| NDVI | B08 NIR, B04 red | $(B08-B04)/(B08+B04)$ |

For the selected scenes, the audit recorded source-raster dimensions, CRS, resolution, affine transform, dtype, nodata, scale, offset, units, and raster mask. It also saved one valid pixel's 12 band values and recomputed all three index formulas independently, then compared those calculations with the generated maps.

## Findings

- **Band selection and arithmetic:** the implementation uses the intended band mapping. Across the validation split, all finite index values were within [-1, 1]. No out-of-range values were found. The maximum difference between the hand-calculated pixel formulas and the map values was approximately $2.9\times10^{-8}$, consistent with float32 rounding.
- **Shared map grid:** the source bands have different native resolutions and grid origins. The dataset adapter resamples all 12 onto the B05-anchored 20 m grid; therefore, RGB previews and index maps in the audit have the same output pixels. The contact sheet shows broad features in the indices in the same locations as the RGB composites. This is a visual sanity check, not a formal geolocation-accuracy test.
- **Coverage:** the common valid mask covered 93.1% to 98.8% of each validation scene (median 96.5%). This is source support after reprojection, not a cloud-free fraction. The selected source raster masks report all source pixels valid, while their differing footprints leave some output pixels without common support.
- **Important zero-data finding:** **12 of 210 scenes** have nonempty common-valid masks, yet all three indices have zero valid pixels. For each of those 12, every common-mask pixel has zero values across all 12 bands. A spot check of one scene confirmed the entire returned image tensor is zero. Its source rasters have no declared nodata value, so the current adapter has no metadata signal that these zero-filled scenes are unusable. Their scene IDs are listed in `artifacts/milestone_11/spectral_input_audit/audit_summary.json`.
- **Quality masks:** the 2,236 source STAC items expose the twelve band assets, with no cloud, shadow, SCL, QA, or mask assets. This audit did not add cloud masking. Clouds and shadows therefore remain possible causes of misleading index patterns.
- **Radiometry:** the selected TIFF headers report `uint16`, scale 1, offset 0, and no nodata value. The inspected STAC band assets contain no per-band scale, offset, unit, or nodata metadata. The adapter's values should be described as stored/raw digital numbers; this audit does not establish calibrated surface reflectance.

The current retrieval code already skips scenes with no valid MNDWI pixels. The audit did not modify that behavior, the index equations, or the ranking. Still, the 12 all-zero scenes expose a distinction the data layer should make explicit: a nonempty raster mask does not guarantee useful spectral signal.

## Interpretation and next gate

The index arithmetic and band wiring pass this first software check, and the sample maps appear co-registered at the adapter's output grid. However, the input-validity check found a real archive/data-quality problem. Before running scene-label association, temporal-change metrics, or making claims about spectral evidence, inspect the 12 all-zero source scenes and decide a documented no-signal policy. The cause is not established here; do not assume whether the zeros originate in the download, the source archive, or preprocessing.

This is not evidence that the indices identify floods. Scene labels describe scenes/dates, not flooded pixels, and clouds, shadows, permanent water, soil, and vegetation can affect the same indices.

## Reproduction

From the repository root, with the local SEN12-FLOOD archive in the configured path:

```powershell
uv run python scripts/audit_spectral_inputs.py
```

The command uses `configs/milestone_8.toml`, `data/splits/sen12flood_s2_seed_17.csv`, the validation split, seed 17, and four examples per available stratum. It writes local, ignored artifacts to `artifacts/milestone_11/spectral_input_audit/`:

- `audit_summary.json` — counts, split fingerprint, band/index mapping, metadata inventory, mask coverage, value ranges, zero-data scenes, and hand-check error.
- `scene_measurements.json` — per-scene validity and index summaries for all 210 validation scenes.
- `selected_examples.csv` — the 16 deterministic visual-QA examples and strata.
- `selected_raster_metadata.csv` — source-header details for all 12 bands in the selected scenes.
- `hand_checked_pixels.csv` — band values and independent index calculations for one valid pixel per selected scene.
- `spectral_input_contact_sheet.png` — RGB preview, NDWI, MNDWI, NDVI, and common-valid mask for each selected example.

Split manifest SHA-256: `d1ef22ad1dc00e65099c98eec9ee8134b3d75405fc5a743a1bcf9549ce11785f`.

## Validation

Focused checks passed: 14 tests across spectral audit, spectral-index, and SEN12-FLOOD dataset behavior. The only emitted warnings were Rasterio's pending deprecation warning in existing synthetic raster tests.
