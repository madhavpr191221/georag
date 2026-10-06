# Local data

Raw datasets are intentionally excluded from Git.

Milestone 1 uses the official 13-band EuroSAT archive:

- Record: <https://zenodo.org/records/7711097>
- File: `EuroSAT_MS.zip` (about 2.1 GB)
- Published MD5: `e2ddcb955c32ff6ff3fa929111dabd9e`

Extract it so the class directories live under `data/EuroSAT_MS/EuroSAT_MS`, matching the archive's nested directory layout. The project does not download or silently transform the corpus.

`scripts/prepare_eurosat.py` scans the GeoTIFF headers, validates the band and shape contract, and writes a split manifest under `data/splits/`. Generated manifests are reproducible from the IDs, seed, fractions, and labels, so they are excluded by default; commit one deliberately when freezing a published experiment.

The same command writes a relocatable JSONL metadata catalog under `data/catalogs/`. Its fingerprint includes every TIFF's relative path, size, and modification time. A changed corpus invalidates the cache rather than silently reusing stale metadata.

EuroSAT's GeoTIFF storage order is `B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A`. This differs from placing `B8A` between `B08` and `B09` in spectral order. The adapter enforces the storage order explicitly; see the maintained [TorchGeo EuroSAT adapter](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/eurosat.py).

## Agriculture-Vision for Milestone 6

M6 uses the labeled Agriculture-Vision challenge distribution. Obtain it from the [official dataset page](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021) and comply with its access and use terms. Do not commit the data. Extract the archive into `data/AgricultureVision/`. Its files will be nested one level below the archive name; the loader root is `data/AgricultureVision/Agriculture-Vision-2021/`, with this structure:

```text
data/AgricultureVision/Agriculture-Vision-2021/
  train/
    images/rgb/<tile>.(jpg|png|tif)
    images/nir/<same-tile>.(png|tif)
    labels/<pattern>/<same-tile>.png
    masks/<same-tile>.png
    boundaries/<same-tile>.png
  val/
    images/rgb/...
    images/nir/...
    labels/<pattern>/...
    masks/...
    boundaries/...
```

The downloaded 2021 archive has nine labeled challenge patterns, which the adapter recognizes `double_plant`, `drydown`, `endrow`, `nutrient_deficiency`, `planter_skip`, `storm_damage`, `water`, `waterway`, and `weed_cluster`; all nine annotation masks plus the valid-pixel mask and field-boundary mask must exist for every labeled tile. A missing mask is treated as incomplete data and raises an error. Positive pattern pixels are counted only where both valid-pixel and field-boundary masks are nonzero. It reads channel-first raster data and returns either RGB `[3,128,128]` or RGB+NIR `[4,128,128]`, scaled by 255. The official challenge test set is not used because its anomaly labels are not released. The training and validation field assignments are supplied by Agriculture-Vision and should be preserved. The adapter records the field ID parsed from the documented `<field-id>_<crop-coordinates>` filename.

The official [Agriculture-Vision repository](https://github.com/SHI-Labs/Agriculture-Vision) documents the RGB/NIR imagery, per-pattern masks, and filename convention. The AWS Open Data Registry describes the original 94,986-tile release and field-disjoint 56,944/18,334/19,708 train/validation/test split; M6 uses the downloaded 2021 supervised challenge archive instead, whose split and mask folders the adapter reads directly.

M6 writes compressed per-split annotation catalogs under `artifacts/milestone_6/catalog/`. Each catalog stores tile identity, field, and pattern labels. A versioned fingerprint of source-relative paths, file sizes, and modification times invalidates stale catalogs. The cache contains no imagery and is ignored by Git.

## SEN12-FLOOD for the next retrieval task

Download the [Kaggle SEN12-FLOOD dataset](https://www.kaggle.com/datasets/virajkadam/sen12flood) manually and extract it under `data/sen12flood/`. The observed STAC data root is `data/sen12flood/sen12flood/`. Preserve its internal folder structure. Do not add the archive, extracted imagery, or image-containing result galleries to Git. The Kaggle listing currently reports its license as `Unknown`; GeoRAG records that status and does not redistribute the data.

The Kaggle listing describes the data as a copy of SEN12-FLOOD and cites the original dataset. Compare the observed content with the [SEN12-FLOOD dataset documentation](https://radiantearth.blob.core.windows.net/mlhub/sen12floods/documentation.pdf). Do not assume that directory names, channel order, metadata keys, or official split details from another copy match this archive.

The local adapter finds 2,236 Sentinel-2 and 3,331 Sentinel-1 source/label pairs covering 335 sequence IDs. Source and label item IDs have different collection prefixes and are paired by their exact sequence/date suffix. Flood labels are scene/date-level `"True"`/`"False"` strings in STAC properties; the sampled `labels.geojson` files have no features, so they are not pixel flood masks. Ninety-eight Sentinel-2 items reference all twelve optical band assets in STAC but have none of those TIFF files in the local extraction. The remaining 2,138 items have all twelve TIFFs. The metadata catalog preserves incomplete rows, while dataset iteration excludes them from model inputs.

The implemented tensor contract uses B05 as each scene's 20 m target grid and returns twelve float32 channels in the order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12. Finer bands are area-averaged, coarser bands use bilinear interpolation, and matching-resolution bands use nearest-neighbor resampling. Nodata is filled with zero in the tensor and exposed through the sample's common `valid_mask`; future normalization and spatial augmentation must use that mask. Pixel values remain raw DN by default (`value_scale = 1.0`); this is not a claim of calibrated reflectance.

Sequence tokens must remain exact strings: `0001` and `1` have different bounding boxes. The upstream documentation describes 337 locations while this copy contains 335 observed text tokens; the discrepancy remains open. The fixed local split uses only the 335 observed IDs. Run `uv run python scripts/prepare_sen12flood.py --config configs/milestone_8.toml` to regenerate it. The split manifest is grouped by sequence and includes all metadata rows, including unavailable scenes; it is written under ignored `data/splits/`. Consult [the M8 audit report](../reports/milestone_8_sen12flood_data_audit.md) for split counts and limitations.

The repository's `data/sen12flood/` ignore rule keeps the corpus local. M8 data plumbing and the M9 self-supervised RGB versus twelve-band representation experiment are implemented; M9's label-neighbor scores are diagnostics rather than human relevance judgments. M10 is the next milestone: natural-language EO query parsing and exact evidence retrieval.
