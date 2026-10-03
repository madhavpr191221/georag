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
