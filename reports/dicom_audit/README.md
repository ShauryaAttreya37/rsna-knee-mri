# DICOM Audit Reports

The full slice-level manifest, file paths, UIDs, checkpoints, decoder results,
and per-file errors remain outside Git under:

```text
/home/shaurya/rsna-data/manifests/dicom_audit_v1/
```

Only aggregate, identifier-free summaries may be copied into this repository
after review. Never commit the generated Parquet parts, unexpected-file list,
or error rows.

## Orientation-variation follow-up

The identifier-free follow-up analysis completed successfully on 2026-08-27
against manifest fingerprint
`9e95dd790bbdf2294f1cc026bc3cfb80b45eeceb777603070676eb1cef686abc`.
All validation checks passed.

- 123 of 24,386 series (0.50%) had more than one
  `ImageOrientationPatient` profile after five-decimal rounding, covering 4,735
  of 819,635 slices.
- Every flagged series was in the training split: 7 axial, 44 coronal, and 72
  sagittal series.
- All 123 had a maximum pairwise slice-normal difference at or below 0.001
  degrees. The observed maximum was 0.000573 degrees and the 95th percentile
  was 0.000572 degrees.
- The largest orientation-component range was approximately `1e-5`.
- The maximum centered difference between per-slice-normal and
  consensus-normal coordinates was 0.001855 mm; the 95th percentile was
  0.001365 mm.
- Projection onto an aligned consensus series normal preserved slice order in
  all 123 flagged series.

These differences are numerical jitter rather than meaningful slice
angulation. Preprocessing should retain and validate every slice's orientation,
align the slice-normal signs within each series, form a normalized consensus
normal, and sort all slice positions by projection onto that consensus normal.
It must not require the raw orientation components to be byte-for-byte or
decimal-for-decimal identical.

The reviewed aggregate JSON belongs at
`reports/dicom_audit/orientation_variation_summary.json`; the source copy
remains under the external manifest directory.

## Physical-geometry follow-up

`scripts/05_physical_geometry_audit.py` reads the same existing Parquet
manifest without opening raw DICOMs. It exhaustively accounts for pixel
spacing and image dimensions at both row and series level, produces
series-weighted physical-field-of-view distributions, evaluates candidate
square physical detail windows, and emits a data-derived resampling
recommendation. The required primary view must retain the complete field of
view through aspect-preserving fit-and-pad; a fixed physical window may be an
optional detail view but must never be the sole model input.

The VM source report belongs at:

```text
/home/shaurya/rsna-data/manifests/dicom_audit_v1/physical_geometry_summary.json
```

The successful aggregate report is stored in this directory. The primary
loader transform now uses spacing-aware, full-FOV aspect fitting with symmetric
zero padding at `224 x 224`, while the selected 140 mm physical window remains
an optional detail view only. Representative low-FOV, high-FOV, and off-center
series still require visual review before training.

## Exhaustive pixel integrity

`scripts/08_pixel_integrity_policy.py` converts exhaustive decode failures into
an external series-level exclusion manifest and writes only aggregate counts,
failure classes, policy checks, and opaque manifest fingerprints to
`pixel_integrity_summary.json` in this directory. Never commit
`excluded_series.csv`, `unsafe_instances.csv`, paths, UIDs, or per-file errors.
