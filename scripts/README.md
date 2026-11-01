# DICOM Audit

`01_dicom_audit.py` performs an exhaustive, read-only DICOM header inventory
and a configurable pixel-decoder audit. It never writes under
`/home/shaurya/rsna-data/raw/competition`; all generated files go to the
separate manifest directory.

## VM setup

Use a dedicated Python 3.12 environment so codec pins cannot disturb the
Jupyter or training environment:

```bash
python3.12 -m venv /home/shaurya/venvs/rsna-audit
source /home/shaurya/venvs/rsna-audit/bin/activate
python -m pip install --upgrade pip
cd /home/shaurya/rsna-project
python -m pip install -r requirements-audit.txt
```

The requirements include pydicom plus the libjpeg and OpenJPEG plugins needed
for the documented JPEG Lossless and JPEG 2000 transfer syntaxes.

## Recommended first run

Run inside `tmux` so an SSH disconnect does not stop the audit:

```bash
tmux new-session -s rsna-dicom-audit
```

```bash
cd /home/shaurya/rsna-project
source /home/shaurya/venvs/rsna-audit/bin/activate
python scripts/01_dicom_audit.py \
  --competition-root /home/shaurya/rsna-data/raw/competition \
  --output-root /home/shaurya/rsna-data/manifests/dicom_audit_v1 \
  --expected-dicom-count 819635 \
  --workers 8 \
  --batch-size 5000 \
  --decode sample \
  --samples-per-transfer-syntax 64 \
  --resume
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach-session -t rsna-dicom-audit
```

The command exits with status `0` only if all strict inventory checks and the
requested decoder checks pass. Status `2` means the outputs were saved but at
least one integrity check failed or the run configuration was unsafe.

The aggregate result is:

```text
/home/shaurya/rsna-data/manifests/dicom_audit_v1/audit_summary.json
```

## Orientation-variation follow-up

After the inventory passes, characterize series whose per-slice orientation
changes after five-decimal rounding:

```bash
python scripts/02_orientation_audit.py \
  --audit-root /home/shaurya/rsna-data/manifests/dicom_audit_v1
```

This command reads only the existing Parquet manifest; it does not reopen the
DICOMs. It writes an identifier-free aggregate report to:

```text
/home/shaurya/rsna-data/manifests/dicom_audit_v1/orientation_variation_summary.json
```

The report measures maximum row, column, and slice-normal angular variation,
rounded-profile patterns, projection differences, and whether projection onto
a consensus series normal changes slice order. No study, series, SOP, or file
identifiers are included, so the reviewed summary can be copied into
`reports/dicom_audit/` for version control.

## Official metric and submission smoke test

Run the offline smoke test after the metadata CSV audit passes:

```bash
python scripts/03_submission_smoke_test.py \
  --competition-root /home/shaurya/rsna-data/raw/competition \
  --output /home/shaurya/rsna-project/submission.csv \
  --report-output /home/shaurya/rsna-project/reports/submission_smoke_test/summary.json
```

This command reproduces the official unweighted macro ROC AUC on the fully
labeled training rows, where constant predictions must score exactly `0.5` for
every target. It then generates a `0.5` example-test submission and strictly
validates the exact 13-column schema, study count, identifier uniqueness and
test-set membership, authoritative sample-submission order, finite
probabilities, and the inclusive `[0, 1]` range.

Raw reports are parsed only to traverse `train.csv` correctly and are never
displayed or written. The generated JSON report contains aggregate values only.
The command refuses to place either generated output inside the raw competition
tree.

## Complete example-test inference path

After the submission smoke test passes, decode and preprocess every DICOM in
the three-study example test set:

```bash
python scripts/04_example_test_pipeline.py
```

For every series, the loader validates DICOM/path UIDs and image metadata,
aligns per-slice normal signs, sorts positions along a normalized consensus
normal, decodes and rescales every pixel array, handles `MONOCHROME1`, applies
bounded-memory series-level 1st/99th-percentile normalization, samples 16
uniformly spaced slices, validates positive and consistent DICOM
`PixelSpacing`, and maps each complete physical field of view into `224 x 224`.
The image is resized to fit with its physical aspect ratio intact and the
shorter dimension is padded symmetrically with normalized zero; no anatomy is
cropped. Geometry-free series may use a unique `InstanceNumber` fallback for
slice ordering, but physical in-plane spacing remains mandatory. Partial or
ambiguous geometry fails closed.

The command processes one series at a time, discards pixel tensors after their
checks, regenerates a validated placeholder submission, and writes only
identifier-free aggregates to
`reports/example_test_pipeline/summary.json`. Its runtime is a correctness
measurement for 3 studies and must not be extrapolated directly to the roughly
1,300-study hidden test set. The report records physical FOV, effective output
spacing, resized shapes, and padded fractions so the transform remains
auditable. A representative train-derived cohort must still be visually
reviewed and benchmarked before training.

## Physical pixel-spacing and field-of-view audit

Before choosing the final image transform, analyze the completed manifest
without reopening any raw DICOMs:

```bash
python scripts/05_physical_geometry_audit.py \
  --audit-root /home/shaurya/rsna-data/manifests/dicom_audit_v1 \
  --candidate-crop-mm 120 140 160 180 200 \
  --minimum-coverage 0.90 \
  --target-pixels 224
```

The audit processes every manifest row, verifies the row and series counts
against `inventory_summary.json`, and emits series-weighted pixel-spacing and
physical-field-of-view distributions overall and by split, anatomical plane,
sequence flags, and their plane/sequence combinations. It explicitly counts
missing, nonpositive/nonfinite, and within-series inconsistent spacing, as
well as missing or inconsistent dimensions. A series with any such issue is
excluded from crop selection but remains fully accounted for in the report.

For each candidate square detail window, coverage is the fraction of series
whose row and column physical fields of view are both at least that size, so
no padding would be required. The recommendation selects the largest candidate
meeting the configured threshold overall and in every observed anatomical
plane, then derives its output pixel spacing. If no candidate qualifies, the
report marks the smallest candidate as an uncertified fallback.

The fixed physical window is never recommended as the sole input: any center
crop could discard anatomy in a larger or off-center field of view. The
required primary transform retains the complete physical field of view,
resizes it to fit the target square with its aspect ratio intact, and pads the
shorter dimension. The selected fixed window is an optional high-resolution
detail view only, subject to representative visual QA.

The identifier-free aggregate report is written atomically to:

```text
/home/shaurya/rsna-data/manifests/dicom_audit_v1/physical_geometry_summary.json
```

After reviewing it for aggregate-only content, copy it to
`reports/dicom_audit/physical_geometry_summary.json`. Do not copy any Parquet
manifest part, path, UID, or per-file error output into Git.

## Representative preprocessing benchmark and visual QA

Benchmark the full-FOV transform on 18 deterministic training series: six per
anatomical plane, covering both observed sequence-flag combinations, low,
median, and high physical FOV anchors, and rectangular edge cases.

```bash
python scripts/06_preprocessing_benchmark.py
```

The script exhaustively validates the source manifest before selecting the
cohort, decodes every image in each selected series, generates 16-slice PNG
montages, and records runtime, physical FOV, output spacing, resized shapes,
and padded fractions. Identifier-bearing artifacts and derived pixels remain
outside Git under:

```text
/home/shaurya/rsna-data/manifests/preprocessing_benchmark_v1/
```

That directory contains `cohort_manifest.csv`, `load_failures.csv`,
`visual_review_checklist.csv`, and anonymous `visuals/case-*.png` montages. The
series loader remains strict: it never reshapes a corrupt payload or silently
skips a bad instance. If a selected series cannot be loaded, the benchmark
records the identifier-bearing failure externally, selects a deterministic
replacement from the same anatomical plane while prioritizing matching sequence
flags and physical geometry, and exposes aggregate failure and replacement
counts in the repository report.
The repository receives only the identifier-free aggregate report at
`reports/preprocessing_benchmark/summary.json`. The report remains marked
`pending_human_review` until every montage is checked for complete anatomy,
acceptable padding, and legibility.

After a human has inspected every tile in all 18 montages and confirmed that
all cases pass, record that explicit attestation with:

```bash
python scripts/09_finalize_preprocessing_review.py --all-cases-pass
```

The finalizer verifies the anonymous cohort IDs, PNG count, and expected montage
dimensions before marking the aggregate report `visual_review.status` as
`pass`. It also completes the external checklist; identifiers and derived
pixels remain outside Git.

## Gold-label validation folds

Create five deterministic study-level folds from the 58 fully gold-labeled
studies:

```bash
python scripts/07_gold_folds.py
```

The rare-label-first multilabel assignment uses 512 deterministic restarts and
fails unless every target has both positive and negative examples in every
fold, fold sizes differ by at most one, and all gold rows are assigned exactly
once. The identifier-bearing assignment is stored only at:

```text
/home/shaurya/rsna-data/manifests/gold_folds_v1/gold_folds.csv
```

Only aggregate fold sizes, target prevalence, class-coverage checks, and an
opaque manifest fingerprint are written to
`reports/validation_folds/summary.json`. The available schema permits
study-level grouping only; because it has no stable patient identifier, a
separate duplicate-image/report risk audit remains necessary before treating
the folds as proven patient-independent.

## Exhaustive pixel decode

After the sampled audit passes, the same manifest can be reused for an
exhaustive decoder check. This reads and decompresses every header-valid image
and can take many hours. Treat this as a required pretraining gate if any
representative benchmark series fails pixel decoding:

```bash
python scripts/01_dicom_audit.py \
  --competition-root /home/shaurya/rsna-data/raw/competition \
  --output-root /home/shaurya/rsna-data/manifests/dicom_audit_v1 \
  --expected-dicom-count 819635 \
  --workers 8 \
  --batch-size 5000 \
  --decode all \
  --resume
```

Interrupted runs are safe to restart with the identical command. Completed
Parquet parts are validated by schema, row count, input digest, and boundary
paths before being reused. A changed configuration is rejected instead of
overwriting prior results.

## Pixel-integrity exclusion policy

If the exhaustive decode reports any unsafe instances, convert them into a
strict series-level exclusion policy:

```bash
python scripts/08_pixel_integrity_policy.py
```

The command reconciles every decode part with `decode_summary.json`, checks the
failed series against `train_series.csv` and `test_series.csv`, and fails if an
affected study would have no eligible series left. It excludes the entire
series containing an unsafe instance; it never skips a slice, repairs a pixel
payload, or reshapes data to fit incorrect header dimensions.

Identifier-bearing outputs stay outside Git under:

```text
/home/shaurya/rsna-data/manifests/pixel_integrity_v1/
```

Training and inference must filter series using `excluded_series.csv` before
loading. Per-instance paths and errors remain in `unsafe_instances.csv`. Only
the aggregate report at
`reports/dicom_audit/pixel_integrity_summary.json` is safe to commit.

## Gold-label overfit training gate

After the preprocessing visual review, gold folds, and pixel-integrity policy
all pass, run a same-cohort memorization test on class-complete fold 0. Use the
dedicated training environment rather than the decoder-audit environment:

```bash
./scripts/setup_training_env.sh
```

The setup script creates `/home/shaurya/venvs/rsna-train`, installs the pinned
Python dependencies and the official PyTorch 2.7.0 CUDA 12.8 wheel, prepares
the external cache/model directories, and runs the training preflight. It does
not read DICOM pixel data. The preflight must finish with `Training preflight
PASS`; it checks the NVIDIA driver, an actual CUDA matrix multiplication, GPU
memory, storage, path isolation, exact dependency versions, manifest validity,
fold-0 class coverage, and every selected series directory.

To repeat the preflight without reinstalling the environment:

```bash
source /home/shaurya/venvs/rsna-train/bin/activate
python scripts/training_preflight.py --device cuda
```

After it passes, run the gate inside `tmux`:

```bash
python scripts/10_gold_overfit.py --device cuda
```

The first run decodes only eligible fold-0 series through the validated
full-FOV loader and caches `16 x 224 x 224` float16 tensors under:

```text
/home/shaurya/rsna-data/processed/gold_overfit_cache_v1/
```

Later runs strictly validate and reuse that cache without decoding DICOMs
again. The tiny model uses a convolutional slice encoder, mean slice pooling,
series metadata, and learned series attention. Model checkpoints remain under:

```text
/home/shaurya/rsna-data/models/gold_overfit_v1/
```

The command passes only when the same training cohort reaches BCE loss at or
below `0.05`, binary accuracy of `1.0`, and macro ROC AUC of at least `0.99`.
These are wiring and memorization checks, not validation results. Failure is
useful evidence of a loader, label, aggregation, optimization, or capacity bug;
do not proceed to fold validation until it is understood.

Only identifier-free aggregate metrics are written to
`reports/training/gold_overfit_summary.json`. Study IDs, tensor mappings,
derived pixels, and checkpoints remain external.

## Full-cohort 2.5D tensor cache

The audited training metadata contains 24,371 series across 4,407 studies.
The three pixel-integrity exclusions leave 24,368 eligible series. Before
weak-label vision training, materialize every series that survives the
pixel-integrity exclusion policy as a `16 x 224 x 224` float16 NumPy tensor:

```bash
source /home/shaurya/venvs/rsna-train/bin/activate
cd /home/shaurya/rsna-project
python scripts/13_cache_training_tensors.py --dry-run
```

The dry run validates exact study/series counts, report presence, metadata,
exclusions, source fingerprints, path isolation, and storage headroom without
decoding pixels. It exits with status `3` because the deliberately partial
cache is not training-ready.

Run a bounded smoke batch before the full build:

```bash
/usr/bin/time -v python scripts/13_cache_training_tensors.py \
  --workers 4 \
  --max-new-series 64
```

Inspect its aggregate rate and memory use, then resume the same cache without
the limit:

```bash
python scripts/13_cache_training_tensors.py --workers 4
```

The expected tensor payload is approximately 36.5 GiB before NumPy headers and
small metadata sidecars. Four CPU workers are the conservative starting point
for the eight-vCPU VM; the T4 is not used for preprocessing. Each tensor and
its metadata sidecar are written atomically. The cache configuration pins the
source CSV and exclusion hashes, tensor dimensions, dtype, normalization,
spatial transform, and slice-sampling policy. A nonblocking cache lock prevents
two builders from writing the same root.

The identifier-bearing artifacts remain outside Git under:

```text
/home/shaurya/rsna-data/processed/training_cache_v1/
  cache_config.json
  cache_manifest.csv
  failures.csv
  progress.json
  studies/<StudyInstanceUID>/<SeriesInstanceUID>.npy
  studies/<StudyInstanceUID>/<SeriesInstanceUID>.json
```

Every resume validates completed tensor shape, float16 dtype, finite `[0, 1]`
range, SHA-256 fingerprint, metadata, and configuration before treating it as a
cache hit. Corrupt or stale completed entries fail closed. A tensor left
without its metadata sidecar by an interruption is safely recomputed and
atomically replaced. The completed external manifest records tensor hashes and
series geometry for downstream training.

Only identifier-free counts, rates, source fingerprints, and storage totals are
written to `reports/preprocessing_cache/summary.json`. Exit status `0` means the
entire eligible cohort is complete, `3` means a requested partial run completed
cleanly, and `2` means validation or preprocessing failed.

## Public weak-label benchmark

Compare public report-derived label CSVs against the 58 official gold studies:

```bash
python scripts/14_benchmark_public_weak_labels.py \
  --candidate source_a=/external/path/source_a.csv \
  --candidate source_b=/external/path/source_b.csv
```

Candidate files and identifiers stay external. The committed report contains only
aggregate AUC, accuracy, and Brier metrics. Treat the result as a label-source gate;
58 studies are too few for an unbiased leaderboard estimate.

Pass `--ensemble-output /external/path/weak_labels.csv` to materialize the
equal-weight candidate mean in the training pipeline's `_prob` schema. The command
refuses to write this identifier-bearing artifact inside the repository.

## Public-label imaging gate

After building the external three-source label ensemble, run the controlled
same-architecture comparison on two representative folds:

```bash
source /home/shaurya/venvs/rsna-train/bin/activate
python scripts/15_run_public_label_gate.py --folds 0 3
```

The command is resumable. It changes only weak supervision versus imaging v1, writes
checkpoints under `imaging_v2_public_labels`, and emits identifier-free aggregate
reports for the weak stage and requested folds.

## Offline imaging-v2 Kaggle package

Run `scripts/16_package_kaggle.py` in the Python 3.12 training environment.
It exports all five promoted fold checkpoints, the exact project and cached
DINOv2 source (including licenses), an autorun notebook, and offline DICOM wheels.
Original checkpoints are preserved; optimizer state and training history are
omitted from inference copies. Fold identity, target order, source-data hashes,
and finite weights are checked before writing the completion manifest.

Prepare one wheel per package for Kaggle's Python/architecture. The current
wheelhouse targets Python 3.12 on Linux x86-64. NumPy and CUDA PyTorch come from
the Kaggle runtime; their observed versions are recorded in the inference report.

```bash
python -m pip download --only-binary=:all: --no-deps \
  --dest /home/shaurya/rsna-data/staging/kaggle-wheels \
  pydicom==3.0.2 pylibjpeg==2.1.0 \
  pylibjpeg-libjpeg==2.4.0 pylibjpeg-openjpeg==2.5.0

python scripts/16_package_kaggle.py \
  --project-root /home/shaurya/rsna-project \
  --model-root /home/shaurya/rsna-data/models/imaging_v2_public_labels \
  --dinov2-repo /home/shaurya/rsna-data/models/torch_hub/facebookresearch_dinov2_main \
  --wheelhouse /home/shaurya/rsna-data/staging/kaggle-wheels \
  --output-root /home/shaurya/rsna-data/staging/imaging-v2-kaggle
```

Use a new output directory for each build. An interrupted build has no
`rsna-package.json` completion manifest. Keep the artifact outside Git.
Upload the bundle as a private Kaggle input, import its `kernel.ipynb`, attach
the competition and bundle, select a GPU, and disable internet. The notebook
verifies file hashes, installs only bundled DICOM wheels using `--no-index`,
then writes `/kaggle/working/submission.csv` and `imaging_v2_summary.json`.
It runs automatically; the paired VM notebook remains guarded unless
`RSNA_RUN_INFERENCE=1` is set.

The package is a deployment gate, not a new model improvement. Measure the
Kaggle runtime and public score before promoting the next geometry experiment.

## Controlled sampling and augmentation gate

Run `scripts/17_run_geometry_gate.py` in the Python 3.12 training environment.
Both arms reuse the promoted imaging-v2 weak-pretraining checkpoint and change
only the gold fine-tuning stage, so each fold is directly comparable to the
imaging-v2 fold result without paying for another weak-pretraining run. Neither
arm re-reads raw DICOMs; the existing tensor cache is used unchanged.

| Arm | Change | Everything else |
|---|---|---|
| `slices16` | `slices_per_series` 12 to 16, using the full cached depth | Identical to imaging v2 |
| `augment` | Training-only slice-sampling and intensity jitter at 12 slices | Identical to imaging v2 |

Augmentation applies a per-series gamma, scale, and shift clipped back into
`[0, 1]`, plus slice-center jitter bounded by half the center spacing. It is
seeded from the stage seed, epoch, and study index, so an interrupted run
resumes with identical inputs. Validation and offline inference never augment.

Horizontal flips are deliberately excluded. Medial and lateral compartments are
distinguished by their position in coronal and axial knee MRI, and the cache
carries no laterality tag, so flipping would silently relabel five targets.

```bash
python scripts/17_run_geometry_gate.py --arm slices16 --folds 0 3
python scripts/17_run_geometry_gate.py --arm augment  --folds 0 3
```

Each arm writes checkpoints under `models/imaging_v3_<arm>` and an
identifier-free `reports/training/imaging_v3_<arm>_gate.json` carrying the
per-fold delta against imaging v2. The command is resumable. Promote an arm to
folds 1, 2, and 4 only when it improves both gate folds; `slices16` also raises
offline inference cost by roughly one third, so re-measure the runtime
projection before packaging it.
