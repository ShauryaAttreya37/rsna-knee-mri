# RSNA Knee MRI Abnormality Detection

**Author:** [Shaurya Attreya](https://github.com/ShauryaAttreya37) · **Project page:** https://shauryaattreya37.github.io/rsna-knee-mri/ · **License:** [MIT](LICENSE)

An end-to-end pipeline for the 2026 Kaggle
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
challenge: predict the probability of twelve knee abnormalities (ACL, MCL,
meniscal tears, compartmental osteoarthritis, effusion, synovitis, Baker's
cyst, bone contusion, and fracture) from multi-series knee MRI studies.

All code in this repository was designed and written by Shaurya Attreya.

## Highlights

- **Scale:** 4,407 training studies, 24,371 MRI series, and 819,078 DICOM
  slices audited end to end; only 58 studies carry expert (gold) labels.
- **Weak supervision from reports:** multilingual radiology reports were
  converted into soft labels for the 4,349 unlabeled studies, first with a
  local Qwen2.5-3B-Instruct extractor (gold macro AUC 0.753), then with an
  equal-mean ensemble of three public CC0 label sources (gold macro AUC 0.896).
- **Model:** a DINOv2 ViT-S/14 slice encoder with target-specific attention
  pooling over slices and over series, conditioned on series metadata
  (plane, fluid sensitivity, fat suppression).
- **Leakage-safe validation:** deterministic multilabel-stratified five-fold
  splits over the 58 gold studies, which are excluded from weak pretraining.
- **Result:** pooled out-of-fold macro AUC improved from **0.686 to 0.784**
  (five-fold mean 0.724 to 0.812) when the weak-label source was replaced
  with architecture, cache, and folds held fixed.
- **Offline inference:** five fold checkpoints, rank-averaged per target,
  packaged as a self-contained Kaggle input with strict submission validation.

## Repository layout

| Path | Contents |
|---|---|
| [`src/rsna/`](src/rsna) | Library code: DICOM audit and loading, preprocessing cache, label extraction, folds, training, evaluation, inference, packaging |
| [`scripts/`](scripts) | Numbered command-line entry points for each pipeline stage |
| [`notebooks/`](notebooks) | Jupytext notebooks that drive audit, training, OOF evaluation, and inference |
| [`tests/`](tests) | pytest suite covering loaders, folds, metrics, training contracts, inference, and packaging |
| [`reports/`](reports) | Identifier-free aggregate summaries of every audit, benchmark, and training stage |

## Running

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-training.txt
pytest
```

The competition data is not included; see [`data/README.md`](data/README.md).

## Acknowledgements

- Competition data: Radiological Society of North America (RSNA) and Kaggle.
  Not redistributed here.
- Slice encoder: [DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI).
- Report extractor: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- Three public CC0 report-label datasets shared by fellow Kaggle competitors
  were benchmarked and ensembled as weak labels.

This is a competition project. It is not a medical device and has not been
validated for clinical use.

© 2026 Shaurya Attreya. Released under the MIT License; reuse must retain the
copyright notice.
