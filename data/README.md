# Data

No competition data is stored in this repository.

The official dataset (knee MRI DICOM studies, series descriptors, and training
radiology reports) is available to registered participants of the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
competition on Kaggle and is subject to the competition's data license.

The pipeline expects the extracted competition files under a data root that
defaults to `/home/shaurya/rsna-data` and can be overridden through the
environment variables and command-line options documented in each script.
Raw images, derived labels, tensor caches, checkpoints, and per-study
predictions are deliberately kept out of version control; only
identifier-free aggregate summaries are committed under `reports/`.
