# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: RSNA (Python 3.12)
#     language: python
#     name: rsna
# ---

# %% [markdown]
# # RSNA Knee Dataset Audit
#
# This notebook audits schemas, sparse gold-label coverage, series metadata, and
# cross-file identifier integrity. It intentionally never displays radiology
# report text.

# %%
import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path(
    os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
).expanduser()
DATA_ROOT = Path(
    os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")
).expanduser()
COMPETITION_ROOT = DATA_ROOT / "raw" / "competition"
REPORT_ROOT = PROJECT_ROOT / "reports" / "data_audit"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

csv_paths = {
    "train": COMPETITION_ROOT / "train.csv",
    "train_series": COMPETITION_ROOT / "train_series.csv",
    "test": COMPETITION_ROOT / "test.csv",
    "test_series": COMPETITION_ROOT / "test_series.csv",
    "sample_submission": COMPETITION_ROOT / "sample_submission.csv",
}

missing_files = [str(path) for path in csv_paths.values() if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        "Required competition metadata is missing:\n" + "\n".join(missing_files)
    )

frames = {name: pd.read_csv(path) for name, path in csv_paths.items()}
train = frames["train"]
train_series = frames["train_series"]
test = frames["test"]
test_series = frames["test_series"]
submission = frames["sample_submission"]

for name, frame in frames.items():
    print(f"{name:20s} shape={frame.shape}")

# %% [markdown]
# ## Schemas
#
# Only column names are displayed. Report contents remain private.

# %%
for name, frame in frames.items():
    print(f"\n{name}")
    print(frame.columns.tolist())

# %% [markdown]
# ## Gold-label coverage

# %%
labels = [
    column for column in submission.columns if column != "StudyInstanceUID"
]

label_records = []
for label in labels:
    known = train[label].dropna()
    positives = int(known.sum())
    negatives = int(len(known) - positives)
    label_records.append(
        {
            "label": label,
            "known": int(len(known)),
            "positive": positives,
            "negative": negatives,
            "prevalence": (
                positives / len(known) if len(known) else float("nan")
            ),
        }
    )

label_summary = pd.DataFrame(label_records).set_index("label")
display(label_summary)

known_mask = train[labels].notna()
gold_label_rows = {
    "any_gold_label": int(known_mask.any(axis=1).sum()),
    "every_gold_label": int(known_mask.all(axis=1).sum()),
    "no_gold_labels": int((~known_mask.any(axis=1)).sum()),
    "missing_reports": int(train["Report"].isna().sum()),
}

for key, value in gold_label_rows.items():
    print(f"{key}: {value}")

# %% [markdown]
# ## Series distributions and identifier integrity

# %%
series_overview = {
    "training_studies": int(train["StudyInstanceUID"].nunique()),
    "training_studies_in_series_csv": int(
        train_series["StudyInstanceUID"].nunique()
    ),
    "training_series": int(train_series["SeriesInstanceUID"].nunique()),
    "test_studies": int(test["StudyInstanceUID"].nunique()),
    "test_series": int(test_series["SeriesInstanceUID"].nunique()),
}

for key, value in series_overview.items():
    print(f"{key}: {value}")

for column in ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]:
    print(f"\n{column}")
    display(train_series[column].value_counts(dropna=False).to_frame("count"))

train_ids = set(train["StudyInstanceUID"])
train_series_ids = set(train_series["StudyInstanceUID"])
test_ids = set(test["StudyInstanceUID"])
test_series_ids = set(test_series["StudyInstanceUID"])

integrity = {
    "train_studies_missing_series": len(train_ids - train_series_ids),
    "series_studies_missing_train_rows": len(train_series_ids - train_ids),
    "test_studies_missing_series": len(test_ids - test_series_ids),
    "series_studies_missing_test_rows": len(test_series_ids - test_ids),
}

integrity

# %% [markdown]
# ## Save sanitized summaries
#
# These outputs contain only aggregate counts and are safe to share within the
# private project repository. Raw reports and images remain outside the project.

# %%
label_summary.to_csv(REPORT_ROOT / "label_summary.csv")

series_metadata_counts = (
    train_series.groupby(
        ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"],
        dropna=False,
    )
    .size()
    .rename("count")
    .reset_index()
)
series_metadata_counts.to_csv(
    REPORT_ROOT / "series_metadata_counts.csv", index=False
)

audit_overview = {
    "shapes": {
        name: [int(frame.shape[0]), int(frame.shape[1])]
        for name, frame in frames.items()
    },
    "columns": {name: frame.columns.tolist() for name, frame in frames.items()},
    "gold_label_rows": gold_label_rows,
    "series_overview": series_overview,
    "identifier_integrity": integrity,
}

with (REPORT_ROOT / "audit_overview.json").open("w", encoding="utf-8") as file:
    json.dump(audit_overview, file, indent=2, ensure_ascii=False)

print(f"Saved sanitized audit summaries to {REPORT_ROOT}")

# %%

# %%
