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
#     display_name: RSNA (Training PyTorch CUDA)
#     language: python
#     name: rsna-train
# ---

# %% [markdown]
# # RSNA Knee: Phase 3 Multilingual Report Weak Supervision Extraction
#
# This notebook extracts weak supervision pseudo-labels from **4,407 clinical radiology reports** across 9 languages (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French).
#
# ### Key Features:
# 1. **Local Multilingual LLM**: `Qwen/Qwen2.5-3B-Instruct` quantized in 4-bit NF4 (VRAM usage ~3.5 GiB on Tesla T4).
# 2. **High-Throughput Batching**: Generates predictions in batches of 8 (~2.1s per report, 4.5x faster than sequential).
# 3. **Crash-Proof Checkpointing**: Saves progress to disk every batch. If interrupted or restarted, it automatically resumes from where it left off.
# 4. **12 Target Conditions**: Generates calibrated continuous probabilities ($0.0$ to $1.0$) for ACL, MCL, Menisci, Compartment OA, Effusion, Synovitis, Baker's cyst, Contusion, and Fracture.
# 5. **Visual Analytics**: Plots pseudo-label prevalence and probability distributions across the cohort.

# %%
import json
import os
import sys
import time
from pathlib import Path

# Paths configuration
PROJECT_ROOT = Path(os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")).expanduser()
DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")).expanduser()
COMPETITION_ROOT = DATA_ROOT / "raw" / "competition"
OUTPUT_CSV = DATA_ROOT / "manifests" / "weak_labels_v1" / "weak_labels.csv"
REPORT_SUMMARY = PROJECT_ROOT / "reports" / "report_supervision" / "weak_supervision_summary.json"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rsna.report_extractor import ClinicalReportExtractor
from rsna.submission import TARGET_COLUMNS

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
else:
    print("WARNING: CUDA is not available. Ensure you selected the 'rsna-train' kernel.")

# %% [markdown]
# ## 1. Inspect Training Cohort & Reports
# Let's inspect `train.csv` to review the dataset size and sample reports across languages.

# %%
train_path = COMPETITION_ROOT / "train.csv"
df = pd.read_csv(train_path)
total_studies = len(df)
gold_count = int(df["ACL"].notna().sum())
unannotated_count = total_studies - gold_count

print(f"Total studies in train.csv  : {total_studies}")
print(f"Gold-annotated studies       : {gold_count}")
print(f"Unannotated studies          : {unannotated_count}")

# Check for any existing progress
if OUTPUT_CSV.is_file():
    existing_df = pd.read_csv(OUTPUT_CSV)
    print(f"\n[INFO] Found existing output file: {OUTPUT_CSV}")
    print(f"[INFO] Already extracted: {len(existing_df)} / {total_studies} studies.")
else:
    print(f"\n[INFO] Starting fresh extraction to: {OUTPUT_CSV}")

# Display a sample report
sample_report = df.loc[0, "Report"]
print(f"\nSample Clinical Report (First 300 chars):\n{sample_report[:300]}...")

# %% [markdown]
# ## 2. Execute Weak Supervision Extraction
#
# Run the extraction engine.
# - **Batch size**: 8 (optimal throughput and memory efficiency on T4 GPU).
# - **Checkpointing**: Automatic after each batch.
# - **Resume**: If you stop this cell and run it again, it skips already completed studies automatically.
# - *Tip*: To test on a small slice first, you can pass `max_studies=16`. For the full run, keep `max_studies=None`.

# %%
from rsna.weak_supervision import extract_weak_supervision

# Set max_studies=None for the complete 4,407 run (~2.5 hours on T4)
# Or set max_studies=50 for a quick test
MAX_STUDIES = None
BATCH_SIZE = 8

print(f"Launching extraction (batch_size={BATCH_SIZE}, max_studies={MAX_STUDIES})...\n")
summary = extract_weak_supervision(
    competition_root=COMPETITION_ROOT,
    output_csv=OUTPUT_CSV,
    report_summary=REPORT_SUMMARY,
    batch_size=BATCH_SIZE,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_studies=MAX_STUDIES,
)

print("\nExtraction Summary:")
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 3. Analyze Extracted Pseudo-Labels & Prevalence
#
# Inspect the distribution of generated pseudo-labels across all 12 target conditions.

# %%
if OUTPUT_CSV.is_file():
    weak_df = pd.read_csv(OUTPUT_CSV)
    print(f"Total labeled rows in {OUTPUT_CSV.name}: {len(weak_df)}")
    
    # Compute prevalence (probability >= 0.5)
    prevalence = {}
    mean_probs = {}
    for col in TARGET_COLUMNS:
        pcol = f"{col}_prob"
        if pcol in weak_df.columns:
            prevalence[col] = float((weak_df[pcol] >= 0.5).mean()) * 100
            mean_probs[col] = float(weak_df[pcol].mean())
            
    prev_s = pd.Series(prevalence).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(prev_s.index, prev_s.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.ylabel("Prevalence (% >= 0.5)", fontsize=12)
    plt.title(f"Extracted Pseudo-Label Prevalence across {len(weak_df)} Studies", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
                    
    plt.tight_layout()
    plt.show()

    # Show first 5 extracted records
    print("\nSample Extracted Records:")
    display_cols = ["StudyInstanceUID"] + [f"{col}_prob" for col in TARGET_COLUMNS[:4]]
    print(weak_df[display_cols].head())
else:
    print(f"Output file {OUTPUT_CSV} not found. Run Section 2 first.")
