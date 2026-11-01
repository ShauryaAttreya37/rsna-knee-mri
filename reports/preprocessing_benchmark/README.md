# Preprocessing Benchmark Reports

Only identifier-free aggregate benchmark results belong in this directory.
The selected study/series mapping, derived image montages, visual-review
checklist, and any raw-image diagnostics must remain outside Git under:

```text
/home/shaurya/rsna-data/manifests/preprocessing_benchmark_v1/
```

Run `scripts/06_preprocessing_benchmark.py` on the VM. Do not mark visual QA as
complete solely because the automated benchmark passes; inspect every
anonymous montage and complete the external checklist first.
