# Validation Fold Reports

Only aggregate, identifier-free fold statistics belong in this directory.
The row-level `StudyInstanceUID` assignment must remain outside Git under:

```text
/home/shaurya/rsna-data/manifests/gold_folds_v1/gold_folds.csv
```

`scripts/07_gold_folds.py` creates deterministic multilabel folds and refuses
assignments that leave either class absent for any target/fold combination.
Study-level isolation is enforced, but patient-level independence cannot be
proven because the supplied schema has no stable patient identifier.
