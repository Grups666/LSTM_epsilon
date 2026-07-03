# Epsilon Change Analysis Summary

Version: `v0.1.0`

This repository publishes the Tereon-based catchment epsilon explorer and the reproducible analysis materials for the current 1950-2019 pure GCIN experiment.

Primary documentation:

- [Reader-facing summary](../paper_repo/docs/SUMMARY.md)
- [Technical methods](../paper_repo/docs/TECHNICAL_METHODS.md)

Public explorer data:

```text
public/modules/epsilon-change/data/epsilon-catchment-distributions.json
```

Current production scope:

```text
pre-change:  1950-1990
post-change: 1991-2019
model-ready catchments: 2,511
explorer catchments with valid epsilon contrasts: 2,297
regimes:     all recession days, low flow, high flow
```

Five-fold physics-informed LSTM-epsilon training and cross-fitted inference are complete.

```text
median catchment NSE: 0.466
median catchment KGE: 0.663
pooled NSE:           0.574
pooled KGE:           0.707
mean delta epsilon:   0.080
median delta epsilon: 0.019
```

Legacy/GridCode/Catchment_ID mixed products are excluded from this production run because their numeric identifiers and catchment boundaries were not reliably matched.

Private processing outputs, model checkpoints, credentials, meeting notes, and internal logs are excluded from this repository.
