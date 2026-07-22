# LSTM-Epsilon Climate Experiment

This repository contains the reproducible code and public documentation for testing whether catchment recession behavior changed around 1990.

The active model follows the physics-informed LSTM-epsilon formulation and infers daily `epsilon` directly inside the recession equation:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

`epsilon` is the model's daily recession coefficient and is interpreted only through this physics-constrained formulation.

The active experiment uses pure GCIN catchments over 1950-2019 and five-fold basin cross-fitting. Each fold uses about 70% of catchments for training, 10% for validation, and 20% for independent testing; all roles retain the full time record so the 1950-1990 versus 1991-2019 contrast is not confounded with a temporal train/test split.

The completed production run evaluates every one of the 2,511 catchments once as held-out test data. Median catchment NSE is 0.327 (p10-p90: -1.045 to 0.651), and pooled NSE is 0.343. Of 2,297 catchments with a valid pre/post epsilon contrast, 566 have NSE > 0.5 in both periods and form the default reliability subset for interpretation.

The public explorer is available at https://grups666.github.io/LSTM_epsilon/. Its Overview panel applies NSE or KGE filtering in the browser; the underlying public data are not hard-filtered at 0.5.

Private daily data, checkpoints, credentials, and raw downloads are excluded from publication. The maintained documentation is:

```text
docs/SUMMARY.md
docs/TECHNICAL_METHODS.md
```
