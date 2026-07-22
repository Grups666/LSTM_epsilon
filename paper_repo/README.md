# LSTM-Epsilon Climate Experiment

This repository contains the reproducible code and public documentation for testing whether catchment recession behavior changed around 1990.

The active model follows the physics-informed LSTM-epsilon formulation and infers daily `epsilon` directly inside the recession equation:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

`epsilon` is the model's daily recession coefficient and is interpreted only through this physics-constrained formulation.

The active experiment uses pure GCIN catchments over 1950-2019 and five-fold temporal cross-fitting. The pre and post periods are each divided into five contiguous blocks. Every fold trains one shared model on four pre blocks and four post blocks, then tests the paired held-out blocks for all catchments. There is no validation set or test-guided checkpoint selection.

Five rotations give every eligible date one out-of-fold streamflow and epsilon estimate. Catchment-level pre/post NSE is used only as an indirect reliability diagnostic for latent epsilon; the scientific result is the within-catchment pre/post epsilon contrast.

The audited run contains 9,192,715 out-of-fold recession-day predictions. Median catchment NSE is 0.581 overall, 0.555 before 1991, and 0.626 after 1990; pooled NSE is 0.577. Of 2,297 catchments with valid epsilon contrasts, 1,304 exceed NSE 0.5 in both periods.

The public explorer is available at https://grups666.github.io/LSTM_epsilon/. Its Overview panel applies NSE or KGE filtering in the browser; the underlying public data are not hard-filtered at 0.5.

Private daily data, checkpoints, credentials, and raw downloads are excluded from publication. The maintained documentation is:

```text
docs/SUMMARY.md
docs/TECHNICAL_METHODS.md
```
