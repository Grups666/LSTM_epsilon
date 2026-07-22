# LSTM Epsilon v0.1.0

Tereon module for catchment-scale epsilon change analysis.

Public site:

`https://grups666.github.io/LSTM_epsilon/`

## Structure

- `public/index.html` - local Tereon shell with the epsilon module loaded by default.
- `public/module.json` - remote module manifest for loading from `https://grups666.github.io/tereon/`.
- `public/modules/epsilon-change/` - module manifest, entry script, and generated data.
- `public/tereon-embed.html` - Hydro-Imbalance-style iframe entry pointing to the published module.
- `paper_repo/` - reproducible analysis code, configuration, and Markdown documentation.
- `paper_repo/docs/SUMMARY.md` - reader-facing result summary.
- `paper_repo/docs/TECHNICAL_METHODS.md` - technical method and run details.
- `gh-pages` branch - publishes `public/` to GitHub Pages.

## Local Preview

```powershell
conda run -n hydro python -m http.server 8766 --directory public
```

Open `http://127.0.0.1:8766/`.

## Data

The module uses the pure GCIN paired temporal cross-fit. It summarizes out-of-time daily epsilon inference by catchment for:

- pre period: 1950-1990;
- post period: 1991-2019;
- all-recession, low-flow (`Q_obs <= Q10`), and high-flow (`Q_obs >= Q90`) regimes.

Each of five rotations holds out one contiguous pre-1990 block and one contiguous post-1990 block for all catchments. The other four blocks from each period train one shared model. There is no validation set or test-guided checkpoint selection.

The generated data file is:

`public/modules/epsilon-change/data/epsilon-catchment-distributions.json`

The training dataset contains 2,511 pure GCIN catchments. Five-fold out-of-fold predictions are concatenated before calculating each catchment's pre-period and post-period NSE/KGE and epsilon contrast.

The explorer JSON retains all 2,297 catchments with a valid epsilon contrast. The Overview panel applies NSE or KGE filtering in the browser; `NSE > 0.5` in both periods is only the default display filter, not a hard-coded export filter.

Audited out-of-fold performance:

- median catchment NSE: `0.581` (p10-p90: `0.128` to `0.775`)
- pooled NSE: `0.577`
- median catchment KGE: `0.642`
- pre/post median NSE: `0.555 / 0.626`
- pre/post NSE above `0.5` in both periods: `1,304` catchments
