# LSTM Epsilon v0.1.0

Tereon module for catchment-scale epsilon change analysis.

Public site:

`https://grups666.github.io/LSTM_epsilon/`

## Structure

- `public/index.html` - local Tereon shell with the epsilon module loaded by default.
- `public/module.json` - remote module manifest for loading from `https://grups666.github.io/tereon/`.
- `public/modules/epsilon-change/` - module manifest, entry script, and generated data.
- `public/tereon-embed.html` - Hydro-Imbalance-style iframe entry pointing to the published module.
- `gh-pages` branch - publishes `public/` to GitHub Pages.

## Local Preview

```powershell
conda run -n hydro python -m http.server 8766 --directory public
```

Open `http://127.0.0.1:8766/`.

## Data

The module uses cross-fitted daily epsilon inference summarized by catchment for:

- pre period: 1982-1990;
- post period: 1991-2019;
- all-recession, low-flow (`Q_obs <= Q10`), mid-flow (`Q10 < Q_obs < Q90`), and high-flow (`Q_obs >= Q90`) regimes.

The generated data file is:

`public/modules/epsilon-change/data/epsilon-catchment-distributions.json`
