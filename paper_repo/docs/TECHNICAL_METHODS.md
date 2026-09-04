# Technical Methods

## Active Experiment

The maintained production workflow is the pure-GCIN paired temporal cross-fit:

```text
run label: temporal_crossfit_1990
config:    paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml
status:    completed and audited
```

The scientific target is the within-catchment change in inferred recession behavior around 1990. The split therefore tests unseen time blocks in the same gauged catchments. It is not an ungauged-basin regionalization experiment.

## Data Contract

The model-ready dataset contains 2,511 GCIN catchments and daily records from 1950 through 2019.

```text
identifier:   GCIN
pre period:   1950-01-01 to 1990-12-31
post period:  1991-01-01 to 2019-12-31
record type:  catchment-day time series
```

Private inputs are restricted to:

```text
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/
```

Observed streamflow and boundary identity use the original GCIN identifier. Private daily data and checkpoints are not published.

The archived run used a thickness-weighted `swvl1-swvl4` model input but an equal-weight `swvl1-swvl3` LP/gamma fit. The locked rerun contract uses the same 0-100 cm root-zone definition in both paths: `SM_RZ = 0.07*swvl1 + 0.21*swvl2 + 0.72*swvl3`. A shared source implementation prevents the two paths from diverging again. The fourth ERA5-Land layer (100-289 cm) is excluded from the primary AET limitation variable.

## Temporal Cross-Fitting

The pre and post periods are independently divided into five contiguous calendar blocks. Fold `k` holds out pre block `k` and post block `k` together.

| Fold | Pre test block | Post test block |
|---:|:---|:---|
| 0 | 1950-1958 | 1991-1996 |
| 1 | 1959-1966 | 1997-2002 |
| 2 | 1967-1974 | 2003-2008 |
| 3 | 1975-1982 | 2009-2014 |
| 4 | 1983-1990 | 2015-2019 |

Every fold uses all 2,511 catchments. Approximately 80% of calendar years train the model and 20% are tested. There is no validation set.

Each training example contains 365 warm-up days followed by a 365-day target sequence. A training example is eligible only if its complete 730-day span belongs to training years. Consequently, no held-out test date can appear in a training target or in a training example's model-input context.

Dynamic normalization statistics are estimated from training years only. The reference workflow's Q-derived `low_high_ratio` static attribute is recomputed from each fold's training years before static normalization. Test-period Q therefore cannot enter model fitting through that attribute.

## Recession Selection

Training and inference retain the reference-compatible recession rules:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
remove temperature_C <= 0 deg C: true
```

Flow regimes are calculated within each catchment from observed out-of-fold recession-day streamflow:

```text
low flow:  Qobs <= catchment Q10
mid flow:  catchment Q10 < Qobs < catchment Q90
high flow: Qobs >= catchment Q90
```

## Model

The model follows the reference physics-informed LSTM-epsilon formulation and directly infers daily `epsilon` inside:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The network receives dynamic forcing and state variables plus static catchment attributes. It predicts epsilon, reset flow, alpha, LP, and gamma. The reference state-reset integration, physical bounds, and four-term optimization objective are retained.

```text
L = 25.0 * L_path
  + 10.0 * L_rhs
  +  0.1 * L_smooth
  +  5.0 * L_q0
```

### Reference-implementation audit

The retained physics core was checked against `arabayati/LSTM-epsilon` commit
`c38830895002284fe47e1679ad956f0f15d704f7`. The comparison covers the model,
loss, and production training settings.

Retained without changing the physical formulation:

- one-layer LSTM with dynamic `epsilon_t` and `q_base_t` heads;
- bounded static `alpha`, `LP`, and `gamma` heads;
- in-model AET calculation and the AET <= PET constraint;
- recession-start state reset and piecewise closed-form daily Q update;
- ten component recession paths collapsed by their mean;
- log-Q Huber path loss, observed-Q RHS loss, epsilon smoothness loss, and Q0 anchor loss;
- the reference production weights `25`, `10`, `0.1`, and `5`, Adam learning rate `1e-4`, dropout `0.4`, hidden size `256`, and gradient clipping at `1.0`.

An identical-weight numerical parity test returned zero maximum absolute
difference for `q_hat`, component Q, `q_base`, epsilon, AET, alpha, LP, gamma,
and all five reported loss values.

Study-specific adaptations are the pure-GCIN inputs, the cold-day recession
mask, paired pre/post temporal cross-fitting, batch size `512`, and a fixed
30-epoch schedule. These adaptations change the data and evaluation protocol,
not the governing equation, state update, or physics-loss construction.

## Training

```text
epochs:              30 fixed
batch size:          512
learning rate:       1e-4
hidden size:         256
LSTM layers:         1
dropout:             0.4
mixture components:  10
target sequence:     365 days
warm-up context:     365 days
checkpoint:          final epoch
```

There is no early stopping or test-guided checkpoint selection. The epoch count and all architecture and loss settings are fixed before test inference. Reducing the reference maximum from 150 to 30 epochs is a fixed computational setting, not a fold-specific choice.

Each completed fold immediately reports held-out catchment NSE for progress monitoring. These reports are evaluation-only and are never used to change the fixed epoch count, model, or hyperparameters.

## Out-of-Fold Inference

Inference uses the same 365-day context plus at most 365 target days used in training. Each held-out pre/post block is tiled into non-overlapping target windows; every eligible recession day is emitted once by the model that did not train on its temporal fold.

The five fold outputs are concatenated by `GCIN` and date. For every catchment, pre-period and post-period NSE are calculated once from the concatenated out-of-fold streamflow predictions. NSE is not calculated as the arithmetic mean of five fold NSE values because each block has a different observed-flow variance. Fold-level NSE mean and standard deviation are retained as supplementary stability diagnostics.

NSE and KGE assess the physics-constrained streamflow reconstruction. They are indirect reliability diagnostics for latent epsilon, not observations of epsilon itself. The scientific contrast is:

```text
delta epsilon = mean out-of-fold epsilon in 1991-2019
              - mean out-of-fold epsilon in 1950-1990
```

The public explorer retains all catchments with valid contrasts and applies the user-selected NSE or KGE threshold in the browser. Its default reliability rule is that both pre-period and post-period NSE exceed 0.5.

## GQ / Q Component Attribution

The out-of-fold recession-day table stores `epsilon_effective`, simulated Q,
and observed Q on the same catchment dates. Effective daily GQ is reconstructed
without retraining:

```text
GQ_effective(t) = epsilon_effective(t) * Qsim(t)
```

Observed Q defines each catchment's low- and high-flow regimes; simulated Q is
used in the GQ reconstruction specified by the study attribution protocol. The
pre/post decomposition uses period geometric means, equivalently period means
of daily logs, so that it closes numerically:

```text
delta log epsilon = delta log GQ - delta log Qsim
```

For the descriptive attribution snapshot, GQ and `-Q` contributions with the
same sign are labelled GQ-dominant, Q-dominant, or combined using two-thirds
and one-third absolute-contribution boundaries. Opposing contributions are
labelled offsetting. This is an algebraic component decomposition, not a causal
attribution to climate forcing and not yet a statistical significance test.

## Primary Era-Shift Inference

The primary inferential unit is the catchment-year-flow-regime. Out-of-fold
daily epsilon is reduced to an annual median when at least three eligible
recession days exist. For each catchment and regime, the fitted model is:

```text
log(annual epsilon) = fold fixed effect + beta_post * I(year >= 1991) + error
era shift (%) = 100 * (exp(beta_post) - 1)
```

The fold intercepts absorb scale differences among the five independently
trained OOF models. Consequently, `beta_post` is identified by pre/post years
in folds that contain both eras, rather than by comparing unrelated model
scales. Eligibility is fixed before testing:

```text
at least 3 recession days per annual median
at least 10 valid annual medians in each era
at least 5 pre and 5 post annual medians inside paired folds
```

Coefficient uncertainty uses a one-year Newey-West HAC covariance. Two-sided
p-values are converted to Benjamini-Hochberg q-values across all statistically
eligible catchments separately for each flow regime. A positive or negative
coefficient is labelled Increase or Decrease only when `q < 0.05`; all other
estimable coefficients are labelled Unresolved. Unresolved is not evidence of
stability. A Stable class would require a separately justified equivalence
margin and is therefore not used.

All-recession, low-flow, and high-flow inference uses independent eligible samples. The
low/high intersection is a coverage diagnostic and does not restrict either
single-regime result. The public NSE/KGE control is a reliability display
filter; it does not recalculate q-values or alter the fixed FDR family.

## Global Field-Level Inference

Per-catchment FDR inference and the global field analysis answer different questions. Local FDR controls false discoveries among catchment labels reported in the inspector. The field analysis evaluates whether the distribution of catchment effects contains a spatially reproducible post-1990 pattern even when many individual coefficients are imprecise.

The maintained field protocol is `global_story_v2`:

```text
config:                    paper_repo/configs/global_story_analysis_v2.yaml
primary reliability:      NSE > 0.5 in both eras
annual support:            >= 5 recession days and >= 10 years per era
spatial assignment:        deterministic 10-degree blocks
discovery / confirmation:  40% / 60% of spatial blocks
confirmation bootstrap:    2,000 spatial-block resamples
candidate correction:      Holm family-wise error control
```

For each catchment-year, all-recession epsilon is summarized by `q25`, `q50`, `q75`, and `spread = q75 / q25`. Each statistic receives the same fold-adjusted log-era model used for the primary catchment analysis. Catchment coefficients and standard errors are combined with a REML random-effects estimator. Spatial-block bootstrap intervals replace an independence-based standard error.

The discovery sample selected candidates without reading confirmation outcomes. The preregistered median statistic entered confirmation regardless of discovery p-value; additional statistics and hydroclimate predictors required discovery p below 0.10, at least 150 catchments, and at least 10 occupied spatial blocks. Ten locked candidates were tested once in the confirmation sample. A candidate required the same sign, a spatial interval excluding zero, and Holm-adjusted p below 0.05.

The field result that reproduced most consistently was annual epsilon distribution spread:

```text
discovery:    +5.17% (95% spatial CI +1.33% to +10.27%; n = 461)
confirmation: +17.62% (+3.66% to +21.33%; Holm p = 0.012; n = 719)
full sample:  +11.67% (+2.75% to +18.95%; descriptive; n = 1,180)
```

The preregistered annual median also confirmed at +16.17% (+4.48% to +18.88%; Holm p = 0.010), but discovery was only +1.41% with an interval crossing zero. It is therefore secondary to the replicated spread result. Full-sample leave-one-20-degree-block estimates were positive for all 30 omitted blocks for both median and spread.

## Hydroclimate Association Model

Climate-era predictors are computed from catchment-year forcing means only when both eras contain at least 10 years. Difference is used for temperature and root-zone soil moisture; positive-valued precipitation and PET use log post/pre ratios. Predictor scale is fixed from discovery data before confirmation.

The primary association model regresses the catchment log-era epsilon coefficient on one standardized climate-era predictor using random-effects precision weights. Confirmation uncertainty uses the same spatial bootstrap and Holm family. Post-confirmation validation adds 10-degree spatial-block fixed effects and enters precipitation and soil-moisture changes jointly.

Only soil moisture retained independent interval evidence:

```text
all recession, univariate confirmation:  -8.10% per discovery SD
  95% spatial CI:                        -12.55% to -6.12%
  Holm p:                                0.010
all recession, joint + block FE:         -4.09%
  95% block-bootstrap CI:                -9.26% to -1.66%
high flow, joint + block FE:             -3.13%
  95% block-bootstrap CI:                -6.46% to -0.64%
```

The precipitation and soil-moisture changes correlate at `r = 0.57`. Precipitation loses interval evidence in the joint model and is not reported as an independent driver. Soil-moisture coefficients retain their negative sign under every leave-one-20-degree-block analysis. These regressions are associative, not causal attribution tests.

The public explorer exposes the inferential hierarchy directly. **Epsilon Change** maps the primary fold-adjusted annual-median era coefficient for the selected All/Low/High condition and reports its local BH-FDR evidence in the inspector. **GQ / Q Decomposition** and **Temporal Robustness** are separate modules because they answer interpretation and sensitivity questions rather than redefining the primary effect. The 1,180 eligible all-recession spread coefficients remain a separate field-level analysis reported in Overview; they are not used as the default catchment map or as a replacement for local FDR inference.

The original `global_story_v1` audit recorded the 10-year climate-support rule but failed to apply it while constructing climate predictors. This was detected after confirmation. `global_story_v2` enforces the configured rule and preserves a protocol-deviation record. The corrected confirmation table is byte-identical to v1 because the affected short-record catchments did not enter the locked confirmation tests.

## Sensitivity Analyses

Annual-support sensitivity checks repeat the era model with one and five days
per annual median around the three-day primary rule. Breakpoint sensitivity
uses 1985 and 1995 in addition to the fixed 1990 primary breakpoint. These
results are summarized without choosing the scenario that produces the most
significant result.

Continuous trends are retained as a secondary check. All-recession, low-flow,
and high-flow epsilon, GQ, and simulated Q are reduced to annual medians with at
least five recession days and at least 20 annual values. Fold-centered log
series receive a Theil-Sen slope; trend-free prewhitening precedes Kendall's
test and BH-FDR correction. This asks whether change is monotonic through time
and never determines the primary pre/post map class.

## Audited Results

```text
OOF recession days:       9,192,715
catchments evaluated:     2,511
valid epsilon contrasts:  2,297
median catchment NSE:     0.581 (p10-p90: 0.128 to 0.775)
pooled NSE:               0.577
median catchment KGE:     0.642
pre median NSE / KGE:     0.555 / 0.622
post median NSE / KGE:    0.626 / 0.666
both-period NSE > 0.5:    1,304
both-period KGE > 0.5:    1,447
NSE > 0.5 low eligible:   791 (139 increase, 19 decrease, 633 unresolved)
NSE > 0.5 high eligible:  1,007 (13 increase, 2 decrease, 992 unresolved)
NSE > 0.5 low/high overlap: 745
```

Among the 1,304 catchments passing NSE 0.5 in both periods, mean delta epsilon is -0.0192 (catchment-bootstrap 95% CI -0.0348 to -0.00521), while median delta epsilon is +0.00230 (95% CI +0.00151 to +0.00324). The opposite signs show a strongly skewed change distribution: negative outliers move the mean, whereas the typical catchment has a small positive shift. Both summaries must be reported.

## Outputs

```text
_private/results/epsilon_pure_gcin_1950_2019/temporal_crossfit_1990/
  fold_<k>/
    final_model.pt
    metrics.csv
    run_metadata.json
    heldout_epsilon_change_summary.parquet
    heldout_skill_summary.csv
    recession_day_simulations.parquet
  analysis/
    temporal_crossfit_epsilon_change_summary.parquet
    temporal_fold_catchment_skill.csv
    epsilon_change_inference.csv
    gq_q_attribution_by_catchment.csv
    gq_q_attribution_by_catchment.parquet
    prepost_shifts_long.csv
    prepost_shifts_long.parquet
    prepost_shifts_by_catchment.csv
    prepost_shifts_by_catchment.parquet
    prepost_shift_sensitivity_summary.csv
    annual_epsilon_gq_q_by_regime.parquet
    continuous_trends_long.csv
    continuous_trends_long.parquet
    continuous_trends_by_catchment.csv
    continuous_trends_by_catchment.parquet
  production_audit.csv

_private/audits/global_story_v2/
  global_story_dataset.parquet
  locked_confirmation_candidates.json
  confirmation_tests.csv
  sensitivity_tests.csv
  robust_story_candidates.csv
  validation_distribution_tests.csv
  validation_leave_one_block_distribution.csv
  validation_leave_one_block_climate.csv
  global_story_validation.json
```

## Reproduction

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\prepare_experiment_inputs.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml

powershell -ExecutionPolicy Bypass -File paper_repo\scripts\run_crossfit_training.ps1 `
  -RunLabel temporal_crossfit_1990

conda run -n hydro python paper_repo\src\epsilon_model\run_full_postprocess.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml `
  --run-label temporal_crossfit_1990 `
  --figures-dir paper_repo\docs\assets\epsilon_pure_gcin_1950_2019

conda run -n hydro python paper_repo\src\epsilon_model\analyze_global_story.py `
  --analysis-config paper_repo\configs\global_story_analysis_v2.yaml --phase all

conda run -n hydro python paper_repo\src\epsilon_model\analyze_global_story_sensitivity.py `
  --analysis-config paper_repo\configs\global_story_analysis_v2.yaml

conda run -n hydro python paper_repo\src\epsilon_model\validate_global_story.py `
  --analysis-config paper_repo\configs\global_story_analysis_v2.yaml
```

Field-level inference follows the logic of spatial block resampling and guarded multiple testing; see Lahiri and Zhu (2006), [doi:10.1214/009053606000000551](https://doi.org/10.1214/009053606000000551), and Wilks (2016), [doi:10.1175/BAMS-D-15-00267.1](https://doi.org/10.1175/BAMS-D-15-00267.1).
