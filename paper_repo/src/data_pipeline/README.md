# Data Pipeline Scripts

The public data lineage and retained products are documented in:

```text
paper_repo/docs/TECHNICAL_METHODS.md
```

Private working notes are kept outside the release repository under `_private/docs/`.

Current retained private products:

```text
D:/MyPaper/papers/Climate_change_gQ/_private/processed/epsilon_training_daily_parquet/
D:/MyPaper/papers/Climate_change_gQ/_private/processed/epsilon_physics_daily_parquet/
D:/MyPaper/papers/Climate_change_gQ/_private/processed/epsilon_model_inputs/
D:/MyPaper/papers/Climate_change_gQ/_private/processed/qobs_yearly_parquet/
```

Primary scripts used for the current products:

```text
prepare_gee_catchment_asset.py
gee_era5land_export_timeseries.py
gee_drive_exports.py
sync_gee_era5land_exports.py
assemble_gee_era5land_timeseries.py
filter_gee_training_catchments.py
build_epsilon_static_attributes.py
fit_epsilon_lp_gamma.py
```
