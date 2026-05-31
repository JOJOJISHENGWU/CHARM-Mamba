# Logs Description

Current release includes:

- `pems_bay_summary.json`: paper-aligned summary for the PeMS-BAY target setting.

## `pems_bay_summary.json` fields

- `run_id`: run identifier
- `dataset`: target dataset name
- `setting`: transfer setting
- `sources`: source domains used in this run
- `target`: target domain
- `few_shot_days`: target adaptation budget (days)
- `best_epoch`: epoch selected by validation
- `best_val_15min`: validation metrics at 15-min horizon
- `test`: test metrics across horizons
- `checkpoint`: path to selected checkpoint

## Consistency requirement

`logs/*.json`, `configs/*.yaml`, and `checkpoints/MANIFEST.json` should agree on:

- source domains
- transfer setting
- target domain
- few-shot budget
