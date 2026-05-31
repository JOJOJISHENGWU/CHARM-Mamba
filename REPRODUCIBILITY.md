# Reproducibility Protocol (Paper-Aligned)

This repository is aligned to the experimental setting in `sample-sigconf-authordraft.tex`.

## 1. Paper-aligned setting

- Task: multi-source to single-target cross-city transfer
- Main target in this release: `PeMS-BAY`
- Sources: `METR-LA`, `Chengdu`, `Shenzhen` (M, C, S -> PeMS-BAY)
- Temporal split: `train:val:test = 7:2:1` (chronological)
- Few-shot target adaptation budget: `3 days`
- Input length: `12`
- Reported horizons:
  - 5-min datasets: `[5, 15, 30]`
  - 10-min datasets: `[10, 30, 60]`
- Metrics: `MAE`, `RMSE`, `MAPE`

## 2. Single source of truth files

- Target config: `configs/pems_bay.yaml`
- Source configs: `configs/metr_la.yaml`, `configs/chengdu.yaml`, `configs/shenzhen.yaml`
- Artifact manifest: `checkpoints/MANIFEST.json`
- Run summary (current release): `logs/pems_bay_summary.json`

## 3. Artifact validation

Run:

```bash
python scripts/verify_artifacts.py
```

The script checks required files and manifest structure.

## 4. Metric consistency check

Use `utils/reproducibility.py` helper:

- Compare actual metrics against `results/expected_metrics.json`
- Use per-metric tolerance to account for numerical/runtime variation

## 5. Determinism notes

- Default seed in current configs: `42`
- Full bitwise reproducibility across hardware/software stacks is not guaranteed.
- Minor fluctuations are expected; use tolerance-based checks.

## 6. Scope note

This repository currently provides paper-aligned configuration and artifact metadata for reproducibility review and verification workflows.
