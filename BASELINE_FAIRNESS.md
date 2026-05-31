# Baseline Fairness Protocol

This document records the shared protocol used to ensure fair comparison conditions, consistent with the paper text.

## Shared protocol

- Chronological split ratio: `7:2:1`
- Target few-shot adaptation budget: `3 days`
- Input sequence length: `12`
- Metrics: `MAE`, `RMSE`, `MAPE`
- Reported horizons follow dataset interval:
  - 5-min datasets: `[5, 15, 30]`
  - 10-min datasets: `[10, 30, 60]`
- Same adaptation split/evaluation protocol across methods
- Repeated runs with different seeds; report averaged results (as stated in paper)

## Notes

- This file describes fairness constraints and reporting conventions.
- Model-specific implementation details should follow each baseline's official design while preserving the shared protocol above.
