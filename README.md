# CHARM-Mamba

**CHARM-Mamba: Calibrated Hierarchical Adaptive Routing Multi-Source Mamba for Cross-City Traffic Flow Prediction**

This repository follows the paper setting in `sample-sigconf-authordraft.tex`, with a focus on consistency between paper claims, configuration, and released artifacts.

## Overview

CHARM-Mamba contains three components:

1. **DC-Mamba Backbone** for coupled temporal-spatial representation learning.
2. **CHPR Routing** for calibrated hierarchical source-pattern selection.
3. **HEA Adaptation** for parameter-efficient target adaptation with frozen backbone.

## Paper-Aligned Experimental Setting

### Multi-source transfer setting

For the PeMS-BAY target experiment, the official source set is:

- **METR-LA**
- **Chengdu**
- **Shenzhen**

This corresponds to the paper setting: **M, C, S \(\rightarrow\) PeMS-BAY**.

### Data split protocol

All dataset configurations in this release follow chronological split:

- **train : val : test = 7 : 2 : 1**

Few-shot adaptation uses **3 days** of target-domain training data, consistent with the paper description.

## Dataset Statistics (paper-aligned)

| Dataset | Nodes | Interval | #Timestamps |
|---|---:|---|---:|
| PeMS-BAY | 325 | 5 min | 52116 |
| METR-LA | 207 | 5 min | 34272 |
| Chengdu | 524 | 10 min | 17280 |
| Shenzhen | 627 | 10 min | 17280 |

## Repository Structure

- `sample-sigconf-authordraft.tex`: main paper source
- `model/charm_mamba.py`: model structure stub
- `configs/*.yaml`: experiment configurations
- `utils/`: routing, graph, metrics, reproducibility helpers
- `logs/`: run summaries
- `checkpoints/MANIFEST.json`: artifact manifest
- `scripts/verify_artifacts.py`: artifact integrity checks

## Notes on Current Release

- This release prioritizes **paper/config/artifact consistency**.
- Core utilities for CHPR routing and dynamic congestion graph are included in `utils/`.
- Please use configuration files as the single source of truth for experimental protocol.

## Minimal Verification

You can verify artifact structure with:

```bash
python scripts/verify_artifacts.py
```

If files required by the verifier are missing, add them before claiming full reproducibility packaging.
