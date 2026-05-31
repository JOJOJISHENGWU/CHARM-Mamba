# CHARM-Mamba

**CHARM-Mamba: Calibrated Hierarchical Adaptive Routing Multi-Source Mamba for Cross-City Traffic Prediction**

> A parameter-efficient framework integrated with Deep Coupled Mamba backbone, Calibrated Hierarchical Adaptive Routing, and Hypernetwork-based Efficient Adaptation.

## 🎯 Overview

**CHARM-Mamba** is a Calibrated Hierarchical Adaptive Routing Multi-source framework designed to address three structural limitations in existing cross-city transfer learning: decoupled spatio-temporal extraction, negative transfer from indiscriminate aggregation, and prohibitive fine-tuning costs.

By synergizing a **Deep Coupled Mamba** backbone where State Space Models and dynamic spatial graphs mutually modulate, the model learns joint representations. It further leverages a **Calibrated Hierarchical Prototype Routing (CHPR)** mechanism to selectively retrieve compatible source prototypes via uncertainty-based calibration, prioritizing transferable knowledge while discarding noise. Finally, a **Hypernetwork-based Efficient Adaptation (HEA)** mechanism dynamically generates target-specific adapter parameters, facilitating pattern-aware adaptation with the backbone frozen.

## ✨ Key Features

- **🐍 Deep Coupled Mamba Backbone**: Simultaneously models entangled traffic dynamics and congestion propagation across heterogeneous domains, surpassing decoupled GNN-RNN architectures.
- **🛣️ Calibrated Hierarchical Adaptive Routing**: Mitigates negative transfer by dynamically retrieving context-specific prototypes based on normalized feature spaces and uncertainty gates.
- **⚡ Hypernetwork-based Efficient Adaptation**: Achieves parameter-efficient transfer (<5% trainable parameters) by synthesizing instance-specific adapter weights.
- **🌐 Robust Few-Shot Generalization**: Significantly outperforms baselines in data-scarce scenarios (e.g., 1-3 days of target data) by effectively filtering irrelevant source noise.

## 🏗️ Architecture

### Core Components

1.  **Deep Coupled Mamba Backbone**
    - **Joint Modeling**: Mutually modulates State Space Models (SSM) and dynamic spatial graphs to capture topology-dependent congestion propagation.
    - **Linear Complexity**: Maintains Mamba's efficiency $O(T)$ for processing long historical sequences.

2.  **Calibrated Hierarchical Prototype Routing (CHPR)**
    - **Prototype Retrieval**: Constructs pattern prototypes at multiple granularities.
    - **Uncertainty Calibration**: Uses distance-based uncertainty gates to prioritize structurally compatible knowledge (e.g., Highway vs. Urban).
    - **Negative Transfer Mitigation**: Actively suppresses irrelevant source domains.

3.  **Hypernetwork-based Efficient Adaptation (HEA)**
    - **Dynamic Weight Generation**: A lightweight hypernetwork generates target-specific adapter parameters driven by routed patterns.
    - **Frozen Backbone Adaptation**: Facilitates rapid adaptation to target cities without determining global parameters, ensuring scalability.

## 📊 Supported Datasets

The framework is evaluated on standard cross-city benchmarks:

| Dataset | Type | Sensors | Time Steps | Description |
|---------|------|---------|------------|-------------|
| **METR-LA** | Source | 207 | 34,272 | Highway traffic speed in Los Angeles |
| **PEMS-BAY** | Target | 325 | 52,116 | Highway traffic speed in Bay Area |
| **Shenzhen** | Source | 627 | 17,280 | Urban traffic speed in Shenzhen |
| **Chengdu** | Source | 524 | 17,280 | Urban traffic speed in Chengdu |

### Data Format
- **Tensor Input**: $\mathbf{X} \in \mathbb{R}^{B \times T \times N \times C}$
- **Graph Inputs**: Dynamic congestion graphs and static road network graphs.

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.9
PyTorch >= 1.12
Mamba-ssm >= 1.0.1
causal-conv1d >= 1.2.0
```

### Installation

```bash
# Clone repository

cd CHARM-Mamba

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from model.charm_mamba import CharmMamba

# 1. Initialize model (paper-aligned skeleton)
model = CharmMamba(
    num_nodes=325,
    input_dim=1,
    output_dim=1,
    d_model=128,
    num_layers=3,
    num_sources=3,
    num_prototypes=10,
    pred_len=12,
    d_neck=32,
)

# 2. Build minimal inputs
# x shape: [B, T, N, C]
B, T, N, C = 4, 12, 325, 1
x = torch.randn(B, T, N, C)
a_geo = torch.eye(N)
a_func = torch.eye(N)
prototypes = torch.randn(3, 10, 128)  # [num_sources, num_prototypes, d_model]

# 3. Forward pass
output = model(x, a_geo=a_geo, a_func=a_func, prototypes=prototypes)
```

## ⚙️ Configuration

### Model Parameters

```python
MODEL_CONFIG = {
    'd_model': 128,              # Embedding dimension
    'd_state': 16,              # SSM state dimension
    'n_layers': 3,              # Backbone depth
    'dropout': 0.1,             # Dropout rate
    'n_prototypes': 20,         # Prototype library size
    'adapter_rank': 8,          # HEA adapter rank
}
```
