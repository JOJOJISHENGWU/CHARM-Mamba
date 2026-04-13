from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    denom = np.maximum(np.abs(target), eps)
    return float(np.mean(np.abs((pred - target) / denom)) * 100.0)


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "mape": mape(pred, target),
    }
