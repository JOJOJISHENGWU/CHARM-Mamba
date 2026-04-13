from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def calibrated_routing(
    query: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 0.5,
    delta: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uncertainty-gated cosine routing.

    Args:
        query: [D]
        prototypes: [K, D]
        temperature: softmax temperature
        delta: uncertainty threshold

    Returns:
        weights alpha [K], uncertainty u [K], context c [D]
    """
    if query.ndim != 1:
        raise ValueError("query must be 1D [D].")
    if prototypes.ndim != 2:
        raise ValueError("prototypes must be 2D [K, D].")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    qn = _normalize(query[None, :])[0]
    pn = _normalize(prototypes)

    cos = pn @ qn
    dist = np.linalg.norm(prototypes - query[None, :], axis=1)
    u = 1.0 / (1.0 + np.exp(-(dist - delta)))

    logits = (cos / temperature) * (1.0 - u)
    logits = logits - np.max(logits)
    ex = np.exp(logits)
    alpha = ex / np.sum(ex)

    context = alpha @ prototypes
    return alpha.astype(np.float32), u.astype(np.float32), context.astype(np.float32)
