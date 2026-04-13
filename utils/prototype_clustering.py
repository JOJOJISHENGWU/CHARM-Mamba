from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def build_prototypes(
    features: np.ndarray,
    num_prototypes: int,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Offline K-means prototype extraction.

    Args:
        features: shape [M, D]
        num_prototypes: number of prototypes K
        seed: random seed for reproducibility
        n_init: KMeans n_init
        max_iter: KMeans max_iter

    Returns:
        prototype matrix P, shape [K, D]
    """
    if features.ndim != 2:
        raise ValueError("features must be 2D [M, D].")
    if num_prototypes < 1:
        raise ValueError("num_prototypes must be >= 1")

    kmeans = KMeans(
        n_clusters=num_prototypes,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    kmeans.fit(features)
    return kmeans.cluster_centers_.astype(np.float32)
