from __future__ import annotations

import numpy as np


def build_dynamic_congestion_graph(
    speed_t: np.ndarray,
    vmax: float,
    tau_cong: float,
) -> np.ndarray:
    """
    Build dynamic congestion graph consistent with Eq. A_cong^(t).

    Args:
        speed_t: node speed vector at time t, shape [N]
        vmax: maximum speed constant used in congestion intensity
        tau_cong: threshold for sparsification

    Returns:
        adjacency matrix A_cong^(t), shape [N, N]
    """
    if vmax <= 0:
        raise ValueError("vmax must be positive.")

    c_t = 1.0 - (speed_t / vmax)
    c_t = np.clip(c_t, 0.0, 1.0)

    outer = np.outer(c_t, c_t)
    norm = np.linalg.norm(c_t) + 1e-8
    sim = outer / (norm * norm)

    a_cong = np.maximum(sim - tau_cong, 0.0)
    np.fill_diagonal(a_cong, 0.0)
    return a_cong
