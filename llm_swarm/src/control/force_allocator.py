"""Allocate desired object wrench to per-robot planar forces."""

from __future__ import annotations

import numpy as np


def allocate_wrench_to_robots(
    rel_points: np.ndarray,
    desired_fx: float,
    desired_fy: float,
    desired_tau: float,
    force_max: float,
    ridge_lambda: float = 1e-3,
) -> np.ndarray:
    """Solve least-squares force allocation with per-robot norm clipping.

    Args:
        rel_points: Relative attach points of shape [n, 2] in object frame.
        desired_fx: Desired net force x.
        desired_fy: Desired net force y.
        desired_tau: Desired net torque around object center.
        force_max: Max norm per robot force.
        ridge_lambda: Tikhonov regularization.

    Returns:
        Per-robot force array with shape [n, 2].
    """
    n = int(rel_points.shape[0])
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # A @ f = w, where f = [f1x, f1y, ..., fnx, fny].
    a = np.zeros((3, 2 * n), dtype=np.float64)
    for i in range(n):
        rx, ry = float(rel_points[i, 0]), float(rel_points[i, 1])
        a[0, 2 * i] = 1.0
        a[1, 2 * i + 1] = 1.0
        a[2, 2 * i] = -ry
        a[2, 2 * i + 1] = rx

    w = np.array([desired_fx, desired_fy, desired_tau], dtype=np.float64)

    # Ridge least squares: [A; sqrt(lam)I] f ~= [w; 0]
    if ridge_lambda > 0.0:
        reg = np.sqrt(ridge_lambda) * np.eye(2 * n, dtype=np.float64)
        a_aug = np.vstack([a, reg])
        w_aug = np.concatenate([w, np.zeros(2 * n, dtype=np.float64)])
        f_vec, *_ = np.linalg.lstsq(a_aug, w_aug, rcond=None)
    else:
        f_vec, *_ = np.linalg.lstsq(a, w, rcond=None)

    f = f_vec.reshape(n, 2)

    # Per-robot saturation.
    norms = np.linalg.norm(f, axis=1)
    for i in range(n):
        if norms[i] > force_max and norms[i] > 1e-8:
            f[i] = f[i] * (force_max / norms[i])

    return f.astype(np.float32)
