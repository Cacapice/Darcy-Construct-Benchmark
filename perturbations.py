"""Minimal-pair perturbation operators on the coefficient space.

Coordinate intervention:   T_{j,delta}(theta)   = theta + delta e_j
Subspace intervention:     T_{S,delta v}(theta) = theta + delta v,  ||v||_2 = 1

A *minimal pair* is (theta, T(theta)) with both members admissible
(inside the ball ||theta||_2 <= a when a is finite). If the nominal
delta would exit the admissible region, the sign is flipped first and
the magnitude shrunk only as a last resort; the delta actually used is
recorded so no silent modification occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def is_admissible(theta: np.ndarray, radius_a: Optional[float]) -> bool:
    return radius_a is None or float(np.linalg.norm(theta)) <= radius_a


def coordinate_direction(m: int, j: int) -> np.ndarray:
    e = np.zeros(m)
    e[j] = 1.0
    return e


@dataclass(frozen=True)
class MinimalPair:
    theta: np.ndarray
    theta_pert: np.ndarray
    direction: np.ndarray      # unit vector v
    delta: float               # signed delta actually applied
    label: str


def make_minimal_pair(
    theta: np.ndarray,
    v: np.ndarray,
    delta: float,
    radius_a: Optional[float],
    label: str,
) -> MinimalPair:
    """Build (theta, theta + delta v) with both members admissible."""
    theta = np.asarray(theta, dtype=float)
    v = np.asarray(v, dtype=float)
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        raise ValueError("direction v must be nonzero")
    v = v / nv
    if not is_admissible(theta, radius_a):
        raise ValueError("base theta is not admissible")

    for cand in (delta, -delta):
        if is_admissible(theta + cand * v, radius_a):
            return MinimalPair(theta, theta + cand * v, v, cand, label)

    # Shrink toward the largest admissible magnitude along +v.
    assert radius_a is not None
    lo, hi = 0.0, abs(delta)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if is_admissible(theta + mid * v, radius_a):
            lo = mid
        else:
            hi = mid
    if lo <= 0.0:
        raise ValueError("no admissible perturbation along v")
    return MinimalPair(theta, theta + lo * v, v, lo, label)


def coordinate_pair(
    theta: np.ndarray, j: int, delta: float, radius_a: Optional[float]
) -> MinimalPair:
    v = coordinate_direction(theta.size, j)
    return make_minimal_pair(theta, v, delta, radius_a, label=f"e_{j}")
