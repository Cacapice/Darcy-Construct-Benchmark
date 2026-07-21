"""Observation operator O and the exact parameter-to-observation map G_m.

G_m : Theta_m -> R^{n_obs},   G_m(theta) = O( p(theta) )   (noiseless).

Sensors are defined in *physical coordinates* (an interior uniform grid),
then mapped to the nearest mesh node. This keeps the observation operator
comparable when mesh resolution changes as a nuisance variable.
"""

from __future__ import annotations

import numpy as np

from .config import NuisanceConfig, ObservationConfig
from .kl import KLBasis
from .solver import DarcySolver


class Observation:
    def __init__(self, cfg: ObservationConfig, n: int):
        self.cfg = cfg
        ns = cfg.n_sensors_per_side
        coords = (np.arange(1, ns + 1)) / (ns + 1)          # strictly interior
        SX, SY = np.meshgrid(coords, coords, indexing="ij")
        self.sensor_xy = np.column_stack([SX.ravel(), SY.ravel()])
        idx = np.rint(self.sensor_xy * (n - 1)).astype(int)
        self.idx_i, self.idx_j = idx[:, 0], idx[:, 1]
        self.n_obs = self.idx_i.size

    def observe(self, p: np.ndarray) -> np.ndarray:
        return p[self.idx_i, self.idx_j]

    def add_noise(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return y + self.cfg.noise_sigma * rng.standard_normal(y.shape)


class ForwardMap:
    """Exact discretized parameter-to-observation map for one nuisance setting."""

    def __init__(self, basis: KLBasis, nuisance: NuisanceConfig):
        self.basis = basis
        self.nuisance = nuisance
        self.solver = DarcySolver(nuisance.solver)
        self.observation = Observation(nuisance.observation, nuisance.solver.n)
        self.n_obs = self.observation.n_obs
        self.noise_sigma = nuisance.observation.noise_sigma

    @property
    def design(self) -> NuisanceConfig:
        """The experiment E this map's identifiability statements are indexed by."""
        return self.nuisance

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        u = self.basis.log_conductivity(theta, self.solver.n)
        p = self.solver.solve(u)
        return self.observation.observe(p)
