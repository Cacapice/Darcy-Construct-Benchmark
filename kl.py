"""Karhunen-Loeve parameterization of the log-conductivity field.

The target parameter theta is defined *once*, on a fixed reference grid:
discrete eigenpairs (lambda_j, phi_j) of the Matern covariance operator
(nodal quadrature). Under nuisance changes of mesh, the modes are
interpolated to the new grid, so theta_j always refers to the same
spatial mode. This is what keeps the target parameter's meaning invariant
while nuisance variables move.

Truncation level m is the smallest m with
    sum_{j<=m} lambda_j / sum_j lambda_j >= rho,
capped at m_max. The achieved proportion is recorded; if it falls short
of the preregistered rho, the basis reports that honestly rather than
silently proceeding.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import gamma as gamma_fn
from scipy.special import kv

from .config import PriorConfig


def matern_kernel(r: np.ndarray, nu: float, ell: float, variance: float) -> np.ndarray:
    """Matern covariance as a function of distance r >= 0."""
    r = np.asarray(r, dtype=float)
    if nu == 0.5:
        return variance * np.exp(-r / ell)
    if nu == 1.5:
        s = np.sqrt(3.0) * r / ell
        return variance * (1.0 + s) * np.exp(-s)
    if nu == 2.5:
        s = np.sqrt(5.0) * r / ell
        return variance * (1.0 + s + s * s / 3.0) * np.exp(-s)
    # General nu via modified Bessel function; guard r=0.
    s = np.sqrt(2.0 * nu) * np.maximum(r, 1.0e-12) / ell
    out = variance * (2.0 ** (1.0 - nu) / gamma_fn(nu)) * (s ** nu) * kv(nu, s)
    return np.where(r <= 0.0, variance, out)


class KLBasis:
    """Discrete KL basis on the unit square, defined on a reference grid."""

    def __init__(self, prior: PriorConfig):
        self.prior = prior
        n = prior.n_ref
        xs = np.linspace(0.0, 1.0, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel()])

        C = matern_kernel(cdist(pts, pts), prior.nu, prior.length_scale, prior.variance)
        w = (1.0 / (n - 1)) ** 2  # uniform nodal quadrature weight (documented approximation)

        eigval, eigvec = eigh(C)
        eigval = eigval[::-1] * w
        eigvec = eigvec[:, ::-1]
        pos = eigval > 1.0e-12
        eigval, eigvec = eigval[pos], eigvec[:, pos]

        total = float(eigval.sum())
        cum = np.cumsum(eigval) / total
        m_rho = int(np.searchsorted(cum, prior.rho) + 1)
        self.m = min(m_rho, prior.m_max, eigval.size)
        self.achieved_rho = float(cum[self.m - 1])
        self.requested_rho_met = self.achieved_rho >= prior.rho

        self.lam = eigval[: self.m]                       # (m,)
        # Normalize so sum_i phi_j(x_i)^2 * w = 1.
        self.phi_ref = (eigvec[:, : self.m] / np.sqrt(w)).reshape(n, n, self.m)
        self.xs_ref = xs
        self._interp = RegularGridInterpolator(
            (xs, xs), self.phi_ref, bounds_error=False, fill_value=None
        )

    # ------------------------------------------------------------------ fields

    def modes_on_grid(self, n: int) -> np.ndarray:
        """Evaluate all m modes on an n x n nodal grid (interpolated). -> (n, n, m)"""
        if n == self.prior.n_ref:
            return self.phi_ref
        xs = np.linspace(0.0, 1.0, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel()])
        return self._interp(pts).reshape(n, n, self.m)

    def log_conductivity(self, theta: np.ndarray, n: int) -> np.ndarray:
        """u_m(x; theta) = ubar + sum_j sqrt(lambda_j) theta_j phi_j(x) on an n x n grid."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.m,):
            raise ValueError(f"theta must have shape ({self.m},), got {theta.shape}")
        modes = self.modes_on_grid(n)
        u = self.prior.mean_value + modes @ (np.sqrt(self.lam) * theta)
        return u

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.m)
