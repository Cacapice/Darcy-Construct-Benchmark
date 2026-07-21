"""Steady Darcy-flow forward solver on the unit square.

    -div( kappa(x) grad p(x) ) = f(x)  in D = (0,1)^2
    p = 0                              on the boundary (homogeneous Dirichlet)

Discretization: nodal 5-point finite differences with harmonic-mean
face transmissibilities (standard for heterogeneous conductivity).
Mesh resolution, forcing, and boundary conditions are *nuisance*
variables of the benchmark, not target parameters.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .config import SolverConfig


def forcing_field(cfg: SolverConfig) -> np.ndarray:
    n = cfg.n
    xs = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    if cfg.forcing == "constant":
        return np.ones((n, n))
    if cfg.forcing == "gaussian_bump":
        s = 0.15
        return np.exp(-(((X - 0.5) ** 2) + ((Y - 0.5) ** 2)) / (2.0 * s * s))
    raise ValueError(f"unknown forcing '{cfg.forcing}'")


class DarcySolver:
    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.n = cfg.n
        self.h = 1.0 / (cfg.n - 1)
        self.f = forcing_field(cfg)

    def solve(self, u_field: np.ndarray) -> np.ndarray:
        """Solve for pressure given log-conductivity u on the nodal grid.

        Parameters
        ----------
        u_field : (n, n) log-conductivity at nodes.

        Returns
        -------
        p : (n, n) pressure, zero on the boundary.
        """
        n, h = self.n, self.h
        if u_field.shape != (n, n):
            raise ValueError(f"u_field must be ({n},{n}), got {u_field.shape}")
        kappa = np.exp(u_field)

        # Face transmissibilities (harmonic means of adjacent nodal kappa).
        Tx = 2.0 / (1.0 / kappa[:, :-1] + 1.0 / kappa[:, 1:])   # (n, n-1): (i,j)-(i,j+1)
        Ty = 2.0 / (1.0 / kappa[:-1, :] + 1.0 / kappa[1:, :])   # (n-1, n): (i,j)-(i+1,j)

        ii, jj = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1), indexing="ij")
        k = (ii - 1) * (n - 2) + (jj - 1)

        diag = Ty[ii - 1, jj] + Ty[ii, jj] + Tx[ii, jj - 1] + Tx[ii, jj]
        rows = [k.ravel()]
        cols = [k.ravel()]
        vals = [diag.ravel()]

        west = jj - 1 >= 1
        rows.append(k[west]); cols.append(((ii - 1) * (n - 2) + (jj - 2))[west])
        vals.append(-Tx[ii, jj - 1][west])

        east = jj + 1 <= n - 2
        rows.append(k[east]); cols.append(((ii - 1) * (n - 2) + jj)[east])
        vals.append(-Tx[ii, jj][east])

        north = ii - 1 >= 1
        rows.append(k[north]); cols.append(((ii - 2) * (n - 2) + (jj - 1))[north])
        vals.append(-Ty[ii - 1, jj][north])

        south = ii + 1 <= n - 2
        rows.append(k[south]); cols.append((ii * (n - 2) + (jj - 1))[south])
        vals.append(-Ty[ii, jj][south])

        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=((n - 2) ** 2, (n - 2) ** 2),
        ).tocsr()
        b = self.f[1:-1, 1:-1].ravel() * h * h

        p = np.zeros((n, n))
        p[1:-1, 1:-1] = spsolve(A, b).reshape(n - 2, n - 2)
        return p
