"""Dataset generation for training and evaluating learned surrogates.

Produces the public Darcy benchmark dataset: (theta, u, p, y) tuples with
theta drawn from the prior, split into train / test / certification sets
from independent seed streams. The certification split exists so that the
identifiability gate is never run on data the surrogate trained on.

Reproducibility contract:
* every split is a deterministic function of (prior, nuisance, dataset
  config, package version) — regeneration is bit-identical;
* a ``manifest.json`` records all three configs, the package version,
  array shapes, and SHA-256 hashes of the saved arrays, so a downloaded
  dataset can be verified against the generating code.

Observations ``y`` are stored NOISELESS. Noise is added downstream at
training / evaluation time so that a single dataset supports multiple
noise levels without regeneration; the benchmark's sigma lives in
``ObservationConfig`` and is recorded in the manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import __version__
from .config import NuisanceConfig
from .forward import ForwardMap
from .kl import KLBasis


@dataclass(frozen=True)
class DatasetConfig:
    n_train: int = 1024
    n_test: int = 128
    n_cert: int = 64
    seed: int = 2026
    include_pressure_fields: bool = True   # FNO/DeepONet field targets
    # Optional confound in the TRAINING DISTRIBUTION (train + test splits):
    # (j_used, j_ignored, r) => theta[j_ignored] = r*theta[j_used]
    #                            + sqrt(1-r^2)*z, marginals still N(0,1).
    # The certification split is ALWAYS drawn from the uncorrelated prior:
    # the gate must be able to realize coefficient combinations off the
    # training correlation manifold, or the shortcut is undetectable in
    # principle. This is the mechanism for studying shortcut surrogates.
    theta_correlation: Optional[Tuple[int, int, float]] = None


def _sample_theta(rng: np.random.Generator, size: int, m: int,
                  correlation: Optional[Tuple[int, int, float]]) -> np.ndarray:
    theta = rng.standard_normal((size, m))
    if correlation is not None:
        a, b, r = correlation
        theta[:, b] = r * theta[:, a] + np.sqrt(1.0 - r * r) * theta[:, b]
    return theta


def _array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def generate_dataset(
    basis: KLBasis,
    nuisance: NuisanceConfig,
    cfg: DatasetConfig,
    out_dir: str | Path,
) -> dict:
    """Generate train/test/cert splits; write NPZ files + manifest. Returns manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    G = ForwardMap(basis, nuisance)
    n_grid = nuisance.solver.n

    streams = np.random.SeedSequence(cfg.seed).spawn(3)
    sizes = {"train": cfg.n_train, "test": cfg.n_test, "cert": cfg.n_cert}
    manifest: dict = {
        "package_version": __version__,
        "prior": dataclasses.asdict(basis.prior),
        "nuisance": dataclasses.asdict(nuisance),
        "dataset": dataclasses.asdict(cfg),
        "m": basis.m,
        "achieved_rho": basis.achieved_rho,
        "n_grid": n_grid,
        "n_obs": G.n_obs,
        "note_noise": "y is noiseless; add N(0, sigma^2 I) downstream",
        "splits": {},
    }

    for (name, size), ss in zip(sizes.items(), streams):
        rng = np.random.default_rng(ss)
        corr = cfg.theta_correlation if name in ("train", "test") else None
        theta = _sample_theta(rng, size, basis.m, corr)
        u = np.empty((size, n_grid, n_grid))
        y = np.empty((size, G.n_obs))
        p = np.empty((size, n_grid, n_grid)) if cfg.include_pressure_fields else None
        for i in range(size):
            u[i] = basis.log_conductivity(theta[i], n_grid)
            pi = G.solver.solve(u[i])
            if p is not None:
                p[i] = pi
            y[i] = G.observation.observe(pi)

        arrays = {"theta": theta, "u": u, "y": y}
        if p is not None:
            arrays["p"] = p
        path = out / f"darcy_{name}.npz"
        np.savez_compressed(path, **arrays)
        manifest["splits"][name] = {
            "file": path.name,
            "n": size,
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "sha256": {k: _array_sha256(v) for k, v in arrays.items()},
        }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
