"""Configuration for the Darcy-flow construct-validity benchmark.

Design principle (mirrors the construct-validity framework):

* The *target parameter* is the KL coefficient vector theta in Theta_m.
* Everything else that affects observations but is NOT a target parameter
  (mesh resolution, forcing, boundary conditions, sensor placement,
  measurement noise) is declared explicitly as a *nuisance variable*
  via ``NuisanceConfig``. Nuisance variables may be varied; the meaning
  of theta must not change when they are.
* All thresholds used in accept/reject decisions are preregistered here
  (``Thresholds``) rather than tuned after observing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class PriorConfig:
    """Matern-class Gaussian prior for the log-conductivity field u."""

    nu: float = 1.5              # Matern smoothness
    length_scale: float = 0.25   # correlation length ell
    variance: float = 1.0        # marginal variance sigma_u^2
    mean_value: float = 0.0      # constant prior mean \bar{u}
    rho: float = 0.95            # preregistered proportion of prior variance retained
    m_max: int = 64              # hard cap on KL truncation level
    n_ref: int = 33              # reference grid (per side) on which KL modes are defined


@dataclass(frozen=True)
class SolverConfig:
    """Nuisance: discretization and forcing of the forward PDE."""

    n: int = 33                       # nodes per side (includes Dirichlet boundary)
    forcing: str = "constant"         # 'constant' | 'gaussian_bump'


@dataclass(frozen=True)
class ObservationConfig:
    """Nuisance: observation operator O and measurement noise."""

    n_sensors_per_side: int = 8       # interior sensor grid
    noise_sigma: float = 1.0e-3       # i.i.d. Gaussian noise s.d.
    seed: int = 0


@dataclass(frozen=True)
class NuisanceConfig:
    """A named bundle of nuisance-variable settings = the EXPERIMENT.

    This object plays two roles at once, and the distinction is the point:

    * It is NUISANCE to the *meaning* of theta: changing the fields here
      must never change what any coefficient represents. KL modes are
      always evaluated on the reference grid and interpolated, so theta
      indexes the same spatial modes under every variant.
    * It is CONSTITUTIVE of the *identifiability* of theta: detection
      power, the identifiability spectrum, and the information functional
      are properties of the full experiment tuple

          E = (Theta_m, prior, G_m, observation operator, noise model),

      not of the forward map alone. The nuisance sweep demonstrates this
      (changing the forcing changes which directions are identifiable).

    ``Experiment`` is provided as an alias to make the second role
    explicit at call sites; both names refer to the same class.
    """

    label: str = "baseline"
    solver: SolverConfig = field(default_factory=SolverConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)


@dataclass(frozen=True)
class Thresholds:
    """Preregistered decision thresholds for the null hypotheses.

    H0_global(v):  the perturbation delta*v produces observation changes
                   indistinguishable from measurement noise.
                   Rejected iff detection power >= power_min at level alpha.
    H0_comp(j):    H0_global with v = e_j (componentwise identifiability).
    H0_preserve(j): the surrogate preserves the response to T_{j,delta};
                   *violated* iff attenuation is outside attenuation_band
                   or relative response error exceeds rel_error_max.
    """

    alpha: float = 0.01                       # test level for detection
    power_min: float = 0.80                   # minimum power to declare detectable
    attenuation_band: Tuple[float, float] = (0.80, 1.25)
    rel_error_max: float = 0.20
    delta: float = 0.5                        # preregistered intervention size (prior s.d. units)
    jacobian_eps: float = 1.0e-3              # FD step for local Jacobian
    informed_sv_cut: float = 1.0              # whitened singular value > 1 => data-informed


@dataclass(frozen=True)
class BenchmarkConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    baseline: NuisanceConfig = field(default_factory=NuisanceConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    radius_a: Optional[float] = None          # admissible ball radius; None => Theta_m = R^m
    theta_seed: int = 12345                   # seed for the known ground-truth coefficient


# Explicit name for the experiment/observation-design role (see docstring).
Experiment = NuisanceConfig
