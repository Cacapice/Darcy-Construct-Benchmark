"""Null hypotheses and diagnostic metrics.

Null hypotheses (preregistered; thresholds in ``config.Thresholds``)
--------------------------------------------------------------------

H0_global(v):   O G_m(theta + delta v) is indistinguishable from
                O G_m(theta) under the noise model N(0, sigma^2 I).
                Test: with two independent noisy observations of the pair,
                the statistic ||y' - y||^2 / (2 sigma^2) is chi^2(n_obs)
                under H0 and noncentral chi^2(n_obs, ||dy||^2/(2 sigma^2))
                under the alternative. H0 is *rejected* (the direction is
                declared practically identifiable) iff the analytic power
                of the level-alpha test is >= power_min.

H0_comp(j):     H0_global with v = e_j. Rejection = componentwise
                practical identifiability at intervention size delta.

H0_preserve(j): the surrogate preserves the response to T_{j,delta}:
                attenuation ||O dG~|| / ||O dG|| inside the preregistered
                band AND relative error ||O(dG~ - dG)|| / ||O dG|| below
                rel_error_max. Violation = suppressed / distorted component.

Negative control: delta = 0 gives noncentrality 0, so reported power
equals alpha by construction - the pipeline cannot manufacture detections
from nothing. Random directions serve as reference perturbations that
were not chosen using any surrogate output.

Local identifiability spectrum: singular values of the noise-whitened
Jacobian J/sigma at theta*. Right singular vectors with whitened singular
value > informed_sv_cut span the *data-informed* subspace; the orthogonal
complement is *prior-dominated*. These directions are derived from the
exact map only, never from surrogate behavior, so testing the surrogate
on them involves no post-hoc selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.stats import chi2, ncx2, norm

from .config import Thresholds
from .perturbations import MinimalPair

Map = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------- detection

def detection_power(delta_y: np.ndarray, sigma: float, alpha: float) -> float:
    """Analytic power of the level-alpha two-sample distinguishability test."""
    n = delta_y.size
    nc = float(delta_y @ delta_y) / (2.0 * sigma * sigma)
    crit = chi2.ppf(1.0 - alpha, df=n)
    if nc == 0.0:
        return alpha
    return float(ncx2.sf(crit, df=n, nc=nc))


def matched_direction_power(delta_y: np.ndarray, sigma: float, alpha: float) -> float:
    """Matched-direction test: project y' - y onto v = dy/||dy||.

    The statistic v^T (y' - y) / (sigma sqrt(2)) is N(mu, 1) with
    mu = ||dy|| / (sigma sqrt(2)); two-sided level-alpha power follows
    analytically. Against the realized simple alternative this is the
    Neyman-Pearson-type benchmark (the NP most-powerful test is the
    one-sided version of the same linear statistic; the two-sided form
    used here is slightly conservative), so it upper-bounds achievable
    per-direction power and removes the chi^2(n_obs) degrees-of-freedom
    dilution of the aggregate test. The matching direction is available
    because the exact map is; in deployment it comes from the numerical
    solver at certification points.

    SUPPLEMENTARY, NOT DECISIONAL: the preregistered reject/retain decision
    remains the aggregate test in ``detection_power``. Promoting this metric
    to decisional status is a protocol amendment, to be logged as such.
    """
    mu = float(np.linalg.norm(delta_y)) / (sigma * np.sqrt(2.0))
    z = norm.ppf(1.0 - alpha / 2.0)
    if mu == 0.0:
        return alpha
    return float(norm.cdf(-z - mu) + norm.sf(z - mu))


# Deprecated alias (v0.2/v0.3 name); prefer matched_direction_power.
directional_power = matched_direction_power


@dataclass(frozen=True)
class DirectionResult:
    label: str
    delta: float
    response_norm: float      # ||O G(theta') - O G(theta)||_2
    power: float              # aggregate test (DECISIONAL, preregistered)
    power_directional: float  # 1-dof oracle bound (supplementary)
    reject_h0: bool           # True => practically identifiable (aggregate)


def evaluate_direction(
    G: Map, pair: MinimalPair, sigma: float, thresholds: Thresholds
) -> DirectionResult:
    dy = G(pair.theta_pert) - G(pair.theta)
    power = detection_power(dy, sigma, thresholds.alpha)
    return DirectionResult(
        label=pair.label,
        delta=pair.delta,
        response_norm=float(np.linalg.norm(dy)),
        power=power,
        power_directional=directional_power(dy, sigma, thresholds.alpha),
        reject_h0=power >= thresholds.power_min,
    )


# ---------------------------------------------------------------- Jacobian

def jacobian_fd(G: Map, theta: np.ndarray, eps: float) -> np.ndarray:
    """Central-difference Jacobian of G at theta. -> (n_obs, m)"""
    m = theta.size
    cols = []
    for j in range(m):
        e = np.zeros(m)
        e[j] = eps
        cols.append((G(theta + e) - G(theta - e)) / (2.0 * eps))
    return np.column_stack(cols)


@dataclass(frozen=True)
class Subspaces:
    singular_values: np.ndarray       # whitened, descending
    directions: np.ndarray            # (m, m) right singular vectors as columns
    rank_informed: int

    @property
    def informed(self) -> np.ndarray:
        return self.directions[:, : self.rank_informed]

    @property
    def prior_dominated(self) -> np.ndarray:
        return self.directions[:, self.rank_informed:]


def identifiability_spectrum(J: np.ndarray, sigma: float, sv_cut: float) -> Subspaces:
    _, s, Vt = np.linalg.svd(J / sigma, full_matrices=True)
    rank = int(np.sum(s > sv_cut))
    return Subspaces(singular_values=s, directions=Vt.T, rank_informed=rank)


# ---------------------------------------------------------------- surrogate

@dataclass(frozen=True)
class PreservationResult:
    label: str
    exact_norm: float
    surrogate_norm: float
    attenuation: float            # ||dG~|| / ||dG||
    rel_error: float              # ||dG~ - dG|| / ||dG||
    preserved: bool               # H0_preserve survives


def evaluate_preservation(
    G: Map, G_sur: Map, pair: MinimalPair, thresholds: Thresholds
) -> PreservationResult:
    dG = G(pair.theta_pert) - G(pair.theta)
    dS = G_sur(pair.theta_pert) - G_sur(pair.theta)
    en = float(np.linalg.norm(dG))
    sn = float(np.linalg.norm(dS))
    if en == 0.0:
        # Exact response is null; preservation is vacuous but flagged.
        return PreservationResult(pair.label, 0.0, sn, np.nan, np.nan, preserved=sn == 0.0)
    att = sn / en
    rel = float(np.linalg.norm(dS - dG)) / en
    lo, hi = thresholds.attenuation_band
    ok = (lo <= att <= hi) and (rel <= thresholds.rel_error_max)
    return PreservationResult(pair.label, en, sn, att, rel, preserved=ok)


# ---------------------------------------------------------------- nuisance

@dataclass(frozen=True)
class NuisanceCell:
    nuisance_label: str
    results: List[DirectionResult]


def decision_flips(
    baseline: Sequence[DirectionResult], variant: Sequence[DirectionResult]
) -> List[str]:
    """Labels whose reject/retain decision changed between nuisance settings."""
    base = {r.label: r.reject_h0 for r in baseline}
    return [r.label for r in variant if r.label in base and base[r.label] != r.reject_h0]


# ------------------------------------------------- identifiability profile

def identifiability_profile(
    G: Map,
    theta: np.ndarray,
    sigma: float,
    thresholds: Thresholds,
    radius_a: Optional[float] = None,
) -> np.ndarray:
    """I_j(delta, sigma): detection power per coefficient, as a spectrum.

    Returns an (m, 2) array: column 0 the aggregate-test power (decisional
    scale), column 1 the directional oracle power (supplementary). This
    replaces the binary identifiable / not-identifiable summary with a
    curve analogous to singular-value decay: linear observability (Jacobian
    rank) and practical detectability (power at finite delta under noise)
    are different objects, and the profile is where they visibly diverge.
    Purely descriptive; the preregistered thresholds still make decisions.
    """
    from .perturbations import coordinate_pair

    rows = []
    for j in range(theta.size):
        pair = coordinate_pair(theta, j, thresholds.delta, radius_a)
        dy = G(pair.theta_pert) - G(pair.theta)
        rows.append((
            detection_power(dy, sigma, thresholds.alpha),
            directional_power(dy, sigma, thresholds.alpha),
        ))
    return np.asarray(rows)


# ---------------------------------------------------- information geometry

@dataclass(frozen=True)
class GeometryReport:
    """Differential-geometric comparison of exact and surrogate maps at theta*.

    Identifiability can survive while the geometry degrades: if both maps
    are injective but the surrogate collapses one direction's derivative by
    a factor of 100, optimization conditioning, posterior geometry, and
    credible intervals all change while pass/fail identifiability does not.
    These quantities measure that degradation directly, along the exact
    map's right singular directions v_i:

    * ``deriv_attenuation[i]`` = ||J_sur v_i|| / ||J v_i||  — first-order
      response preservation per direction (1 = preserved, 0 = collapsed);
    * ``coordinate_attenuation[j]`` = ||J_sur e_j|| / ||J e_j|| — the
      per-coefficient first-order analogue of the minimal-pair attenuation
      metric; the right lens for coordinate-aligned pathologies, since
      singular directions v_i mix coordinates and can mask them;
    * ``posterior_std_ratio[i]`` — ratio of Laplace posterior standard
      deviations along v_i under the surrogate vs the exact map, in the
      linear-Gaussian approximation at theta* with the KL prior
      theta ~ N(0, I_m). Values >> 1 mean the surrogate has surrendered
      the data's information along v_i back to the prior — the Bayesian
      signature of geometric (not binary) identifiability loss, and the
      quantity that bridges to posterior-perturbation bounds.

    Descriptive diagnostics: attaching decision thresholds to them would
    be a protocol amendment.
    """

    sv_exact: np.ndarray
    sv_surrogate_along_exact: np.ndarray   # ||J_sur v_i|| (same basis)
    deriv_attenuation: np.ndarray          # along exact right singular dirs v_i
    coordinate_attenuation: np.ndarray     # ||J_sur e_j|| / ||J e_j|| per coeff
    posterior_std_ratio: np.ndarray        # along v_i
    directions: np.ndarray                 # v_i as columns


def posterior_geometry(J: np.ndarray, J_sur: np.ndarray, sigma: float) -> GeometryReport:
    m = J.shape[1]
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    V = Vt.T                                   # (m, m) when n_obs >= m
    JV = J @ V
    JsV = J_sur @ V
    nJV = np.linalg.norm(JV, axis=0)
    nJsV = np.linalg.norm(JsV, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        att = np.where(nJV > 0.0, nJsV / nJV, np.nan)
    nJ = np.linalg.norm(J, axis=0)
    nJs = np.linalg.norm(J_sur, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coord_att = np.where(nJ > 0.0, nJs / nJ, np.nan)

    eye = np.eye(m)
    C = np.linalg.inv(J.T @ J / sigma**2 + eye)        # Laplace posterior cov,
    Cs = np.linalg.inv(J_sur.T @ J_sur / sigma**2 + eye)  # prior theta ~ N(0, I)
    num = np.einsum("ji,jk,ki->i", V, Cs, V)
    den = np.einsum("ji,jk,ki->i", V, C, V)
    ratio = np.sqrt(num / den)

    return GeometryReport(
        sv_exact=s,
        sv_surrogate_along_exact=nJsV,
        deriv_attenuation=att,
        coordinate_attenuation=coord_att,
        posterior_std_ratio=ratio,
        directions=V,
    )


# ------------------------------------------------ information preservation

@dataclass(frozen=True)
class InformationReport:
    """Information-preservation functional I(G_sur, G) at theta*.

    One aspect of the benchmark's measured quantity, inferential
    fidelity: whether the surrogate supports the same inferences the exact
    map supports. This functional measures the information aspect; the
    minimal-pair nulls measure parameter alignment (limitation stated below). In the
    linearized (Laplace) Gaussian setting at theta* with the KL prior
    theta ~ N(0, I_m), the expected information gain of the experiment
    (Lindley information; log Bayesian D-optimality criterion) is

        EIG = (1/2) log det( I + J^T J / sigma^2 )   [nats].

    * ``preserved_fraction`` = EIG(G_sur)/EIG(G): the scalar functional
      I(G_sur, G).
    * ``bits_lost_total`` = EIG(G) - EIG(G_sur) in bits: total information
      the surrogate surrenders back to the prior.
    * ``bits_lost_by_direction[i]`` = log2 of the posterior-std ratio
      along the exact map's singular direction v_i: the spectral
      decomposition of the loss (approximate — directions are the exact
      map's, so terms need not sum exactly to the total).

    Scope and an important limitation, stated deliberately:

    * LOCAL: everything is the linear-Gaussian approximation at theta*.
    * INFORMATION WITHOUT ALIGNMENT: EIG depends on J_sur only through its
      singular spectrum, so it is blind to output rotations — a surrogate
      can preserve the information while redirecting it toward the
      wrong coefficients (the ShortcutSurrogate's contaminated e_{j_used}
      response is exactly this: attenuation ~1, relative response error
      large, modest information change). Information preservation is
      therefore NECESSARY for trustworthy inversion, not sufficient; the
      H0_preserve minimal-pair nulls test the PARAMETER ALIGNMENT of the preserved
      information with the correct coefficients. The two aspects are
      complementary by design.

    Descriptive diagnostic: attaching decision thresholds would be a
    protocol amendment.
    """

    eig_exact_nats: float
    eig_surrogate_nats: float
    preserved_fraction: float
    bits_lost_total: float
    bits_lost_by_direction: np.ndarray


def information_functional(
    J: np.ndarray, J_sur: np.ndarray, sigma: float
) -> InformationReport:
    m = J.shape[1]
    eye = np.eye(m)

    def eig_nats(Jac: np.ndarray) -> float:
        sign, logdet = np.linalg.slogdet(eye + Jac.T @ Jac / sigma**2)
        if sign <= 0.0:
            raise np.linalg.LinAlgError("posterior precision not SPD")
        return 0.5 * logdet

    e_exact = eig_nats(J)
    e_sur = eig_nats(J_sur)
    geom = posterior_geometry(J, J_sur, sigma)
    return InformationReport(
        eig_exact_nats=e_exact,
        eig_surrogate_nats=e_sur,
        preserved_fraction=e_sur / e_exact if e_exact > 0 else float("nan"),
        bits_lost_total=(e_exact - e_sur) / np.log(2.0),
        bits_lost_by_direction=np.log2(geom.posterior_std_ratio),
    )
