"""Tests for v0.4: information-preservation functional, the
aggregation-dilution proposition, and terminology/API changes."""

import numpy as np
import pytest

from darcy_construct_benchmark import (
    Experiment,
    ForwardMap,
    KLBasis,
    ModeSuppressingSurrogate,
    NuisanceConfig,
    ObservationConfig,
    PriorConfig,
    SolverConfig,
    Thresholds,
    detection_power,
    information_functional,
    matched_direction_power,
)
from darcy_construct_benchmark.diagnostics import directional_power, jacobian_fd
from darcy_construct_benchmark.surrogates import ShortcutSurrogate


@pytest.fixture(scope="module")
def setup():
    prior = PriorConfig(n_ref=21, m_max=16, rho=0.90)
    basis = KLBasis(prior)
    nuis = NuisanceConfig(
        solver=SolverConfig(n=21),
        observation=ObservationConfig(n_sensors_per_side=6, noise_sigma=1e-3),
    )
    G = ForwardMap(basis, nuis)
    rng = np.random.default_rng(0)
    theta = basis.sample_theta(rng)
    th = Thresholds()
    J = jacobian_fd(G, theta, th.jacobian_eps)
    return basis, G, theta, th, J


# --------------------------------------------------- information functional

def test_information_identity_for_exact(setup):
    _, G, _, _, J = setup
    info = information_functional(J, J, G.noise_sigma)
    assert info.preserved_fraction == pytest.approx(1.0)
    assert info.bits_lost_total == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(info.bits_lost_by_direction, 0.0, atol=1e-8)


def test_information_monotone_under_suppression(setup):
    """Suppressing more coordinates loses more information; the loss is
    strictly positive and the per-direction decomposition is nonnegative
    (the suppressor cannot create information)."""
    basis, G, theta, th, J = setup
    losses = []
    for k in (basis.m - 2, basis.m // 2, 2):
        S = ModeSuppressingSurrogate(G, k_keep=k)
        J_s = jacobian_fd(S, theta, th.jacobian_eps)
        info = information_functional(J, J_s, G.noise_sigma)
        losses.append(info.bits_lost_total)
        assert info.preserved_fraction < 1.0
        assert np.nanmin(info.bits_lost_by_direction) > -0.2  # no created info
    assert losses[0] < losses[1] < losses[2]


def test_information_capacity_blind_to_alignment(setup):
    """The stated limitation, enforced as a test so it is never forgotten:
    an output ROTATION of the exact Jacobian preserves EIG exactly while
    completely misaligning which coefficient the information is about.
    This is why I(G_sur, G) is necessary but not sufficient, and why the
    H0_preserve alignment layer exists."""
    _, G, _, _, J = setup
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((J.shape[0], J.shape[0])))
    J_rot = Q @ J
    info = information_functional(J, J_rot, G.noise_sigma)
    assert info.preserved_fraction == pytest.approx(1.0, abs=1e-9)
    # yet the response directions no longer match the exact map's:
    assert np.linalg.norm(J_rot - J) / np.linalg.norm(J) > 0.5


def test_information_shortcut_loses_capacity_and_alignment(setup):
    basis, G, theta, th, J = setup
    S = ShortcutSurrogate(G, j_used=0, j_ignored=5, corr=0.98)
    J_s = jacobian_fd(S, theta, th.jacobian_eps)
    info = information_functional(J, J_s, G.noise_sigma)
    assert 0.0 < info.bits_lost_total  # capacity strictly lost (zero column)
    assert info.preserved_fraction < 1.0


# ------------------------------------------- aggregation-dilution proposition

def test_aggregation_dilution_proposition():
    """Fix dy != 0 and sigma; append uninformative sensors (zero components).
    The omnibus chi^2 test's power strictly decreases toward alpha, while
    the matched-direction test's power is invariant. Hence loss of omnibus
    power under sensor addition reflects test choice, not loss of
    identifiability."""
    sigma, alpha = 1e-3, 0.01
    signal = 5e-3
    agg, matched = [], []
    for n in (16, 64, 256, 1024):
        dy = np.zeros(n)
        dy[0] = signal
        agg.append(detection_power(dy, sigma, alpha))
        matched.append(matched_direction_power(dy, sigma, alpha))
    assert all(a > b for a, b in zip(agg, agg[1:])), agg      # strictly down
    assert agg[-1] < 3 * alpha       # approaching alpha (convergence is slow)
    assert np.allclose(matched, matched[0])                    # invariant
    assert matched[0] > 0.8                                    # signal is real


# ----------------------------------------------------------- API/terminology

def test_matched_direction_alias_backcompat():
    dy = np.zeros(8)
    dy[2] = 3e-3
    assert directional_power(dy, 1e-3, 0.01) == matched_direction_power(dy, 1e-3, 0.01)


def test_experiment_alias_is_nuisance_config():
    assert Experiment is NuisanceConfig
