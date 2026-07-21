"""Tests for v0.3 additions: ShortcutSurrogate (accuracy/identifiability
gap), the identifiability profile, and posterior information geometry."""

import numpy as np
import pytest

from darcy_construct_benchmark import (
    BenchmarkConfig,
    DatasetConfig,
    ExactSurrogate,
    ForwardMap,
    KLBasis,
    ModeSuppressingSurrogate,
    NuisanceConfig,
    ObservationConfig,
    PriorConfig,
    SolverConfig,
    Thresholds,
    certify_surrogate,
    coordinate_pair,
    freeze_protocol,
    generate_dataset,
)
from darcy_construct_benchmark.diagnostics import (
    evaluate_preservation,
    identifiability_profile,
    jacobian_fd,
    posterior_geometry,
)
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
    cfg = BenchmarkConfig(prior=prior, baseline=nuis, radius_a=None)
    rng = np.random.default_rng(0)
    theta = basis.sample_theta(rng)
    return basis, G, cfg, theta


# --------------------------------------------------------------- shortcut

def _shortcut(G, basis):
    # j_ignored chosen mid-spectrum: identifiable by the exact map at the
    # small fixture (m<=16), sensitivity modest so training error is small.
    return ShortcutSurrogate(G, j_used=0, j_ignored=5, corr=0.98)


def test_shortcut_accurate_on_training_distribution(setup):
    """The gap that motivates the program: small forward error on the
    training distribution..."""
    basis, G, cfg, _ = setup
    S = _shortcut(G, basis)
    rng = np.random.default_rng(3)
    n = 48
    T = rng.standard_normal((n, basis.m))
    T[:, 5] = 0.98 * T[:, 0] + np.sqrt(1 - 0.98**2) * T[:, 5]
    err2 = sig2 = 0.0
    for i in range(n):
        y = G(T[i])
        err2 += float(np.sum((S(T[i]) - y) ** 2))
        sig2 += float(np.sum(y ** 2))
    rel_rmse = np.sqrt(err2 / sig2)
    assert rel_rmse < 0.05, f"training-distribution rel RMSE too large: {rel_rmse}"


def test_shortcut_fails_gate_on_ignored_component(setup, tmp_path):
    """...while the certification gate fails, because minimal pairs break
    the training correlation."""
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    S = _shortcut(G, basis)

    # Attenuation on the ignored coordinate is exactly zero.
    pair = coordinate_pair(theta, 5, th.delta, cfg.radius_a)
    p = evaluate_preservation(G, S, pair, th)
    assert p.surrogate_norm == pytest.approx(0.0, abs=1e-15)
    assert not p.preserved

    # Gate over a protocol containing all coordinate directions: FAIL,
    # with the ignored component among decisional violations.
    directions = [(f"e_{j}", np.eye(basis.m)[j]) for j in range(basis.m)]
    frozen = freeze_protocol(G, theta, directions, cfg, tmp_path / "p.json")
    e5 = next(e for e in frozen["directions"] if e["label"] == "e_5")
    assert e5["decisional"], "fixture assumption: e_5 identifiable by exact map"
    rep = certify_surrogate(G, S, frozen)
    assert not rep.passed
    assert any((not r.preserved) and r.label == "e_5" for r in rep.decisional)


def test_shortcut_distorts_used_component(setup):
    """Perturbing theta_{j_used} leaks a spurious response through the
    imputed coefficient: relative response error strictly positive."""
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    S = _shortcut(G, basis)
    pair = coordinate_pair(theta, 0, th.delta, cfg.radius_a)
    p = evaluate_preservation(G, S, pair, th)
    assert p.rel_error > 0.01


def test_dataset_correlation_train_only(setup, tmp_path):
    basis, G, cfg, _ = setup
    dcfg = DatasetConfig(n_train=400, n_test=50, n_cert=400, seed=11,
                         include_pressure_fields=False,
                         theta_correlation=(0, 5, 0.9))
    generate_dataset(basis, cfg.baseline, dcfg, tmp_path)
    tr = np.load(tmp_path / "darcy_train.npz")["theta"]
    ce = np.load(tmp_path / "darcy_cert.npz")["theta"]
    r_tr = np.corrcoef(tr[:, 0], tr[:, 5])[0, 1]
    r_ce = np.corrcoef(ce[:, 0], ce[:, 5])[0, 1]
    assert abs(r_tr - 0.9) < 0.06, f"train corr {r_tr}"
    assert abs(r_ce) < 0.2, f"cert split must stay prior-distributed, corr {r_ce}"
    # marginal variance preserved under the confound
    assert abs(np.std(tr[:, 5]) - 1.0) < 0.15


# ---------------------------------------------------------------- profile

def test_identifiability_profile_shape_and_consistency(setup):
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    prof = identifiability_profile(G, theta, G.noise_sigma, th, cfg.radius_a)
    assert prof.shape == (basis.m, 2)
    assert np.all((prof >= 0.0) & (prof <= 1.0))
    # Leading coefficient detectable in this fixture; oracle >= aggregate
    # is not a theorem pointwise, but at the extremes it must hold sanely:
    assert prof[0, 0] > 0.99 and prof[0, 1] > 0.99
    # Directional oracle never below alpha.
    assert np.all(prof[:, 1] >= th.alpha - 1e-12)


# --------------------------------------------------------------- geometry

def test_geometry_exact_is_identity(setup):
    basis, G, cfg, theta = setup
    J = jacobian_fd(G, theta, cfg.thresholds.jacobian_eps)
    geom = posterior_geometry(J, J, G.noise_sigma)
    assert np.allclose(geom.deriv_attenuation, 1.0, atol=1e-12)
    assert np.allclose(geom.posterior_std_ratio, 1.0, atol=1e-10)


def test_geometry_detects_collapse_beyond_pass_fail(setup):
    """Both maps injective on the retained subspace is irrelevant: the
    suppressor collapses directions, and the geometry shows the posterior
    surrendering those directions back to the prior."""
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    k = basis.m // 2
    S = ModeSuppressingSurrogate(G, k_keep=k)
    J = jacobian_fd(G, theta, th.jacobian_eps)
    J_s = jacobian_fd(S, theta, th.jacobian_eps)
    geom = posterior_geometry(J, J_s, G.noise_sigma)
    # suppressed coordinates collapsed exactly at first order...
    assert np.all(geom.coordinate_attenuation[k:] < 1e-12)
    # ...while singular-direction attenuation can mask this (v_i mix
    # coordinates), which is precisely why both bases are reported:
    assert np.nanmin(geom.deriv_attenuation) > 0.05
    # ...and posterior std inflates along exact-map directions
    # (information surrendered to the prior). In this fixture k = m/2, so
    # every exact singular direction overlaps the suppressed span and ALL
    # ratios exceed 1 — global geometric degradation from a coordinate-
    # local pathology, itself a finding worth reporting.
    assert np.nanmax(geom.posterior_std_ratio) > 3.0
    # ratios never meaningfully below 1: a surrogate cannot manufacture
    # information the exact map lacks along these directions
    # (J_s columns k: are zero, so precision only decreases).
    assert np.nanmin(geom.posterior_std_ratio) > 0.9


def test_geometry_shortcut_posterior_inflation_on_ignored(setup):
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    S = _shortcut(G, basis)
    J = jacobian_fd(G, theta, th.jacobian_eps)
    J_s = jacobian_fd(S, theta, th.jacobian_eps)
    geom = posterior_geometry(J, J_s, G.noise_sigma)
    # The surrogate's Jacobian column for the ignored coefficient is zero
    # exactly; posterior std along SOME exact-map direction must inflate.
    assert geom.coordinate_attenuation[5] == pytest.approx(0.0, abs=1e-12)
    assert np.nanmax(geom.posterior_std_ratio) > 2.0
