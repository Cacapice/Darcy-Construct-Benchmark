"""Tests for the construct-validity benchmark.

The tests are organized around the framework's own claims:

* the forward machinery is correct (solver, KL basis);
* minimal pairs perturb exactly one declared direction;
* the delta = 0 negative control cannot produce a detection;
* the exact surrogate passes every preservation null (negative control);
* the mode-suppressing surrogate is caught on suppressed components and
  passes on retained ones (positive control);
* the target parameter's meaning is invariant to mesh nuisance.
"""

import numpy as np
import pytest

from darcy_construct_benchmark import (
    BenchmarkConfig,
    DarcySolver,
    ExactSurrogate,
    ForwardMap,
    KLBasis,
    ModeSuppressingSurrogate,
    NuisanceConfig,
    ObservationConfig,
    PriorConfig,
    SolverConfig,
    Thresholds,
    coordinate_pair,
    detection_power,
    make_minimal_pair,
    evaluate_direction,
    evaluate_preservation,
)


@pytest.fixture(scope="module")
def small_setup():
    prior = PriorConfig(n_ref=21, m_max=16, rho=0.90)
    basis = KLBasis(prior)
    nuis = NuisanceConfig(
        solver=SolverConfig(n=21),
        observation=ObservationConfig(n_sensors_per_side=6, noise_sigma=1e-3),
    )
    G = ForwardMap(basis, nuis)
    rng = np.random.default_rng(0)
    theta = basis.sample_theta(rng)
    return basis, G, theta


# ----------------------------------------------------------------- solver

def test_solver_matches_analytic_constant_kappa():
    """kappa = 1, f = 1: compare to the Fourier series solution of -Lap p = 1."""
    cfg = SolverConfig(n=41, forcing="constant")
    solver = DarcySolver(cfg)
    p = solver.solve(np.zeros((41, 41)))

    xs = np.linspace(0, 1, 41)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    exact = np.zeros_like(X)
    for i in range(1, 40, 2):
        for j in range(1, 40, 2):
            exact += (16.0 / (np.pi**4 * i * j * (i**2 + j**2))
                      * np.sin(i * np.pi * X) * np.sin(j * np.pi * Y))
    assert np.max(np.abs(p - exact)) < 5e-4
    assert np.all(p[1:-1, 1:-1] > 0)          # maximum principle, f > 0
    assert np.allclose(p, p.T, atol=1e-10)     # symmetry of the setup


def test_solver_scaling_in_kappa():
    """Multiplying kappa by c scales p by 1/c."""
    cfg = SolverConfig(n=21)
    solver = DarcySolver(cfg)
    p1 = solver.solve(np.zeros((21, 21)))
    p2 = solver.solve(np.full((21, 21), np.log(2.0)))
    assert np.allclose(p2, p1 / 2.0, atol=1e-12)


# -------------------------------------------------------------------- KL

def test_kl_orthonormality_and_truncation(small_setup):
    basis, _, _ = small_setup
    n = basis.prior.n_ref
    w = (1.0 / (n - 1)) ** 2
    P = basis.phi_ref.reshape(n * n, basis.m)
    gram = (P.T @ P) * w
    assert np.allclose(gram, np.eye(basis.m), atol=1e-8)
    assert np.all(np.diff(basis.lam) <= 1e-12)          # descending
    assert basis.m <= basis.prior.m_max
    assert 0.0 < basis.achieved_rho <= 1.0


def test_field_linearity_in_theta(small_setup):
    basis, _, theta = small_setup
    u1 = basis.log_conductivity(theta, 21)
    u2 = basis.log_conductivity(2.0 * theta, 21)
    mean = basis.prior.mean_value
    assert np.allclose(u2 - mean, 2.0 * (u1 - mean), atol=1e-10)


# ------------------------------------------------------------ minimal pairs

def test_minimal_pair_perturbs_single_component(small_setup):
    basis, _, theta = small_setup
    pair = coordinate_pair(theta, 3, 0.5, radius_a=None)
    diff = pair.theta_pert - pair.theta
    assert diff[3] == pytest.approx(pair.delta)
    assert np.count_nonzero(diff) == 1


def test_minimal_pair_respects_admissible_ball():
    theta = np.zeros(4)
    theta[0] = 1.9
    pair = make_minimal_pair(theta, np.eye(4)[0], 0.5, radius_a=2.0, label="e_0")
    # +0.5 would exit the ball; sign must flip.
    assert pair.delta == pytest.approx(-0.5)
    assert np.linalg.norm(pair.theta_pert) <= 2.0


def test_minimal_pair_shrinks_when_both_signs_exit():
    theta = np.zeros(2)
    pair = make_minimal_pair(theta, np.eye(2)[0], 5.0, radius_a=1.0, label="e_0")
    assert 0.0 < abs(pair.delta) <= 1.0 + 1e-9
    assert np.linalg.norm(pair.theta_pert) <= 1.0 + 1e-9


# ------------------------------------------------------------ null controls

def test_zero_perturbation_power_equals_alpha():
    th = Thresholds()
    assert detection_power(np.zeros(64), 1e-3, th.alpha) == pytest.approx(th.alpha)


def test_leading_component_detected_exact_map(small_setup):
    basis, G, theta = small_setup
    th = Thresholds()
    pair = coordinate_pair(theta, 0, th.delta, radius_a=None)
    r = evaluate_direction(G, pair, G.noise_sigma, th)
    assert r.reject_h0, "leading KL component should be practically identifiable"


# ------------------------------------------------------ surrogate controls

def test_exact_surrogate_preserves_everything(small_setup):
    basis, G, theta = small_setup
    th = Thresholds()
    S = ExactSurrogate(G)
    for j in range(basis.m):
        pair = coordinate_pair(theta, j, th.delta, radius_a=None)
        p = evaluate_preservation(G, S, pair, th)
        assert p.preserved, f"exact surrogate flagged on component {j}"
        assert p.attenuation == pytest.approx(1.0)
        assert p.rel_error == pytest.approx(0.0)


def test_mode_suppressing_surrogate_caught(small_setup):
    basis, G, theta = small_setup
    th = Thresholds()
    k = basis.m // 2
    S = ModeSuppressingSurrogate(G, k_keep=k)
    # Suppressed components: response should be (near) annihilated.
    for j in (k, basis.m - 1):
        pair = coordinate_pair(theta, j, th.delta, radius_a=None)
        p = evaluate_preservation(G, S, pair, th)
        assert not p.preserved, f"suppression of component {j} not detected"
        assert p.attenuation < 0.1
    # A retained *leading* component: perturbation passes through unchanged,
    # though the base point differs (suppressed coords zeroed), so responses
    # agree only approximately; the preregistered band must absorb that.
    pair = coordinate_pair(theta, 0, th.delta, radius_a=None)
    p = evaluate_preservation(G, S, pair, th)
    assert p.preserved, "retained leading component wrongly flagged"


# ------------------------------------------------- nuisance invariance

def test_theta_meaning_invariant_under_mesh_nuisance(small_setup):
    """Same theta, finer mesh: the log-conductivity field converges to the
    same function (interpolated modes), so nodal values on shared points
    must agree closely. This is the operational meaning of 'mesh is a
    nuisance variable, not a target parameter'."""
    basis, _, theta = small_setup
    u_coarse = basis.log_conductivity(theta, 21)
    u_fine = basis.log_conductivity(theta, 41)
    assert np.allclose(u_fine[::2, ::2], u_coarse, atol=1e-8)


def test_run_benchmark_smoke():
    from darcy_construct_benchmark.run_benchmark import run
    cfg = BenchmarkConfig(
        prior=PriorConfig(n_ref=21, m_max=12, rho=0.85),
        radius_a=None,
    )
    out = run(cfg, verbose=False)
    # Negative control: exact surrogate preserves all preregistered directions.
    exact_rows = out["preservation"]["exact (negative control)"]
    assert all(r.preserved for r in exact_rows)
    # Positive control: at least one direction violated by the suppressor.
    sup_key = next(k for k in out["preservation"] if "suppressing" in k)
    assert any(not r.preserved for r in out["preservation"][sup_key])
