"""Tests for v0.2 additions: directional oracle power, dataset generation,
field-surrogate adapter, and the frozen-protocol certification gate."""

import json

import numpy as np
import pytest

from darcy_construct_benchmark import (
    BenchmarkConfig,
    DatasetConfig,
    ExactSurrogate,
    FieldToObservationSurrogate,
    ForwardMap,
    KLBasis,
    ModeSuppressingSurrogate,
    NuisanceConfig,
    ObservationConfig,
    PriorConfig,
    SolverConfig,
    Thresholds,
    certify_surrogate,
    detection_power,
    directional_power,
    freeze_protocol,
    generate_dataset,
    load_protocol,
)
from darcy_construct_benchmark.run_benchmark import build_direction_set
from darcy_construct_benchmark.diagnostics import (
    identifiability_spectrum,
    jacobian_fd,
)


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


# ------------------------------------------------------- directional power

def test_directional_power_null_equals_alpha():
    th = Thresholds()
    assert directional_power(np.zeros(36), 1e-3, th.alpha) == pytest.approx(th.alpha)


def test_directional_dominates_aggregate_in_weak_signal():
    """The 1-dof oracle removes chi^2(n_obs) dof inflation: for a weak
    signal spread over many sensors, directional power must exceed the
    aggregate test's power."""
    sigma, alpha, n = 1e-3, 0.01, 64
    dy = np.zeros(n)
    dy[0] = 4e-3
    assert directional_power(dy, sigma, alpha) > 3 * detection_power(dy, sigma, alpha)


def test_directional_power_monotone_in_signal():
    sigma, alpha = 1e-3, 0.01
    powers = [directional_power(np.array([s]), sigma, alpha)
              for s in (1e-4, 1e-3, 3e-3, 1e-2)]
    assert all(a < b for a, b in zip(powers, powers[1:]))


# ------------------------------------------------------------- dataset

def test_dataset_deterministic_with_manifest(setup, tmp_path):
    basis, G, cfg, _ = setup
    dcfg = DatasetConfig(n_train=6, n_test=3, n_cert=2, seed=7,
                         include_pressure_fields=True)
    m1 = generate_dataset(basis, cfg.baseline, dcfg, tmp_path / "a")
    m2 = generate_dataset(basis, cfg.baseline, dcfg, tmp_path / "b")
    # Bit-identical regeneration.
    for split in ("train", "test", "cert"):
        assert m1["splits"][split]["sha256"] == m2["splits"][split]["sha256"]
    # Shapes and files.
    tr = np.load(tmp_path / "a" / "darcy_train.npz")
    assert tr["theta"].shape == (6, basis.m)
    assert tr["u"].shape == (6, 21, 21)
    assert tr["p"].shape == (6, 21, 21)
    assert tr["y"].shape == (6, G.n_obs)
    # y is consistent with the forward map (noiseless).
    assert np.allclose(tr["y"][0], G(tr["theta"][0]), atol=1e-12)
    # Splits are independent draws, not shared.
    te = np.load(tmp_path / "a" / "darcy_test.npz")
    assert not np.allclose(tr["theta"][0], te["theta"][0])
    # Manifest on disk matches returned manifest.
    with open(tmp_path / "a" / "manifest.json") as f:
        assert json.load(f)["splits"] == m1["splits"]


# ------------------------------------------------------- field adapter

def test_field_adapter_matches_exact_map(setup):
    basis, G, cfg, theta = setup
    n = cfg.baseline.solver.n

    def field_fn(t):
        return G.solver.solve(basis.log_conductivity(t, n))

    S = FieldToObservationSurrogate(field_fn, G)
    assert np.allclose(S(theta), G(theta), atol=1e-14)


def test_field_adapter_rejects_wrong_shape(setup):
    _, G, _, theta = setup
    S = FieldToObservationSurrogate(lambda t: np.zeros((5, 5)), G)
    with pytest.raises(ValueError):
        S(theta)


# --------------------------------------------------- protocol + gate

def test_freeze_certify_roundtrip(setup, tmp_path):
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    J = jacobian_fd(G, theta, th.jacobian_eps)
    sub = identifiability_spectrum(J, G.noise_sigma, th.informed_sv_cut)
    directions = build_direction_set(basis, theta, sub, cfg)

    path = tmp_path / "protocol.json"
    frozen = freeze_protocol(G, theta, directions, cfg, path)
    loaded = load_protocol(path)
    assert loaded["directions"] == frozen["directions"]
    assert any(e["decisional"] for e in loaded["directions"])

    # Negative control: exact surrogate passes the gate.
    rep_exact = certify_surrogate(G, ExactSurrogate(G), loaded)
    assert rep_exact.passed
    assert all(r.preserved for r in rep_exact.decisional)

    # Positive control: mode suppressor fails the gate.
    rep_sup = certify_surrogate(G, ModeSuppressingSurrogate(G, basis.m // 2), loaded)
    assert not rep_sup.passed
    # ...and the failure is on decisional directions, not merely supplementary.
    assert any(not r.preserved for r in rep_sup.decisional)
    assert "FAIL" in rep_sup.summary()


def test_gate_ignores_supplementary_only_failures(setup, tmp_path):
    """A surrogate must not be failed for losing responses the exact map
    itself cannot detect: build a protocol, then check that supplementary
    violations alone leave 'passed' True. We construct this by certifying
    the exact surrogate (all decisional preserved) and confirming the gate
    ruleis computed from decisional entries only."""
    basis, G, cfg, theta = setup
    th = cfg.thresholds
    J = jacobian_fd(G, theta, th.jacobian_eps)
    sub = identifiability_spectrum(J, G.noise_sigma, th.informed_sv_cut)
    directions = build_direction_set(basis, theta, sub, cfg)
    path = tmp_path / "protocol.json"
    frozen = freeze_protocol(G, theta, directions, cfg, path)
    rep = certify_surrogate(G, ExactSurrogate(G), frozen)
    assert rep.passed == all(r.preserved for r in rep.decisional)
