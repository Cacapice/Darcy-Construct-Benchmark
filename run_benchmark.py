"""Run the full construct-validity benchmark with a known ground truth.

Usage:
    python -m darcy_construct_benchmark.run_benchmark

Pipeline
--------
1. Build the KL basis (target parameter space Theta_m; m fixed by the
   preregistered variance proportion rho).
2. Draw a known ground truth theta* (seeded; projected into the
   admissible ball if one is configured).
3. Assemble the preregistered direction set: leading coefficients,
   truncation-boundary coefficients, low/high-frequency blocks, random
   directions, data-informed and prior-dominated directions (the latter
   two derived from the exact map's Jacobian, not from any surrogate).
4. Componentwise / subspace identifiability of the exact map (H0_comp,
   H0_global), plus the delta=0 negative control.
5. Surrogate preservation (H0_preserve) for the exact surrogate
   (negative control) and a mode-suppressing surrogate (positive control).
6. Nuisance sweep: repeat step 4 under mesh / forcing / sensor variants;
   report decision flips.
"""

from __future__ import annotations

import numpy as np

from .config import BenchmarkConfig, NuisanceConfig, ObservationConfig, SolverConfig
from .diagnostics import (
    DirectionResult,
    identifiability_profile,
    information_functional,
    posterior_geometry,
    decision_flips,
    detection_power,
    identifiability_spectrum,
    jacobian_fd,
    evaluate_direction,
    evaluate_preservation,
)
from .forward import ForwardMap
from .kl import KLBasis
from .perturbations import coordinate_pair, make_minimal_pair
from .surrogates import ExactSurrogate, ModeSuppressingSurrogate, ShortcutSurrogate


def build_direction_set(basis, theta_star, subspaces, cfg: BenchmarkConfig):
    """Preregistered directions; returns list of (label, unit_vector)."""
    m = basis.m
    rng = np.random.default_rng(777)  # fixed seed: random directions are preregistered
    dirs = []

    # Individual leading coefficients and truncation-boundary coefficients.
    for j in range(min(3, m)):
        dirs.append((f"e_{j}", np.eye(m)[j]))
    for j in range(max(0, m - 3), m):
        dirs.append((f"e_{j}", np.eye(m)[j]))

    # Low- and high-frequency blocks (uniform combinations).
    half = m // 2
    lo = np.zeros(m); lo[:half] = 1.0
    hi = np.zeros(m); hi[half:] = 1.0
    dirs.append(("block_low", lo / np.linalg.norm(lo)))
    dirs.append(("block_high", hi / np.linalg.norm(hi)))

    # Random directions.
    for k in range(3):
        v = rng.standard_normal(m)
        dirs.append((f"random_{k}", v / np.linalg.norm(v)))

    # Data-informed and prior-dominated directions (from the exact map only).
    if subspaces.rank_informed > 0:
        dirs.append(("informed_top", subspaces.informed[:, 0]))
    if subspaces.rank_informed < m:
        dirs.append(("prior_dom_last", subspaces.prior_dominated[:, -1]))

    # De-duplicate labels (leading/boundary overlap when m <= 6).
    seen, out = set(), []
    for label, v in dirs:
        if label not in seen:
            seen.add(label)
            out.append((label, v))
    return out


def run(cfg: BenchmarkConfig | None = None, verbose: bool = True) -> dict:
    cfg = cfg or BenchmarkConfig()
    th = cfg.thresholds
    log = print if verbose else (lambda *a, **k: None)

    # 1. Target parameter space -------------------------------------------
    basis = KLBasis(cfg.prior)
    log(f"KL truncation: m = {basis.m}  "
        f"(achieved rho = {basis.achieved_rho:.4f}, "
        f"requested {cfg.prior.rho}, met = {basis.requested_rho_met})")

    # 2. Known ground truth ------------------------------------------------
    rng = np.random.default_rng(cfg.theta_seed)
    theta_star = basis.sample_theta(rng)
    if cfg.radius_a is not None:
        nrm = np.linalg.norm(theta_star)
        if nrm > 0.9 * cfg.radius_a:
            theta_star *= 0.9 * cfg.radius_a / nrm
    log(f"ground truth ||theta*|| = {np.linalg.norm(theta_star):.3f}"
        + (f"  (admissible ball a = {cfg.radius_a})" if cfg.radius_a else "  (Theta_m = R^m)"))

    G = ForwardMap(basis, cfg.baseline)
    sigma = G.noise_sigma
    log(f"baseline nuisance: n = {cfg.baseline.solver.n}, "
        f"forcing = {cfg.baseline.solver.forcing}, "
        f"n_obs = {G.n_obs}, sigma = {sigma:g}\n")

    # 3. Local identifiability spectrum & preregistered directions ---------
    J = jacobian_fd(G, theta_star, th.jacobian_eps)
    sub = identifiability_spectrum(J, sigma, th.informed_sv_cut)
    log(f"whitened singular values (top 5): "
        f"{np.array2string(sub.singular_values[:5], precision=2)}")
    log(f"data-informed rank (sv > {th.informed_sv_cut}): "
        f"{sub.rank_informed} of {basis.m}\n")

    directions = build_direction_set(basis, theta_star, sub, cfg)

    # 4. Identifiability of the exact map ---------------------------------
    log("=== H0_comp / H0_global: identifiability of the exact map ===")
    log("(decision uses the aggregate test; 'matched' is the supplementary")
    log(" matched-direction (NP-type) upper bound and is NOT decisional)")
    log(f"{'direction':<16}{'delta':>7}{'||dy||':>12}{'power':>8}{'matched':>9}  decision")
    id_results: list[DirectionResult] = []
    for label, v in directions:
        pair = make_minimal_pair(theta_star, v, th.delta, cfg.radius_a, label)
        r = evaluate_direction(G, pair, sigma, th)
        id_results.append(r)
        log(f"{r.label:<16}{r.delta:>7.2f}{r.response_norm:>12.3e}{r.power:>8.3f}"
            f"{r.power_directional:>9.3f}  "
            + ("REJECT H0 (identifiable)" if r.reject_h0 else "retain H0"))

    null_power = detection_power(np.zeros(G.n_obs), sigma, th.alpha)
    log(f"\nnegative control (delta = 0): power = {null_power:.3f} "
        f"(equals alpha = {th.alpha} by construction)\n")

    # Identifiability spectrum I_j(delta, sigma) over all m coefficients:
    # a curve, not a cut point. Decisions still use the aggregate column
    # against the preregistered power floor.
    profile = identifiability_profile(G, theta_star, sigma, th, cfg.radius_a)
    n_ident = int(np.sum(profile[:, 0] >= th.power_min))
    log(f"identifiability spectrum I_j(delta={th.delta}, sigma={sigma:g}) "
        f"[aggregate | matched-direction]:")
    for start in range(0, basis.m, 8):
        blk = profile[start:start + 8]
        log("  j=%2d..%2d  agg: %s" % (start, start + len(blk) - 1,
            " ".join(f"{p:5.2f}" for p in blk[:, 0])))
        log("            dir: %s" % " ".join(f"{p:5.2f}" for p in blk[:, 1]))
    log(f"{n_ident}/{basis.m} coefficients cross the preregistered power floor "
        f"(aggregate test); "
        f"{int(np.sum(profile[:,1] >= th.power_min))}/{basis.m} under the "
        f"matched-direction test (supplementary)\n")

    # 5. Freeze protocol, then certify surrogates --------------------------
    from .protocol import certify_surrogate, freeze_protocol

    # Protocol direction set = curated preregistered directions UNION all
    # coordinate directions e_0..e_{m-1}. Componentwise identifiability of
    # every coefficient is Level-2 recoverability in the framework document;
    # v0.2's frozen protocol omitted e_3..e_{m-4} and this is corrected here
    # BEFORE any learned surrogate exists (logged as protocol change v0.3).
    seen = {label for label, _ in directions}
    protocol_directions = list(directions) + [
        (f"e_{j}", np.eye(basis.m)[j]) for j in range(basis.m)
        if f"e_{j}" not in seen
    ]
    protocol_path = "frozen_protocol.json"
    protocol = freeze_protocol(G, theta_star, protocol_directions, cfg, protocol_path)
    n_dec = sum(e["decisional"] for e in protocol["directions"])
    log(f"protocol frozen -> {protocol_path}  "
        f"({n_dec}/{len(protocol['directions'])} directions decisional)\n")

    k_keep = max(1, basis.m // 4)
    j_used, j_ignored, r_corr = 1, 8, 0.98
    shortcut = ShortcutSurrogate(G, j_used=j_used, j_ignored=j_ignored, corr=r_corr)
    surrogates = {
        "exact (negative control)": ExactSurrogate(G),
        f"mode-suppressing k={k_keep} (positive control)":
            ModeSuppressingSurrogate(G, k_keep),
        f"shortcut e_{j_used}~e_{j_ignored}, r={r_corr} (confound control)": shortcut,
    }

    # Shortcut accuracy demonstration: on ITS OWN training distribution
    # (theta_b = r theta_a + sqrt(1-r^2) z), forward error is small --
    # the point is that aggregate accuracy cannot see the confound.
    rng_d = np.random.default_rng(99)
    n_demo = 128
    T = rng_d.standard_normal((n_demo, basis.m))
    T[:, j_ignored] = r_corr * T[:, j_used] + np.sqrt(1 - r_corr**2) * T[:, j_ignored]
    err2 = sig2 = 0.0
    for i in range(n_demo):
        yi = G(T[i]); yi_s = shortcut(T[i])
        err2 += float(np.sum((yi_s - yi) ** 2)); sig2 += float(np.sum(yi ** 2))
    rms_err = np.sqrt(err2 / n_demo); rms_sig = np.sqrt(sig2 / n_demo)
    noise_rms = sigma * np.sqrt(G.n_obs)
    log("shortcut surrogate on its training distribution "
        f"(n={n_demo}, corr(theta_{j_used}, theta_{j_ignored})={r_corr}):")
    log(f"  relative forward RMSE = {rms_err / rms_sig:.4f}   "
        f"RMS error / noise RMS = {rms_err / noise_rms:.2f}")
    log("  (aggregate accuracy metrics look fine; watch the gate below)\n")
    preservation = {}
    certifications = {}
    for name, S in surrogates.items():
        log(f"=== H0_preserve: {name} ===")
        log(f"{'direction':<16}{'atten':>8}{'rel_err':>9}  verdict")
        rows = []
        for label, v in directions:
            pair = make_minimal_pair(theta_star, v, th.delta, cfg.radius_a, label)
            p = evaluate_preservation(G, S, pair, th)
            rows.append(p)
            log(f"{p.label:<16}{p.attenuation:>8.3f}{p.rel_error:>9.3f}  "
                + ("preserved" if p.preserved else "VIOLATED"))
        preservation[name] = rows
        report = certify_surrogate(G, S, protocol)
        certifications[name] = report
        log(report.summary())
        J_sur = jacobian_fd(S, theta_star, th.jacobian_eps)
        geom = posterior_geometry(J, J_sur, sigma)
        info = information_functional(J, J_sur, sigma)
        worst_c = int(np.nanargmin(geom.coordinate_attenuation))
        infl = int(np.nanargmax(geom.posterior_std_ratio))
        log("geometry at theta*: coordinate deriv attenuation "
            f"min = {np.nanmin(geom.coordinate_attenuation):.3f} (e_{worst_c}); "
            f"max posterior-std inflation = "
            f"{np.nanmax(geom.posterior_std_ratio):.1f}x "
            f"(exact-map direction v_{infl})")
        log(f"information: exact {info.eig_exact_nats/np.log(2):.1f} bits -> "
            f"surrogate {info.eig_surrogate_nats/np.log(2):.1f} bits  "
            f"({100*info.preserved_fraction:.1f}% preserved, "
            f"{info.bits_lost_total:.1f} bits lost; "
            f"worst direction v_{int(np.nanargmax(info.bits_lost_by_direction))}: "
            f"{np.nanmax(info.bits_lost_by_direction):.1f} bits)")
        log("")

    # 6. Nuisance sweep ----------------------------------------------------
    variants = [
        NuisanceConfig(label="finer_mesh",
                       solver=SolverConfig(n=49, forcing=cfg.baseline.solver.forcing),
                       observation=cfg.baseline.observation),
        NuisanceConfig(label="bump_forcing",
                       solver=SolverConfig(n=cfg.baseline.solver.n,
                                           forcing="gaussian_bump"),
                       observation=cfg.baseline.observation),
        NuisanceConfig(label="sparser_sensors",
                       solver=cfg.baseline.solver,
                       observation=ObservationConfig(
                           n_sensors_per_side=6,
                           noise_sigma=cfg.baseline.observation.noise_sigma)),
    ]
    log("=== nuisance sweep: decision stability of H0_comp / H0_global ===")
    nuisance_report = {}
    for var in variants:
        Gv = ForwardMap(basis, var)
        rows = []
        for label, v in directions:
            pair = make_minimal_pair(theta_star, v, th.delta, cfg.radius_a, label)
            rows.append(evaluate_direction(Gv, pair, Gv.noise_sigma, th))
        flips = decision_flips(id_results, rows)
        nuisance_report[var.label] = flips
        log(f"{var.label:<18} decision flips vs baseline: "
            + (", ".join(flips) if flips else "none"))
    log("\nnote: under 'bump_forcing' the observations change entirely; "
        "flips there measure sensitivity of *identifiability* to the "
        "experimental design, not error. The target parameter theta is "
        "identical across all variants by construction (shared KL basis).")

    return {
        "protocol": protocol,
        "certifications": certifications,
        "basis": basis,
        "theta_star": theta_star,
        "subspaces": sub,
        "identifiability": id_results,
        "identifiability_profile": profile,
        "preservation": preservation,
        "nuisance_flips": nuisance_report,
    }


if __name__ == "__main__":
    run()
