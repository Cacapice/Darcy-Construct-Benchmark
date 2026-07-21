# Darcy-Flow Construct-Validity Benchmark (Work Package 2)

A benchmark for surrogate-based inverse problems in which the true coefficient
is known by construction: log-conductivity Darcy flow on the unit square with a
Matérn/KL parameterization.

## The object being measured

| Concept | Mathematical object | Role |
|---|---|---|
| **Inferential fidelity** | the two functionals below, evaluated against the exact map under experiment E | quantity measured |
| — information aspect | EIG ratio 𝓘(G̃,G), EIG = ½ log det(I + JᵀJ/σ²); spectral refinement: per-direction posterior-variance ratios (bits lost) | how much information survives — rotation-invariant, hence necessary but not sufficient |
| — parameter-alignment aspect | minimal-pair response fidelity: attenuation and relative response error per preregistered direction | whether surviving information remains attached to the correct coefficients — the independent axis |
| **Identifiability** | detection power ≥ preregistered floor at (δ, α, σ) | derived property; the scientific claim |
| **Certification** | frozen decision rule on the exact map's detectable set | operational outcome; PASS/FAIL is one summary of the continuous quantities |

Two structural facts organize the table. The posterior-geometry quantities
are the *spectral refinement* of the scalar information functional (bits
lost per direction = log₂ posterior-std ratio), not a third independent
component. And the two aspects are genuinely independent in one direction
only: the information aspect is provably blind to output rotations
(item 13 below), so parameter alignment cannot be recovered from it —
which is why both are measured. All statements are indexed by the experiment tuple

    E = (Θ_m, prior, G_m, 𝒪, noise model)

— identifiability is a property of E, not of the forward map alone (the
nuisance sweep demonstrates this: changing the forcing changes which
directions are identifiable). The experiment plays a deliberate dual role,
made explicit in `config.Experiment` (alias of `NuisanceConfig`): it is
*nuisance* to the semantic meaning of θ, and *constitutive* of the
identifiability of θ.

The diagnostics decompose into two complementary layers:

* **Information aspect** — `information_functional` computes the
  linearized Lindley information 𝓘(G̃,G) = EIG(G̃)/EIG(G) with
  EIG = ½ log det(I + JᵀJ/σ²), its total bits lost, and its spectral
  decomposition along exact-map directions. Baseline run: suppressor 40.6%
  preserved (74.1 bits lost); **shortcut 96.7% preserved (4.1 bits lost)
  yet gate-FAILED** — the EIG ratio is blind to output rotations, so a
  surrogate can keep the information while redirecting it toward the
  wrong coefficients.
* **Parameter-alignment aspect** — the H0_preserve minimal-pair nulls test
  whether the preserved information is still *about the right
  coefficients*. Information preservation is necessary for trustworthy
  inversion; parameter alignment makes it sufficient at the tested
  directions. Neither aspect subsumes the other, and the shortcut
  surrogate is the constructed witness.

**Aggregation dilution, operationalized.** The statistical phenomenon is
well established: at fixed noncentrality, the power of a level-α omnibus
χ²(n) test is strictly decreasing in n and → α, while a matched-direction
(Neyman–Pearson-type) test's power is invariant — elementary from the
monotonicity of noncentral-χ² power in df, enforced numerically in
`test_aggregation_dilution_proposition`. No novelty is claimed for the
proposition. The benchmark's contribution is to **isolate and
operationalize it within surrogate certification for inverse problems**,
where — under the stated assumptions (fixed response Δy, appended sensors
carrying no signal, this omnibus test) — it would otherwise masquerade as
non-identifiability. Within this setting, the baseline gap — 16/32
identifiable under the aggregate test vs 23/32 under the matched-direction
bound — reflects test choice under sensor aggregation, not loss of
information about the coefficient.

**Terminology.** The supplementary per-direction bound is the
**matched-direction test** (`matched_direction_power`; the earlier name
`directional_power` remains as a deprecated alias). Against the realized
simple alternative it is the Neyman–Pearson-type benchmark (two-sided,
hence slightly conservative relative to the one-sided NP test).

## Narrative map (feature → work package)

Every feature traces to exactly one objective:

* **WP1 — Define identifiability.** `config` (thresholds, admissible region,
  Experiment), `kl` (target parameterization, semantic invariance),
  `perturbations` (minimal-pair operators), `protocol.freeze_protocol`
  (the definition, frozen as an artifact).
* **WP2 — Measure identifiability.** `diagnostics` (detection power,
  matched-direction bound, identifiability spectrum I_j(δ,σ), posterior
  geometry, information functional), `dataset` (measurement material,
  incl. confounded training distributions), `solver`/`forward` (the exact
  reference the measurements are against).
* **WP3 — Certify surrogate-based inversion.** `protocol.certify_surrogate`
  (the gate), `surrogates` (negative/positive/confound controls; the
  field adapter that admits real operator networks), posterior-std and
  bits-lost quantities (the bridge to posterior-perturbation bounds).

Features that cannot be traced to one of these three do not belong in the
benchmark. It operationalizes the construct-validity framework of Technical Note 1 — the estimator-independent layer (parameterization,
nuisance definitions, perturbation operators, null hypotheses, thresholds) is
fixed up front, while representational transfer to any particular surrogate is
tested empirically, never assumed.

## Mathematical conclusions demonstrated by the test suite

Each statement below is verified by the named test(s). Their epistemic
status differs, and the grouping makes it explicit: **[A]** exact algebraic
identities, verified to machine precision — the proof is elementary and the
test guards the implementation against regression; **[S]** analytic
statistical facts, verified numerically at stated instances; **[C]**
discretization-convergence claims, verified at stated resolution and
tolerance; **[W]** existence results, where the test itself is the witness.
No test proves a theorem; tests verify identities, instances, and witnesses.

**Forward map and parameterization**

1. **[C]** The discrete solution of −∇·(κ∇p) = f with κ ≡ 1, f ≡ 1 agrees
   with the Fourier-series solution of the Poisson problem to max nodal
   error < 5·10⁻⁴ at n = 41, satisfies the discrete maximum principle
   (p > 0 in the interior for f > 0), and preserves the problem's symmetry
   (`test_solver_matches_analytic_constant_kappa`).
2. **[A]** The discrete solution map is homogeneous of degree −1 in κ:
   p[cκ] = p[κ]/c exactly, by linearity of the discrete operator in the
   transmissibilities (`test_solver_scaling_in_kappa`).
3. **[A]** The discrete KL eigenpairs are orthonormal in the weighted ℓ²
   inner product, ⟨φᵢ, φⱼ⟩_w = δᵢⱼ, with nonincreasing eigenvalues; the map
   θ ↦ u(·;θ) − ū is exactly linear
   (`test_kl_orthonormality_and_truncation`, `test_field_linearity_in_theta`).
4. **[A]** Semantic invariance of θ under mesh nuisance: evaluation of the
   KL field commutes with mesh refinement on shared nodes — u_fine
   restricted to the coarse grid equals u_coarse (to 10⁻⁸, the
   interpolator's identity on nodes)
   (`test_theta_meaning_invariant_under_mesh_nuisance`).
5. **[A]** T_{j,δ} changes exactly one coordinate by exactly δ; under the
   admissible ball Θ_m^(a), the applied intervention (after sign flip or
   magnitude reduction) yields a pair with both members admissible
   (`test_minimal_pair_*`).

**Statistical calibration of the diagnostics**

6. **[S]** Size equals level by construction: at Δy = 0 both the aggregate
   χ²(n_obs) test and the matched-direction test report power exactly α —
   the pipeline cannot manufacture detections
   (`test_zero_perturbation_power_equals_alpha`,
   `test_directional_power_null_equals_alpha`).
7. **[S]** Matched-direction power is strictly increasing in ‖Δy‖/σ
   (`test_directional_power_monotone_in_signal`).
8. **[S]** **Aggregation dilution (well-established phenomenon,
   operationalized here).** At fixed noncentrality λ = ‖Δy‖²/(2σ²) > 0,
   the level-α noncentral-χ²(n) power is strictly decreasing in n
   (verified over n ∈ {16, 64, 256, 1024}, approaching α), while
   matched-direction power is invariant in n; the proof is the one-line
   monotonicity of noncentral-χ² power in df, and the test pins the
   implementation to it. Scope: under these assumptions — fixed Δy,
   appended sensors carrying no signal, this omnibus test — loss of
   omnibus power does not indicate loss of information, so the baseline
   16/32-vs-23/32 gap is an instance of test choice within this setting.
   The contribution is the operationalization in surrogate certification,
   not the proposition
   (`test_aggregation_dilution_proposition`,
   `test_directional_dominates_aggregate_in_weak_signal`).

**Gate correctness relative to constructed controls**

9. **[A]** Reflexivity: G̃ = G implies attenuation ≡ 1, relative response
   error ≡ 0, all preservation nulls survive, 𝓘 = 1, and zero bits lost —
   the gate and both functionals have no false positives on the identity
   (`test_exact_surrogate_preserves_everything`,
   `test_geometry_exact_is_identity`, `test_information_identity_for_exact`).
10. **[A]** If a surrogate is constant in coefficient j (zero Jacobian
    column), its minimal-pair response along e_j is exactly zero and the
    corresponding preservation null is violated; the frozen protocol's JSON
    round-trip is the identity; and the verdict is a function of the
    decisional entries only — a surrogate cannot be failed on directions
    the exact experiment cannot itself resolve
    (`test_mode_suppressing_surrogate_caught`,
    `test_freeze_certify_roundtrip`,
    `test_gate_ignores_supplementary_only_failures`).

**Information and posterior geometry**

11. **[A/W]** Information monotonicity: with matched base points, zeroing
    Jacobian columns Loewner-dominates the posterior precision, so
    posterior variance cannot decrease along any direction and bits lost
    are nonnegative — exact in the linearized setting. Under the nonlinear
    map the suppressor's effective base point shifts, so the domination is
    approximate; the tests assert per-direction posterior-std ratios > 0.9
    and per-direction bits lost > −0.2 (the slack quantifies the base-point
    effect), and total bits lost increasing strictly as the suppressed
    coordinate set grows (nested case)
    (`test_geometry_detects_collapse_beyond_pass_fail`,
    `test_information_monotone_under_suppression`).
12. **[A]** Basis dependence of first-order attenuation: exact coordinate
    collapse (coordinate attenuation ≡ 0 on suppressed coefficients) can
    coexist with singular-direction attenuation bounded away from zero,
    because exact-map singular directions mix coordinates — both bases are
    therefore reported, and neither suffices alone
    (`test_geometry_detects_collapse_beyond_pass_fail`).
13. **[A]** Rotation invariance of the information functional: for any
    orthogonal Q, EIG(QJ) = EIG(J) exactly, since (QJ)ᵀ(QJ) = JᵀJ, while
    ‖QJ − J‖ can be large. Hence 𝓘(G̃, G) is blind to redirected
    information: information preservation is necessary, not sufficient, and
    the minimal-pair response-direction layer is not redundant with it
    (`test_information_capacity_blind_to_alignment`).
14. **[W]** Local pathology, global degradation: in the test fixture,
    suppressing half the coordinates inflates posterior std along *every*
    exact-map singular direction (all ratios > 1, max > 3)
    (`test_geometry_detects_collapse_beyond_pass_fail`).

**The accuracy–identifiability gap (existence results)**

15. **[W]** There exists a surrogate whose relative forward RMSE on its own
    training distribution is < 5% (baseline configuration: 1.1%, RMS error
    at 0.65× the noise floor) and which nevertheless fails the
    certification gate, with the response to one exact-map-identifiable
    coefficient annihilated exactly and the correlated coefficient's
    response direction distorted (relative error > 0.01 at attenuation
    ≈ 1). Accuracy in the training-measure norm therefore does not control
    identifiability; the witness is constructed, not trained, so the gap is
    exact rather than an optimization artifact
    (`test_shortcut_accurate_on_training_distribution`,
    `test_shortcut_fails_gate_on_ignored_component`,
    `test_shortcut_distorts_used_component`,
    `test_information_shortcut_loses_capacity_and_alignment`).
16. **[A/W]** The confounded sampler realizes the stated joint law —
    corr(θ_a, θ_b) ≈ r on train/test with unit marginals preserved, while
    the certification split remains prior-distributed — so minimal pairs
    evaluated at certification necessarily realize coefficient combinations
    off the training correlation manifold, which is the mechanism by which
    the gate sees what training-measure accuracy cannot
    (`test_dataset_correlation_train_only`).

**Reproducibility as a verified property**

17. **[A]** Dataset generation is a deterministic function of
    (prior, experiment, dataset config): regeneration is bit-identical
    (SHA-256 equality), stored observations satisfy y = G(θ) exactly, and
    the field-surrogate adapter satisfies the composition identity
    O(p̂(θ)) = G(θ) when p̂ is the exact solver
    (`test_dataset_deterministic_with_manifest`,
    `test_field_adapter_matches_exact_map`).

## Mapping from the framework to the code

| Framework element | Module | Object |
|---|---|---|
| Coefficient parameterization θ ∈ Θ_m | `kl.py` | `KLBasis` (Matérn eigenpairs, truncation by preregistered ρ) |
| Admissible region Θ_m or Θ_m^(a) | `config.py`, `perturbations.py` | `radius_a`, `is_admissible` |
| Nuisance variable definition | `config.py` | `NuisanceConfig` (mesh, forcing, sensors, noise) |
| Forward map G_m | `solver.py`, `forward.py` | `DarcySolver`, `ForwardMap` |
| Minimal-pair operators T_{j,δ}, T_{S,δv} | `perturbations.py` | `coordinate_pair`, `make_minimal_pair` |
| Null hypotheses H0_global, H0_comp, H0_preserve | `diagnostics.py` | docstring + `evaluate_direction`, `evaluate_preservation` |
| Diagnostic metrics | `diagnostics.py` | detection power, whitened-Jacobian spectrum, attenuation / relative response error |
| Preregistered thresholds | `config.py` | `Thresholds` |
| Surrogate slot G̃_m | `surrogates.py` | any `theta -> y` callable |

## Design commitments

1. **The semantic meaning of θ is invariant to nuisance transformations.**
   Not θ's value — what each coefficient *represents*. KL modes are computed
   once on a reference grid and interpolated to whatever mesh a nuisance
   variant uses, so θ_j indexes the same spatial mode under every mesh,
   forcing, and sensor configuration
   (`test_theta_meaning_invariant_under_mesh_nuisance`). This is the
   construct-validity commitment imported from the interpretability
   framework: nuisance transformations may change what is *observable*
   about the construct, never what the construct *is*.
2. **No post-hoc direction selection.** The direction set (leading and
   truncation-boundary coefficients, low/high-frequency blocks, seeded random
   directions, data-informed and prior-dominated directions) is fixed before
   any surrogate is evaluated. Data-informed directions come from the exact
   map's Jacobian, not from surrogate behavior.
3. **Built-in controls.** δ = 0 yields power exactly α (the pipeline cannot
   manufacture detections). `ExactSurrogate` must pass every preservation
   null; `ModeSuppressingSurrogate` must fail on suppressed components. Both
   are enforced by tests, so the diagnostics are themselves validated before
   any learned surrogate is scored.

## v0.3 — information geometry, identifiability spectrum, shortcut control

Motivated by review feedback: identifiability can survive as pass/fail while
the *geometry* of the inverse problem degrades — both maps injective, but one
derivative direction collapsed 100-fold changes optimization conditioning,
posterior geometry, and credible intervals. v0.3 measures that directly.

| Addition | Module | Purpose |
|---|---|---|
| Identifiability spectrum | `diagnostics.identifiability_profile` | I_j(δ,σ) = detection power per coefficient as a curve (aggregate + directional oracle), analogous to singular-value decay. Replaces the binary cut point — correctly, since the baseline profile is non-monotone (j=13 fails while j=14–16 pass). Descriptive; thresholds still decide. |
| Posterior geometry | `diagnostics.posterior_geometry` | Along exact-map singular directions v_i and coordinates e_j: derivative attenuation and Laplace posterior-std ratios (KL prior N(0,I)). Ratio ≫ 1 = information surrendered to the prior — the Bayesian signature of geometric identifiability loss, and the bridge quantity to posterior-perturbation bounds (RQ2). |
| Shortcut control | `surrogates.ShortcutSurrogate` | Imputes θ_b from θ_a via the training-distribution regression E[θ_b|θ_a]=rθ_a. **On its own training distribution: relative forward RMSE ≈ 1.1%, RMS error 0.65× the noise floor. Gate: FAIL (16/21), e_b attenuation exactly 0.** Unlike the mode suppressor, nothing is broken relative to its training objective — imputation is risk-minimizing on that distribution; the pathology is the distribution, exposed only because minimal pairs realize coefficient combinations off the correlation manifold. |
| Correlated training data | `dataset.DatasetConfig.theta_correlation` | Confounded train/test distributions for studying shortcut learning in real operator networks. The certification split always stays prior-distributed — otherwise the shortcut is undetectable in principle. |

**Protocol change (logged):** the frozen protocol now includes all coordinate
directions e_0..e_{m-1}, per the framework document's Level-2 definition of
componentwise recoverability for *each* coefficient. v0.2's protocol omitted
e_3..e_{m-4}; corrected before any learned surrogate exists. Post-training,
this same change would require a formal amendment.

A geometry finding from the baseline run: the mode suppressor (a
coordinate-local pathology, k=8 of 32) inflates posterior std along exact-map
directions up to 45×, and the shortcut (two coefficients confounded) up to
31× — local representational failures produce global posterior-geometry
degradation. Attenuation and rel-error are complementary, not redundant: the
shortcut's e_1 response has attenuation 0.998 (magnitude preserved) with
rel_err 0.297 (direction rotated by the spurious imputed sensitivity).

## v0.2 — next-phase machinery (learned-surrogate certification)

| Addition | Module | Purpose |
|---|---|---|
| Directional oracle power | `diagnostics.directional_power` | 1-dof test on the exact response direction; supplementary upper bound on per-direction power, removing χ²(n_obs) dof inflation. **Not decisional** — the preregistered aggregate test still decides; promoting it would be a protocol amendment. |
| Dataset generator | `dataset.generate_dataset` | Train / test / **certification** splits from independent seed streams; bit-identical regeneration; `manifest.json` with configs, version, and SHA-256 hashes. `y` stored noiseless (noise added downstream). |
| Frozen protocol | `protocol.freeze_protocol` | Serializes thresholds, configs, θ*, and the direction set **with each direction's exact-map decisional status** to JSON *before* any surrogate is trained. The file is the preregistration. |
| Certification gate | `protocol.certify_surrogate` | PASS iff H0_preserve survives on every decisional direction; supplementary directions reported but cannot fail a surrogate (the observation design cannot resolve them for the exact map either). |
| Field adapter | `surrogates.FieldToObservationSurrogate` | Wraps DeepONet/FNO-style field-valued predictors (θ → pressure field) with the benchmark's own observation operator, so no sensor-placement mismatch can leak in. |

Generating the public dataset:

```python
from darcy_construct_benchmark import *
basis = KLBasis(PriorConfig())
generate_dataset(basis, NuisanceConfig(), DatasetConfig(), "dataset/")
```

Empirical note from the v0.2 baseline run: `block_high` has aggregate power
0.117 but directional oracle power 0.849 — above the preregistered 0.8 floor.
Under the oracle test it *would* be identifiable; under the frozen aggregate
test it is not. The discrepancy is reported, not resolved: changing which
test decides is an amendment, not a code edit.

## Usage

```bash
python -m darcy_construct_benchmark.run_benchmark   # full report
python -m pytest darcy_construct_benchmark/tests/   # 13 tests
```

To evaluate a learned surrogate, implement `__call__(theta) -> y` matching the
`ForwardMap` observation layout and pass it through `evaluate_preservation`
over the preregistered direction set (see `run_benchmark.py`, step 5).

## Interpretation notes and known limitations

* **Local vs. finite-perturbation identifiability differ, and both are
  reported.** At σ = 10⁻³ the whitened Jacobian can have full informed rank
  while the finite-δ two-sample test retains H0 for late components: the
  test's χ²(n_obs) null inflates the detection threshold relative to the
  per-direction linearized signal. Neither number is "the" identifiability;
  the benchmark deliberately reports both.
* Attenuation for retained components under the mode-suppressing surrogate is
  not exactly 1: the surrogate evaluates at a different base point (suppressed
  coordinates zeroed), so responses agree only up to the nonlinearity of
  exp(u). The preregistered band absorbs this; the code does not hide it.
* The KL quadrature uses uniform nodal weights (documented approximation);
  eigenpairs are those of the discretized operator, and totals in the ρ
  criterion are discrete sums, not the analytic trace.
* The solver is validated against the Fourier-series solution for constant κ
  (max error < 5·10⁻⁴ at n = 41); heterogeneous cases are verified only
  through scaling/positivity/symmetry properties, not manufactured solutions.
* `bump_forcing` changes the observations entirely; decision flips there
  measure sensitivity of identifiability to experimental design, not error.
