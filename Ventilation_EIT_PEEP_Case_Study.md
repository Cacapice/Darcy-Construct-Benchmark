# Technical Note — Ventilation EIT for PEEP Titration:
# Target Parameter Space and a Concrete Clinical Case Study

This note translates the construct-validity framework to bedside
ventilation-EIT, mirroring the Darcy target-parameter-space document and
connecting directly to the certification gate. It closes with a
constructed clinical vignette showing where a shortcut-learning failure
would manifest as a wrong PEEP decision. Physiological ranges below are
illustrative — typical of the ARDS/EIT literature (Costa et al.
overdistension–collapse methodology; Frerichs et al. consensus on clinical
EIT) — not a specific trial's reported data; the vignette is constructed
for the same reason the benchmark's `ShortcutSurrogate` is constructed
rather than trained: it is a designed witness, not an empirical claim.

**Two references incorporated in this revision, used for different
purposes and at different evidentiary weight:**

* Semler et al., *Oxygen-Saturation Targets for Critically Ill Adults
  Receiving Mechanical Ventilation* (PILOT trial), *N Engl J Med* 2022;
  387:1759–1769 — a peer-reviewed pragmatic RCT (n = 2,541), used in §8 as
  a structural precedent for why an aggregate signal and an aggregate
  outcome can be jointly underpowered to detect regionally heterogeneous
  physiology, and to fix the field's accepted primary outcome measure
  (ventilator-free days through day 28) against which any EIT-driven
  clinical claim should be benchmarked.
* Ring, *Electrical Impedance Technology (EIT): Research & Clinical
  Applications* (CME lecture; course objectives: mechanism of EIT
  monitoring, research-tool vs. bedside-clinical-adjunct use, role across
  the continuum of care for mechanically ventilated patients) — an
  educational source, not a primary study; used in §2 and new §7 only for
  its stated framing (the research/bedside distinction and the
  continuum-of-care structure), which independently motivates two design
  decisions this note already needed to make explicit. No specific
  findings are attributed to this source beyond its stated objectives, as
  its content beyond them has not been reviewed.

## 1. Domain and forward model

Let Ω ⊂ ℝ² (thoracic cross-section at the electrode plane, typically the
4th–5th intercostal space) or Ω ⊂ ℝ³ (multi-plane / volumetric EIT). The
complete electrode model (CEM) replaces the Darcy PDE:

    −∇·(σ(x)∇u(x)) = 0   in Ω
    u(x) + z_ℓ σ(x) ∂u/∂n(x) = U_ℓ   on electrode ℓ
    ∫_{e_ℓ} σ ∂u/∂n ds = I_ℓ,   Σ I_ℓ = 0   (current conservation)
    σ ∂u/∂n = 0   on ∂Ω \ ∪ℓ e_ℓ   (insulating boundary elsewhere)

with σ(x) > 0 the tissue conductivity field, z_ℓ the (measurable,
nuisance) contact impedance at electrode ℓ, and (I_ℓ, U_ℓ) the injected
current and measured voltage at each of L electrodes (typically L = 16 or
32, mounted as a belt). This is the same elliptic coefficient inverse
problem as Darcy flow — σ plays the role of κ — with discrete
electrode-pair injection replacing continuous forcing, and boundary
voltage replacing interior pressure sensors.

## 2. Target parameter: regional, not spectral

For static conductivity recovery (tumor-EIT), a KL/Matérn expansion of
log σ is the natural target, exactly as in the Darcy case. For
**ventilation** EIT, this is the wrong primary parameterization, for a
reason worth stating plainly: clinicians do not act on spatial-mode
coefficients, they act on **regional physiological summaries** computed
from the time series of reconstructed images. The target parameter space
is therefore redefined as a physiological coordinate vector rather than a
spectral one:

    θ = (θ_1, ..., θ_R) ∈ Θ_R,   R = number of predefined lung regions
                                     (canonical choice: R = 4 —
                                     right/left × ventral/dorsal
                                     quadrants; finer grids, e.g. R = 36,
                                     are used for pixel-level indices)

with each θ_r a **regional ventilation state** at a given point in the
respiratory cycle: fractional tidal impedance change (a surrogate for
regional tidal volume), end-expiratory lung impedance (EELI, a surrogate
for regional aeration/recruitment), and, when a PEEP trial is performed,
regional compliance dC_r/dPEEP estimated from the impedance–pressure
relationship at each step of a stepwise PEEP trial.

This is a **design choice, not a derivation** — the map from the raw
conductivity field σ(x,t) to θ is itself a reduction (typically: fixed
anatomical or pixel-based ROI masks applied to the reconstructed image
sequence), and the identifiability question the gate asks is downstream
of that reduction. Two consequences follow, both flagged rather than
absorbed silently:

* **Semantic invariance of θ is not free here.** In the Darcy case, θ_j's
  meaning is invariant to mesh nuisance by construction (reference-grid
  interpolation). Here, patient thorax geometry varies (body habitus,
  chest wall edema, prior thoracotomy, pleural effusion) not as a mesh
  refinement of one domain but as genuinely different domains per patient.
  Making "the dorsal-right region's compliance" mean the same thing across
  differently shaped thoraces requires an explicit registration step
  (anatomical landmarking or a population-averaged atlas warp) that has no
  Darcy-benchmark analogue and is real methodological work, stated here as
  an open problem rather than assumed solved.
* **R is a preregistered granularity choice**, exactly like the Darcy
  benchmark's truncation level m, and is subject to the same discipline:
  report sensitivity across R ∈ {4, 16, 36}, do not select R after seeing
  which grid makes the surrogate look best.

### 2.1 The regional parameterization, as an explicit object

Made precise, θ is not one vector but a **family indexed by granularity**,
because — as below — research and bedside use of EIT operate at different
points in this family, and conflating them would silently mix two
different target parameter spaces under one symbol.

    θ^(R) = (θ_1, ..., θ_R) ∈ Θ_R,   θ_r = (E_r, V_r, C_r) ∈ ℝ³

for region r = 1, ..., R, with:

* **E_r** — end-expiratory lung impedance (EELI) relative to a reference
  breath, a surrogate for regional aeration/recruitment state;
* **V_r** — fractional tidal impedance change, a surrogate for regional
  tidal volume (Σ_r V_r = 1 by construction, a constraint the gate's
  admissibility check must enforce exactly as it enforces Θ_m^(a) in the
  Darcy case);
* **C_r** — regional compliance dC_r/dPEEP, estimated by finite difference
  across steps of a stepwise PEEP trial, defined only when such a trial is
  performed (see §2.2).

R is fixed by which of two use modes θ is serving:

| Granularity | R | ROI definition | Use mode (§7) |
|---|---|---|---|
| Coarse | 4 | anatomical quadrants (ventral/dorsal × left/right) | bedside clinical adjunct |
| Fine | 36 | pixel-block grid over the reconstructed image | research / offline certification |

This is not two arbitrary choices on the same sensitivity sweep — it is
the formal version of the research-vs-bedside distinction below (§7),
and the certification question in §5 must therefore be asked and answered
**separately at each R**: identifiability at R = 36 does not imply, and
is not implied by, identifiability of the coarser R = 4 regional
aggregates a clinician actually reads at the bedside, since the two are
related by a (further) aggregation step with its own dilution behavior
(§8's cited precedent is the general version of this same phenomenon).

### 2.2 When each component of θ is defined

E_r and V_r are defined breath-by-breath from a single EIT recording at
fixed ventilator settings. C_r requires a **stepwise PEEP trial** — a
deliberate, time-limited maneuver, not continuous monitoring — and is
therefore itself a research-mode quantity in the sense of §7 even when R
is coarse: a bedside clinician may watch E_r and V_r trend continuously
between trials, but C_r is only ever recomputed at the discrete PEEP-trial
checkpoints, structurally identical to the amortized (not per-breath)
schedule already specified for the certification gate itself.

## 3. Nuisance variables (the experiment E)

| Nuisance | Darcy analogue | EIT/ventilation instance |
|---|---|---|
| Mesh / discretization | `SolverConfig.n` | FEM mesh of the CEM forward solve; thorax segmentation resolution |
| Forcing | `SolverConfig.forcing` | Current injection pattern (adjacent, opposite, or "skip" pairing; drive frequency, typically 50–100 kHz) |
| Observation operator | `ObservationConfig` (sensor grid) | Electrode count and belt position (intercostal level); which voltage pairs are used for reconstruction |
| Noise model | `ObservationConfig.noise_sigma` | Measurement noise plus **motion artifact** (cardiac-related impedance change, patient movement) — a nuisance with structure, not i.i.d., unlike the Darcy benchmark's clean Gaussian assumption |
| — (no analogue) | — | **Patient geometry** (thorax shape, electrode belt slippage) — the genuinely new nuisance axis, discussed in §2 |

The injection-pattern row deserves the same treatment as sensor placement
in the Darcy nuisance sweep: which patterns actually resolve dorsal
atelectasis is an empirical question the gate can answer directly, by
running the identifiability profile I_j(δ,σ) under each injection protocol
and reporting where it shifts.

## 4. Preregistered intervention directions

Mirroring the Darcy direction set (leading/boundary KL coefficients,
frequency blocks, random directions, data-informed/prior-dominated
subspaces), the clinically motivated direction set for ventilation EIT is:

* **Per-region minimal pairs**: T_{r,δ}(θ) = θ + δe_r — does the gate
  detect a change confined to one region (e.g., dorsal-right EELI drops,
  simulating regional derecruitment) as distinguishable from the
  unperturbed state, at the noise and motion-artifact levels present in a
  real ICU recording?
* **Ventral–dorsal contrast direction**: v = (dorsal regions) −
  (ventral regions), normalized — the single most clinically load-bearing
  direction, since ventral/dorsal dissociation is the specific pattern
  that both distinguishes recruitable-collapse ARDS from overdistension
  and is the pattern a ventral–dorsal correlation shortcut would destroy.
* **Data-informed subspace**: derived from the CEM Jacobian at the
  patient- and injection-pattern-specific operating point, exactly as in
  the Darcy benchmark — this is where the specific electrode configuration
  and body habitus determine what is resolvable at all, independent of
  any surrogate.
* **Random directions**: unchanged role — a calibration baseline.

## 5. The certification question, stated for this domain

Does a learned EIT reconstruction network (CNN/U-Net class, mapping
boundary voltages directly to a conductivity image or to θ directly)
preserve the ventral–dorsal contrast direction as identifiable, or does it
reproduce it accurately in aggregate while attenuating or misattributing
the specific minimal pair that would reveal dorsal atelectasis dissociated
from ventral aeration — exactly the shortcut-surrogate failure mode,
relabeled?

## 6. Constructed clinical case study

**Patient.** 58-year-old, moderate-to-severe ARDS (Berlin criteria),
mechanically ventilated, undergoing a decremental PEEP trial to identify
the "best PEEP" balancing recruitment against overdistension (Costa et al.
overdistension–collapse method). This is a **research-mode** maneuver in
the sense of §7 — a stepwise trial computing C_r, not passive bedside
trending — presented at the coarse R = 4 granularity of §2.1 for
legibility; the fine-grained (R = 36) version of the same trial is what
the certification gate in §5 actually runs against. EIT belt placed at the
5th intercostal space, 16 electrodes, adjacent injection, images
reconstructed at each of five PEEP steps (20 → 16 → 12 → 8 → 4 cmH₂O), a
learned reconstruction network used in place of the classical
(GREIT/linearized) algorithm for real-time bedside display.

**Illustrative regional state at PEEP 12 cmH₂O** (the exact map's
reconstruction, i.e., ground truth in this constructed case):

| Region | EELI (relative) | Tidal fraction | Overdistension % | Collapse % |
|---|---|---|---|---|
| Ventral-right | high | 0.31 | 18 | 2 |
| Ventral-left | high | 0.29 | 15 | 3 |
| Dorsal-right | low | 0.14 | 1 | 34 |
| Dorsal-left | low | 0.26 | 2 | 9 |

Global collapse+overdistension minimized near PEEP 12 in this construction
— the classical "best PEEP" — driven substantially by the dorsal-right
region's collapse fraction, which is *dissociated* from the ventral
regions' near-normal state: exactly the ventral–dorsal contrast direction
in §4.

**The shortcut, translated.** Suppose the reconstruction network was
trained predominantly on a cohort with milder, more homogeneous disease,
where ventral and dorsal regional states are correlated (corr ≈ 0.9, the
same structural form as the benchmark's confound-imitating surrogate).
The network learns to impute dorsal state substantially from the ventral
signal. On its training distribution this is close to risk-minimizing and
validation-set RMSE looks excellent. On *this* patient — where dorsal
collapse is real and ventral aeration is preserved, i.e., precisely off
the training correlation manifold — the network under-reports dorsal-right
collapse, because it is partly inferring dorsal state from ventral
measurements rather than resolving it independently. The displayed image
looks clinically plausible (smooth, anatomically shaped, globally
consistent) and the reported global inhomogeneity index may look
acceptable, because the error is concentrated exactly in the one
minimal-pair direction the shortcut collapses, not spread diffusely across
the image the way ordinary reconstruction noise would be.

**Consequence for the PEEP decision.** If the network under-reports
dorsal-right collapse at PEEP 8–12, the apparent overdistension–collapse
curve shifts toward favoring a *lower* best-PEEP than the true regional
state supports — because the algorithm no longer "sees" the collapse that
would have penalized derecruitment at lower pressures. The clinician
titrates to a PEEP that leaves the dorsal-right region under-recruited:
persistent atelectrauma and shunt, worse oxygenation, and — the ICU-burden
framing — a patient who remains ventilator-dependent longer and is a worse
candidate for extubation or NIV trial than the displayed EIT trace
suggests, precisely because the failure is silent rather than showing up
as visible reconstruction noise the clinician would discount.

**What the gate would have shown.** Applied at certification (not
per-breath): a minimal pair perturbing dorsal-right EELI in the
constructed evaluation set, off the training correlation manifold, would
show attenuation well below 1 in the network's response to that direction
while the ventral-region minimal pairs pass — the network-specific
analogue of the benchmark's e₈ result (response annihilated exactly on
the ignored coefficient, forward RMSE still near the noise floor overall).
The certification decision is made once, against a validated bank of
synthetic and heterogeneous-cohort minimal pairs, before the network is
trusted for bedside PEEP titration — not recomputed per patient or per
breath, consistent with the amortized-gate design already stated for the
EIT work package generally.

## 7. Research tool versus bedside clinical adjunct — where certification runs

EIT's clinical literature draws exactly the distinction §2.1 needed
formally: EIT used as a **research tool** (fine spatial resolution, full
identifiability characterization, deliberate maneuvers like a stepwise
PEEP trial) versus EIT used as a **bedside clinical adjunct** (coarse,
continuously trended regional summaries feeding a real-time display).
These are not two performance tiers of the same measurement — they are
two different points in the θ^(R) family (§2.1) with two different
certification obligations:

* **Research mode (R = 36, or finer).** This is where the certification
  gate itself belongs, run offline against the full preregistered
  direction set of §4 — including the ventral–dorsal contrast direction
  and the data-informed subspace derived from the patient- and
  protocol-specific Jacobian. This is expensive (a full stepwise PEEP
  trial, dense injection patterns, the complete minimal-pair battery) and
  is exactly why the gate is designed to run at discrete certification
  checkpoints rather than per-breath — a design constraint stated
  generally for the EIT work package and now given its concrete reason:
  research-mode characterization is not something one would want to
  repeat continuously even if it were cheap, because it requires a
  deliberate maneuver (the PEEP trial), not passive observation.
* **Bedside mode (R = 4).** This is what a clinician actually watches
  between certifications: continuously trended E_r and V_r on a coarse
  regional display. The network deployed here is the one *certified* in
  research mode; bedside mode consumes the certification decision, it
  does not re-derive it. If the coarse aggregate looks reassuring while
  the fine-grained certification would have failed — the ventral–dorsal
  shortcut of §6, expressed at R = 4 — bedside mode alone offers no way to
  detect this, which is precisely the argument for certifying before
  deployment rather than trusting the bedside display's own apparent
  plausibility.

**Continuum of care.** The same distinction recurs across the ventilation
episode rather than only within one PEEP trial: at intubation and early
ARDS management, a research-mode characterization (a full stepwise trial)
is plausible and clinically motivated; through the bulk of the ICU stay,
only bedside-mode trending is practical; at a weaning or extubation
decision point, a second, lighter research-mode check (fewer PEEP steps,
still off the passive-monitoring correlation manifold) would be the
natural point to re-certify, since a network trained on the
early-illness-heavy data available up to that point is exactly where a
correlation shortcut is most likely to have formed and least likely to
have been exercised. This gives the "reducing ICU burden" claim from the
motivating discussion a concrete shape: not a single certification event,
but certification checkpoints placed at the clinically natural
decision points across the continuum — intubation, PEEP retitration,
weaning trial, extubation — each a discrete, deliberate moment rather than
a per-breath computation.

## 8. A structural precedent for why aggregation can hide what matters

The PILOT trial (Semler et al., *NEJM* 2022) titrated oxygen therapy using
a single global signal (pulse oximetry SpO₂) against a single aggregate
outcome (ventilator-free days through day 28) in 2,541 mechanically
ventilated ICU/ED adults, comparing lower (90%), intermediate (94%), and
higher (98%) SpO₂ targets. It found no difference in the primary outcome
across targets (median 20 vs. 21 vs. 21 days, P = 0.81) and no difference
in 28-day mortality. The trial's own conclusion is that, within this
range, target choice does not affect these outcomes — that conclusion is
not disputed here and nothing about EIT bears on it directly.

What is borrowed is structural, not clinical: an aggregate signal driving
an aggregate outcome is a specific, limited instrument, and a null result
under it does not establish that no regionally or physiologically
resolved signal would have found something an aggregate one could not —
the same logical point as the benchmark's aggregation-dilution
proposition (loss of power under an aggregate test does not imply loss of
underlying information), transposed from a statistical test to a
measurement-and-outcome pairing. This motivates, rather than proves, why
a spatially resolved technology aimed at regional heterogeneity — EIT
titrating PEEP by regional recruitment/overdistension rather than a
global SpO₂ number — is a mechanistically distinct proposition from PILOT,
not a restatement of it, and why ventilator-free days through day 28
(PILOT's endpoint) is the right accepted outcome measure against which
any eventual EIT-driven clinical claim from this program should actually
be tested, rather than an invented substitute metric.

## 9. What is genuinely new relative to the Darcy/EIT (tumor) case

1. Regional-physiological θ instead of spectral θ (§2) — a reduction with
   its own semantic-invariance burden across patient geometries, and now
   an explicit dual-granularity structure (§2.1) rather than a single
   scale.
2. Structured, non-i.i.d. nuisance (cardiac and motion artifact) rather
   than clean Gaussian noise (§3).
3. A directly stated clinical decision (PEEP level) downstream of the
   identifiability question, giving the gate's PASS/FAIL a legible
   clinical consequence rather than an abstract one — useful for
   communicating the program's stakes to a non-technical reviewer, and
   worth stating with exactly this much precision and no more: the
   vignette is constructed to be illustrative of the mechanism, not a
   report of a validated clinical finding.
4. A genuine research/bedside bifurcation in *where the gate runs* (§7),
   with its own continuum-of-care placement — absent from the Darcy
   benchmark, where certification and deployment share one setting.

## References

Costa ELV, Borges JB, Melo A, et al. Bedside estimation of recruitable
alveolar collapse and hyperdistension by electrical impedance tomography.
*Intensive Care Med*. 2009;35(6):1132–1137. — overdistension–collapse
methodology referenced in §6.

Frerichs I, Amato MBP, van Kaam AH, et al. Chest electrical impedance
tomography examination, data analysis, terminology, clinical use and
recommendations: consensus statement of the TRanslational EIT developmENt
stuDy group. *Thorax*. 2017;72(1):83–93. — clinical EIT terminology and
practice referenced throughout.

Ring B. *Electrical Impedance Technology (EIT): Research & Clinical
Applications*. CME lecture. Course objectives: mechanism of EIT
monitoring; research-tool vs. bedside-clinical-adjunct use; role across
the continuum of care for mechanically ventilated patients. — educational
source; used in §2.1 and §7 for its stated framing only, not as a source
of specific empirical findings, since its content beyond the stated
objectives has not been reviewed.

Semler MW, Casey JD, Lloyd BD, et al.; PILOT Investigators and the
Pragmatic Critical Care Research Group. Oxygen-saturation targets for
critically ill adults receiving mechanical ventilation. *N Engl J Med*.
2022;387(19):1759–1769. doi:10.1056/NEJMoa2208415. — pragmatic RCT (PILOT
trial), used in §8 as a structural precedent for aggregate-signal/
aggregate-outcome limitations and to fix the accepted primary outcome
(ventilator-free days through day 28) referenced there.
