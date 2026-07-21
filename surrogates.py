"""Surrogate maps G~_m and built-in positive controls.

The benchmark evaluates any callable theta -> R^{n_obs}. Two built-in
surrogates make the diagnostics testable *before* any learned surrogate
exists:

* ``ExactSurrogate``           - wraps G_m itself; every preservation
                                 null should survive (negative control).
* ``ModeSuppressingSurrogate`` - internally projects theta onto its first
                                 k KL modes before solving. This is the
                                 canonical failure the framework warns
                                 about: aggregate pressure fields look
                                 plausible while low-variance components
                                 are silently destroyed. Preservation
                                 nulls for j >= k should be violated
                                 (positive control).

A learned surrogate (NN, GP, reduced basis, ...) plugs in by exposing
the same ``__call__(theta) -> y`` signature.
"""

from __future__ import annotations

import numpy as np

from .forward import ForwardMap


class ExactSurrogate:
    def __init__(self, forward: ForwardMap):
        self._G = forward

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self._G(theta)


class ModeSuppressingSurrogate:
    """Zeroes KL coefficients j >= k_keep before evaluating the exact map."""

    def __init__(self, forward: ForwardMap, k_keep: int):
        if not (0 < k_keep <= forward.basis.m):
            raise ValueError("k_keep must be in (0, m]")
        self._G = forward
        self.k_keep = k_keep

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        t = np.array(theta, dtype=float, copy=True)
        t[self.k_keep:] = 0.0
        return self._G(t)


class FieldToObservationSurrogate:
    """Adapter for field-valued learned surrogates (DeepONet, FNO, ...).

    Operator-learning models typically predict the full pressure field
    p_hat(theta) on the solver grid rather than the sensor vector. This
    adapter composes any callable ``theta -> (n, n) field`` with the
    benchmark's observation operator, so the certification gate sees the
    standard ``theta -> y`` signature. The observation operator is taken
    from the SAME ForwardMap the gate uses, so no sensor-placement
    mismatch can leak in.
    """

    def __init__(self, field_fn, forward: ForwardMap):
        self._field_fn = field_fn
        self._observe = forward.observation.observe
        self._n = forward.solver.n

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        p = np.asarray(self._field_fn(theta), dtype=float)
        if p.shape != (self._n, self._n):
            raise ValueError(
                f"field surrogate returned {p.shape}, expected ({self._n},{self._n})"
            )
        return self._observe(p)


class ShortcutSurrogate:
    """The failure mode motivating the program, constructed analytically.

    Model of a surrogate trained on a distribution where theta_{j_ignored}
    is correlated with theta_{j_used} (corr = r): the surrogate never learns
    the ignored coefficient's independent effect and instead imputes it via
    the training-distribution regression E[theta_b | theta_a] = r theta_a,
    then evaluates the exact physics.

    On the training distribution (theta_b = r theta_a + sqrt(1-r^2) z) the
    forward error is driven only by the imputation residual sqrt(1-r^2)
    times one mid-spectrum coefficient's sensitivity — small, typically at
    or below the noise floor, so RMSE and test error look excellent. But:

    * the response to a minimal pair along e_{j_ignored} is exactly zero
      (attenuation 0) — the coefficient is no longer identifiable;
    * the response to e_{j_used} is contaminated by a spurious r-weighted
      e_{j_ignored} sensitivity — the used direction is distorted.

    Minimal pairs detect this precisely because they break the training
    correlation: they realize (theta_a, theta_b) combinations off the
    correlation manifold. Aggregate accuracy on the training distribution
    cannot, by construction. Unlike ModeSuppressingSurrogate, nothing here
    is 'deliberately broken' relative to its training objective — imputing
    from the correlation is the risk-minimizing thing to do on that
    distribution. The pathology is the distribution, exposed by the gate.
    """

    def __init__(self, forward: ForwardMap, j_used: int, j_ignored: int, corr: float):
        m = forward.basis.m
        if not (0 <= j_used < m and 0 <= j_ignored < m and j_used != j_ignored):
            raise ValueError("j_used and j_ignored must be distinct valid indices")
        if not (-1.0 < corr < 1.0):
            raise ValueError("corr must be in (-1, 1)")
        self._G = forward
        self.j_used = j_used
        self.j_ignored = j_ignored
        self.corr = corr

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        t = np.array(theta, dtype=float, copy=True)
        t[self.j_ignored] = self.corr * t[self.j_used]
        return self._G(t)
