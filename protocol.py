"""Protocol freezing and the certification gate.

``freeze_protocol`` serializes everything a certification decision depends
on — thresholds, prior and nuisance configs, ground truth theta*, and the
preregistered direction set with the exact-map identifiability decision for
each direction — to a JSON artifact. Freezing happens BEFORE any learned
surrogate is trained or evaluated; the file is the preregistration.

``certify_surrogate`` runs the gate against a frozen protocol.

Gate rule (fixed at freeze time):
* DECISIONAL directions are the preregistered directions on which the
  EXACT map rejects H0 (the data can see the perturbation). The gate
  passes iff H0_preserve survives on every decisional direction.
* Directions the exact map itself cannot detect are SUPPLEMENTARY: they
  are evaluated and reported, but a surrogate is not failed for losing a
  response the observation design cannot resolve in the first place.

Any change to the gate rule, thresholds, or direction set after freezing
is a protocol amendment and belongs in an amendment log, not in code.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .config import BenchmarkConfig, Thresholds
from .diagnostics import (
    PreservationResult,
    evaluate_direction,
    evaluate_preservation,
)
from .forward import ForwardMap
from .kl import KLBasis
from .perturbations import make_minimal_pair

Map = Callable[[np.ndarray], np.ndarray]


def freeze_protocol(
    G: ForwardMap,
    theta_star: np.ndarray,
    directions: Sequence[Tuple[str, np.ndarray]],
    cfg: BenchmarkConfig,
    path: str | Path,
) -> dict:
    """Evaluate exact-map identifiability per direction and write the frozen protocol."""
    th = cfg.thresholds
    entries = []
    for label, v in directions:
        pair = make_minimal_pair(theta_star, v, th.delta, cfg.radius_a, label)
        r = evaluate_direction(G, pair, G.noise_sigma, th)
        entries.append({
            "label": label,
            "vector": np.asarray(v, dtype=float).tolist(),
            "delta_applied": r.delta,
            "exact_response_norm": r.response_norm,
            "exact_power": r.power,
            "exact_power_directional": r.power_directional,
            "decisional": bool(r.reject_h0),
        })
    protocol = {
        "package_version": __version__,
        "frozen_on": date.today().isoformat(),
        "thresholds": dataclasses.asdict(th),
        "prior": dataclasses.asdict(cfg.prior),
        "nuisance": dataclasses.asdict(cfg.baseline),
        "radius_a": cfg.radius_a,
        "theta_star": np.asarray(theta_star, dtype=float).tolist(),
        "gate_rule": ("pass iff H0_preserve survives on every direction with "
                      "decisional=true; others supplementary"),
        "directions": entries,
    }
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2)
    return protocol


def load_protocol(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class CertificationReport:
    passed: bool
    decisional: List[PreservationResult]
    supplementary: List[PreservationResult]

    def summary(self) -> str:
        nd = len(self.decisional)
        ok = sum(r.preserved for r in self.decisional)
        verdict = "PASS" if self.passed else "FAIL"
        failed = [r.label for r in self.decisional if not r.preserved]
        line = f"certification: {verdict}  ({ok}/{nd} decisional directions preserved)"
        if failed:
            line += f"  violated: {', '.join(failed)}"
        return line


def certify_surrogate(
    G: ForwardMap,
    surrogate: Map,
    protocol: dict,
    thresholds: Optional[Thresholds] = None,
) -> CertificationReport:
    """Run the frozen gate. ``thresholds`` defaults to the frozen ones."""
    th = thresholds or Thresholds(**protocol["thresholds"])
    theta_star = np.asarray(protocol["theta_star"], dtype=float)
    radius_a = protocol["radius_a"]

    decisional, supplementary = [], []
    for entry in protocol["directions"]:
        v = np.asarray(entry["vector"], dtype=float)
        pair = make_minimal_pair(theta_star, v, th.delta, radius_a, entry["label"])
        res = evaluate_preservation(G, surrogate, pair, th)
        (decisional if entry["decisional"] else supplementary).append(res)

    return CertificationReport(
        passed=all(r.preserved for r in decisional),
        decisional=decisional,
        supplementary=supplementary,
    )
