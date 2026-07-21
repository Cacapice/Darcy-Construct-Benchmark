"""Construct-validity benchmark for surrogate-based Darcy-flow inversion.

Target parameter: KL coefficient vector theta of the log-conductivity
field. Ground truth is known by construction, so every diagnostic has
an unambiguous reference.
"""

__version__ = "0.4.4"

from .config import (
    BenchmarkConfig,
    Experiment,
    NuisanceConfig,
    ObservationConfig,
    PriorConfig,
    SolverConfig,
    Thresholds,
)
from .dataset import DatasetConfig, generate_dataset
from .diagnostics import (
    detection_power,
    directional_power,
    identifiability_profile,
    information_functional,
    matched_direction_power,
    posterior_geometry,
    identifiability_spectrum,
    jacobian_fd,
    evaluate_direction,
    evaluate_preservation,
)
from .forward import ForwardMap, Observation
from .kl import KLBasis, matern_kernel
from .perturbations import coordinate_pair, is_admissible, make_minimal_pair
from .protocol import (
    CertificationReport,
    certify_surrogate,
    freeze_protocol,
    load_protocol,
)
from .solver import DarcySolver
from .surrogates import (
    ExactSurrogate,
    ShortcutSurrogate,
    FieldToObservationSurrogate,
    ModeSuppressingSurrogate,
)

__all__ = [
    "BenchmarkConfig", "NuisanceConfig", "ObservationConfig", "PriorConfig",
    "SolverConfig", "Thresholds", "KLBasis", "matern_kernel", "DarcySolver",
    "ForwardMap", "Observation", "ExactSurrogate", "ModeSuppressingSurrogate",
    "coordinate_pair", "make_minimal_pair", "is_admissible",
    "detection_power", "directional_power", "identifiability_spectrum", "jacobian_fd",
    "DatasetConfig", "generate_dataset", "FieldToObservationSurrogate",
    "CertificationReport", "certify_surrogate", "freeze_protocol", "load_protocol",
    "ShortcutSurrogate", "identifiability_profile", "posterior_geometry",
    "Experiment", "information_functional", "matched_direction_power",
    "evaluate_direction", "evaluate_preservation",
]
