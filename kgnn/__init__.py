"""KGNN-RFFI: Knownness-Gated Nearest Neighbor RF Fingerprint Identification."""

from .envelope import KgnnModel, build_kgnn_model, predict_kgnn
from .perturbation import (
    PerturbationSafetyResult,
    PerturbationSpec,
    apply_perturbation_batch,
    classify_perturbation_safety,
    default_perturbation_specs,
    select_specs,
)
from .phantom import PerturbationConfig, PerturbationEngine

__all__ = [
    "KgnnModel",
    "PerturbationConfig",
    "PerturbationEngine",
    "PerturbationSafetyResult",
    "PerturbationSpec",
    "apply_perturbation_batch",
    "build_kgnn_model",
    "classify_perturbation_safety",
    "default_perturbation_specs",
    "predict_kgnn",
    "select_specs",
]
