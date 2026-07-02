"""KGNN-RFFI: Knownness-Gated Nearest Neighbor RF Fingerprint Identification."""

from .envelope import IpGateModel, build_ip_gate_model, predict_ip_gate
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
    "IpGateModel",
    "PerturbationConfig",
    "PerturbationEngine",
    "PerturbationSafetyResult",
    "PerturbationSpec",
    "apply_perturbation_batch",
    "build_ip_gate_model",
    "classify_perturbation_safety",
    "default_perturbation_specs",
    "predict_ip_gate",
    "select_specs",
]
