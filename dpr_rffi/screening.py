from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .perturbations import PerturbationEngine, PerturbationSpec, default_perturbation_specs


PerturbationRole = Literal["low-impact", "neutral", "high-impact"]


@dataclass(frozen=True)
class PerturbationScreeningResult:
    spec: PerturbationSpec
    clean_accuracy: float
    perturbed_accuracy: float
    retention_score: float
    role: PerturbationRole
    samples: int


def screen_perturbations(
    predict_labels: Callable[[np.ndarray], np.ndarray],
    source_x: np.ndarray,
    source_y: np.ndarray,
    *,
    num_classes: int,
    specs: list[PerturbationSpec] | None = None,
    theta_low: float = 0.50,
    theta_high: float = 0.90,
    max_samples_per_class: int = 25,
    epsilon: float = 1e-6,
    seed: int = 0,
) -> list[PerturbationScreeningResult]:
    """Screen perturbations using only a stratified source-validation subset."""

    if int(num_classes) <= 1:
        raise ValueError("num_classes must be greater than one.")
    if not 0.0 <= float(theta_low) < float(theta_high) <= 1.0:
        raise ValueError("Require 0 <= theta_low < theta_high <= 1.")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive.")
    x = np.asarray(source_x, dtype=np.float32)
    y = np.asarray(source_y, dtype=np.int64).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("source_x and source_y must contain the same number of samples.")
    indices = stratified_subset(
        y,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    x_screen, y_screen = x[indices], y[indices]
    clean_prediction = np.asarray(predict_labels(x_screen), dtype=np.int64).reshape(-1)
    if clean_prediction.size != y_screen.size:
        raise ValueError("predict_labels returned an unexpected number of predictions.")
    clean_accuracy = float(np.mean(clean_prediction == y_screen))
    chance_accuracy = 1.0 / float(num_classes)
    degenerate = clean_accuracy <= chance_accuracy + float(epsilon)
    denominator = max(clean_accuracy - chance_accuracy, float(epsilon))
    output: list[PerturbationScreeningResult] = []
    for index, spec in enumerate(specs or default_perturbation_specs()):
        engine = PerturbationEngine(seed=int(seed) + 104729 * (index + 1))
        perturbed = engine.apply_batch(x_screen, spec)
        prediction = np.asarray(predict_labels(perturbed), dtype=np.int64).reshape(-1)
        accuracy = float(np.mean(prediction == y_screen))
        eta = (accuracy - chance_accuracy) / denominator
        if degenerate:
            role: PerturbationRole = "neutral"
        elif eta >= float(theta_high):
            role = "low-impact"
        elif eta < float(theta_low):
            role = "high-impact"
        else:
            role = "neutral"
        output.append(
            PerturbationScreeningResult(
                spec=spec,
                clean_accuracy=clean_accuracy,
                perturbed_accuracy=accuracy,
                retention_score=float(eta),
                role=role,
                samples=int(y_screen.size),
            )
        )
    return output


def select_specs(
    results: list[PerturbationScreeningResult],
    role: PerturbationRole,
) -> list[PerturbationSpec]:
    return [item.spec for item in results if item.role == role]


def stratified_subset(
    labels: np.ndarray,
    *,
    max_samples_per_class: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0:
        raise ValueError("Cannot screen an empty validation set.")
    if int(max_samples_per_class) <= 0:
        raise ValueError("max_samples_per_class must be positive.")
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for cls in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        parts.append(indices[: min(indices.size, int(max_samples_per_class))])
    selected = np.concatenate(parts).astype(np.int64)
    rng.shuffle(selected)
    return selected
