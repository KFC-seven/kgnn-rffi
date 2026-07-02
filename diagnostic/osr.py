from __future__ import annotations

import math

import numpy as np


def msp_unknown_score(logits: np.ndarray) -> np.ndarray:
    probs = _softmax(np.asarray(logits, dtype=np.float32))
    return (1.0 - np.max(probs, axis=1)).astype(np.float32)


def energy_unknown_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    scaled = logits / float(temperature)
    return (-float(temperature) * _logsumexp(scaled, axis=1)).astype(np.float32)


def prototype_distance_score(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    distances = np.linalg.norm(embeddings[:, None, :] - prototypes[None, :, :], axis=2)
    return np.min(distances, axis=1).astype(np.float32)


def fit_opensvdd(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    radius_quantile: float = 1.0,
    eps: float = 1e-6,
) -> dict:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if embeddings.shape[0] != labels.size:
        raise ValueError("embeddings and labels must have equal sample count.")
    if not (0.0 < float(radius_quantile) <= 1.0):
        raise ValueError("radius_quantile must be in (0, 1].")
    centers = fit_prototypes(embeddings, labels, num_classes=num_classes)
    radii = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        distances = np.linalg.norm(embeddings[mask] - centers[cls][None, :], axis=1)
        radius = float(np.quantile(distances, float(radius_quantile)))
        radii.append(max(radius, float(eps)))
    return {
        "centers": centers.astype(np.float32),
        "radii": np.asarray(radii, dtype=np.float32),
    }


def opensvdd_unknown_score(embeddings: np.ndarray, model: dict) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centers = np.asarray(model["centers"], dtype=np.float32)
    radii = np.asarray(model["radii"], dtype=np.float32).reshape(1, -1)
    distances = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
    normalized = distances / radii
    return np.min(normalized, axis=1).astype(np.float32)


def calibrate_threshold(source_scores: np.ndarray, frr: float = 0.05) -> float:
    scores = np.sort(np.asarray(source_scores, dtype=np.float32).reshape(-1))
    if scores.size == 0:
        raise ValueError("Cannot calibrate threshold from an empty score array.")
    if not (0.0 <= frr < 1.0):
        raise ValueError(f"frr must be in [0, 1), got {frr}.")
    keep_count = max(1, int(math.ceil((1.0 - float(frr)) * scores.size)))
    threshold_index = min(scores.size - 1, keep_count - 1)
    return float(scores[threshold_index])


def reject_by_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=np.float32).reshape(-1) > float(threshold)


def fit_prototypes(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    prototypes = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError(f"Cannot fit prototype for class {cls}: no samples.")
        prototypes.append(np.mean(embeddings[mask], axis=0))
    return np.stack(prototypes, axis=0).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return (
        np.squeeze(max_values, axis=axis)
        + np.log(np.sum(np.exp(values - max_values), axis=axis))
    )
