from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .perturbations import PerturbationEngine, PerturbationSpec
from .screening import PerturbationScreeningResult, screen_perturbations, select_specs


ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DPRConfig:
    """Paper configuration for DPR and CCA."""

    theta_low: float = 0.50
    theta_high: float = 0.90
    screening_samples_per_class: int = 25
    low_augmentations_per_sample: int = 4
    high_augmentations_per_sample: int = 1
    kappa: float = 0.30
    low_reference_limit_per_class: int = 1000
    high_reference_limit: int = 5000
    knn_k: int = 5
    cca_quantile: float = 0.90
    cca_expansion: float = 1.75
    dpr_weight_low: float = 0.50
    dpr_weight_high: float = 0.95
    source_frr: float = 0.03
    epsilon: float = 1e-6
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 <= self.theta_low < self.theta_high <= 1.0:
            raise ValueError("Require 0 <= theta_low < theta_high <= 1.")
        if self.screening_samples_per_class <= 0:
            raise ValueError("screening_samples_per_class must be positive.")
        if self.low_augmentations_per_sample < 0 or self.high_augmentations_per_sample < 0:
            raise ValueError("augmentation counts must be nonnegative.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.low_reference_limit_per_class <= 0 or self.high_reference_limit < 0:
            raise ValueError("reference limits are invalid.")
        if self.knn_k <= 0:
            raise ValueError("knn_k must be positive.")
        if not 0.0 < self.cca_quantile < 1.0:
            raise ValueError("cca_quantile must lie in (0, 1).")
        if self.cca_expansion <= 1.0:
            raise ValueError("cca_expansion must be greater than one.")
        if not 0.0 <= self.dpr_weight_low <= self.dpr_weight_high <= 1.0:
            raise ValueError("DPR fusion weights are invalid.")
        if not 0.0 < self.source_frr < 1.0:
            raise ValueError("source_frr must lie in (0, 1).")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")


@dataclass(frozen=True)
class Prediction:
    """DPR-RFFI inference output."""

    label: np.ndarray
    rejected: np.ndarray
    score: np.ndarray
    dpr_score: np.ndarray
    nearest_neighbor_score: np.ndarray
    cca: np.ndarray
    dpr_weight: np.ndarray


class DPRRFFI:
    """Source-only DPR-RFFI detector and enrolled-device identifier."""

    def __init__(self, config: DPRConfig | None = None):
        self.config = config or DPRConfig()
        self.config.validate()
        self.screening_results: list[PerturbationScreeningResult] = []
        self.reference_summary: dict[str, int | float] = {}
        self._fitted = False

    def fit(
        self,
        *,
        source_train_x: np.ndarray,
        source_train_y: np.ndarray,
        source_val_x: np.ndarray,
        source_val_y: np.ndarray,
        encode: ArrayFunction,
        predict_labels: ArrayFunction,
        perturbation_specs: list[PerturbationSpec] | None = None,
    ) -> "DPRRFFI":
        """Fit all reference sets and calibration statistics from source data."""

        train_x, train_y = _validate_signal_label_pair(source_train_x, source_train_y)
        val_x, val_y = _validate_signal_label_pair(source_val_x, source_val_y)
        classes = np.unique(train_y)
        if not np.array_equal(classes, np.arange(classes.size)):
            raise ValueError("Source labels must be contiguous integers starting at zero.")
        if not set(np.unique(val_y)).issubset(set(classes)):
            raise ValueError("Validation labels must be present in source training labels.")
        self.num_classes = int(classes.size)
        self._encode = encode
        self.screening_results = screen_perturbations(
            predict_labels,
            val_x,
            val_y,
            num_classes=self.num_classes,
            specs=perturbation_specs,
            theta_low=self.config.theta_low,
            theta_high=self.config.theta_high,
            max_samples_per_class=self.config.screening_samples_per_class,
            epsilon=self.config.epsilon,
            seed=self.config.seed,
        )
        low_specs = select_specs(self.screening_results, "low-impact")
        high_specs = select_specs(self.screening_results, "high-impact")

        train_embeddings = _as_embeddings(encode(train_x), train_x.shape[0])
        val_embeddings = _as_embeddings(encode(val_x), val_x.shape[0])
        self.source_embeddings = train_embeddings
        self.source_labels = train_y
        self.class_centers = class_centers(train_embeddings, train_y, self.num_classes)
        self.class_scales = class_scales(
            train_embeddings,
            train_y,
            self.class_centers,
            epsilon=self.config.epsilon,
        )
        self.class_radii = class_radii(
            train_embeddings,
            train_y,
            self.class_centers,
            quantile=self.config.cca_quantile,
            epsilon=self.config.epsilon,
        )
        low_reference, low_labels, low_augmented = self._build_low_reference(
            train_x,
            train_y,
            train_embeddings,
            low_specs,
        )
        high_reference = self._build_high_reference(train_x, high_specs)
        self.low_reference = low_reference
        self.low_reference_labels = low_labels
        self.high_reference = high_reference
        self._low_index = NearestNeighbors(n_neighbors=1, metric="cosine").fit(low_reference)
        self._high_index = None
        if high_reference.shape[0]:
            self._high_index = NearestNeighbors(n_neighbors=1, metric="cosine").fit(high_reference)
        effective_k = min(self.config.knn_k, train_embeddings.shape[0])
        self._source_index = NearestNeighbors(n_neighbors=effective_k, metric="cosine").fit(
            train_embeddings
        )
        self._effective_k = int(effective_k)

        raw = self._raw_components(val_embeddings)
        self.dpr_mean, self.dpr_std = _mean_std(raw["dpr"], self.config.epsilon)
        self.nn_mean, self.nn_std = _mean_std(raw["nn"], self.config.epsilon)
        validation_score = self._compose_from_raw(raw)
        self.threshold = calibrate_source_threshold(validation_score, self.config.source_frr)
        self.reference_summary = {
            "screening_samples": int(self.screening_results[0].samples),
            "low_impact_settings": int(len(low_specs)),
            "high_impact_settings": int(len(high_specs)),
            "neutral_settings": int(
                len(self.screening_results) - len(low_specs) - len(high_specs)
            ),
            "low_reference_features": int(low_reference.shape[0]),
            "low_impact_augmented_features": int(low_augmented),
            "high_reference_features": int(high_reference.shape[0]),
            "source_memory_features": int(train_embeddings.shape[0]),
            "source_threshold": float(self.threshold),
        }
        self._fitted = True
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        self._check_fitted()
        embeddings = _as_embeddings(self._encode(np.asarray(x, dtype=np.float32)), len(x))
        raw = self._raw_components(embeddings)
        score = self._compose_from_raw(raw)
        rejected = score > float(self.threshold)
        label = np.where(rejected, -1, raw["candidate"]).astype(np.int64)
        return Prediction(
            label=label,
            rejected=rejected.astype(bool),
            score=score.astype(np.float32),
            dpr_score=raw["dpr"].astype(np.float32),
            nearest_neighbor_score=raw["nn"].astype(np.float32),
            cca=raw["cca"].astype(np.float32),
            dpr_weight=raw["weight"].astype(np.float32),
        )

    def score_embeddings(self, embeddings: np.ndarray) -> Prediction:
        """Score precomputed features with the fitted source-side structures."""

        self._check_fitted()
        values = _as_embeddings(embeddings)
        raw = self._raw_components(values)
        score = self._compose_from_raw(raw)
        rejected = score > float(self.threshold)
        label = np.where(rejected, -1, raw["candidate"]).astype(np.int64)
        return Prediction(
            label=label,
            rejected=rejected.astype(bool),
            score=score.astype(np.float32),
            dpr_score=raw["dpr"].astype(np.float32),
            nearest_neighbor_score=raw["nn"].astype(np.float32),
            cca=raw["cca"].astype(np.float32),
            dpr_weight=raw["weight"].astype(np.float32),
        )

    def _build_low_reference(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        source_embeddings: np.ndarray,
        low_specs: list[PerturbationSpec],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        rng = np.random.default_rng(self.config.seed)
        augmented_x: list[np.ndarray] = []
        augmented_labels: list[int] = []
        if low_specs and self.config.low_augmentations_per_sample:
            engine = PerturbationEngine(self.config.seed + 15485863)
            for index, sample in enumerate(source_x):
                for _ in range(self.config.low_augmentations_per_sample):
                    spec = low_specs[int(rng.integers(0, len(low_specs)))]
                    augmented_x.append(engine.apply(sample, spec))
                    augmented_labels.append(int(source_y[index]))
        if augmented_x:
            augmented_embeddings = _as_embeddings(
                self._encode(np.stack(augmented_x, axis=0)),
                len(augmented_x),
            )
            augmented_labels_array = np.asarray(augmented_labels, dtype=np.int64)
        else:
            augmented_embeddings = np.empty((0, source_embeddings.shape[1]), dtype=np.float32)
            augmented_labels_array = np.empty(0, dtype=np.int64)

        reference_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        retained_augmented = 0
        for cls in range(self.num_classes):
            base = source_embeddings[source_y == cls]
            candidate = augmented_embeddings[augmented_labels_array == cls]
            if candidate.shape[0]:
                distance = cosine_distance_matrix(
                    candidate,
                    self.class_centers[cls][None, :],
                )[:, 0]
                candidate = candidate[
                    distance <= self.config.kappa * self.class_scales[cls]
                ]
            limit = self.config.low_reference_limit_per_class
            if base.shape[0] >= limit:
                selected = rng.choice(base.shape[0], size=limit, replace=False)
                class_reference = base[selected]
            else:
                remaining = limit - base.shape[0]
                if candidate.shape[0] > remaining:
                    selected = rng.choice(candidate.shape[0], size=remaining, replace=False)
                    candidate = candidate[selected]
                class_reference = (
                    np.concatenate([base, candidate], axis=0) if candidate.shape[0] else base
                )
                retained_augmented += int(candidate.shape[0])
            reference_parts.append(class_reference.astype(np.float32))
            label_parts.append(np.full(class_reference.shape[0], cls, dtype=np.int64))
        return (
            np.concatenate(reference_parts, axis=0),
            np.concatenate(label_parts, axis=0),
            retained_augmented,
        )

    def _build_high_reference(
        self,
        source_x: np.ndarray,
        high_specs: list[PerturbationSpec],
    ) -> np.ndarray:
        feature_dim = self.source_embeddings.shape[1]
        if (
            not high_specs
            or self.config.high_augmentations_per_sample == 0
            or self.config.high_reference_limit == 0
        ):
            return np.empty((0, feature_dim), dtype=np.float32)
        rng = np.random.default_rng(self.config.seed + 32452843)
        source_order = np.arange(source_x.shape[0])
        rng.shuffle(source_order)
        engine = PerturbationEngine(self.config.seed + 49979687)
        augmented: list[np.ndarray] = []
        for index in source_order:
            for _ in range(self.config.high_augmentations_per_sample):
                spec = high_specs[int(rng.integers(0, len(high_specs)))]
                augmented.append(engine.apply(source_x[index], spec))
                if len(augmented) >= self.config.high_reference_limit:
                    break
            if len(augmented) >= self.config.high_reference_limit:
                break
        return _as_embeddings(self._encode(np.stack(augmented, axis=0)), len(augmented))

    def _raw_components(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        d_low = self._low_index.kneighbors(embeddings, return_distance=True)[0][:, 0]
        if self._high_index is None:
            dpr = d_low
        else:
            d_high = self._high_index.kneighbors(embeddings, return_distance=True)[0][:, 0]
            dpr = d_low / np.maximum(d_high, self.config.epsilon)
        distances, indices = self._source_index.kneighbors(
            embeddings,
            n_neighbors=self._effective_k,
            return_distance=True,
        )
        nn_score = np.mean(distances, axis=1)
        candidate = majority_vote(self.source_labels[indices])
        candidate_distance = cosine_distance_matrix(
            embeddings,
            self.class_centers,
        )[np.arange(embeddings.shape[0]), candidate]
        radius = self.class_radii[candidate]
        cca = class_consistency(
            candidate_distance,
            radius,
            expansion=self.config.cca_expansion,
            epsilon=self.config.epsilon,
        )
        weight = self.config.dpr_weight_high - (
            self.config.dpr_weight_high - self.config.dpr_weight_low
        ) * cca
        return {
            "dpr": dpr.astype(np.float32),
            "nn": nn_score.astype(np.float32),
            "candidate": candidate,
            "cca": cca.astype(np.float32),
            "weight": weight.astype(np.float32),
        }

    def _compose_from_raw(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        dpr_z = (raw["dpr"] - self.dpr_mean) / self.dpr_std
        nn_z = (raw["nn"] - self.nn_mean) / self.nn_std
        return (raw["weight"] * dpr_z + (1.0 - raw["weight"]) * nn_z).astype(
            np.float32
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit before inference.")


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = _as_embeddings(a)
    right = _as_embeddings(b)
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.clip(1.0 - left_norm @ right_norm.T, 0.0, 2.0).astype(np.float32)


def class_centers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    centers = []
    for cls in range(int(num_classes)):
        values = embeddings[labels == cls]
        if values.shape[0] == 0:
            raise ValueError(f"No source training feature for class {cls}.")
        centers.append(np.mean(values, axis=0))
    return np.stack(centers, axis=0).astype(np.float32)


def class_scales(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    scales = []
    for cls in range(centers.shape[0]):
        distance = cosine_distance_matrix(
            embeddings[labels == cls],
            centers[cls][None, :],
        )[:, 0]
        scales.append(max(float(np.sqrt(np.mean(distance**2))), float(epsilon)))
    return np.asarray(scales, dtype=np.float32)


def class_radii(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    quantile: float,
    epsilon: float,
) -> np.ndarray:
    radii = []
    for cls in range(centers.shape[0]):
        distance = cosine_distance_matrix(
            embeddings[labels == cls],
            centers[cls][None, :],
        )[:, 0]
        radii.append(max(float(np.quantile(distance, float(quantile))), float(epsilon)))
    return np.asarray(radii, dtype=np.float32)


def class_consistency(
    candidate_distance: np.ndarray,
    candidate_radius: np.ndarray,
    *,
    expansion: float,
    epsilon: float,
) -> np.ndarray:
    distance = np.asarray(candidate_distance, dtype=np.float32)
    radius = np.maximum(np.asarray(candidate_radius, dtype=np.float32), float(epsilon))
    expanded = float(expansion) * radius
    return np.where(
        distance <= radius,
        1.0,
        np.where(
            distance >= expanded,
            0.0,
            (expanded - distance) / ((float(expansion) - 1.0) * radius + float(epsilon)),
        ),
    ).astype(np.float32)


def majority_vote(neighbor_labels: np.ndarray) -> np.ndarray:
    output = []
    for labels in np.asarray(neighbor_labels, dtype=np.int64):
        values, counts = np.unique(labels, return_counts=True)
        output.append(int(values[np.argmax(counts)]))
    return np.asarray(output, dtype=np.int64)


def calibrate_source_threshold(scores: np.ndarray, source_frr: float) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float32).reshape(-1))
    if values.size == 0:
        raise ValueError("Cannot calibrate a threshold from no source scores.")
    keep_count = max(1, int(np.ceil((1.0 - float(source_frr)) * values.size)))
    return float(values[min(values.size - 1, keep_count - 1)])


def _mean_std(values: np.ndarray, epsilon: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float32)
    return float(np.mean(array)), max(float(np.std(array)), float(epsilon))


def _validate_signal_label_pair(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if signals.ndim != 3 or signals.shape[-1] != 2:
        raise ValueError(f"Expected signals with shape (N, L, 2), got {signals.shape}.")
    if signals.shape[0] != labels.size or labels.size == 0:
        raise ValueError("Signals and labels must be nonempty and aligned.")
    return signals, labels


def _as_embeddings(values: np.ndarray, expected_rows: int | None = None) -> np.ndarray:
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a feature matrix, got shape {embeddings.shape}.")
    if expected_rows is not None and embeddings.shape[0] != int(expected_rows):
        raise ValueError("Encoder returned an unexpected number of features.")
    if embeddings.shape[0] == 0 or not np.all(np.isfinite(embeddings)):
        raise ValueError("Feature matrix must be nonempty and finite.")
    return embeddings
