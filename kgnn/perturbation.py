from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnostic.sourceonly import infer_logits_embeddings
from kgnn.phantom import PerturbationEngine


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    family: str
    level: int
    params: dict[str, float | int]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "level": int(self.level),
            "params": {key: _json_scalar(value) for key, value in self.params.items()},
        }


@dataclass(frozen=True)
class PerturbationSafetyResult:
    spec: PerturbationSpec
    accuracy: float
    clean_accuracy: float
    retention: float
    role: str
    samples: int
    mean_relative_distance: float = 0.0
    distance_safe_mult: float = 0.0
    distance_destructive_mult: float = 0.0
    decision_rule: str = "classifier"
    distance_fallback_active: bool = False

    def to_dict(self) -> dict:
        return {
            **self.spec.to_dict(),
            "accuracy": float(self.accuracy),
            "clean_accuracy": float(self.clean_accuracy),
            "retention": float(self.retention),
            "role": self.role,
            "samples": int(self.samples),
            "mean_relative_distance": float(self.mean_relative_distance),
            "distance_safe_mult": float(self.distance_safe_mult),
            "distance_destructive_mult": float(self.distance_destructive_mult),
            "decision_rule": self.decision_rule,
            "distance_fallback_active": bool(self.distance_fallback_active),
        }


def default_perturbation_specs() -> list[PerturbationSpec]:
    specs: list[PerturbationSpec] = []
    for level, degrees in enumerate([5.0, 15.0, 30.0, 60.0, 90.0], start=1):
        for sign in [-1.0, 1.0]:
            theta = sign * np.deg2rad(degrees)
            specs.append(
                PerturbationSpec(
                    name=f"phase_{int(sign * degrees):+d}deg",
                    family="phase",
                    level=level,
                    params={"theta": float(theta)},
                )
            )
    for level, cycles in enumerate([0.02, 0.08, 0.16, 0.32, 0.50], start=1):
        for sign in [-1.0, 1.0]:
            signed = sign * cycles
            specs.append(
                PerturbationSpec(
                    name=f"cfo_{signed:+.2f}cyc",
                    family="cfo",
                    level=level,
                    params={"cycles": float(signed)},
                )
            )
    for level, offset in enumerate([1.0, 3.0, 8.0, 15.0, 30.0], start=1):
        for sign in [-1.0, 1.0]:
            signed = sign * offset
            specs.append(
                PerturbationSpec(
                    name=f"timing_{signed:+.0f}",
                    family="timing",
                    level=level,
                    params={"offset": float(signed)},
                )
            )
    for level, alpha in enumerate([0.90, 0.75, 0.60, 0.40, 1.50, 2.00], start=1):
        specs.append(
            PerturbationSpec(
                name=f"amplitude_{alpha:.2f}",
                family="amplitude",
                level=level,
                params={"alpha": float(alpha)},
            )
        )
    iq_grid = [
        (1.03, 2.0),
        (1.08, 5.0),
        (1.15, 8.0),
        (1.35, 15.0),
        (1.60, 25.0),
    ]
    for level, (gain, degrees) in enumerate(iq_grid, start=1):
        specs.append(
            PerturbationSpec(
                name=f"iq_gain{gain:.2f}_phase{degrees:.0f}deg",
                family="iq_imbalance",
                level=level,
                params={"gain": float(gain), "phase": float(np.deg2rad(degrees))},
            )
        )
    for level, snr_db in enumerate([30.0, 20.0, 10.0, 5.0, 0.0, -3.0], start=1):
        specs.append(
            PerturbationSpec(
                name=f"noise_{snr_db:+.0f}db",
                family="noise",
                level=level,
                params={"snr_db": float(snr_db)},
            )
        )
    for level, taps in enumerate([2, 3, 4, 6, 8], start=1):
        specs.append(
            PerturbationSpec(
                name=f"multipath_{taps}tap",
                family="multipath",
                level=level,
                params={"n_taps": int(taps)},
            )
        )
    return specs


def classify_perturbation_safety(
    encoder,
    source_x: np.ndarray,
    source_labels: np.ndarray,
    perturbation_engine: PerturbationEngine,
    specs: list[PerturbationSpec] | None = None,
    safe_accuracy: float = 0.90,
    destructive_accuracy: float = 0.50,
    threshold_mode: str = "absolute",
    max_samples_per_class: int = 25,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 0,
    use_distance_fallback: bool = True,
    fallback_min_safe_specs: int = 2,
    fallback_clean_accuracy: float = 0.85,
    safe_distance_mult: float = 1.5,
    destructive_distance_mult: float = 3.0,
) -> list[PerturbationSafetyResult]:
    if threshold_mode not in {"absolute", "relative"}:
        raise ValueError(f"Unknown threshold_mode={threshold_mode!r}.")
    specs = specs or default_perturbation_specs()
    labels = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    idx = _stratified_limit(labels, max_samples_per_class=max_samples_per_class, seed=seed)
    x_eval = np.asarray(source_x, dtype=np.float32)[idx]
    y_eval = labels[idx]
    clean_logits, clean_embeddings = infer_logits_embeddings(
        encoder,
        x_eval,
        batch_size=batch_size,
        device=device,
    )
    clean_pred = np.argmax(clean_logits, axis=1).astype(np.int64)
    clean_accuracy = _accuracy(clean_pred, y_eval)
    centers, spreads = _class_geometry(clean_embeddings, y_eval)
    pending: list[dict] = []
    for i, spec in enumerate(specs):
        local_engine = PerturbationEngine(
            config=perturbation_engine.config,
            seed=seed + 104729 * (i + 1),
        )
        perturbed = apply_perturbation_batch(x_eval, spec, local_engine)
        logits, embeddings = infer_logits_embeddings(
            encoder,
            perturbed,
            batch_size=batch_size,
            device=device,
        )
        pred = np.argmax(logits, axis=1).astype(np.int64)
        accuracy = _accuracy(pred, y_eval)
        retention = accuracy / max(clean_accuracy, 1e-6)
        decision_value = accuracy if threshold_mode == "absolute" else retention
        if decision_value >= float(safe_accuracy):
            classifier_role = "safe"
        elif decision_value < float(destructive_accuracy):
            classifier_role = "destructive"
        else:
            classifier_role = "neutral"
        relative_distance = _mean_relative_class_distance(
            embeddings=embeddings,
            labels=y_eval,
            centers=centers,
            spreads=spreads,
        )
        pending.append(
            {
                "spec": spec,
                "accuracy": float(accuracy),
                "retention": float(retention),
                "classifier_role": classifier_role,
                "relative_distance": float(relative_distance),
            }
        )
    classifier_safe_count = sum(1 for item in pending if item["classifier_role"] == "safe")
    fallback_active = bool(
        use_distance_fallback
        and (
            clean_accuracy < float(fallback_clean_accuracy)
            or classifier_safe_count < int(fallback_min_safe_specs)
        )
    )
    results: list[PerturbationSafetyResult] = []
    for item in pending:
        role = str(item["classifier_role"])
        decision_rule = f"classifier_{role}"
        if fallback_active:
            relative_distance = float(item["relative_distance"])
            if relative_distance <= float(safe_distance_mult):
                role = "safe"
                decision_rule = "distance_safe"
            elif relative_distance >= float(destructive_distance_mult):
                role = "destructive"
                decision_rule = "distance_destructive"
        results.append(
            PerturbationSafetyResult(
                spec=item["spec"],
                accuracy=float(item["accuracy"]),
                clean_accuracy=float(clean_accuracy),
                retention=float(item["retention"]),
                role=role,
                samples=int(y_eval.shape[0]),
                mean_relative_distance=float(item["relative_distance"]),
                distance_safe_mult=float(safe_distance_mult),
                distance_destructive_mult=float(destructive_distance_mult),
                decision_rule=decision_rule,
                distance_fallback_active=fallback_active,
            )
        )
    return results


def select_specs(results: list[PerturbationSafetyResult], role: str) -> list[PerturbationSpec]:
    return [item.spec for item in results if item.role == role]


def apply_perturbation_batch(
    x: np.ndarray,
    spec: PerturbationSpec,
    perturbation_engine: PerturbationEngine,
) -> np.ndarray:
    return np.stack(
        [apply_perturbation_one(sample, spec, perturbation_engine) for sample in np.asarray(x, dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def apply_perturbation_one(
    iq: np.ndarray,
    spec: PerturbationSpec,
    perturbation_engine: PerturbationEngine,
) -> np.ndarray:
    params = spec.params
    if spec.family == "phase":
        return perturbation_engine.phase_rotate(iq, theta=float(params["theta"]))
    if spec.family == "cfo":
        return perturbation_engine.carrier_offset(iq, cycles=float(params["cycles"]))
    if spec.family == "timing":
        return perturbation_engine.timing_offset(iq, offset=float(params["offset"]))
    if spec.family == "amplitude":
        return perturbation_engine.amplitude_scale(iq, alpha=float(params["alpha"]))
    if spec.family == "iq_imbalance":
        return perturbation_engine.iq_imbalance(
            iq,
            gain=float(params["gain"]),
            phase=float(params["phase"]),
        )
    if spec.family == "noise":
        return perturbation_engine.add_noise(iq, snr_db=float(params["snr_db"]))
    if spec.family == "multipath":
        return perturbation_engine.multipath_filter(iq, n_taps=int(params["n_taps"]))
    raise ValueError(f"Unknown perturbation family={spec.family!r}.")


def _stratified_limit(labels: np.ndarray, max_samples_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        parts.append(idx[: min(idx.size, int(max_samples_per_class))])
    out = np.concatenate(parts, axis=0).astype(np.int64)
    rng.shuffle(out)
    return out


def _accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred, dtype=np.int64) == np.asarray(labels, dtype=np.int64))) if labels.size else 0.0


def _class_geometry(embeddings: np.ndarray, labels: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    labels = np.asarray(labels, dtype=np.int64)
    centers: dict[int, np.ndarray] = {}
    spreads: dict[int, float] = {}
    for cls in sorted(np.unique(labels).tolist()):
        cls_embeddings = np.asarray(embeddings[labels == cls], dtype=np.float32)
        center = np.mean(cls_embeddings, axis=0).astype(np.float32)
        dist = _cosine_distance(cls_embeddings, center[None, :])[:, 0]
        centers[int(cls)] = center
        spreads[int(cls)] = max(float(np.sqrt(np.mean(dist * dist))), 1e-6)
    return centers, spreads


def _mean_relative_class_distance(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: dict[int, np.ndarray],
    spreads: dict[int, float],
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    values = []
    for cls in sorted(np.unique(labels).tolist()):
        mask = labels == cls
        dist = _cosine_distance(
            np.asarray(embeddings[mask], dtype=np.float32),
            centers[int(cls)][None, :],
        )[:, 0]
        values.append(dist / max(float(spreads[int(cls)]), 1e-6))
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values, axis=0)))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (1.0 - a_n @ b_n.T).astype(np.float32)


def _json_scalar(value: float | int) -> float | int:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
