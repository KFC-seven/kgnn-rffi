from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from diagnostic.sourceonly import infer_logits_embeddings
from kgnn.phantom import PerturbationEngine

from .perturbation import PerturbationSpec, apply_perturbation_one


@dataclass
class KgnnModel:
    distance: str
    score_mode: str
    threshold: float
    support_embeddings: np.ndarray
    support_labels: np.ndarray
    destructive_embeddings: np.ndarray
    class_centers: np.ndarray
    class_spreads: np.ndarray
    nearest_support: NearestNeighbors
    nearest_destructive: NearestNeighbors | None
    support_k: int = 1
    destructive_k: int = 1
    class_norm_alpha: float = 1.0
    destructive_balance: str = "none"
    score_calibration: str = "none"
    calibration_strength: float = 1.0
    calibration_max_factor: float = 0.0
    calibration_quantile: float = 0.97
    calibration_centers: np.ndarray | None = None
    calibration_scales: np.ndarray | None = None
    calibration_global_center: float = 0.0
    calibration_global_scale: float = 1.0


def build_kgnn_model(
    encoder,
    source_x: np.ndarray,
    source_labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    num_classes: int,
    safe_specs: list[PerturbationSpec],
    destructive_specs: list[PerturbationSpec],
    perturbation_engine: PerturbationEngine,
    augment_count: int = 4,
    sigma_mult: float = 2.0,
    max_support_per_class: int = 1000,
    max_destructive_bank: int = 5000,
    destructive_per_sample: int = 1,
    batch_size: int = 256,
    device: str = "cpu",
    distance: str = "cosine",
    score_mode: str = "ratio",
    support_k: int = 1,
    destructive_k: int = 1,
    class_norm_alpha: float = 1.0,
    destructive_balance: str = "none",
    score_calibration: str = "none",
    calibration_strength: float = 1.0,
    calibration_max_factor: float = 0.0,
    calibration_quantile: float | None = None,
    frr: float = 0.05,
    gate_support: bool = True,
    seed: int = 0,
) -> tuple[KgnnModel, dict]:
    if distance not in {"euclidean", "cosine"}:
        raise ValueError(f"Unknown distance={distance!r}.")
    if score_mode not in {"ratio", "known", "class_norm_ratio", "knn_ratio", "balanced_destructive_ratio"}:
        raise ValueError(f"Unknown score_mode={score_mode!r}.")
    if destructive_balance not in {"none", "family", "family_level"}:
        raise ValueError(f"Unknown destructive_balance={destructive_balance!r}.")
    if score_calibration not in {"none", "class_iqr", "class_tail_z", "class_tail_ratio"}:
        raise ValueError(f"Unknown score_calibration={score_calibration!r}.")
    support_k = max(1, int(support_k))
    destructive_k = max(1, int(destructive_k))
    class_norm_alpha = max(0.0, float(class_norm_alpha))
    calibration_strength = min(1.0, max(0.0, float(calibration_strength)))
    calibration_max_factor = max(0.0, float(calibration_max_factor))
    if calibration_quantile is None:
        calibration_quantile = 1.0 - float(frr)
    calibration_quantile = min(0.999, max(0.5, float(calibration_quantile)))
    labels = np.asarray(source_labels, dtype=np.int64)
    train_idx = np.asarray(train_indices, dtype=np.int64)
    val_idx = np.asarray(val_indices, dtype=np.int64)
    _logits, source_embeddings = infer_logits_embeddings(
        encoder,
        source_x,
        batch_size=batch_size,
        device=device,
    )
    train_embeddings = source_embeddings[train_idx]
    train_labels = labels[train_idx]
    val_embeddings = source_embeddings[val_idx]
    centers = _fit_centers(train_embeddings, train_labels, num_classes)
    spreads = _class_spreads(
        embeddings=train_embeddings,
        labels=train_labels,
        centers=centers,
        distance=distance,
    )
    support_embeddings, support_labels, support_info = _build_gated_support_embeddings(
        encoder=encoder,
        source_x=source_x[train_idx],
        source_embeddings=train_embeddings,
        source_labels=train_labels,
        centers=centers,
        spreads=spreads,
        safe_specs=safe_specs,
        perturbation_engine=perturbation_engine,
        augment_count=augment_count,
        sigma_mult=sigma_mult,
        max_support_per_class=max_support_per_class,
        batch_size=batch_size,
        device=device,
        distance=distance,
        gate_support=gate_support,
        seed=seed,
    )
    nearest_support = NearestNeighbors(
        n_neighbors=min(support_k, support_embeddings.shape[0]),
        metric=distance,
    )
    nearest_support.fit(support_embeddings)
    effective_destructive_balance = destructive_balance
    if score_mode == "balanced_destructive_ratio" and effective_destructive_balance == "none":
        effective_destructive_balance = "family"
    destructive_embeddings, destructive_info = _build_destructive_bank(
        encoder=encoder,
        source_x=source_x[train_idx],
        destructive_specs=destructive_specs,
        perturbation_engine=perturbation_engine,
        max_destructive_bank=max_destructive_bank,
        destructive_per_sample=destructive_per_sample,
        destructive_balance=effective_destructive_balance,
        batch_size=batch_size,
        device=device,
        seed=seed + 271828,
    )
    nearest_destructive = None
    if destructive_embeddings.shape[0] > 0:
        nearest_destructive = NearestNeighbors(
            n_neighbors=min(destructive_k, destructive_embeddings.shape[0]),
            metric=distance,
        )
        nearest_destructive.fit(destructive_embeddings)
    model = KgnnModel(
        distance=distance,
        score_mode=score_mode,
        threshold=0.0,
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        destructive_embeddings=destructive_embeddings,
        class_centers=centers,
        class_spreads=spreads,
        nearest_support=nearest_support,
        nearest_destructive=nearest_destructive,
        support_k=support_k,
        destructive_k=destructive_k,
        class_norm_alpha=class_norm_alpha,
        destructive_balance=effective_destructive_balance,
        score_calibration="none",
        calibration_strength=calibration_strength,
        calibration_max_factor=calibration_max_factor,
        calibration_quantile=calibration_quantile,
    )
    raw_val_scores, raw_val_pred = _score_embeddings(model, val_embeddings)
    calibration_info = _fit_score_calibration(
        scores=raw_val_scores,
        pred=raw_val_pred,
        num_classes=num_classes,
        mode=score_calibration,
        quantile=calibration_quantile,
    )
    model.score_calibration = score_calibration
    model.calibration_centers = calibration_info.pop("calibration_centers_array")
    model.calibration_scales = calibration_info.pop("calibration_scales_array")
    model.calibration_global_center = float(calibration_info["calibration_global_center"])
    model.calibration_global_scale = float(calibration_info["calibration_global_scale"])
    val_scores, _pred = _score_embeddings(model, val_embeddings)
    model.threshold = _calibrate_threshold(val_scores, frr=frr)
    info = {
        **support_info,
        "destructive_bank_size": int(destructive_embeddings.shape[0]),
        "threshold": float(model.threshold),
        "score_mode": score_mode,
        "distance": distance,
        "support_k": int(support_k),
        "destructive_k": int(destructive_k),
        "class_norm_alpha": float(class_norm_alpha),
        "destructive_balance": effective_destructive_balance,
        "score_calibration": score_calibration,
        "calibration_strength": float(calibration_strength),
        "calibration_max_factor": float(calibration_max_factor),
        "calibration_quantile": float(calibration_quantile),
        "sigma_mult": float(sigma_mult),
        "safe_spec_count": int(len(safe_specs)),
        "destructive_spec_count": int(len(destructive_specs)),
        **destructive_info,
        **calibration_info,
    }
    return model, info


def predict_kgnn(model: KgnnModel, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, pred = _score_embeddings(model, embeddings)
    rejected = scores > float(model.threshold)
    return pred.astype(np.int64), rejected.astype(bool), scores.astype(np.float32)


def known_distance_scores(
    embeddings: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:
    nearest = NearestNeighbors(n_neighbors=1, metric=distance)
    nearest.fit(support_embeddings)
    dist, idx = nearest.kneighbors(np.asarray(embeddings, dtype=np.float32), n_neighbors=1)
    return dist[:, 0].astype(np.float32), support_labels[idx[:, 0]].astype(np.int64)


def prototype_scores(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:
    dist = distance_matrix(embeddings, prototypes, distance=distance)
    pred = np.argmin(dist, axis=1).astype(np.int64)
    return dist[np.arange(dist.shape[0]), pred].astype(np.float32), pred


def distance_matrix(a: np.ndarray, b: np.ndarray, distance: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if distance == "euclidean":
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)
    if distance == "cosine":
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return (1.0 - a_n @ b_n.T).astype(np.float32)
    raise ValueError(f"Unknown distance={distance!r}.")


def calibrate_threshold(scores: np.ndarray, frr: float) -> float:
    return _calibrate_threshold(scores, frr=frr)


def _score_embeddings(model: KgnnModel, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    known_dist, idx = model.nearest_support.kneighbors(
        np.asarray(embeddings, dtype=np.float32),
        n_neighbors=_effective_neighbors(model.nearest_support, model.support_k),
    )
    pred = model.support_labels[idx[:, 0]].astype(np.int64)
    known = _aggregate_distances(known_dist, model.score_mode, model.support_k)
    if model.score_mode == "known" or model.nearest_destructive is None:
        return known.astype(np.float32), pred
    destructive_dist, _destructive_idx = model.nearest_destructive.kneighbors(
        np.asarray(embeddings, dtype=np.float32),
        n_neighbors=_effective_neighbors(model.nearest_destructive, model.destructive_k),
    )
    destructive = _aggregate_distances(destructive_dist, model.score_mode, model.destructive_k)
    if model.score_mode == "class_norm_ratio":
        class_scale, global_scale = _class_scale(model.class_spreads, pred, model.class_norm_alpha)
        known = known / np.maximum(class_scale, 1e-6)
        destructive = destructive / max(float(global_scale), 1e-6)
    scores = known / np.maximum(destructive, 1e-6)
    scores = _apply_score_calibration(model, scores, pred)
    return scores.astype(np.float32), pred


def _apply_score_calibration(model: KgnnModel, scores: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mode = getattr(model, "score_calibration", "none")
    values = np.asarray(scores, dtype=np.float32)
    if mode == "none":
        return values
    strength = min(1.0, max(0.0, float(getattr(model, "calibration_strength", 1.0))))
    if strength <= 0.0:
        return values
    centers = model.calibration_centers
    scales = model.calibration_scales
    if centers is None or scales is None or centers.size == 0 or scales.size == 0:
        center = float(getattr(model, "calibration_global_center", 0.0))
        scale = max(float(getattr(model, "calibration_global_scale", 1.0)), 1e-6)
        if mode == "class_tail_ratio":
            factor = float((1.0 / scale) ** strength)
            max_factor = float(getattr(model, "calibration_max_factor", 0.0))
            if max_factor > 0.0:
                factor = min(factor, max_factor)
            return values * factor
        calibrated = (values - center) / scale
        return values if strength >= 1.0 else ((1.0 - strength) * values + strength * calibrated)
    cls = np.clip(np.asarray(pred, dtype=np.int64), 0, scales.shape[0] - 1)
    scale = np.maximum(scales[cls].astype(np.float32), 1e-6)
    if mode == "class_tail_ratio":
        global_scale = max(float(getattr(model, "calibration_global_scale", 1.0)), 1e-6)
        factor = np.power(global_scale / scale, strength).astype(np.float32)
        max_factor = float(getattr(model, "calibration_max_factor", 0.0))
        if max_factor > 0.0:
            factor = np.minimum(factor, max_factor).astype(np.float32)
        return values * factor
    center = centers[np.clip(cls, 0, centers.shape[0] - 1)].astype(np.float32)
    calibrated = (values - center) / scale
    return calibrated if strength >= 1.0 else ((1.0 - strength) * values + strength * calibrated)


def _fit_score_calibration(
    scores: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    mode: str,
    quantile: float,
) -> dict:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(pred, dtype=np.int64).reshape(-1)
    if values.size == 0:
        centers = np.zeros(int(num_classes), dtype=np.float32)
        scales = np.ones(int(num_classes), dtype=np.float32)
        return {
            "calibration_centers_array": centers,
            "calibration_scales_array": scales,
            "calibration_global_center": 0.0,
            "calibration_global_scale": 1.0,
            "calibration_min_class_samples": 0,
            "calibration_max_class_samples": 0,
        }
    q = min(0.999, max(0.5, float(quantile)))
    global_center, global_scale = _calibration_center_scale(values, mode=mode, quantile=q)
    centers = np.full(int(num_classes), float(global_center), dtype=np.float32)
    scales = np.full(int(num_classes), float(global_scale), dtype=np.float32)
    class_counts = []
    for cls in range(int(num_classes)):
        cls_values = values[labels == cls]
        class_counts.append(int(cls_values.size))
        if cls_values.size < 3:
            continue
        center, scale = _calibration_center_scale(cls_values, mode=mode, quantile=q)
        centers[cls] = float(center)
        scales[cls] = float(scale)
    scales = np.maximum(scales, 1e-6).astype(np.float32)
    return {
        "calibration_centers_array": centers,
        "calibration_scales_array": scales,
        "calibration_global_center": float(global_center),
        "calibration_global_scale": float(global_scale),
        "calibration_min_class_samples": int(min(class_counts) if class_counts else 0),
        "calibration_max_class_samples": int(max(class_counts) if class_counts else 0),
    }


def _calibration_center_scale(values: np.ndarray, mode: str, quantile: float) -> tuple[float, float]:
    safe = np.asarray(values, dtype=np.float32).reshape(-1)
    if safe.size == 0:
        return 0.0, 1.0
    if mode == "class_tail_ratio":
        scale = float(np.quantile(safe, quantile))
        return 0.0, max(scale, 1e-6)
    median = float(np.median(safe))
    if mode == "class_tail_z":
        upper = float(np.quantile(safe, quantile))
        return median, max(upper - median, 1e-6)
    q25 = float(np.quantile(safe, 0.25))
    q75 = float(np.quantile(safe, 0.75))
    return median, max(q75 - q25, 1e-6)


def _effective_neighbors(nearest: NearestNeighbors, requested: int) -> int:
    fit_count = int(getattr(nearest, "n_samples_fit_", nearest._fit_X.shape[0]))
    return max(1, min(int(requested), fit_count))


def _aggregate_distances(distances: np.ndarray, score_mode: str, requested_k: int) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        return values.reshape(values.shape[0], -1)[:, 0].astype(np.float32)
    if score_mode == "knn_ratio" or int(requested_k) > 1:
        return np.mean(values, axis=1).astype(np.float32)
    return values[:, 0].astype(np.float32)


def _class_scale(spreads: np.ndarray, pred: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    if float(alpha) <= 0.0:
        return np.ones_like(pred, dtype=np.float32), 1.0
    safe_spreads = np.maximum(np.asarray(spreads, dtype=np.float32), 1e-6)
    global_scale = float(np.median(safe_spreads) ** float(alpha))
    class_scale = safe_spreads[np.asarray(pred, dtype=np.int64)] ** float(alpha)
    return class_scale.astype(np.float32), global_scale


def _build_gated_support_embeddings(
    encoder,
    source_x: np.ndarray,
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    centers: np.ndarray,
    spreads: np.ndarray,
    safe_specs: list[PerturbationSpec],
    perturbation_engine: PerturbationEngine,
    augment_count: int,
    sigma_mult: float,
    max_support_per_class: int,
    batch_size: int,
    device: str,
    distance: str,
    gate_support: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    emb_parts = []
    label_parts = []
    augmented_attempted = 0
    augmented_kept = 0
    augmented_dropped = 0
    expanded_classes = 0
    per_class_counts: dict[str, dict[str, int]] = {}
    for cls in sorted(np.unique(source_labels).tolist()):
        cls_idx = np.where(source_labels == cls)[0]
        base_embeddings = source_embeddings[cls_idx].astype(np.float32)
        augmented_embeddings = np.empty((0, base_embeddings.shape[1]), dtype=np.float32)
        if safe_specs and int(augment_count) > 0 and cls_idx.size > 0:
            augmented_x = []
            for local_idx in cls_idx.tolist():
                for _ in range(int(augment_count)):
                    spec = safe_specs[int(rng.integers(0, len(safe_specs)))]
                    augmented_x.append(apply_perturbation_one(source_x[local_idx], spec, perturbation_engine))
            if augmented_x:
                augmented_attempted += len(augmented_x)
                perturbed = np.stack(augmented_x, axis=0).astype(np.float32)
                _logits, emb = infer_logits_embeddings(
                    encoder,
                    perturbed,
                    batch_size=batch_size,
                    device=device,
                )
                dist = distance_matrix(emb, centers[int(cls)][None, :], distance=distance)[:, 0]
                if gate_support:
                    keep = dist <= (float(sigma_mult) * float(spreads[int(cls)]))
                else:
                    keep = np.ones(emb.shape[0], dtype=bool)
                augmented_embeddings = emb[keep].astype(np.float32)
                augmented_kept += int(keep.sum())
                augmented_dropped += int((~keep).sum())
        cls_support = _limit_class_support(
            base_embeddings=base_embeddings,
            augmented_embeddings=augmented_embeddings,
            max_support_per_class=max_support_per_class,
            rng=rng,
        )
        if cls_support.shape[0] > base_embeddings.shape[0]:
            expanded_classes += 1
        emb_parts.append(cls_support)
        label_parts.append(np.full(cls_support.shape[0], int(cls), dtype=np.int64))
        per_class_counts[str(int(cls))] = {
            "base": int(base_embeddings.shape[0]),
            "support": int(cls_support.shape[0]),
            "augmented_kept": int(max(0, cls_support.shape[0] - base_embeddings.shape[0])),
        }
    support_embeddings = np.concatenate(emb_parts, axis=0).astype(np.float32)
    support_labels = np.concatenate(label_parts, axis=0).astype(np.int64)
    return support_embeddings, support_labels, {
        "base_support_samples": int(source_embeddings.shape[0]),
        "support_samples": int(support_embeddings.shape[0]),
        "augmented_attempted": int(augmented_attempted),
        "augmented_kept": int(augmented_kept),
        "augmented_dropped": int(augmented_dropped),
        "support_expanded_classes": int(expanded_classes),
        "support_class_count": int(len(per_class_counts)),
        "support_per_class": per_class_counts,
    }


def _build_destructive_bank(
    encoder,
    source_x: np.ndarray,
    destructive_specs: list[PerturbationSpec],
    perturbation_engine: PerturbationEngine,
    max_destructive_bank: int,
    destructive_per_sample: int,
    destructive_balance: str,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    info = {
        "destructive_balance_effective": destructive_balance,
        "destructive_balance_groups": 0,
        "destructive_balance_min_group_samples": 0,
        "destructive_balance_max_group_samples": 0,
    }
    if not destructive_specs or int(max_destructive_bank) <= 0:
        return np.empty((0, 0), dtype=np.float32), info
    rng = np.random.default_rng(seed)
    sample_indices = np.arange(source_x.shape[0], dtype=np.int64)
    rng.shuffle(sample_indices)
    grouped_specs = _group_destructive_specs(destructive_specs, destructive_balance)
    group_keys = sorted(grouped_specs)
    group_counts = {key: 0 for key in group_keys}
    info["destructive_balance_groups"] = int(len(group_keys))
    out = []
    step = 0
    for idx in sample_indices.tolist():
        for _ in range(int(destructive_per_sample)):
            if destructive_balance == "none":
                spec = destructive_specs[int(rng.integers(0, len(destructive_specs)))]
                group_key = "all"
            else:
                group_key = group_keys[step % len(group_keys)]
                specs = grouped_specs[group_key]
                spec = specs[group_counts[group_key] % len(specs)]
                group_counts[group_key] += 1
                step += 1
            out.append(apply_perturbation_one(source_x[idx], spec, perturbation_engine))
            if len(out) >= int(max_destructive_bank):
                break
        if len(out) >= int(max_destructive_bank):
            break
    if not out:
        return np.empty((0, 0), dtype=np.float32), info
    if destructive_balance != "none" and group_counts:
        counts = list(group_counts.values())
        info["destructive_balance_min_group_samples"] = int(min(counts))
        info["destructive_balance_max_group_samples"] = int(max(counts))
    perturbed = np.stack(out, axis=0).astype(np.float32)
    _logits, embeddings = infer_logits_embeddings(
        encoder,
        perturbed,
        batch_size=batch_size,
        device=device,
    )
    return embeddings.astype(np.float32), info


def _group_destructive_specs(
    destructive_specs: list[PerturbationSpec],
    destructive_balance: str,
) -> dict[str, list[PerturbationSpec]]:
    if destructive_balance == "none":
        return {"all": list(destructive_specs)}
    groups: dict[str, list[PerturbationSpec]] = {}
    for spec in destructive_specs:
        if destructive_balance == "family":
            key = spec.family
        elif destructive_balance == "family_level":
            key = f"{spec.family}:{int(spec.level)}"
        else:
            raise ValueError(f"Unknown destructive_balance={destructive_balance!r}.")
        groups.setdefault(key, []).append(spec)
    return groups


def _limit_class_support(
    base_embeddings: np.ndarray,
    augmented_embeddings: np.ndarray,
    max_support_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    limit = int(max_support_per_class)
    if limit <= 0:
        return base_embeddings.astype(np.float32)
    if base_embeddings.shape[0] >= limit:
        chosen = rng.choice(base_embeddings.shape[0], size=limit, replace=False)
        return base_embeddings[chosen].astype(np.float32)
    remaining = limit - base_embeddings.shape[0]
    if augmented_embeddings.shape[0] > remaining:
        chosen = rng.choice(augmented_embeddings.shape[0], size=remaining, replace=False)
        augmented_embeddings = augmented_embeddings[chosen]
    if augmented_embeddings.shape[0] == 0:
        return base_embeddings.astype(np.float32)
    return np.concatenate([base_embeddings, augmented_embeddings], axis=0).astype(np.float32)


def _fit_centers(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.stack(
        [np.mean(embeddings[labels == cls], axis=0) for cls in range(int(num_classes))],
        axis=0,
    ).astype(np.float32)


def _class_spreads(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    distance: str,
) -> np.ndarray:
    spreads = []
    for cls in range(centers.shape[0]):
        cls_embeddings = embeddings[labels == cls]
        if cls_embeddings.shape[0] == 0:
            spreads.append(1.0)
            continue
        dist = distance_matrix(cls_embeddings, centers[cls][None, :], distance=distance)[:, 0]
        spreads.append(max(float(np.sqrt(np.mean(dist * dist))), 1e-6))
    return np.asarray(spreads, dtype=np.float32)


def _calibrate_threshold(scores: np.ndarray, frr: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float32).reshape(-1))
    if scores.size == 0:
        return 0.0
    keep_count = max(1, int(np.ceil((1.0 - float(frr)) * scores.size)))
    return float(scores[min(scores.size - 1, keep_count - 1)])
