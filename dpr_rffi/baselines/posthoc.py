from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto
from scipy.stats import weibull_min
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class KNNScore:
    scores: np.ndarray
    predicted_label: np.ndarray


@dataclass(frozen=True)
class MahalanobisModel:
    centers: np.ndarray
    precision: np.ndarray


@dataclass(frozen=True)
class OpenMaxModel:
    mavs: np.ndarray
    weibull_params: list[tuple[float, float, float]]
    alpha: int


@dataclass(frozen=True)
class ReActModel:
    clip_value: float
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray
    temperature: float


@dataclass(frozen=True)
class ViMModel:
    origin: np.ndarray
    null_space: np.ndarray
    alpha: float
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray


@dataclass(frozen=True)
class RbfOpenSvddModel:
    models: list[OneClassSVM]


@dataclass(frozen=True)
class HmeGpdModel:
    centroids: np.ndarray
    tail_start: float
    gpd_shape: float
    gpd_scale: float


@dataclass(frozen=True)
class MeDaeCenterModel:
    centers: np.ndarray
    feature_dim: int


@dataclass(frozen=True)
class SyntheticGOpenMaxModel:
    mean: np.ndarray
    scale: np.ndarray
    classifier: LogisticRegression
    openmax: OpenMaxModel
    num_known_classes: int


def max_logit_unknown_score(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return (-np.max(logits, axis=1)).astype(np.float32)


def energy_unknown_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    scaled = logits / float(temperature)
    return (-float(temperature) * _logsumexp(scaled, axis=1)).astype(np.float32)


def knn_unknown_score(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    k: int = 5,
    metric: str = "euclidean",
) -> KNNScore:
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    n_neighbors = max(1, min(int(k), train_embeddings.shape[0]))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(train_embeddings)
    distances, indices = nn.kneighbors(query_embeddings, return_distance=True)
    score = np.mean(distances, axis=1).astype(np.float32)
    predicted = _majority_label(train_labels[indices])
    return KNNScore(scores=score, predicted_label=predicted)


def nndr_unknown_score(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    metric: str = "cosine",
    epsilon: float = 1e-6,
) -> KNNScore:
    """Nearest-neighbor distance ratio with a different-class denominator."""

    train = np.asarray(train_embeddings, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    query = np.asarray(query_embeddings, dtype=np.float32)
    if train.shape[0] != labels.size or np.unique(labels).size < 2:
        raise ValueError("NNDR requires aligned source features from at least two classes.")
    nearest = NearestNeighbors(n_neighbors=train.shape[0], metric=metric).fit(train)
    distances, indices = nearest.kneighbors(query, return_distance=True)
    ordered_labels = labels[indices]
    prediction = ordered_labels[:, 0].astype(np.int64)
    numerator = distances[:, 0]
    denominator = np.empty(query.shape[0], dtype=np.float32)
    for row in range(query.shape[0]):
        different = np.flatnonzero(ordered_labels[row] != prediction[row])
        if different.size == 0:
            raise RuntimeError("NNDR could not find a source sample from another class.")
        denominator[row] = float(distances[row, different[0]])
    ratio = numerator / np.maximum(denominator, float(epsilon))
    return KNNScore(scores=ratio.astype(np.float32), predicted_label=prediction)


def fit_ledoit_mahalanobis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> MahalanobisModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centers = _class_centers(embeddings, labels, num_classes=num_classes)
    residuals = embeddings - centers[labels]
    covariance = LedoitWolf().fit(residuals)
    return MahalanobisModel(
        centers=centers.astype(np.float32),
        precision=covariance.precision_.astype(np.float32),
    )


def mahalanobis_unknown_score(
    embeddings: np.ndarray,
    model: MahalanobisModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    diff = embeddings[:, None, :] - model.centers[None, :, :]
    projected = np.einsum("ncd,df->ncf", diff, model.precision)
    dist2 = np.sum(projected * diff, axis=2)
    dist = np.sqrt(np.maximum(dist2, 0.0))
    pred = np.argmin(dist, axis=1).astype(np.int64)
    return np.min(dist, axis=1).astype(np.float32), pred


def fit_openmax(
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    tail_size: int = 20,
    alpha: int = 10,
) -> OpenMaxModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = np.argmax(logits, axis=1).astype(np.int64)
    mavs = []
    weibull_params: list[tuple[float, float, float]] = []
    for cls in range(int(num_classes)):
        cls_mask = labels == cls
        correct_mask = cls_mask & (predicted == cls)
        fit_mask = correct_mask if np.any(correct_mask) else cls_mask
        cls_embeddings = embeddings[fit_mask]
        if cls_embeddings.size == 0:
            raise ValueError(f"Cannot fit OpenMax class {cls}: no samples.")
        mav = np.mean(cls_embeddings, axis=0)
        distances = np.linalg.norm(cls_embeddings - mav[None, :], axis=1)
        tail_count = max(2, min(int(tail_size), distances.size))
        tail = np.sort(distances)[-tail_count:]
        params = weibull_min.fit(np.maximum(tail, 1e-6), floc=0.0)
        mavs.append(mav.astype(np.float32))
        weibull_params.append(tuple(float(x) for x in params))
    return OpenMaxModel(
        mavs=np.stack(mavs, axis=0).astype(np.float32),
        weibull_params=weibull_params,
        alpha=max(1, int(alpha)),
    )


def openmax_unknown_score(
    embeddings: np.ndarray,
    logits: np.ndarray,
    model: OpenMaxModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    distances = np.linalg.norm(embeddings[:, None, :] - model.mavs[None, :, :], axis=2)
    ranked = np.argsort(logits, axis=1)[:, ::-1]
    activations = logits - np.min(logits, axis=1, keepdims=True)
    activations = np.maximum(activations, 0.0).astype(np.float32)
    revised = activations.copy()
    unknown_mass = np.zeros(logits.shape[0], dtype=np.float32)
    alpha_count = min(model.alpha, logits.shape[1])
    for sample_idx in range(logits.shape[0]):
        for rank, cls in enumerate(ranked[sample_idx, :alpha_count]):
            shape, loc, scale = model.weibull_params[int(cls)]
            tail_prob = float(weibull_min.cdf(max(distances[sample_idx, cls], 1e-6), shape, loc=loc, scale=scale))
            weight = (alpha_count - rank) / float(alpha_count)
            removed = revised[sample_idx, cls] * tail_prob * weight
            revised[sample_idx, cls] -= removed
            unknown_mass[sample_idx] += removed
    openmax_logits = np.concatenate([revised, unknown_mass[:, None]], axis=1)
    probs = _softmax(openmax_logits)
    return probs[:, -1].astype(np.float32), np.argmax(revised, axis=1).astype(np.int64)


def fit_react(
    embeddings: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    *,
    quantile: float = 0.95,
    temperature: float = 1.0,
) -> ReActModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return ReActModel(
        clip_value=float(np.quantile(embeddings, float(quantile))),
        classifier_weight=np.asarray(classifier_weight, dtype=np.float32),
        classifier_bias=np.asarray(classifier_bias, dtype=np.float32),
        temperature=float(temperature),
    )


def react_unknown_score(
    embeddings: np.ndarray,
    model: ReActModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    clipped = np.minimum(embeddings, model.clip_value)
    logits = clipped @ model.classifier_weight.T + model.classifier_bias[None, :]
    return energy_unknown_score(logits, temperature=model.temperature), np.argmax(logits, axis=1).astype(np.int64)


def fit_vim(
    embeddings: np.ndarray,
    logits: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    *,
    explained_variance: float = 0.90,
) -> ViMModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    weight = np.asarray(classifier_weight, dtype=np.float32)
    bias = np.asarray(classifier_bias, dtype=np.float32)
    origin = -(np.linalg.pinv(weight) @ bias).astype(np.float32)
    centered = embeddings - origin[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values * singular_values
    explained = np.cumsum(variance) / max(float(np.sum(variance)), 1e-12)
    principal_dim = int(np.searchsorted(explained, float(explained_variance)) + 1)
    principal_dim = min(max(1, principal_dim), max(1, vh.shape[0] - 1))
    null_space = vh[principal_dim:].T.astype(np.float32)
    residual = np.linalg.norm(centered @ null_space, axis=1)
    logit_scale = np.mean(np.max(logits, axis=1))
    alpha = float(logit_scale / max(float(np.mean(residual)), 1e-6))
    return ViMModel(
        origin=origin,
        null_space=null_space,
        alpha=max(alpha, 1e-6),
        classifier_weight=weight,
        classifier_bias=bias,
    )


def vim_unknown_score(
    embeddings: np.ndarray,
    logits: np.ndarray,
    model: ViMModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    residual = np.linalg.norm((embeddings - model.origin[None, :]) @ model.null_space, axis=1)
    score = model.alpha * residual - _logsumexp(logits, axis=1)
    pred = np.argmax(logits, axis=1).astype(np.int64)
    return score.astype(np.float32), pred


def fit_rbf_opensvdd(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    nu: float = 0.05,
    gamma: str | float = "scale",
) -> RbfOpenSvddModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    models: list[OneClassSVM] = []
    for cls in range(int(num_classes)):
        cls_embeddings = embeddings[labels == cls]
        if cls_embeddings.size == 0:
            raise ValueError(f"Cannot fit RBF OpenSVDD class {cls}: no samples.")
        model = OneClassSVM(kernel="rbf", nu=float(nu), gamma=gamma)
        model.fit(cls_embeddings)
        models.append(model)
    return RbfOpenSvddModel(models=models)


def rbf_opensvdd_unknown_score(
    embeddings: np.ndarray,
    model: RbfOpenSvddModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    decisions = np.stack(
        [svdd.decision_function(embeddings).reshape(-1) for svdd in model.models],
        axis=1,
    )
    pred = np.argmax(decisions, axis=1).astype(np.int64)
    return (-np.max(decisions, axis=1)).astype(np.float32), pred


def fit_hme_gpd(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    tail_quantile: float = 0.99,
    min_tail: int = 20,
) -> HmeGpdModel:
    embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centroids = _l2_normalize(_class_centers(embeddings, labels, num_classes=num_classes))
    distances, _ = _hme_cosine_distance(embeddings, centroids)
    tail_start = float(np.quantile(distances, float(tail_quantile)))
    tail = distances[distances >= tail_start]
    if tail.size < min_tail:
        tail_count = min(max(2, int(min_tail)), distances.size)
        tail = np.sort(distances)[-tail_count:]
        tail_start = float(np.min(tail))
    excess = np.maximum(tail - tail_start, 1e-6)
    shape, _, scale = genpareto.fit(excess, floc=0.0)
    return HmeGpdModel(
        centroids=centroids.astype(np.float32),
        tail_start=tail_start,
        gpd_shape=float(shape),
        gpd_scale=max(float(scale), 1e-6),
    )


def hme_gpd_unknown_score(
    embeddings: np.ndarray,
    model: HmeGpdModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    distances, pred = _hme_cosine_distance(embeddings, model.centroids)
    below_tail = distances <= model.tail_start
    excess = np.maximum(distances - model.tail_start, 0.0)
    tail_prob = genpareto.cdf(
        excess,
        model.gpd_shape,
        loc=0.0,
        scale=model.gpd_scale,
    )
    in_tail_scale = distances / max(float(model.tail_start), 1e-6)
    score = np.where(below_tail, 0.5 * in_tail_scale, 0.5 + 0.5 * tail_prob)
    return score.astype(np.float32), pred


def hme_cosine_unknown_score(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_embeddings = _l2_normalize(np.asarray(train_embeddings, dtype=np.float32))
    query_embeddings = _l2_normalize(np.asarray(query_embeddings, dtype=np.float32))
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    centroids = _l2_normalize(_class_centers(train_embeddings, train_labels, num_classes=num_classes))
    return _hme_cosine_distance(query_embeddings, centroids)


def fit_medae_center(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> MeDaeCenterModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return MeDaeCenterModel(
        centers=_class_centers(embeddings, labels, num_classes=num_classes),
        feature_dim=int(embeddings.shape[1]),
    )


def medae_center_unknown_score(
    embeddings: np.ndarray,
    model: MeDaeCenterModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    distances = np.linalg.norm(embeddings[:, None, :] - model.centers[None, :, :], axis=2)
    pred = np.argmin(distances, axis=1).astype(np.int64)
    scale = np.sqrt(max(1, 3 * model.feature_dim))
    return (np.min(distances, axis=1) / scale).astype(np.float32), pred


def fit_synthetic_gopenmax(
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    tail_size: int = 20,
    alpha: int = 3,
    random_state: int = 42,
) -> SyntheticGOpenMaxModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    synthetic = _synthetic_outliers(embeddings, labels, num_classes=num_classes, random_state=random_state)
    synthetic_labels = np.full(synthetic.shape[0], int(num_classes), dtype=np.int64)
    train_x = np.concatenate([embeddings, synthetic], axis=0)
    train_y = np.concatenate([labels, synthetic_labels], axis=0)
    mean = np.mean(train_x, axis=0).astype(np.float32)
    scale = np.maximum(np.std(train_x, axis=0), 1e-6).astype(np.float32)
    train_z = (train_x - mean[None, :]) / scale[None, :]
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=int(random_state),
        solver="lbfgs",
    )
    classifier.fit(train_z, train_y)
    train_aug_logits = classifier.decision_function(train_z).astype(np.float32)
    if train_aug_logits.ndim == 1:
        train_aug_logits = np.stack([-train_aug_logits, train_aug_logits], axis=1)
    openmax = fit_openmax(
        train_x,
        train_aug_logits,
        train_y,
        num_classes=int(num_classes) + 1,
        tail_size=tail_size,
        alpha=min(int(alpha), int(num_classes) + 1),
    )
    return SyntheticGOpenMaxModel(
        mean=mean,
        scale=scale,
        classifier=classifier,
        openmax=openmax,
        num_known_classes=int(num_classes),
    )


def synthetic_gopenmax_unknown_score(
    embeddings: np.ndarray,
    model: SyntheticGOpenMaxModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    z = (embeddings - model.mean[None, :]) / model.scale[None, :]
    logits = model.classifier.decision_function(z).astype(np.float32)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    probs = _softmax(logits)
    openmax_score, _ = openmax_unknown_score(embeddings, logits, model.openmax)
    known_probs = probs[:, : model.num_known_classes]
    generated_prob = probs[:, model.num_known_classes]
    confidence_score = 1.0 - np.max(known_probs, axis=1)
    score = np.maximum.reduce([openmax_score, generated_prob, confidence_score])
    pred = np.argmax(known_probs, axis=1).astype(np.int64)
    return score.astype(np.float32), pred


def _class_centers(embeddings: np.ndarray, labels: np.ndarray, *, num_classes: int) -> np.ndarray:
    centers = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError(f"Cannot fit center for class {cls}: no samples.")
        centers.append(np.mean(embeddings[mask], axis=0))
    return np.stack(centers, axis=0).astype(np.float32)


def _majority_label(neighbor_labels: np.ndarray) -> np.ndarray:
    out = []
    for labels in np.asarray(neighbor_labels, dtype=np.int64):
        values, counts = np.unique(labels, return_counts=True)
        out.append(int(values[np.argmax(counts)]))
    return np.asarray(out, dtype=np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_values = np.max(values, axis=axis, keepdims=True)
    return (
        np.squeeze(max_values, axis=axis)
        + np.log(np.sum(np.exp(values - max_values), axis=axis))
    )


def _l2_normalize(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, float(eps))


def _hme_cosine_distance(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.asarray(embeddings, dtype=np.float32) @ np.asarray(centroids, dtype=np.float32).T
    pred = np.argmax(similarity, axis=1).astype(np.int64)
    return (1.0 - np.max(similarity, axis=1)).astype(np.float32), pred


def _synthetic_outliers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    centers = _class_centers(embeddings, labels, num_classes=num_classes)
    outliers = []
    for cls in range(int(num_classes)):
        cls_embeddings = embeddings[labels == cls]
        if cls_embeddings.size == 0:
            continue
        center = centers[cls]
        distances = np.linalg.norm(cls_embeddings - center[None, :], axis=1)
        edge_count = min(max(4, cls_embeddings.shape[0] // 4), cls_embeddings.shape[0])
        edge = cls_embeddings[np.argsort(distances)[-edge_count:]]
        outliers.append(center[None, :] + 1.8 * (edge - center[None, :]))
        other_classes = [other for other in range(int(num_classes)) if other != cls]
        if other_classes:
            bridge = []
            for other in other_classes:
                direction = center - centers[other]
                bridge.append(center + 1.2 * direction)
            outliers.append(np.stack(bridge, axis=0))
        radius = max(float(np.quantile(distances, 0.95)), 1e-6)
        random_dirs = rng.normal(0.0, 1.0, size=(edge_count, embeddings.shape[1]))
        random_dirs = random_dirs / np.maximum(np.linalg.norm(random_dirs, axis=1, keepdims=True), 1e-6)
        outliers.append(center[None, :] + random_dirs * radius * rng.uniform(1.2, 1.8, size=(edge_count, 1)))
    if not outliers:
        raise ValueError("Cannot synthesize G-OpenMax outliers: no class embeddings.")
    return np.concatenate(outliers, axis=0).astype(np.float32)
