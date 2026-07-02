from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class SodaThreeStateScoreOnlyModel:
    num_classes: int
    prototypes: np.ndarray
    mu_z: np.ndarray
    var_z: np.ndarray
    ref_sid_emb: np.ndarray
    ref_sid_energy: np.ndarray
    models_kr: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]]
    fallback_k: dict[int, tuple[np.ndarray, np.ndarray, float]]
    source_rx_ids: np.ndarray
    dom_mu: np.ndarray
    dom_std: np.ndarray
    dom_global_mu: float
    dom_global_std: float
    tau_id: float
    temp_id: float
    tau_drift: float
    temp_drift: float
    unknown_threshold: float
    fusion_lambda: float
    frr: float


def fit_soda_three_state_score_only(
    *,
    source_x: np.ndarray,
    source_embeddings: np.ndarray,
    source_logits: np.ndarray,
    source_labels: np.ndarray,
    source_rx: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    num_classes: int,
    frr: float = 0.03,
    false_drift_target: float = 0.05,
    fusion_lambda: float = 0.5,
    d_dim: int = 32,
    min_rx_samples: int = 20,
) -> SodaThreeStateScoreOnlyModel:
    labels = np.asarray(source_labels, dtype=np.int64)
    train_idx = np.asarray(train_indices, dtype=np.int64)
    val_idx = np.asarray(val_indices, dtype=np.int64)

    z_train = np.asarray(source_embeddings[train_idx], dtype=np.float32)
    logits_train = np.asarray(source_logits[train_idx], dtype=np.float32)
    y_train = labels[train_idx]
    z_val = np.asarray(source_embeddings[val_idx], dtype=np.float32)
    logits_val = np.asarray(source_logits[val_idx], dtype=np.float32)
    y_val = labels[val_idx]

    prototypes = fit_class_prototypes(z_train, y_train, int(num_classes), l2norm=True)
    mu_z, var_z = fit_emb_maha_diag(z_train, y_train, int(num_classes))

    emb_train = sid_embmaha(z_train, mu_z, var_z)
    energy_train = sid_energy(logits_train)
    ref_sid_emb = np.asarray([np.mean(emb_train), np.std(emb_train) + 1e-8], dtype=np.float32)
    ref_sid_energy = np.asarray([np.mean(energy_train), np.std(energy_train) + 1e-8], dtype=np.float32)

    d_train = compute_domain_feats_batch(np.asarray(source_x[train_idx], dtype=np.float32), d_dim=d_dim)
    d_val = compute_domain_feats_batch(np.asarray(source_x[val_idx], dtype=np.float32), d_dim=d_dim)
    rx_train, source_rx_ids = encode_rx_ids(np.asarray(source_rx[train_idx], dtype=object))
    models_kr, fallback_k = fit_device_rx_models(
        d_train,
        y_train,
        rx_train,
        int(num_classes),
        source_rx_ids,
        min_n=int(min_rx_samples),
    )

    sdom_train_allk = sdom_mix_nll_allk(d_train, int(num_classes), models_kr, fallback_k, source_rx_ids)
    sdom_train_true = gather_class_scores(sdom_train_allk, y_train)
    dom_mu, dom_std, dom_global_mu, dom_global_std = fit_classwise_dom_stats(
        sdom_train_true,
        y_train,
        int(num_classes),
        min_std=0.10,
    )

    sid_val = sid_fusion_fixed(
        z_val,
        logits_val,
        mu_z,
        var_z,
        ref_sid_emb,
        ref_sid_energy,
        lam=float(fusion_lambda),
    )
    tau_id = float(np.quantile(sid_val, float(frr))) if sid_val.size else 0.0
    temp_id = robust_scale(sid_val, min_scale=0.10)
    p_unknown_val = sigmoid_np((tau_id - sid_val) / max(temp_id, 1e-6))
    unknown_threshold = float(np.quantile(p_unknown_val, 1.0 - float(frr))) if p_unknown_val.size else 0.5

    sdom_val_allk = sdom_mix_nll_allk(d_val, int(num_classes), models_kr, fallback_k, source_rx_ids)
    sdrift_val_allk = normalize_dom_matrix_by_class(
        sdom_val_allk,
        dom_mu,
        dom_std,
        dom_global_mu,
        dom_global_std,
    )
    sdrift_val_true = gather_class_scores(sdrift_val_allk, y_val)
    tau_drift = (
        float(np.quantile(sdrift_val_true, 1.0 - float(false_drift_target)))
        if sdrift_val_true.size
        else 0.0
    )
    temp_drift = robust_scale(sdrift_val_true, min_scale=0.10)

    return SodaThreeStateScoreOnlyModel(
        num_classes=int(num_classes),
        prototypes=prototypes,
        mu_z=mu_z,
        var_z=var_z,
        ref_sid_emb=ref_sid_emb,
        ref_sid_energy=ref_sid_energy,
        models_kr=models_kr,
        fallback_k=fallback_k,
        source_rx_ids=source_rx_ids,
        dom_mu=dom_mu,
        dom_std=dom_std,
        dom_global_mu=float(dom_global_mu),
        dom_global_std=float(dom_global_std),
        tau_id=float(tau_id),
        temp_id=float(temp_id),
        tau_drift=float(tau_drift),
        temp_drift=float(temp_drift),
        unknown_threshold=float(unknown_threshold),
        fusion_lambda=float(fusion_lambda),
        frr=float(frr),
    )


def score_soda_three_state(
    *,
    model: SodaThreeStateScoreOnlyModel,
    x: np.ndarray,
    embeddings: np.ndarray,
    logits: np.ndarray,
    use_gmm: bool = False,
    gmm_lambda_mid: float = 0.25,
    gmm_random_state: int = 0,
    d_dim: int = 32,
) -> dict:
    z = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    d_eval = compute_domain_feats_batch(np.asarray(x, dtype=np.float32), d_dim=d_dim)

    sid = sid_fusion_fixed(
        z,
        logits,
        model.mu_z,
        model.var_z,
        model.ref_sid_emb,
        model.ref_sid_energy,
        lam=float(model.fusion_lambda),
    )
    cos = cosine_to_proto(z, model.prototypes)
    pred = np.argmax(cos, axis=1).astype(np.int64)
    sdom_allk = sdom_mix_nll_allk(
        d_eval,
        model.num_classes,
        model.models_kr,
        model.fallback_k,
        model.source_rx_ids,
    )
    sdrift_allk = normalize_dom_matrix_by_class(
        sdom_allk,
        model.dom_mu,
        model.dom_std,
        model.dom_global_mu,
        model.dom_global_std,
    )
    sdom = gather_class_scores(sdom_allk, pred)
    sdrift = gather_class_scores(sdrift_allk, pred)

    p_unknown = sigmoid_np((model.tau_id - sid) / max(model.temp_id, 1e-6))
    p_known = np.clip(1.0 - p_unknown, 1e-8, 1.0)
    p_shift_given_known = sigmoid_np((sdrift - model.tau_drift) / max(model.temp_drift, 1e-6))
    p_stable = np.clip(p_known * (1.0 - p_shift_given_known), 1e-8, None)
    p_shift = np.clip(p_known * p_shift_given_known, 1e-8, None)
    route_probs = normalize_rows(np.stack([p_stable, p_shift, p_unknown], axis=1))
    gmm_info = {}

    if use_gmm:
        gmm_info = fit_pknown_gmm3(route_probs[:, 0] + route_probs[:, 1], random_state=gmm_random_state)
        gmm_unknown = np.clip(
            np.asarray(gmm_info["p_low"], dtype=np.float32)
            + float(gmm_lambda_mid) * np.asarray(gmm_info["p_mid"], dtype=np.float32),
            0.0,
            1.0,
        )
        p_unknown_gmm = np.maximum(route_probs[:, 2], gmm_unknown).astype(np.float32)
        known_left = np.clip(1.0 - p_unknown_gmm, 1e-8, 1.0)
        known_raw = np.clip(route_probs[:, 0] + route_probs[:, 1], 1e-8, 1.0)
        p_stable = route_probs[:, 0] / known_raw * known_left
        p_shift = route_probs[:, 1] / known_raw * known_left
        route_probs = normalize_rows(np.stack([p_stable, p_shift, p_unknown_gmm], axis=1))

    unknown_score = route_probs[:, 2].astype(np.float32)
    rejected = unknown_score >= float(model.unknown_threshold)
    state = np.argmax(route_probs, axis=1).astype(np.int64)
    return {
        "unknown_score": unknown_score,
        "rejected": rejected.astype(bool),
        "predicted_label": pred,
        "state": state,
        "route_probs": route_probs.astype(np.float32),
        "Sid": sid.astype(np.float32),
        "Sdom": sdom.astype(np.float32),
        "Sdrift": sdrift.astype(np.float32),
        "gmm_info": gmm_info,
        "info": summarize_routes(route_probs, state, model.unknown_threshold, use_gmm),
    }


def summarize_routes(route_probs: np.ndarray, state: np.ndarray, threshold: float, use_gmm: bool) -> dict:
    n = int(route_probs.shape[0])
    return {
        "soda_score_variant": "gmm" if use_gmm else "raw",
        "soda_unknown_threshold": float(threshold),
        "soda_p_stable_mean": float(np.mean(route_probs[:, 0])) if n else 0.0,
        "soda_p_shift_mean": float(np.mean(route_probs[:, 1])) if n else 0.0,
        "soda_p_unknown_mean": float(np.mean(route_probs[:, 2])) if n else 0.0,
        "soda_state_stable_rate": float(np.mean(state == 0)) if n else 0.0,
        "soda_state_shift_rate": float(np.mean(state == 1)) if n else 0.0,
        "soda_state_unknown_rate": float(np.mean(state == 2)) if n else 0.0,
    }


def compute_domain_feats_batch(x: np.ndarray, d_dim: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"Expected IQ shape (N, L, 2), got {x.shape}.")
    xc = (x[..., 0] + 1j * x[..., 1]).astype(np.complex64)
    out = np.empty((xc.shape[0], int(d_dim)), dtype=np.float32)
    for idx in range(xc.shape[0]):
        sig = xc[idx]
        sig = sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12)
        spectrum = np.fft.fft(sig, n=256)
        log_mag = np.log(np.abs(spectrum) + 1e-12)
        cep = np.fft.rfft(log_mag, n=512)
        out[idx] = np.real(cep[: int(d_dim)]).astype(np.float32)
    return out


def fit_emb_maha_diag(z_train: np.ndarray, y_train: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    z_train = np.asarray(z_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    global_mu = np.mean(z_train, axis=0)
    global_var = np.var(z_train, axis=0) + 1e-3
    mu = np.tile(global_mu[None, :], (int(num_classes), 1)).astype(np.float32)
    var = np.tile(global_var[None, :], (int(num_classes), 1)).astype(np.float32)
    for cls in range(int(num_classes)):
        z_cls = z_train[y_train == cls]
        if z_cls.size:
            mu[cls] = np.mean(z_cls, axis=0)
            var[cls] = np.var(z_cls, axis=0) + 1e-3
    return mu, var


def sid_embmaha(z: np.ndarray, mu_z: np.ndarray, var_z: np.ndarray) -> np.ndarray:
    diff = z[:, None, :] - mu_z[None, :, :]
    dist = np.sum((diff * diff) / (var_z[None, :, :] + 1e-6), axis=2)
    return (-np.min(dist, axis=1)).astype(np.float32)


def sid_energy(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(np.asarray(logits, dtype=np.float32), nan=-50.0, posinf=50.0, neginf=-50.0)
    max_logit = np.max(logits, axis=1, keepdims=True)
    energy = max_logit + np.log(np.exp(np.clip(logits - max_logit, -60.0, 60.0)).sum(axis=1, keepdims=True) + 1e-12)
    return energy[:, 0].astype(np.float32)


def zscore_fixed(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = float(ref[0])
    std = max(float(ref[1]), 1e-8)
    return ((x - mean) / std).astype(np.float32)


def sid_fusion_fixed(
    z: np.ndarray,
    logits: np.ndarray,
    mu_z: np.ndarray,
    var_z: np.ndarray,
    ref_sid_emb: np.ndarray,
    ref_sid_energy: np.ndarray,
    lam: float,
) -> np.ndarray:
    return (
        zscore_fixed(sid_embmaha(z, mu_z, var_z), ref_sid_emb)
        + float(lam) * zscore_fixed(sid_energy(logits), ref_sid_energy)
    ).astype(np.float32)


def fit_class_prototypes(z_train: np.ndarray, y_train: np.ndarray, num_classes: int, l2norm: bool = True) -> np.ndarray:
    z = np.asarray(z_train, dtype=np.float32)
    if l2norm:
        z = normalize_vec_rows(z)
    protos = np.zeros((int(num_classes), z.shape[1]), dtype=np.float32)
    global_proto = np.mean(z, axis=0)
    for cls in range(int(num_classes)):
        vals = z[np.asarray(y_train, dtype=np.int64) == cls]
        protos[cls] = np.mean(vals, axis=0) if vals.size else global_proto
    return normalize_vec_rows(protos) if l2norm else protos


def cosine_to_proto(z: np.ndarray, protos: np.ndarray) -> np.ndarray:
    return (normalize_vec_rows(z) @ normalize_vec_rows(protos).T).astype(np.float32)


def encode_rx_ids(rx_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(v) for v in rx_values.tolist()}), dtype=object)
    mapping = {str(v): idx for idx, v in enumerate(unique.tolist())}
    encoded = np.asarray([mapping[str(v)] for v in rx_values.tolist()], dtype=np.int64)
    return encoded, np.arange(len(unique), dtype=np.int64)


def fit_gaussian(d: np.ndarray, reg: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float]:
    d = np.asarray(d, dtype=np.float32)
    if d.ndim != 2 or d.shape[0] == 0:
        raise ValueError("Cannot fit Gaussian on empty data.")
    mu = np.mean(d, axis=0).astype(np.float32)
    if d.shape[0] < 2:
        cov = np.eye(d.shape[1], dtype=np.float32)
    else:
        cov = np.cov(d.T, bias=False).astype(np.float32)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float32)
    cov = cov + float(reg) * np.eye(cov.shape[0], dtype=np.float32)
    inv = np.linalg.pinv(cov).astype(np.float32)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = float(np.log(max(float(np.linalg.det(cov)), 1e-12)))
    return mu, inv, float(logdet)


def fit_device_rx_models(
    d_train: np.ndarray,
    y_train: np.ndarray,
    rx_train: np.ndarray,
    num_classes: int,
    source_rx_ids: np.ndarray,
    reg: float = 1e-3,
    min_n: int = 20,
) -> tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]], dict[int, tuple[np.ndarray, np.ndarray, float]]]:
    models: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    fallback: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for cls in range(int(num_classes)):
        cls_mask = np.asarray(y_train, dtype=np.int64) == cls
        fallback[cls] = fit_gaussian(d_train[cls_mask], reg=reg)
        for rx in source_rx_ids:
            idx = cls_mask & (np.asarray(rx_train, dtype=np.int64) == int(rx))
            if int(np.sum(idx)) >= int(min_n):
                models[(cls, int(rx))] = fit_gaussian(d_train[idx], reg=reg)
    return models, fallback


def logpdf_gaussian(d: np.ndarray, model: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    mu, inv, logdet = model
    x = np.asarray(d, dtype=np.float32) - mu.reshape(1, -1)
    maha = np.einsum("nd,dd,nd->n", x, inv, x)
    dim = x.shape[1]
    return (-0.5 * (maha + float(logdet) + dim * np.log(2.0 * np.pi))).astype(np.float32)


def sdom_mix_nll_allk(
    d_eval: np.ndarray,
    num_classes: int,
    models_kr: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]],
    fallback_k: dict[int, tuple[np.ndarray, np.ndarray, float]],
    source_rx_ids: np.ndarray,
) -> np.ndarray:
    d_eval = np.asarray(d_eval, dtype=np.float32)
    out = np.zeros((d_eval.shape[0], int(num_classes)), dtype=np.float32)
    rx_ids = [int(x) for x in np.asarray(source_rx_ids, dtype=np.int64).tolist()]
    for cls in range(int(num_classes)):
        parts = []
        for rx in rx_ids:
            parts.append(logpdf_gaussian(d_eval, models_kr.get((cls, rx), fallback_k[cls])))
        logps = np.stack(parts, axis=1)
        max_logp = np.max(logps, axis=1, keepdims=True)
        loglik = max_logp[:, 0] + np.log(np.exp(logps - max_logp).sum(axis=1) + 1e-12) - np.log(max(1, len(rx_ids)))
        out[:, cls] = (-loglik).astype(np.float32)
    return out


def gather_class_scores(score_allk: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.clip(np.asarray(labels, dtype=np.int64), 0, score_allk.shape[1] - 1)
    return score_allk[np.arange(score_allk.shape[0]), labels].astype(np.float32)


def fit_classwise_dom_stats(
    sdom_true: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    min_std: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    sdom_true = np.asarray(sdom_true, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    global_mu = float(np.mean(sdom_true)) if sdom_true.size else 0.0
    global_std = float(max(np.std(sdom_true), float(min_std))) if sdom_true.size else float(min_std)
    dom_mu = np.full((int(num_classes),), global_mu, dtype=np.float32)
    dom_std = np.full((int(num_classes),), global_std, dtype=np.float32)
    for cls in range(int(num_classes)):
        vals = sdom_true[y == cls]
        if vals.size >= 8:
            dom_mu[cls] = float(np.mean(vals))
            dom_std[cls] = float(max(np.std(vals), float(min_std)))
    return dom_mu, dom_std, global_mu, global_std


def normalize_dom_matrix_by_class(
    sdom_allk: np.ndarray,
    dom_mu: np.ndarray,
    dom_std: np.ndarray,
    dom_global_mu: float,
    dom_global_std: float,
) -> np.ndarray:
    out = np.zeros_like(np.asarray(sdom_allk, dtype=np.float32))
    for cls in range(out.shape[1]):
        mu = float(dom_mu[cls]) if cls < len(dom_mu) else float(dom_global_mu)
        std = float(dom_std[cls]) if cls < len(dom_std) else float(dom_global_std)
        out[:, cls] = (sdom_allk[:, cls] - mu) / max(std, 1e-6)
    return out.astype(np.float32)


def fit_pknown_gmm3(p_known: np.ndarray, random_state: int = 0, eps: float = 1e-6) -> dict:
    p_raw = np.asarray(p_known, dtype=np.float64).reshape(-1)
    n = int(p_raw.shape[0])
    p_safe = np.clip(np.where(np.isfinite(p_raw), p_raw, np.nanmedian(p_raw)), float(eps), 1.0 - float(eps))
    if n < 3:
        return _fallback_gmm3(p_safe, "n_below_3")
    try:
        x = logit_np(p_safe).reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=3,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=5,
            random_state=int(random_state),
        )
        gmm.fit(x)
        means = sigmoid_np(np.asarray(gmm.means_, dtype=np.float64).reshape(-1))
        order = np.argsort(means)
        resp = gmm.predict_proba(x)[:, order].astype(np.float32)
        weights = np.asarray(gmm.weights_, dtype=np.float64).reshape(-1)[order]
        return {
            "p_low": resp[:, 0],
            "p_mid": resp[:, 1],
            "p_high": resp[:, 2],
            "gmm3_fit_success": True,
            "gmm3_fallback_used": False,
            "gmm3_fallback_reason": "",
            "gmm3_component_mean_pknown_low": float(means[order][0]),
            "gmm3_component_mean_pknown_mid": float(means[order][1]),
            "gmm3_component_mean_pknown_high": float(means[order][2]),
            "gmm3_component_weight_low": float(weights[0]),
            "gmm3_component_weight_mid": float(weights[1]),
            "gmm3_component_weight_high": float(weights[2]),
        }
    except Exception as exc:
        return _fallback_gmm3(p_safe, f"fit_exception:{exc}")


def _fallback_gmm3(p_safe: np.ndarray, reason: str) -> dict:
    tau_low = float(np.quantile(p_safe, 0.10)) if p_safe.size else 0.10
    tau_mid = float(np.quantile(p_safe, 0.50)) if p_safe.size else 0.50
    p_low = (p_safe < tau_low).astype(np.float32)
    p_mid = ((p_safe >= tau_low) & (p_safe < tau_mid)).astype(np.float32)
    p_high = (p_safe >= tau_mid).astype(np.float32)
    return {
        "p_low": p_low,
        "p_mid": p_mid,
        "p_high": p_high,
        "gmm3_fit_success": False,
        "gmm3_fallback_used": True,
        "gmm3_fallback_reason": str(reason),
    }


def robust_scale(x: np.ndarray, min_scale: float = 0.10) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(min_scale)
    q75, q25 = np.quantile(x, [0.75, 0.25])
    iqr = float(q75 - q25)
    std = float(np.std(x))
    return float(max(iqr / 1.349 if iqr > 1e-8 else 0.0, std, float(min_scale)))


def normalize_vec_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.maximum(p, 1e-12)
    return (p / np.sum(p, axis=1, keepdims=True)).astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float32), -30.0, 30.0)))).astype(np.float32)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))
