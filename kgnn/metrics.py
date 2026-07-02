from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class OscrCurve:
    unknown_fpr: np.ndarray
    shifted_known_ccr: np.ndarray


EXTENDED_OSR_METRIC_KEYS = [
    "shifted_known_acceptance_rate",
    "accepted_shifted_known_id_accuracy",
    "known_false_rejection_rate",
    "known_acceptance_rate",
    "known_correct_id_rate",
    "known_wrong_known_rate",
    "accepted_known_id_accuracy",
    "unknown_detection_precision",
    "unknown_detection_recall",
    "unknown_detection_f1",
    "paper_a_open_set_balanced_accuracy",
    "paper_a_open_set_accuracy",
    "paper_a_open_set_error_rate",
    "residual_true_unknown_purity",
    "auosc_shifted_known_vs_unknown",
    "aupr_unknown",
    "aupr_shifted_known",
    "paper_a_auroc_shifted_known_vs_unknown",
    "unknown_fpr_at_shifted_known_ccr_95",
    "true_unknown_rejection_at_shifted_known_ccr_95",
    "unknown_fpr_at_shifted_known_ccr_90",
    "true_unknown_rejection_at_shifted_known_ccr_90",
    "unknown_fpr_at_shifted_known_ccr_80",
    "true_unknown_rejection_at_shifted_known_ccr_80",
    "shifted_known_ccr_at_unknown_fpr_01",
    "shifted_known_ccr_at_unknown_fpr_05",
    "shifted_known_ccr_at_unknown_fpr_10",
    "shifted_known_ccr_at_unknown_fpr_20",
    "target_sweep_best_open_set_h_score",
    "target_sweep_best_unknown_fpr",
    "target_sweep_best_true_unknown_rejection",
]


def compute_osr_extended_metrics(
    *,
    rejected: np.ndarray,
    predicted_label: np.ndarray,
    true_label: np.ndarray,
    is_known: np.ndarray,
    is_shifted_known: np.ndarray,
    unknown_score: np.ndarray,
) -> dict[str, float | int]:
    rejected = np.asarray(rejected, dtype=bool).reshape(-1)
    predicted_label = np.asarray(predicted_label, dtype=np.int64).reshape(-1)
    true_label = np.asarray(true_label, dtype=np.int64).reshape(-1)
    known = np.asarray(is_known, dtype=bool).reshape(-1)
    shifted = np.asarray(is_shifted_known, dtype=bool).reshape(-1)
    score = np.asarray(unknown_score, dtype=np.float32).reshape(-1)
    if not (rejected.shape == predicted_label.shape == true_label.shape == known.shape == shifted.shape == score.shape):
        raise ValueError("All OSR metric inputs must have the same one-dimensional shape.")

    unknown = ~known
    shifted_correct = shifted & (~rejected) & (predicted_label == true_label)
    shifted_wrong_known = shifted & (~rejected) & (predicted_label != true_label)
    known_correct = known & (~rejected) & (predicted_label == true_label)
    known_wrong = known & (~rejected) & (predicted_label != true_label)
    residual_n = int(rejected.sum())
    shifted_correct_rate = _mean(shifted_correct[shifted])
    true_unknown_rejection = _mean(rejected[unknown])
    shifted_frr = _mean(rejected[shifted])
    known_frr = _mean(rejected[known])
    unknown_precision = _divide(int((rejected & unknown).sum()), residual_n)
    accepted_shifted_n = int((shifted & (~rejected)).sum())
    accepted_known_n = int((known & (~rejected)).sum())
    paper_a_mask = shifted | unknown
    paper_a_correct = shifted_correct | (unknown & rejected)
    curve = oscr_curve(
        unknown_score=score,
        predicted_label=predicted_label,
        true_label=true_label,
        is_known=known,
        is_shifted_known=shifted,
    )
    metrics: dict[str, float | int] = {
        "residual_samples": residual_n,
        "shifted_known_false_rejection_rate": shifted_frr,
        "shifted_known_acceptance_rate": 1.0 - shifted_frr,
        "shifted_known_correct_id_rate": shifted_correct_rate,
        "shifted_known_wrong_known_rate": _mean(shifted_wrong_known[shifted]),
        "accepted_shifted_known_id_accuracy": _divide(int(shifted_correct.sum()), accepted_shifted_n),
        "known_false_rejection_rate": known_frr,
        "known_acceptance_rate": 1.0 - known_frr,
        "known_correct_id_rate": _mean(known_correct[known]),
        "known_wrong_known_rate": _mean(known_wrong[known]),
        "accepted_known_id_accuracy": _divide(int(known_correct.sum()), accepted_known_n),
        "true_unknown_rejection_rate": true_unknown_rejection,
        "unknown_acceptance_rate": 1.0 - true_unknown_rejection,
        "unknown_detection_precision": unknown_precision,
        "unknown_detection_recall": true_unknown_rejection,
        "unknown_detection_f1": _hmean(unknown_precision, true_unknown_rejection),
        "sample_open_set_h_score": _hmean(shifted_correct_rate, true_unknown_rejection),
        "paper_a_open_set_balanced_accuracy": 0.5 * (shifted_correct_rate + true_unknown_rejection),
        "paper_a_open_set_accuracy": _mean(paper_a_correct[paper_a_mask]),
        "paper_a_open_set_error_rate": 1.0 - _mean(paper_a_correct[paper_a_mask]),
        "residual_known_contamination_rate": _mean(known[rejected]) if residual_n else 0.0,
        "residual_true_unknown_purity": _mean(unknown[rejected]) if residual_n else 0.0,
        "auosc_shifted_known_vs_unknown": _area_under_curve(curve.unknown_fpr, curve.shifted_known_ccr),
        "aupr_unknown": _safe_average_precision(unknown, score),
        "aupr_shifted_known": _safe_average_precision(shifted[paper_a_mask], -score[paper_a_mask]),
        "paper_a_auroc_shifted_known_vs_unknown": _safe_auc(unknown[paper_a_mask], score[paper_a_mask]),
    }
    for target in [0.95, 0.90, 0.80]:
        fpr = _fpr_at_ccr(curve, target)
        suffix = int(round(target * 100))
        metrics[f"unknown_fpr_at_shifted_known_ccr_{suffix}"] = fpr
        metrics[f"true_unknown_rejection_at_shifted_known_ccr_{suffix}"] = 1.0 - fpr
    for limit in [0.01, 0.05, 0.10, 0.20]:
        suffix = int(round(limit * 100))
        metrics[f"shifted_known_ccr_at_unknown_fpr_{suffix:02d}"] = _ccr_at_fpr(curve, limit)
    best_h, best_fpr = _best_oscr_h(curve)
    metrics["target_sweep_best_open_set_h_score"] = best_h
    metrics["target_sweep_best_unknown_fpr"] = best_fpr
    metrics["target_sweep_best_true_unknown_rejection"] = 1.0 - best_fpr
    return metrics


def oscr_curve(
    *,
    unknown_score: np.ndarray,
    predicted_label: np.ndarray,
    true_label: np.ndarray,
    is_known: np.ndarray,
    is_shifted_known: np.ndarray,
) -> OscrCurve:
    score = np.asarray(unknown_score, dtype=np.float32).reshape(-1)
    pred = np.asarray(predicted_label, dtype=np.int64).reshape(-1)
    label = np.asarray(true_label, dtype=np.int64).reshape(-1)
    known = np.asarray(is_known, dtype=bool).reshape(-1)
    shifted = np.asarray(is_shifted_known, dtype=bool).reshape(-1)
    unknown = ~known
    thresholds = np.concatenate(
        [
            np.asarray([-np.inf], dtype=np.float32),
            np.unique(score),
            np.asarray([np.inf], dtype=np.float32),
        ]
    )
    fprs = []
    ccrs = []
    for threshold in thresholds:
        accepted = score <= float(threshold)
        fprs.append(_mean(accepted[unknown]))
        shifted_correct = shifted & accepted & (pred == label)
        ccrs.append(_mean(shifted_correct[shifted]))
    return _dedupe_monotone_curve(np.asarray(fprs, dtype=np.float64), np.asarray(ccrs, dtype=np.float64))


def _dedupe_monotone_curve(fpr: np.ndarray, ccr: np.ndarray) -> OscrCurve:
    order = np.argsort(fpr, kind="mergesort")
    fpr = fpr[order]
    ccr = ccr[order]
    unique_fpr = []
    max_ccr = []
    for value in np.unique(fpr):
        mask = fpr == value
        unique_fpr.append(float(value))
        max_ccr.append(float(np.max(ccr[mask])))
    x = np.asarray(unique_fpr, dtype=np.float64)
    y = np.maximum.accumulate(np.asarray(max_ccr, dtype=np.float64))
    if x.size == 0 or x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    if x[-1] < 1.0:
        x = np.append(x, 1.0)
        y = np.append(y, y[-1])
    return OscrCurve(unknown_fpr=x, shifted_known_ccr=y)


def _area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.trapz(y, x))


def _fpr_at_ccr(curve: OscrCurve, target_ccr: float) -> float:
    mask = curve.shifted_known_ccr >= float(target_ccr)
    if not np.any(mask):
        return 1.0
    return float(np.min(curve.unknown_fpr[mask]))


def _ccr_at_fpr(curve: OscrCurve, max_fpr: float) -> float:
    mask = curve.unknown_fpr <= float(max_fpr)
    if not np.any(mask):
        return 0.0
    return float(np.max(curve.shifted_known_ccr[mask]))


def _best_oscr_h(curve: OscrCurve) -> tuple[float, float]:
    unknown_rejection = 1.0 - curve.unknown_fpr
    h_values = np.asarray(
        [_hmean(float(ccr), float(rej)) for ccr, rej in zip(curve.shifted_known_ccr, unknown_rejection)],
        dtype=np.float64,
    )
    if h_values.size == 0:
        return 0.0, 1.0
    idx = int(np.argmax(h_values))
    return float(h_values[idx]), float(curve.unknown_fpr[idx])


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if np.unique(y).size < 2:
        return 0.0
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float32).reshape(-1)))


def _safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if np.unique(y).size < 2:
        return 0.0
    return float(average_precision_score(y, np.asarray(score, dtype=np.float32).reshape(-1)))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _divide(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom else 0.0


def _hmean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float(2.0 * a * b / (a + b))
