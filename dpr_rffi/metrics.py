from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class OSCRCurve:
    unknown_false_acceptance_rate: np.ndarray
    known_correct_classification_rate: np.ndarray


def open_set_metrics(
    *,
    unknown_score: np.ndarray,
    rejected: np.ndarray,
    predicted_label: np.ndarray,
    true_label: np.ndarray,
    is_known: np.ndarray,
) -> dict[str, float]:
    """Compute the paper's H-score, OSCR, AUROC, rejection, ACC, and FRR."""

    score = np.asarray(unknown_score, dtype=np.float32).reshape(-1)
    reject = np.asarray(rejected, dtype=bool).reshape(-1)
    prediction = np.asarray(predicted_label, dtype=np.int64).reshape(-1)
    label = np.asarray(true_label, dtype=np.int64).reshape(-1)
    known = np.asarray(is_known, dtype=bool).reshape(-1)
    if len({score.size, reject.size, prediction.size, label.size, known.size}) != 1:
        raise ValueError("All metric inputs must be aligned.")
    unknown = ~known
    correct_known = known & (~reject) & (prediction == label)
    known_correct_rate = _mean(correct_known[known])
    unknown_rejection = _mean(reject[unknown])
    false_rejection = _mean(reject[known])
    accuracy = _mean((correct_known | (unknown & reject)))
    curve = oscr_curve(
        unknown_score=score,
        predicted_label=prediction,
        true_label=label,
        is_known=known,
    )
    return {
        "h_score": harmonic_mean(known_correct_rate, unknown_rejection),
        "oscr": _area(
            curve.unknown_false_acceptance_rate,
            curve.known_correct_classification_rate,
        ),
        "auroc": _safe_auroc(unknown, score),
        "unknown_rejection_rate": unknown_rejection,
        "accuracy": accuracy,
        "false_rejection_rate": false_rejection,
        "known_correct_classification_rate": known_correct_rate,
    }


def oscr_curve(
    *,
    unknown_score: np.ndarray,
    predicted_label: np.ndarray,
    true_label: np.ndarray,
    is_known: np.ndarray,
) -> OSCRCurve:
    score = np.asarray(unknown_score, dtype=np.float32).reshape(-1)
    prediction = np.asarray(predicted_label, dtype=np.int64).reshape(-1)
    label = np.asarray(true_label, dtype=np.int64).reshape(-1)
    known = np.asarray(is_known, dtype=bool).reshape(-1)
    unknown = ~known
    thresholds = np.concatenate(
        [
            np.asarray([-np.inf], dtype=np.float32),
            np.unique(score),
            np.asarray([np.inf], dtype=np.float32),
        ]
    )
    false_acceptance = []
    correct_classification = []
    for threshold in thresholds:
        accepted = score <= float(threshold)
        false_acceptance.append(_mean(accepted[unknown]))
        correct = known & accepted & (prediction == label)
        correct_classification.append(_mean(correct[known]))
    x = np.asarray(false_acceptance, dtype=np.float64)
    y = np.asarray(correct_classification, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    unique_x = np.unique(x)
    maximum_y = np.asarray([np.max(y[x == value]) for value in unique_x])
    maximum_y = np.maximum.accumulate(maximum_y)
    if unique_x[0] > 0.0:
        unique_x = np.insert(unique_x, 0, 0.0)
        maximum_y = np.insert(maximum_y, 0, 0.0)
    if unique_x[-1] < 1.0:
        unique_x = np.append(unique_x, 1.0)
        maximum_y = np.append(maximum_y, maximum_y[-1])
    return OSCRCurve(unique_x, maximum_y)


def harmonic_mean(a: float, b: float) -> float:
    return float(2.0 * a * b / (a + b)) if a > 0.0 and b > 0.0 else 0.0


def _area(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    return float(roc_auc_score(y_true.astype(np.int64), score))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0
