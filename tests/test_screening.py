import numpy as np

from dpr_rffi.perturbations import PerturbationSpec
from dpr_rffi.screening import screen_perturbations


def _signals() -> tuple[np.ndarray, np.ndarray]:
    class_zero = np.zeros((8, 16, 2), dtype=np.float32)
    class_one = np.zeros((8, 16, 2), dtype=np.float32)
    class_zero[:, :, 0] = 1.0
    class_one[:, :, 1] = 1.0
    return (
        np.concatenate([class_zero, class_one], axis=0),
        np.asarray([0] * 8 + [1] * 8, dtype=np.int64),
    )


def _predict(x: np.ndarray) -> np.ndarray:
    means = np.mean(x, axis=1)
    return np.argmax(means, axis=1).astype(np.int64)


def test_retention_score_assigns_low_and_high_impact_roles():
    x, y = _signals()
    specs = [
        PerturbationSpec("identity", "phase", 1, {"theta": 0.0}),
        PerturbationSpec("half_turn", "phase", 2, {"theta": np.pi}),
    ]
    results = screen_perturbations(
        _predict,
        x,
        y,
        num_classes=2,
        specs=specs,
        max_samples_per_class=8,
    )
    assert results[0].retention_score == 1.0
    assert results[0].role == "low-impact"
    assert results[1].retention_score < 0.5
    assert results[1].role == "high-impact"


def test_degenerate_source_classifier_leaves_all_settings_neutral():
    x, y = _signals()
    results = screen_perturbations(
        lambda values: np.zeros(values.shape[0], dtype=np.int64),
        x,
        y,
        num_classes=2,
        specs=[PerturbationSpec("identity", "phase", 1, {"theta": 0.0})],
        max_samples_per_class=8,
    )
    assert results[0].clean_accuracy == 0.5
    assert results[0].role == "neutral"
