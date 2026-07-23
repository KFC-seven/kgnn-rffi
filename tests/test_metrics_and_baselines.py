import numpy as np

from dpr_rffi.baselines.posthoc import nndr_unknown_score
from dpr_rffi.metrics import open_set_metrics


def test_metrics_are_finite_with_numpy_1_and_2():
    metrics = open_set_metrics(
        unknown_score=np.asarray([0.1, 0.2, 0.8, 0.9]),
        rejected=np.asarray([False, False, True, True]),
        predicted_label=np.asarray([0, 1, -1, -1]),
        true_label=np.asarray([0, 1, -1, -1]),
        is_known=np.asarray([True, True, False, False]),
    )
    assert metrics["h_score"] == 1.0
    assert metrics["unknown_rejection_rate"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_nndr_uses_a_different_class_denominator():
    source = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    query = np.asarray([[0.95, 0.05]], dtype=np.float32)
    result = nndr_unknown_score(source, labels, query)
    assert result.predicted_label.tolist() == [0]
    assert 0.0 <= float(result.scores[0]) < 1.0
