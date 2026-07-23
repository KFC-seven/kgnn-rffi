import numpy as np

from dpr_rffi.model import DPRConfig, DPRRFFI, class_consistency
from dpr_rffi.perturbations import PerturbationSpec


def _dataset(samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    class_zero = np.zeros((samples_per_class, 16, 2), dtype=np.float32)
    class_one = np.zeros((samples_per_class, 16, 2), dtype=np.float32)
    class_zero[:, :, 0] = 1.0
    class_one[:, :, 1] = 1.0
    class_zero += rng.normal(0.0, 0.01, size=class_zero.shape)
    class_one += rng.normal(0.0, 0.01, size=class_one.shape)
    return (
        np.concatenate([class_zero, class_one], axis=0),
        np.asarray([0] * samples_per_class + [1] * samples_per_class, dtype=np.int64),
    )


def _encode(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=1).astype(np.float32)


def _classify(x: np.ndarray) -> np.ndarray:
    return np.argmax(_encode(x), axis=1).astype(np.int64)


def test_end_to_end_fit_uses_paper_facing_reference_roles():
    train_x, train_y = _dataset(12)
    val_x, val_y = _dataset(6)
    specs = [
        PerturbationSpec("identity", "phase", 1, {"theta": 0.0}),
        PerturbationSpec("half_turn", "phase", 2, {"theta": np.pi}),
    ]
    model = DPRRFFI(
        DPRConfig(
            screening_samples_per_class=6,
            low_augmentations_per_sample=1,
            low_reference_limit_per_class=100,
            high_reference_limit=100,
            knn_k=3,
        )
    ).fit(
        source_train_x=train_x,
        source_train_y=train_y,
        source_val_x=val_x,
        source_val_y=val_y,
        encode=_encode,
        predict_labels=_classify,
        perturbation_specs=specs,
    )
    prediction = model.predict(val_x)
    assert model.reference_summary["low_impact_settings"] == 1
    assert model.reference_summary["high_impact_settings"] == 1
    assert model.reference_summary["high_reference_features"] > 0
    assert prediction.score.shape == (val_x.shape[0],)
    assert np.all(np.isfinite(prediction.score))
    assert np.all((prediction.cca >= 0.0) & (prediction.cca <= 1.0))


def test_class_consistency_boundary_values():
    value = class_consistency(
        np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        np.ones(4, dtype=np.float32),
        expansion=2.0,
        epsilon=1e-6,
    )
    np.testing.assert_allclose(value[[0, 1]], 1.0)
    assert 0.49 < value[2] < 0.51
    assert value[3] == 0.0
