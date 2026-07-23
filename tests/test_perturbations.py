import numpy as np

from dpr_rffi.perturbations import (
    PerturbationEngine,
    PerturbationSpec,
    default_perturbation_specs,
)


def test_default_grid_contains_52_settings():
    specs = default_perturbation_specs()
    assert len(specs) == 52
    assert len({item.name for item in specs}) == 52
    assert {item.family for item in specs} == {
        "phase",
        "cfo",
        "timing",
        "amplitude",
        "iq_imbalance",
        "noise",
        "multipath",
    }


def test_phase_rotation_preserves_shape_and_power():
    x = np.zeros((32, 2), dtype=np.float32)
    x[:, 0] = 1.0
    spec = PerturbationSpec("quarter_turn", "phase", 1, {"theta": np.pi / 2})
    y = PerturbationEngine(seed=1).apply(x, spec)
    assert y.shape == x.shape
    np.testing.assert_allclose(np.sum(y * y, axis=1), 1.0, atol=1e-6)
