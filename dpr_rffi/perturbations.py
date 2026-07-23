from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerturbationSpec:
    """One RF perturbation type and strength."""

    name: str
    family: str
    level: int
    parameters: dict[str, float | int]


class PerturbationEngine:
    """RF transformations applied to real-valued ``(length, 2)`` I/Q arrays."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(int(seed))

    def apply(self, iq: np.ndarray, spec: PerturbationSpec) -> np.ndarray:
        x = _as_iq(iq)
        p = spec.parameters
        if spec.family == "phase":
            return self.phase_rotation(x, float(p["theta"]))
        if spec.family == "cfo":
            return self.carrier_frequency_offset(x, float(p["cycles"]))
        if spec.family == "timing":
            return self.timing_offset(x, float(p["offset"]))
        if spec.family == "amplitude":
            return self.amplitude_scaling(x, float(p["alpha"]))
        if spec.family == "iq_imbalance":
            return self.iq_imbalance(x, float(p["gain"]), float(p["phase"]))
        if spec.family == "noise":
            return self.additive_noise(x, float(p["snr_db"]))
        if spec.family == "multipath":
            return self.multipath_filter(x, int(p["n_taps"]))
        raise ValueError(f"Unknown perturbation family: {spec.family!r}")

    def apply_batch(self, x: np.ndarray, spec: PerturbationSpec) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError(f"Expected shape (N, L, 2), got {values.shape}.")
        return np.stack([self.apply(sample, spec) for sample in values], axis=0)

    @staticmethod
    def phase_rotation(iq: np.ndarray, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        i, q = iq[:, 0], iq[:, 1]
        return np.stack([c * i - s * q, s * i + c * q], axis=-1).astype(np.float32)

    @staticmethod
    def carrier_frequency_offset(iq: np.ndarray, cycles: float) -> np.ndarray:
        n = iq.shape[0]
        phase = 2.0 * np.pi * cycles * np.arange(n, dtype=np.float32) / max(n - 1, 1)
        c, s = np.cos(phase), np.sin(phase)
        i, q = iq[:, 0], iq[:, 1]
        return np.stack([c * i - s * q, s * i + c * q], axis=-1).astype(np.float32)

    @staticmethod
    def timing_offset(iq: np.ndarray, offset: float) -> np.ndarray:
        base = np.arange(iq.shape[0], dtype=np.float32)
        query = base - offset
        i = np.interp(query, base, iq[:, 0], left=0.0, right=0.0)
        q = np.interp(query, base, iq[:, 1], left=0.0, right=0.0)
        return np.stack([i, q], axis=-1).astype(np.float32)

    @staticmethod
    def amplitude_scaling(iq: np.ndarray, alpha: float) -> np.ndarray:
        return (float(alpha) * iq).astype(np.float32)

    @staticmethod
    def iq_imbalance(iq: np.ndarray, gain: float, phase: float) -> np.ndarray:
        i = float(gain) * iq[:, 0]
        q = iq[:, 1] / max(float(gain), 1e-6)
        q_phase = np.sin(float(phase)) * i + np.cos(float(phase)) * q
        return np.stack([i, q_phase], axis=-1).astype(np.float32)

    def additive_noise(self, iq: np.ndarray, snr_db: float) -> np.ndarray:
        signal_power = float(np.mean(np.sum(iq * iq, axis=-1)))
        noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0) + 1e-12)
        noise = self.rng.normal(0.0, np.sqrt(noise_power / 2.0), size=iq.shape)
        return (iq + noise).astype(np.float32)

    def multipath_filter(self, iq: np.ndarray, n_taps: int) -> np.ndarray:
        taps_count = max(2, int(n_taps))
        signal = iq[:, 0] + 1j * iq[:, 1]
        taps = self.rng.normal(size=taps_count) + 1j * self.rng.normal(size=taps_count)
        taps *= np.exp(-np.arange(taps_count, dtype=np.float32))
        taps /= np.sqrt(np.sum(np.abs(taps) ** 2)) + 1e-12
        filtered = np.convolve(signal, taps, mode="same")
        return np.stack([filtered.real, filtered.imag], axis=-1).astype(np.float32)


def default_perturbation_specs() -> list[PerturbationSpec]:
    """Return the 52 perturbation settings reported in the paper."""

    specs: list[PerturbationSpec] = []
    for level, degrees in enumerate([5, 15, 30, 60, 90], start=1):
        for sign in (-1, 1):
            value = sign * degrees
            specs.append(
                PerturbationSpec(
                    f"phase_{value:+d}deg",
                    "phase",
                    level,
                    {"theta": float(np.deg2rad(value))},
                )
            )
    for level, cycles in enumerate([0.02, 0.08, 0.16, 0.32, 0.50], start=1):
        for sign in (-1, 1):
            value = sign * cycles
            specs.append(
                PerturbationSpec(
                    f"cfo_{value:+.2f}cyc",
                    "cfo",
                    level,
                    {"cycles": float(value)},
                )
            )
    for level, offset in enumerate([1, 3, 8, 15, 30], start=1):
        for sign in (-1, 1):
            value = sign * offset
            specs.append(
                PerturbationSpec(
                    f"timing_{value:+d}",
                    "timing",
                    level,
                    {"offset": float(value)},
                )
            )
    for level, alpha in enumerate([0.40, 0.60, 0.75, 0.90, 1.50, 2.00], start=1):
        specs.append(
            PerturbationSpec(
                f"amplitude_{alpha:.2f}",
                "amplitude",
                level,
                {"alpha": float(alpha)},
            )
        )
    for level, (gain, degrees) in enumerate(
        [(1.03, 2), (1.08, 5), (1.15, 8), (1.35, 15), (1.60, 25)],
        start=1,
    ):
        specs.append(
            PerturbationSpec(
                f"iq_gain{gain:.2f}_phase{degrees:d}deg",
                "iq_imbalance",
                level,
                {"gain": gain, "phase": float(np.deg2rad(degrees))},
            )
        )
    for level, snr_db in enumerate([30, 20, 10, 5, 0, -3], start=1):
        specs.append(
            PerturbationSpec(
                f"noise_{snr_db:+d}db",
                "noise",
                level,
                {"snr_db": float(snr_db)},
            )
        )
    for level, taps in enumerate([2, 3, 4, 6, 8], start=1):
        specs.append(
            PerturbationSpec(
                f"multipath_{taps}tap",
                "multipath",
                level,
                {"n_taps": taps},
            )
        )
    if len(specs) != 52:
        raise AssertionError(f"Expected 52 perturbations, constructed {len(specs)}.")
    return specs


def _as_iq(iq: np.ndarray) -> np.ndarray:
    value = np.asarray(iq, dtype=np.float32)
    if value.ndim != 2 or value.shape[-1] != 2:
        raise ValueError(f"Expected shape (L, 2), got {value.shape}.")
    return value
