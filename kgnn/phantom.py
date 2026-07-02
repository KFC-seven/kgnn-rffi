from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerturbationConfig:
    phase_max: float = np.pi / 6.0
    cfo_cycles_max: float = 0.25
    timing_max: float = 15.0
    amp_min: float = 0.6
    amp_max: float = 1.4
    iq_gain_min: float = 0.85
    iq_gain_max: float = 1.15
    iq_phase_max: float = np.deg2rad(8.0)
    snr_min_db: float = 3.0
    snr_max_db: float = 25.0
    multipath_max_taps: int = 4
    min_ops: int = 2
    max_ops: int = 4


class PerturbationEngine:
    """Source-only RF perturbations on raw IQ arrays."""

    def __init__(self, config: PerturbationConfig | None = None, seed: int = 0):
        self.config = config or PerturbationConfig()
        self.rng = np.random.default_rng(seed)

    def perturb_batch(self, x: np.ndarray, n_perturbations: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if int(n_perturbations) <= 0:
            return np.empty((0,) + x.shape[1:], dtype=np.float32)
        out = []
        for sample in x:
            for _ in range(int(n_perturbations)):
                out.append(self.perturb_one(sample))
        return np.stack(out, axis=0).astype(np.float32) if out else np.empty((0,) + x.shape[1:], dtype=np.float32)

    def perturb_one(self, iq: np.ndarray) -> np.ndarray:
        y = np.asarray(iq, dtype=np.float32).copy()
        ops = [
            self.phase_rotate,
            self.carrier_offset,
            self.timing_offset,
            self.amplitude_scale,
            self.iq_imbalance,
            self.add_noise,
            self.multipath_filter,
        ]
        count = int(self.rng.integers(self.config.min_ops, self.config.max_ops + 1))
        chosen = self.rng.choice(len(ops), size=min(count, len(ops)), replace=False)
        for idx in chosen.tolist():
            y = ops[idx](y)
        return np.asarray(y, dtype=np.float32)

    def phase_rotate(self, iq: np.ndarray, theta: float | None = None) -> np.ndarray:
        theta = float(self.rng.uniform(-self.config.phase_max, self.config.phase_max) if theta is None else theta)
        c = np.cos(theta)
        s = np.sin(theta)
        i = iq[:, 0]
        q = iq[:, 1]
        return np.stack([c * i - s * q, s * i + c * q], axis=-1).astype(np.float32)

    def carrier_offset(self, iq: np.ndarray, cycles: float | None = None) -> np.ndarray:
        cycles = float(self.rng.uniform(-self.config.cfo_cycles_max, self.config.cfo_cycles_max) if cycles is None else cycles)
        n = iq.shape[0]
        phase = 2.0 * np.pi * cycles * np.arange(n, dtype=np.float32) / max(n - 1, 1)
        c = np.cos(phase)
        s = np.sin(phase)
        i = iq[:, 0]
        q = iq[:, 1]
        return np.stack([c * i - s * q, s * i + c * q], axis=-1).astype(np.float32)

    def timing_offset(self, iq: np.ndarray, offset: float | None = None) -> np.ndarray:
        offset = float(self.rng.uniform(-self.config.timing_max, self.config.timing_max) if offset is None else offset)
        n = iq.shape[0]
        base = np.arange(n, dtype=np.float32)
        query = base - offset
        i = np.interp(query, base, iq[:, 0], left=0.0, right=0.0)
        q = np.interp(query, base, iq[:, 1], left=0.0, right=0.0)
        return np.stack([i, q], axis=-1).astype(np.float32)

    def amplitude_scale(self, iq: np.ndarray, alpha: float | None = None) -> np.ndarray:
        alpha = float(self.rng.uniform(self.config.amp_min, self.config.amp_max) if alpha is None else alpha)
        return (alpha * iq).astype(np.float32)

    def iq_imbalance(
        self,
        iq: np.ndarray,
        gain: float | None = None,
        phase: float | None = None,
    ) -> np.ndarray:
        gain = float(self.rng.uniform(self.config.iq_gain_min, self.config.iq_gain_max) if gain is None else gain)
        phase = float(self.rng.uniform(-self.config.iq_phase_max, self.config.iq_phase_max) if phase is None else phase)
        i = gain * iq[:, 0]
        q = iq[:, 1] / max(gain, 1e-6)
        q_phase = np.sin(phase) * i + np.cos(phase) * q
        return np.stack([i, q_phase], axis=-1).astype(np.float32)

    def add_noise(self, iq: np.ndarray, snr_db: float | None = None) -> np.ndarray:
        snr_db = float(self.rng.uniform(self.config.snr_min_db, self.config.snr_max_db) if snr_db is None else snr_db)
        power = float(np.mean(np.sum(iq * iq, axis=-1)))
        noise_power = power / (10.0 ** (snr_db / 10.0) + 1e-12)
        noise = self.rng.normal(0.0, np.sqrt(noise_power / 2.0), size=iq.shape)
        return (iq + noise).astype(np.float32)

    def multipath_filter(self, iq: np.ndarray, n_taps: int | None = None) -> np.ndarray:
        n_taps = int(self.rng.integers(2, self.config.multipath_max_taps + 1) if n_taps is None else n_taps)
        signal = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
        taps = self.rng.normal(0.0, 1.0, size=n_taps) + 1j * self.rng.normal(0.0, 1.0, size=n_taps)
        decay = np.exp(-np.arange(n_taps, dtype=np.float32))
        taps = taps * decay
        taps = taps / (np.sqrt(np.sum(np.abs(taps) ** 2)) + 1e-12)
        filtered = np.convolve(signal, taps, mode="same")
        return np.stack([filtered.real, filtered.imag], axis=-1).astype(np.float32)

