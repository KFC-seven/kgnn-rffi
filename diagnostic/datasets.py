from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .compact import get_leaf


@dataclass(frozen=True)
class SampleBatch:
    x: np.ndarray
    known_label: np.ndarray
    true_tx: np.ndarray
    rx_id: np.ndarray
    day_id: np.ndarray
    split_name: np.ndarray
    domain_type: np.ndarray
    is_known: np.ndarray
    is_shifted_known: np.ndarray


def materialize_records(
    dataset: dict,
    records: list[dict],
    signal_equalized: int = 1,
    max_samples_per_record: int | None = None,
    sample_mode: str = "head",
    sample_seed: int = 0,
) -> SampleBatch:
    if sample_mode not in {"head", "random"}:
        raise ValueError(f"sample_mode must be 'head' or 'random', got {sample_mode!r}.")
    x_parts = []
    metadata = {
        "known_label": [],
        "true_tx": [],
        "rx_id": [],
        "day_id": [],
        "split_name": [],
        "domain_type": [],
        "is_known": [],
        "is_shifted_known": [],
    }
    for record in records:
        signals = np.asarray(
            get_leaf(
                dataset=dataset,
                tx=record["true_tx"],
                rx=record["rx_id"],
                date=record["day_id"],
                equalized=signal_equalized,
            ),
            dtype=np.float32,
        )
        if max_samples_per_record is not None:
            signals = _sample_signals(
                signals=signals,
                max_samples=int(max_samples_per_record),
                mode=sample_mode,
                seed=int(sample_seed),
                record=record,
            )
        if signals.shape[0] == 0:
            continue
        x_parts.append(signals)
        n = int(signals.shape[0])
        for key in metadata:
            metadata[key].extend([record[key]] * n)

    if not x_parts:
        raise ValueError("No samples were materialized from the provided records.")

    return SampleBatch(
        x=np.concatenate(x_parts, axis=0).astype(np.float32),
        known_label=np.asarray(metadata["known_label"], dtype=np.int64),
        true_tx=np.asarray(metadata["true_tx"], dtype=object),
        rx_id=np.asarray(metadata["rx_id"], dtype=object),
        day_id=np.asarray(metadata["day_id"], dtype=object),
        split_name=np.asarray(metadata["split_name"], dtype=object),
        domain_type=np.asarray(metadata["domain_type"], dtype=object),
        is_known=np.asarray(metadata["is_known"], dtype=bool),
        is_shifted_known=np.asarray(metadata["is_shifted_known"], dtype=bool),
    )


def _sample_signals(
    *,
    signals: np.ndarray,
    max_samples: int,
    mode: str,
    seed: int,
    record: dict,
) -> np.ndarray:
    if max_samples < 0:
        raise ValueError(f"max_samples_per_record must be non-negative, got {max_samples}.")
    if signals.shape[0] <= max_samples:
        return signals
    if mode == "head":
        return signals[:max_samples]

    key = "|".join(
        str(record.get(field, ""))
        for field in ("split_name", "true_tx", "rx_id", "day_id", "domain_type")
    )
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    record_seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
    rng = np.random.default_rng(record_seed)
    indices = np.sort(rng.choice(signals.shape[0], size=max_samples, replace=False))
    return signals[indices]
