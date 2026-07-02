from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .compact import load_compact_dataset, paired_count


def build_split_records(
    known_txs: list[str],
    unknown_txs: list[str],
    source_rxs: list[str],
    drift_rxs: list[str],
    source_date: str,
    day_shift_date: str,
) -> list[dict]:
    records: list[dict] = []
    records.extend(
        _records_for(
            split_name="source_train",
            txs=known_txs,
            rxs=source_rxs,
            date=source_date,
            known_txs=known_txs,
            domain_type="source",
            is_known=True,
            is_shifted_known=False,
        )
    )
    records.extend(
        _records_for(
            split_name="shifted_known_rx",
            txs=known_txs,
            rxs=drift_rxs,
            date=source_date,
            known_txs=known_txs,
            domain_type="receiver_shift",
            is_known=True,
            is_shifted_known=True,
        )
    )
    records.extend(
        _records_for(
            split_name="shifted_known_day",
            txs=known_txs,
            rxs=source_rxs,
            date=day_shift_date,
            known_txs=known_txs,
            domain_type="day_shift",
            is_known=True,
            is_shifted_known=True,
        )
    )
    records.extend(
        _records_for(
            split_name="unknown_source_rx",
            txs=unknown_txs,
            rxs=source_rxs,
            date=source_date,
            known_txs=known_txs,
            domain_type="source_unknown",
            is_known=False,
            is_shifted_known=False,
        )
    )
    records.extend(
        _records_for(
            split_name="unknown_drift_rx",
            txs=unknown_txs,
            rxs=drift_rxs,
            date=source_date,
            known_txs=known_txs,
            domain_type="receiver_shift_unknown",
            is_known=False,
            is_shifted_known=False,
        )
    )
    records.extend(
        _records_for(
            split_name="unknown_source_day",
            txs=unknown_txs,
            rxs=source_rxs,
            date=day_shift_date,
            known_txs=known_txs,
            domain_type="day_shift_unknown",
            is_known=False,
            is_shifted_known=False,
        )
    )
    return records


def build_manifest(config: dict) -> dict:
    dataset = load_compact_dataset(config["dataset"]["path"])
    source_date = config["dates"]["source"]
    day_shift_date = config["dates"]["day_shift"]
    required_equalized = list(config.get("required_equalized", dataset["equalized_list"]))
    tx_pool, rx_pool, matrix_summary = select_filtered_matrix(
        dataset=dataset,
        source_date=source_date,
        day_shift_date=day_shift_date,
        required_equalized=required_equalized,
        min_samples_per_triple=int(config["filter"]["min_samples_per_triple"]),
        tx_pool_size=int(config["filter"]["tx_pool_size"]),
        rx_pool_size=int(config["filter"]["rx_pool_size"]),
    )

    protocols = []
    for protocol_index, protocol_cfg in enumerate(config["protocols"]):
        protocols.append(
            _build_protocol_manifest(
                dataset=dataset,
                protocol_cfg=protocol_cfg,
                protocol_index=protocol_index,
                tx_pool=tx_pool,
                rx_pool=rx_pool,
                source_date=source_date,
                day_shift_date=day_shift_date,
                required_equalized=required_equalized,
                tx_split_repeats=int(config.get("tx_split_repeats", 1)),
            )
        )

    capabilities = _capabilities(protocols)
    return {
        "dataset": {
            "name": config["dataset"]["name"],
            "path": str(Path(config["dataset"]["path"])),
            "tx_total": len(dataset["tx_list"]),
            "rx_total": len(dataset["rx_list"]),
            "dates": list(dataset["capture_date_list"]),
            "equalized": list(dataset["equalized_list"]),
        },
        "dates": {"source": source_date, "day_shift": day_shift_date},
        "required_equalized": required_equalized,
        "filter": dict(config["filter"]),
        "matrix_summary": matrix_summary,
        "tx_pool": tx_pool,
        "rx_pool": rx_pool,
        "protocols": protocols,
        "capabilities": capabilities,
    }


def select_filtered_matrix(
    dataset: dict,
    source_date: str,
    day_shift_date: str,
    required_equalized: Iterable[int],
    min_samples_per_triple: int,
    tx_pool_size: int,
    rx_pool_size: int,
) -> tuple[list[str], list[str], dict]:
    txs = list(dataset["tx_list"])
    rxs = list(dataset["rx_list"])
    usable = np.zeros((len(txs), len(rxs)), dtype=bool)
    for tx_i, tx in enumerate(txs):
        for rx_i, rx in enumerate(rxs):
            usable[tx_i, rx_i] = (
                paired_count(dataset, tx, rx, source_date, required_equalized)
                >= min_samples_per_triple
                and paired_count(dataset, tx, rx, day_shift_date, required_equalized)
                >= min_samples_per_triple
            )

    rx_indices = _choose_rx_pool(usable, rx_pool_size)
    complete_tx_indices = [
        tx_i for tx_i in range(len(txs)) if bool(np.all(usable[tx_i, rx_indices]))
    ]
    if len(complete_tx_indices) < tx_pool_size:
        raise ValueError(
            "Filtered matrix has only "
            f"{len(complete_tx_indices)} complete Tx for {rx_pool_size} Rx; "
            f"requested tx_pool_size={tx_pool_size}."
        )

    selected_tx_indices = complete_tx_indices[:tx_pool_size]
    selected_rx_indices = list(rx_indices)
    summary = {
        "usable_pairs": int(usable.sum()),
        "total_pairs": int(usable.size),
        "complete_tx_for_selected_rx": int(len(complete_tx_indices)),
        "selected_tx_count": int(len(selected_tx_indices)),
        "selected_rx_count": int(len(selected_rx_indices)),
    }
    return (
        [txs[i] for i in selected_tx_indices],
        [rxs[i] for i in selected_rx_indices],
        summary,
    )


def _build_protocol_manifest(
    dataset: dict,
    protocol_cfg: dict,
    protocol_index: int,
    tx_pool: list[str],
    rx_pool: list[str],
    source_date: str,
    day_shift_date: str,
    required_equalized: list[int],
    tx_split_repeats: int,
) -> dict:
    source_rx_count = int(protocol_cfg["source_rx_count"])
    drift_rx_count = int(protocol_cfg["drift_rx_count"])
    if source_rx_count + drift_rx_count > len(rx_pool):
        raise ValueError(
            f"{protocol_cfg['name']} requests {source_rx_count + drift_rx_count} Rx "
            f"from a pool of {len(rx_pool)}."
        )

    rng = random.Random(int(protocol_cfg.get("seed", 0)) + 7919 * protocol_index)
    source_rxs = sorted(rng.sample(rx_pool, source_rx_count), key=rx_pool.index)
    remaining_rxs = [rx for rx in rx_pool if rx not in source_rxs]
    if len(remaining_rxs) == drift_rx_count:
        drift_rxs = remaining_rxs
    else:
        drift_rxs = sorted(rng.sample(remaining_rxs, drift_rx_count), key=rx_pool.index)

    tx_splits = []
    for split_id in range(1, tx_split_repeats + 1):
        known_txs, unknown_txs = _draw_tx_split(
            tx_pool=tx_pool,
            known_count=int(protocol_cfg["known_tx_count"]),
            unknown_count=int(protocol_cfg["unknown_tx_count"]),
            seed=int(protocol_cfg.get("seed", 0)) + 104729 * split_id,
        )
        records = build_split_records(
            known_txs=known_txs,
            unknown_txs=unknown_txs,
            source_rxs=source_rxs,
            drift_rxs=drift_rxs,
            source_date=source_date,
            day_shift_date=day_shift_date,
        )
        tx_splits.append(
            {
                "split_id": split_id,
                "known_txs": known_txs,
                "unknown_txs": unknown_txs,
                "sample_counts": _sample_counts(
                    dataset=dataset,
                    records=records,
                    required_equalized=required_equalized,
                ),
            }
        )

    return {
        "name": protocol_cfg["name"],
        "source_rxs": source_rxs,
        "drift_rxs": drift_rxs,
        "source_rx_count": source_rx_count,
        "drift_rx_count": drift_rx_count,
        "known_tx_count": int(protocol_cfg["known_tx_count"]),
        "unknown_tx_count": int(protocol_cfg["unknown_tx_count"]),
        "tx_splits": tx_splits,
    }


def _records_for(
    split_name: str,
    txs: list[str],
    rxs: list[str],
    date: str,
    known_txs: list[str],
    domain_type: str,
    is_known: bool,
    is_shifted_known: bool,
) -> list[dict]:
    records = []
    for tx in txs:
        for rx in rxs:
            records.append(
                {
                    "split_name": split_name,
                    "true_tx": tx,
                    "known_label": known_txs.index(tx) if tx in known_txs else -1,
                    "rx_id": rx,
                    "day_id": date,
                    "domain_type": domain_type,
                    "is_known": is_known,
                    "is_shifted_known": is_shifted_known,
                }
            )
    return records


def _choose_rx_pool(usable: np.ndarray, rx_pool_size: int) -> list[int]:
    if rx_pool_size > usable.shape[1]:
        raise ValueError(
            f"rx_pool_size={rx_pool_size} exceeds available Rx={usable.shape[1]}."
        )
    total_combinations = math.comb(usable.shape[1], rx_pool_size)
    if total_combinations <= 250_000:
        best_combo = None
        best_score = None
        for combo in itertools.combinations(range(usable.shape[1]), rx_pool_size):
            complete_tx = int(np.all(usable[:, combo], axis=1).sum())
            support = int(usable[:, combo].sum())
            score = (complete_tx, support, tuple(-i for i in combo))
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo
        return list(best_combo or [])

    selected: list[int] = []
    remaining = set(range(usable.shape[1]))
    while len(selected) < rx_pool_size:
        best_rx = None
        best_score = None
        for rx_i in sorted(remaining):
            candidate = selected + [rx_i]
            complete_tx = int(np.all(usable[:, candidate], axis=1).sum())
            support = int(usable[:, candidate].sum())
            score = (complete_tx, support, -rx_i)
            if best_score is None or score > best_score:
                best_score = score
                best_rx = rx_i
        selected.append(int(best_rx))
        remaining.remove(int(best_rx))
    return selected


def _draw_tx_split(
    tx_pool: list[str],
    known_count: int,
    unknown_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if known_count + unknown_count > len(tx_pool):
        raise ValueError(
            f"known_count + unknown_count = {known_count + unknown_count} "
            f"exceeds tx_pool size {len(tx_pool)}."
        )
    shuffled = list(tx_pool)
    random.Random(seed).shuffle(shuffled)
    known = shuffled[:known_count]
    unknown = shuffled[known_count : known_count + unknown_count]
    order = {tx: i for i, tx in enumerate(tx_pool)}
    return sorted(known, key=order.get), sorted(unknown, key=order.get)


def _sample_counts(
    dataset: dict,
    records: list[dict],
    required_equalized: list[int],
) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[row["split_name"]] += paired_count(
            dataset=dataset,
            tx=row["true_tx"],
            rx=row["rx_id"],
            date=row["day_id"],
            required_equalized=required_equalized,
        )
    return dict(sorted(counts.items()))


def _capabilities(protocols: list[dict]) -> dict:
    supports_h_a = False
    supports_h_b_clean = False
    for protocol in protocols:
        for split in protocol["tx_splits"]:
            counts = split["sample_counts"]
            shifted_known = counts.get("shifted_known_rx", 0) + counts.get(
                "shifted_known_day", 0
            )
            unknown = (
                counts.get("unknown_source_rx", 0)
                + counts.get("unknown_drift_rx", 0)
                + counts.get("unknown_source_day", 0)
            )
            supports_h_a = supports_h_a or (shifted_known > 0 and unknown > 0)
            supports_h_b_clean = supports_h_b_clean or unknown > 0
    return {
        "supports_h_a": bool(supports_h_a),
        "supports_h_b_clean": bool(supports_h_b_clean),
        "supports_h_b_contam": bool(supports_h_a and supports_h_b_clean),
    }
