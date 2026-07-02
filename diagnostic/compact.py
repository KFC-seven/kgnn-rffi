from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable


def load_compact_dataset(path: str | Path) -> dict:
    dataset_path = Path(path)
    if dataset_path.is_dir():
        dataset_path = dataset_path / f"{dataset_path.stem}.pkl"
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    _validate_compact_dataset(dataset, dataset_path)
    return dataset


def _validate_compact_dataset(dataset: dict, path: Path) -> None:
    required = {
        "tx_list",
        "rx_list",
        "capture_date_list",
        "equalized_list",
        "data",
    }
    missing = sorted(required.difference(dataset))
    if missing:
        raise ValueError(f"{path} is missing compact dataset keys: {missing}")


def get_leaf(dataset: dict, tx: str, rx: str, date: str, equalized: int):
    tx_i = dataset["tx_list"].index(tx)
    rx_i = dataset["rx_list"].index(rx)
    date_i = dataset["capture_date_list"].index(date)
    eq_i = dataset["equalized_list"].index(equalized)
    return dataset["data"][tx_i][rx_i][date_i][eq_i]


def paired_count(
    dataset: dict,
    tx: str,
    rx: str,
    date: str,
    required_equalized: Iterable[int],
) -> int:
    counts = []
    for eq in required_equalized:
        leaf = get_leaf(dataset, tx, rx, date, eq)
        counts.append(int(getattr(leaf, "shape", (0,))[0]))
    return min(counts) if counts else 0
