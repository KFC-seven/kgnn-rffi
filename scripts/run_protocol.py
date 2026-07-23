from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dpr_rffi.data import (
    build_manifest,
    build_split_records,
    load_compact_dataset,
    load_config,
    materialize_records,
)
from dpr_rffi.metrics import open_set_metrics
from dpr_rffi.model import DPRConfig, DPRRFFI
from dpr_rffi.training import (
    classifier_function,
    encoder_function,
    train_source_encoder,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one source-only DPR-RFFI protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--data", help="Override dataset.path in the configuration.")
    parser.add_argument("--architecture", choices=["tiny", "resnet1d"], required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples-per-record", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = load_config(args.config)
    if args.data:
        config["dataset"]["path"] = str(Path(args.data).resolve())
    manifest = build_manifest(config)
    protocol = select_named(manifest["protocols"], args.protocol)
    split = select_split(protocol["tx_splits"], args.split)
    records = build_split_records(
        known_txs=split["known_txs"],
        unknown_txs=split["unknown_txs"],
        source_rxs=protocol["source_rxs"],
        drift_rxs=protocol["drift_rxs"],
        source_date=manifest["dates"]["source"],
        day_shift_date=manifest["dates"]["day_shift"],
    )
    dataset = load_compact_dataset(config["dataset"]["path"])
    source = materialize_records(
        dataset,
        [row for row in records if row["split_name"] == "source_train"],
        signal_equalized=1,
        max_samples_per_record=args.max_samples_per_record,
        sample_mode="head",
        sample_seed=args.seed,
    )
    target = materialize_records(
        dataset,
        [row for row in records if row["split_name"] != "source_train"],
        signal_equalized=1,
        max_samples_per_record=args.max_samples_per_record,
        sample_mode="head",
        sample_seed=args.seed,
    )
    num_classes = len(split["known_txs"])
    training = train_source_encoder(
        source.x,
        source.known_label,
        num_classes=num_classes,
        architecture=args.architecture,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        seed=args.seed,
        device=device,
    )
    encode = encoder_function(training.model, batch_size=args.batch_size, device=device)
    classify = classifier_function(training.model, batch_size=args.batch_size, device=device)
    detector = DPRRFFI(DPRConfig(seed=args.seed)).fit(
        source_train_x=source.x[training.train_indices],
        source_train_y=source.known_label[training.train_indices],
        source_val_x=source.x[training.validation_indices],
        source_val_y=source.known_label[training.validation_indices],
        encode=encode,
        predict_labels=classify,
    )
    prediction = detector.predict(target.x)
    metrics = open_set_metrics(
        unknown_score=prediction.score,
        rejected=prediction.rejected,
        predicted_label=prediction.label,
        true_label=target.known_label,
        is_known=target.is_known,
    )
    payload = {
        "dataset": manifest["dataset"]["name"],
        "protocol": args.protocol,
        "split": args.split,
        "source_only": True,
        "target_labels_used_for_training_or_calibration": False,
        "architecture": args.architecture,
        "embedding_dim": args.embedding_dim,
        "best_source_validation_accuracy": training.best_validation_accuracy,
        "best_epoch": training.best_epoch,
        "trained_epochs": training.trained_epochs,
        "reference_summary": detector.reference_summary,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def select_named(rows: list[dict], name: str) -> dict:
    matches = [row for row in rows if row["name"] == name]
    if len(matches) != 1:
        raise KeyError(f"Expected one protocol named {name!r}, found {len(matches)}.")
    return matches[0]


def select_split(rows: list[dict], split_id: int) -> dict:
    matches = [row for row in rows if int(row["split_id"]) == int(split_id)]
    if len(matches) != 1:
        raise KeyError(f"Expected one split {split_id}, found {len(matches)}.")
    return matches[0]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return requested


if __name__ == "__main__":
    raise SystemExit(main())
