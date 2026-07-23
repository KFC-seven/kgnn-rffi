from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dpr_rffi.baselines.posthoc import (
    energy_unknown_score,
    fit_openmax,
    knn_unknown_score,
    nndr_unknown_score,
    openmax_unknown_score,
)
from dpr_rffi.data import (
    build_manifest,
    build_split_records,
    load_compact_dataset,
    load_config,
    materialize_records,
)
from dpr_rffi.metrics import open_set_metrics
from dpr_rffi.model import calibrate_source_threshold
from dpr_rffi.training import infer, train_source_encoder
from run_protocol import resolve_device, select_named, select_split


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the shared-encoder Energy, kNN, NNDR, and OpenMax baselines."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--data")
    parser.add_argument("--architecture", choices=["tiny", "resnet1d"], required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples-per-record", type=int, required=True)
    parser.add_argument("--source-frr", type=float, default=0.03)
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
    )
    target = materialize_records(
        dataset,
        [row for row in records if row["split_name"] != "source_train"],
        signal_equalized=1,
        max_samples_per_record=args.max_samples_per_record,
    )
    training = train_source_encoder(
        source.x,
        source.known_label,
        num_classes=len(split["known_txs"]),
        architecture=args.architecture,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        seed=args.seed,
        device=device,
    )
    source_logits, source_embeddings = infer(
        training.model,
        source.x,
        batch_size=args.batch_size,
        device=device,
    )
    target_logits, target_embeddings = infer(
        training.model,
        target.x,
        batch_size=args.batch_size,
        device=device,
    )
    train_index = training.train_indices
    validation_index = training.validation_indices
    train_embeddings = source_embeddings[train_index]
    train_labels = source.known_label[train_index]

    rows: dict[str, dict] = {}
    rows["Energy"] = evaluate(
        energy_unknown_score(source_logits[validation_index]),
        energy_unknown_score(target_logits),
        np.argmax(target_logits, axis=1),
        target,
        args.source_frr,
    )
    validation_knn = knn_unknown_score(
        train_embeddings,
        train_labels,
        source_embeddings[validation_index],
        k=5,
        metric="cosine",
    )
    target_knn = knn_unknown_score(
        train_embeddings,
        train_labels,
        target_embeddings,
        k=5,
        metric="cosine",
    )
    rows["kNN"] = evaluate(
        validation_knn.scores,
        target_knn.scores,
        target_knn.predicted_label,
        target,
        args.source_frr,
    )
    validation_nndr = nndr_unknown_score(
        train_embeddings,
        train_labels,
        source_embeddings[validation_index],
    )
    target_nndr = nndr_unknown_score(
        train_embeddings,
        train_labels,
        target_embeddings,
    )
    rows["NNDR"] = evaluate(
        validation_nndr.scores,
        target_nndr.scores,
        target_nndr.predicted_label,
        target,
        args.source_frr,
    )
    openmax = fit_openmax(
        train_embeddings,
        source_logits[train_index],
        train_labels,
        num_classes=len(split["known_txs"]),
        tail_size=20,
        alpha=min(10, len(split["known_txs"])),
    )
    validation_openmax, _ = openmax_unknown_score(
        source_embeddings[validation_index],
        source_logits[validation_index],
        openmax,
    )
    target_openmax, target_openmax_prediction = openmax_unknown_score(
        target_embeddings,
        target_logits,
        openmax,
    )
    rows["OpenMax"] = evaluate(
        validation_openmax,
        target_openmax,
        target_openmax_prediction,
        target,
        args.source_frr,
    )
    payload = {
        "dataset": manifest["dataset"]["name"],
        "protocol": args.protocol,
        "split": args.split,
        "source_only": True,
        "target_labels_used_for_training_or_calibration": False,
        "methods": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def evaluate(
    validation_score: np.ndarray,
    target_score: np.ndarray,
    target_prediction: np.ndarray,
    target,
    source_frr: float,
) -> dict:
    threshold = calibrate_source_threshold(validation_score, source_frr)
    rejected = np.asarray(target_score) > threshold
    metrics = open_set_metrics(
        unknown_score=target_score,
        rejected=rejected,
        predicted_label=target_prediction,
        true_label=target.known_label,
        is_known=target.is_known,
    )
    return {"source_threshold": threshold, **metrics}


if __name__ == "__main__":
    raise SystemExit(main())
