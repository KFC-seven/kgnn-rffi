"""Compare Full KGNN-RFFI vs No-Perturbation-Augmentation variant.

Full:   R_K = source + safe-perturbed embeddings, R_D = destructive-perturbed, u_p = d_K/d_D
NoAug:  R_K = source embeddings only, R_D = empty, u_p = d_K
        (SCE gate, z-normalization, adaptive weighting, kNN score — all retained)
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.compact import load_compact_dataset
from diagnostic.config import load_config
from diagnostic.datasets import materialize_records
from diagnostic.sourceonly import infer_logits_embeddings, train_sourceonly
from diagnostic.splits import build_manifest, build_split_records
from kgnn import (
    build_kgnn_model, classify_perturbation_safety, default_perturbation_specs,
    predict_kgnn, select_specs, PerturbationConfig, PerturbationEngine,
)
from kgnn.baselines.posthoc import knn_unknown_score
from kgnn.metrics import compute_osr_extended_metrics
from kgnn.utils import _configure_torch_determinism, _resolve_device, _select_protocol, _select_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/compare_aug")
    parser.add_argument("--source-frr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    _configure_torch_determinism()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ManyTx protocol where SCE gate matters most
    run_specs = [
        "T1|configs/manytx_owen_v0.yaml|MTX_RX6-6_TX20-20|1|resnet1d|5|30|128",
        "S1|configs/manysig_soda4.yaml|RX9-3_TX2-4|1|tiny|5|100|64",
    ]

    for text in run_specs:
        parts = text.split("|")
        spec = {"run_id": parts[0], "config": parts[1], "protocol": parts[2],
                "split_id": int(parts[3]), "model": parts[4], "epochs": int(parts[5]),
                "max_samples_per_record": int(parts[6]), "embedding_dim": int(parts[7])}

        config = load_config(spec["config"])
        manifest = build_manifest(config)
        dataset = load_compact_dataset(config["dataset"]["path"])
        protocol = _select_protocol(manifest, spec["protocol"])
        split = _select_split(protocol, spec["split_id"])
        records = build_split_records(
            known_txs=split["known_txs"], unknown_txs=split["unknown_txs"],
            source_rxs=protocol["source_rxs"], drift_rxs=protocol["drift_rxs"],
            source_date=manifest["dates"]["source"], day_shift_date=manifest["dates"]["day_shift"],
        )
        source_recs = [r for r in records if r["split_name"] == "source_train"]
        eval_recs = [r for r in records if r["split_name"] != "source_train"]
        source = materialize_records(dataset, records=source_recs, signal_equalized=1,
                                      max_samples_per_record=spec["max_samples_per_record"])
        eval_batch = materialize_records(dataset, records=eval_recs, signal_equalized=1,
                                          max_samples_per_record=spec["max_samples_per_record"])

        train_result = train_sourceonly(
            x=source.x, y=source.known_label, num_classes=len(split["known_txs"]),
            epochs=spec["epochs"], batch_size=args.batch_size, seed=args.seed,
            embedding_dim=spec["embedding_dim"], model_name=spec["model"], device=device,
        )
        train_idx = np.asarray(train_result.train_indices, dtype=np.int64)
        val_idx = np.asarray(train_result.val_indices, dtype=np.int64)
        num_classes = len(split["known_txs"])

        perturb = PerturbationEngine(config=PerturbationConfig(),
                                      seed=args.seed + int(spec["split_id"]) * 7919)
        safety_results = classify_perturbation_safety(
            encoder=train_result.model, source_x=source.x[val_idx],
            source_labels=source.known_label[val_idx], perturbation_engine=perturb,
            specs=default_perturbation_specs(), safe_accuracy=0.90, destructive_accuracy=0.50,
            threshold_mode="absolute", max_samples_per_class=25, batch_size=args.batch_size,
            device=device, seed=args.seed,
        )

        # ── Variant A: Full KGNN-RFFI ──
        safe_specs = select_specs(safety_results, "safe")
        destructive_specs = select_specs(safety_results, "destructive")
        full_model, _ = build_kgnn_model(
            encoder=train_result.model, source_x=source.x, source_labels=source.known_label,
            train_indices=train_idx, val_indices=val_idx, num_classes=num_classes,
            safe_specs=safe_specs, destructive_specs=destructive_specs,
            perturbation_engine=perturb, augment_count=4, sigma_mult=0.3,
            max_support_per_class=1000, max_destructive_bank=5000, destructive_per_sample=1,
            batch_size=args.batch_size, device=device, distance="cosine",
            score_mode="ratio", support_k=1, destructive_k=1, class_norm_alpha=1.0,
            destructive_balance="none", score_calibration="none",
            frr=float(args.source_frr), gate_support=True, seed=args.seed,
        )

        # ── Variant B: No perturbation augmentation ──
        noaug_model, _ = build_kgnn_model(
            encoder=train_result.model, source_x=source.x, source_labels=source.known_label,
            train_indices=train_idx, val_indices=val_idx, num_classes=num_classes,
            safe_specs=[], destructive_specs=[],
            perturbation_engine=perturb, augment_count=0, sigma_mult=0.3,
            max_support_per_class=1000, max_destructive_bank=0, destructive_per_sample=0,
            batch_size=args.batch_size, device=device, distance="cosine",
            score_mode="known", support_k=1, destructive_k=1, class_norm_alpha=1.0,
            destructive_balance="none", score_calibration="none",
            frr=float(args.source_frr), gate_support=True, seed=args.seed,
        )

        _sl, source_embeddings = infer_logits_embeddings(train_result.model, source.x,
                                                           batch_size=args.batch_size, device=device)
        _el, eval_embeddings = infer_logits_embeddings(train_result.model, eval_batch.x,
                                                         batch_size=args.batch_size, device=device)
        train_embeddings = source_embeddings[train_idx]
        train_labels = np.asarray(source.known_label[train_idx], dtype=np.int64)

        knn_result = knn_unknown_score(train_embeddings, train_labels, eval_embeddings, k=5, metric="cosine")

        print(f"\n{'='*70}")
        print(f"  {manifest['dataset']['name']} / {protocol['name']}")
        print(f"  safe={len(safe_specs)}, destr={len(destructive_specs)}")
        print(f"{'='*70}")

        for label, model in [("Full (R_K+R_D ratio)", full_model), ("NoAug (d_K only)", noaug_model)]:
            support_pred, rejected, score = predict_kgnn(model, eval_embeddings)
            pred = knn_result.predicted_label
            metrics = compute_osr_extended_metrics(
                rejected=rejected, predicted_label=pred,
                true_label=eval_batch.known_label, is_known=eval_batch.is_known,
                is_shifted_known=eval_batch.is_shifted_known, unknown_score=score,
            )
            h = float(metrics['sample_open_set_h_score'])
            acc = float(metrics['paper_a_open_set_accuracy'])
            unk = float(metrics['true_unknown_rejection_rate'])
            frr = float(metrics['shifted_known_false_rejection_rate'])
            auroc = float(metrics['paper_a_auroc_shifted_known_vs_unknown'])
            oscr = float(metrics['auosc_shifted_known_vs_unknown'])
            print(f"  {label:<25s} H={h:.4f}  ACC={acc:.4f}  Unk.Rej={unk:.4f}  FRR={frr:.4f}  AUROC={auroc:.4f}  OSCR={oscr:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
