"""Full 33-run ablation: Full KGNN-RFFI vs No Perturbation-Augmentation variant.

Full:   R_K = source + safe-perturbed, R_D = destructive, u_p = d_K/d_D (ratio)
NoAug:  R_K = source only, R_D = empty, u_p = d_K (known-only)
        SCE gate, z-normalization, adaptive weighting, kNN ID — all retained.

Generates per_run.csv and summary_by_dataset.csv for Table insertion.
"""
from __future__ import annotations

import argparse, csv, json, sys
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

# ── Protocol definitions ──
ALL_RUNS = [
    # ManySig (tiny CNN, 100 samples/record, 64-dim)
    ("S_RX9-3_TX4-2",  "configs/manysig_soda4.yaml", "RX9-3_TX4-2",  "tiny",     100, 100, 64),
    ("S_RX9-3_TX2-4",  "configs/manysig_soda4.yaml", "RX9-3_TX2-4",  "tiny",     100, 100, 64),
    ("S_RX6-6_TX3-3",  "configs/manysig_soda4.yaml", "RX6-6_TX3-3",  "tiny",     100, 100, 64),
    ("S_RX3-9_TX2-4",  "configs/manysig_soda4.yaml", "RX3-9_TX2-4",  "tiny",     100, 100, 64),
    # ManyTx (ResNet1D, 30 samples/record, 128-dim)
    ("T_RX9-3_TX40-40","configs/manytx_owen_v0.yaml","MTX_RX9-3_TX40-40","resnet1d",100, 30,  128),
    ("T_RX6-6_TX40-40","configs/manytx_owen_v0.yaml","MTX_RX6-6_TX40-40","resnet1d",100, 30,  128),
    ("T_RX9-3_TX20-20","configs/manytx_owen_v0.yaml","MTX_RX9-3_TX20-20","resnet1d",100, 30,  128),
    ("T_RX6-6_TX20-20","configs/manytx_owen_v0.yaml","MTX_RX6-6_TX20-20","resnet1d",100, 30,  128),
    ("T_RX9-3_TX20-40","configs/manytx_owen_v0.yaml","MTX_RX9-3_TX20-40","resnet1d",100, 30,  128),
    ("T_RX6-6_TX20-40","configs/manytx_owen_v0.yaml","MTX_RX6-6_TX20-40","resnet1d",100, 30,  128),
    ("T_RX3-9_TX20-80","configs/manytx_owen_v0.yaml","MTX_RX3-9_TX20-80","resnet1d",100, 30,  128),
]

METRIC_COLS = [
    "sample_open_set_h_score", "paper_a_open_set_accuracy",
    "true_unknown_rejection_rate", "shifted_known_false_rejection_rate",
    "paper_a_auroc_shifted_known_vs_unknown", "auosc_shifted_known_vs_unknown",
    "shifted_known_correct_id_rate",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Full 33-run perturbation augmentation ablation.")
    parser.add_argument("--output-dir", default="results/ablate_augmentation")
    parser.add_argument("--source-frr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only first ManySig + first ManyTx protocol, 1 split each.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    _configure_torch_determinism()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    progress_path = output_dir / "progress.txt"

    runs = ALL_RUNS
    if args.smoke:
        runs = [ALL_RUNS[0], ALL_RUNS[4]]  # first ManySig + first ManyTx

    for rid, cfg, proto, model_name, epochs, max_samp, emb_dim in runs:
        for split_id in (range(1, 4) if not args.smoke else [1]):
            run_label = f"{rid}_split{split_id}"
            print(f"\n{'='*60}\n  {run_label}  ({model_name}, max_samp={max_samp}, dim={emb_dim})\n{'='*60}")
            progress_path.write_text(f"{run_label} running\n", encoding="utf-8")

            config = load_config(cfg)
            manifest = build_manifest(config)
            dataset = load_compact_dataset(config["dataset"]["path"])
            protocol = _select_protocol(manifest, proto)
            split = _select_split(protocol, split_id)
            records = build_split_records(
                known_txs=split["known_txs"], unknown_txs=split["unknown_txs"],
                source_rxs=protocol["source_rxs"], drift_rxs=protocol["drift_rxs"],
                source_date=manifest["dates"]["source"], day_shift_date=manifest["dates"]["day_shift"],
            )
            source_recs = [r for r in records if r["split_name"] == "source_train"]
            eval_recs = [r for r in records if r["split_name"] != "source_train"]
            source = materialize_records(dataset, records=source_recs, signal_equalized=1,
                                          max_samples_per_record=max_samp)
            eval_batch = materialize_records(dataset, records=eval_recs, signal_equalized=1,
                                              max_samples_per_record=max_samp)

            train_result = train_sourceonly(
                x=source.x, y=source.known_label, num_classes=len(split["known_txs"]),
                epochs=epochs, batch_size=args.batch_size, seed=args.seed,
                embedding_dim=emb_dim, model_name=model_name, device=device,
            )
            train_idx = np.asarray(train_result.train_indices, dtype=np.int64)
            val_idx = np.asarray(train_result.val_indices, dtype=np.int64)

            perturb = PerturbationEngine(config=PerturbationConfig(),
                                          seed=args.seed + split_id * 7919)
            safety_results = classify_perturbation_safety(
                encoder=train_result.model, source_x=source.x[val_idx],
                source_labels=source.known_label[val_idx], perturbation_engine=perturb,
                specs=default_perturbation_specs(), safe_accuracy=0.90, destructive_accuracy=0.50,
                threshold_mode="absolute", max_samples_per_class=25, batch_size=args.batch_size,
                device=device, seed=args.seed,
            )
            safe_specs = select_specs(safety_results, "safe")
            destructive_specs = select_specs(safety_results, "destructive")

            # ── Full KGNN-RFFI ──
            full_model, _ = build_kgnn_model(
                encoder=train_result.model, source_x=source.x, source_labels=source.known_label,
                train_indices=train_idx, val_indices=val_idx,
                num_classes=len(split["known_txs"]),
                safe_specs=safe_specs, destructive_specs=destructive_specs,
                perturbation_engine=perturb, augment_count=4, sigma_mult=0.3,
                max_support_per_class=1000, max_destructive_bank=5000, destructive_per_sample=1,
                batch_size=args.batch_size, device=device, distance="cosine",
                score_mode="ratio", support_k=1, destructive_k=1, class_norm_alpha=1.0,
                destructive_balance="none", score_calibration="none",
                frr=float(args.source_frr), gate_support=True, seed=args.seed,
            )

            # ── NoAug variant ──
            noaug_model, _ = build_kgnn_model(
                encoder=train_result.model, source_x=source.x, source_labels=source.known_label,
                train_indices=train_idx, val_indices=val_idx,
                num_classes=len(split["known_txs"]),
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
            knn_result = knn_unknown_score(train_embeddings, train_labels, eval_embeddings,
                                            k=5, metric="cosine")

            common = {
                "run_id": rid, "dataset": manifest["dataset"]["name"],
                "protocol": proto, "split_id": split_id,
                "safe_regimes": len(safe_specs), "destructive_regimes": len(destructive_specs),
                "source_val_accuracy": float(train_result.best_val_accuracy),
            }

            for variant_label, model in [("Full", full_model), ("NoAug", noaug_model)]:
                support_pred, rejected, score = predict_kgnn(model, eval_embeddings)
                pred = knn_result.predicted_label
                metrics = compute_osr_extended_metrics(
                    rejected=rejected, predicted_label=pred,
                    true_label=eval_batch.known_label, is_known=eval_batch.is_known,
                    is_shifted_known=eval_batch.is_shifted_known, unknown_score=score,
                )
                row = {**common, "variant": variant_label}
                for k in METRIC_COLS:
                    row[k] = float(metrics.get(k, 0.0))
                per_run_rows.append(row)

                h = row["sample_open_set_h_score"]
                acc = row["paper_a_open_set_accuracy"]
                unk = row["true_unknown_rejection_rate"]
                print(f"  {variant_label:<6s}  H={h:.4f}  ACC={acc:.4f}  Unk.Rej={unk:.4f}  "
                      f"FRR={row['shifted_known_false_rejection_rate']:.4f}  "
                      f"AUROC={row['paper_a_auroc_shifted_known_vs_unknown']:.4f}  "
                      f"OSCR={row['auosc_shifted_known_vs_unknown']:.4f}")

            progress_path.write_text(f"{run_label} done\n", encoding="utf-8")

    # ── Save per-run CSV ──
    per_run_path = output_dir / "per_run.csv"
    if per_run_rows:
        fields = list(per_run_rows[0].keys())
        with per_run_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_run_rows)
        print(f"\nWrote {per_run_path} ({len(per_run_rows)} rows)")

    # ── Summary by dataset ──
    summary_path = output_dir / "summary_by_dataset.csv"
    if per_run_rows:
        summary = []
        for ds in sorted(set(r["dataset"] for r in per_run_rows)):
            for variant in ["Full", "NoAug"]:
                items = [r for r in per_run_rows if r["dataset"] == ds and r["variant"] == variant]
                if not items:
                    continue
                srow = {"dataset": ds, "variant": variant, "runs": len(items)}
                for k in METRIC_COLS:
                    vals = [r[k] for r in items]
                    srow[f"{k}_mean"] = float(np.mean(vals))
                    srow[f"{k}_std"] = float(np.std(vals))
                summary.append(srow)

        # Compute deltas
        full_summary = {s["dataset"]: s for s in summary if s["variant"] == "Full"}
        noaug_summary = {s["dataset"]: s for s in summary if s["variant"] == "NoAug"}
        for ds in full_summary:
            fs = full_summary[ds]
            ns = noaug_summary.get(ds)
            if ns:
                delta_row = {"dataset": ds, "variant": "Delta (Full - NoAug)", "runs": fs["runs"]}
                for k in METRIC_COLS:
                    delta_row[f"{k}_mean"] = fs[f"{k}_mean"] - ns[f"{k}_mean"]
                    delta_row[f"{k}_std"] = 0.0
                summary.append(delta_row)

        fields = list(summary[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

        print(f"Wrote {summary_path}\n")
        for ds in sorted(set(r["dataset"] for r in per_run_rows)):
            fs = full_summary.get(ds)
            ns = noaug_summary.get(ds)
            if fs and ns:
                print(f"  {ds}:")
                for k in ["sample_open_set_h_score", "true_unknown_rejection_rate",
                           "paper_a_open_set_accuracy"]:
                    d = fs[f"{k}_mean"] - ns[f"{k}_mean"]
                    print(f"    {k}: {fs[f'{k}_mean']:.4f} -> {ns[f'{k}_mean']:.4f}  "
                          f"Delta = {d:+.4f}")

    # Save manifest for reproducibility
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_frr": float(args.source_frr),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "runs": [{"run_id": r[0], "config": r[1], "protocol": r[2],
                   "model": r[3], "epochs": r[4], "max_samples_per_record": r[5],
                   "embedding_dim": r[6]} for r in runs],
    }, indent=2), encoding="utf-8")

    progress_path.write_text("all done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
