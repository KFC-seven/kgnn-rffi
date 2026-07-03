"""33-run ablation: Pure Ratio vs Full DPRNN-RFFI vs NoAug (d_K only).

Pure Ratio:  d_K/d_D ratio only, threshold at source val 97th percentile
Full:        ratio + z-norm + SCE gate + kNN score + weighted combination
NoAug:       d_K only, no perturbation augmentation, same thresholding
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


ALL_RUNS = [
    ("S_RX9-3_TX4-2",  "configs/manysig_soda4.yaml", "RX9-3_TX4-2",  "tiny",     5, 100, 64),
    ("S_RX9-3_TX2-4",  "configs/manysig_soda4.yaml", "RX9-3_TX2-4",  "tiny",     5, 100, 64),
    ("S_RX6-6_TX3-3",  "configs/manysig_soda4.yaml", "RX6-6_TX3-3",  "tiny",     5, 100, 64),
    ("S_RX3-9_TX2-4",  "configs/manysig_soda4.yaml", "RX3-9_TX2-4",  "tiny",     5, 100, 64),
    ("T_RX9-3_TX40-40","configs/manytx_owen_v0.yaml","MTX_RX9-3_TX40-40","resnet1d",5, 30,  128),
    ("T_RX6-6_TX40-40","configs/manytx_owen_v0.yaml","MTX_RX6-6_TX40-40","resnet1d",5, 30,  128),
    ("T_RX9-3_TX20-20","configs/manytx_owen_v0.yaml","MTX_RX9-3_TX20-20","resnet1d",5, 30,  128),
    ("T_RX6-6_TX20-20","configs/manytx_owen_v0.yaml","MTX_RX6-6_TX20-20","resnet1d",5, 30,  128),
    ("T_RX9-3_TX20-40","configs/manytx_owen_v0.yaml","MTX_RX9-3_TX20-40","resnet1d",5, 30,  128),
    ("T_RX6-6_TX20-40","configs/manytx_owen_v0.yaml","MTX_RX6-6_TX20-40","resnet1d",5, 30,  128),
    ("T_RX3-9_TX20-80","configs/manytx_owen_v0.yaml","MTX_RX3-9_TX20-80","resnet1d",5, 30,  128),
]

METRIC_COLS = [
    "sample_open_set_h_score", "paper_a_open_set_accuracy",
    "true_unknown_rejection_rate", "shifted_known_false_rejection_rate",
    "paper_a_auroc_shifted_known_vs_unknown", "auosc_shifted_known_vs_unknown",
    "shifted_known_correct_id_rate",
]


def cos_dist_matrix(query, bank):
    q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    b_n = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    return 1.0 - q_n @ b_n.T


def main() -> int:
    parser = argparse.ArgumentParser(description="33-run pure ratio vs full vs noaug ablation.")
    parser.add_argument("--output-dir", default="results/ablate_ratio")
    parser.add_argument("--source-frr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    _configure_torch_determinism()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []

    runs = ALL_RUNS
    if args.smoke:
        runs = [ALL_RUNS[0], ALL_RUNS[4]]

    for rid, cfg, proto, model_name, epochs, max_samp, emb_dim in runs:
        for split_id in (range(1, 4) if not args.smoke else [1]):
            run_label = f"{rid}_split{split_id}"
            print(f"\n{'='*60}\n  {run_label}  ({model_name})\n{'='*60}")

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

            # Get embeddings
            _sl, source_embeddings = infer_logits_embeddings(train_result.model, source.x,
                                                               batch_size=args.batch_size, device=device)
            _el, eval_embeddings = infer_logits_embeddings(train_result.model, eval_batch.x,
                                                             batch_size=args.batch_size, device=device)
            train_embeddings = source_embeddings[train_idx]
            train_labels = np.asarray(source.known_label[train_idx], dtype=np.int64)

            # kNN identity predictor
            knn_result = knn_unknown_score(train_embeddings, train_labels, eval_embeddings,
                                            k=5, metric="cosine")
            pred = knn_result.predicted_label

            is_known = np.asarray(eval_batch.is_known, dtype=bool)
            is_shifted = np.asarray(eval_batch.is_shifted_known, dtype=bool)
            num_classes = len(split["known_txs"])

            common = {
                "run_id": rid, "dataset": manifest["dataset"]["name"],
                "protocol": proto, "split_id": split_id,
                "safe_regimes": len(safe_specs), "destructive_regimes": len(destructive_specs),
                "source_val_accuracy": float(train_result.best_val_accuracy),
            }

            # ---- Full model ----
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
            rk = full_model.support_embeddings
            rd = full_model.destructive_embeddings

            # ---- Pure Ratio variant ----
            # Calibrate threshold on source val
            src_val_emb = source_embeddings[val_idx]
            dk_val = cos_dist_matrix(src_val_emb, rk).min(axis=1)
            dd_val = cos_dist_matrix(src_val_emb, rd).min(axis=1)
            ratio_val = dk_val / np.maximum(dd_val, 1e-8)
            pct = 100.0 * (1.0 - float(args.source_frr))
            thresh_ratio = np.percentile(ratio_val, pct)

            # Evaluate on target
            dk_eval = cos_dist_matrix(eval_embeddings, rk).min(axis=1)
            dd_eval = cos_dist_matrix(eval_embeddings, rd).min(axis=1)
            ratio_eval = dk_eval / np.maximum(dd_eval, 1e-8)
            rejected_ratio = ratio_eval > thresh_ratio

            metrics_ratio = compute_osr_extended_metrics(
                rejected=rejected_ratio, predicted_label=pred,
                true_label=eval_batch.known_label, is_known=is_known,
                is_shifted_known=is_shifted, unknown_score=ratio_eval,
            )

            # ---- NoAug model ----
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

            for variant_label, rejected, score in [
                ("PureRatio", rejected_ratio, ratio_eval),
            ]:
                row = {**common, "variant": variant_label}
                m = metrics_ratio
                for k in METRIC_COLS:
                    row[k] = float(m.get(k, 0.0))
                per_run_rows.append(row)
                h = row["sample_open_set_h_score"]
                acc = row["paper_a_open_set_accuracy"]
                unk = row["true_unknown_rejection_rate"]
                print(f"  {variant_label:<12s} H={h:.4f}  ACC={acc:.4f}  Unk.Rej={unk:.4f}  "
                      f"FRR={row['shifted_known_false_rejection_rate']:.4f}")

            for variant_label, model in [("Full", full_model), ("NoAug", noaug_model)]:
                _, rejected, score = predict_kgnn(model, eval_embeddings)
                m = compute_osr_extended_metrics(
                    rejected=rejected, predicted_label=pred,
                    true_label=eval_batch.known_label, is_known=is_known,
                    is_shifted_known=is_shifted, unknown_score=score,
                )
                row = {**common, "variant": variant_label}
                for k in METRIC_COLS:
                    row[k] = float(m.get(k, 0.0))
                per_run_rows.append(row)
                h = row["sample_open_set_h_score"]
                acc = row["paper_a_open_set_accuracy"]
                unk = row["true_unknown_rejection_rate"]
                print(f"  {variant_label:<12s} H={h:.4f}  ACC={acc:.4f}  Unk.Rej={unk:.4f}  "
                      f"FRR={row['shifted_known_false_rejection_rate']:.4f}")

    # ---- Save per-run CSV ----
    per_run_path = output_dir / "per_run.csv"
    if per_run_rows:
        fields = list(per_run_rows[0].keys())
        with per_run_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_run_rows)
        print(f"\nWrote {per_run_path} ({len(per_run_rows)} rows)")

    # ---- Summary by dataset ----
    summary_path = output_dir / "summary_by_dataset.csv"
    if per_run_rows:
        summary = []
        for ds in sorted(set(r["dataset"] for r in per_run_rows)):
            for variant in ["PureRatio", "Full", "NoAug"]:
                items = [r for r in per_run_rows if r["dataset"] == ds and r["variant"] == variant]
                if not items:
                    continue
                srow = {"dataset": ds, "variant": variant, "runs": len(items)}
                for k in METRIC_COLS:
                    vals = [r[k] for r in items]
                    srow[f"{k}_mean"] = float(np.mean(vals))
                    srow[f"{k}_std"] = float(np.std(vals))
                summary.append(srow)

        # Deltas: PureRatio vs Full, PureRatio vs NoAug
        pure_summary = {s["dataset"]: s for s in summary if s["variant"] == "PureRatio"}
        full_summary = {s["dataset"]: s for s in summary if s["variant"] == "Full"}
        noaug_summary = {s["dataset"]: s for s in summary if s["variant"] == "NoAug"}
        for ds in pure_summary:
            ps = pure_summary[ds]
            for ref_name, ref in [("Full", full_summary), ("NoAug", noaug_summary)]:
                rs = ref.get(ds)
                if rs:
                    delta_row = {"dataset": ds, "variant": f"Delta (PureRatio - {ref_name})", "runs": ps["runs"]}
                    for k in METRIC_COLS:
                        delta_row[f"{k}_mean"] = ps[f"{k}_mean"] - rs[f"{k}_mean"]
                        delta_row[f"{k}_std"] = 0.0
                    summary.append(delta_row)

        fields = list(summary[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

        print(f"Wrote {summary_path}\n")
        for ds in sorted(pure_summary.keys()):
            ps = pure_summary[ds]
            fs = full_summary.get(ds)
            ns = noaug_summary.get(ds)
            print(f"  {ds}:")
            for variant, s in [("PureRatio", ps), ("Full", fs), ("NoAug", ns)]:
                if s:
                    print(f"    {variant}: H={s['sample_open_set_h_score_mean']:.4f}  "
                          f"ACC={s['paper_a_open_set_accuracy_mean']:.4f}  "
                          f"Unk.Rej={s['true_unknown_rejection_rate_mean']:.4f}")

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_frr": float(args.source_frr),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "runs": [{"run_id": r[0], "config": r[1], "protocol": r[2],
                   "model": r[3], "epochs": r[4], "max_samples_per_record": r[5],
                   "embedding_dim": r[6]} for r in runs],
    }, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
