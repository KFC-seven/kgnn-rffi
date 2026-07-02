"""Per-sample SCE gate activation analysis.

Runs KGNN-RFFI on selected protocols and saves per-sample gate values
(g_env) together with ground-truth labels for stratified reliability analysis.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.compact import load_compact_dataset
from diagnostic.config import load_config
from diagnostic.datasets import materialize_records
from diagnostic.sourceonly import infer_logits_embeddings, train_sourceonly
from diagnostic.splits import build_manifest, build_split_records
from kgnn import (
    build_kgnn_model,
    classify_perturbation_safety,
    default_perturbation_specs,
    predict_kgnn,
    select_specs,
    PerturbationConfig,
    PerturbationEngine,
)
from kgnn.baselines.posthoc import knn_unknown_score


def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    a_n = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    b_n = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
    return np.clip(1.0 - a_n @ b_n.T, 0.0, 2.0).astype(np.float32)


def _envelope_gate(train_embeddings, train_labels, query_embeddings, predicted_label,
                   num_classes, quantile, max_mult):
    """Compute per-sample class envelope gate g_env (Eq. 8-9 in paper)."""
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    predicted_label = np.asarray(predicted_label, dtype=np.int64).reshape(-1)
    centers, radii = [], []
    for cls in range(int(num_classes)):
        cls_emb = train_embeddings[train_labels == cls]
        if cls_emb.size == 0:
            centers.append(np.zeros(train_embeddings.shape[1], dtype=np.float32))
            radii.append(1.0)
            continue
        center = np.mean(cls_emb, axis=0).astype(np.float32)
        centers.append(center)
        cls_dist = _cosine_distance_matrix(cls_emb, center[None, :])[:, 0]
        radii.append(max(float(np.quantile(cls_dist, float(quantile))), 1e-6))
    centers = np.stack(centers, axis=0).astype(np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    dist = _cosine_distance_matrix(query_embeddings, centers)
    pred = np.clip(predicted_label, 0, int(num_classes) - 1)
    pred_dist = dist[np.arange(dist.shape[0]), pred]
    pred_radius = np.maximum(radii[pred], 1e-6)
    max_radius = np.maximum(float(max_mult) * pred_radius, pred_radius + 1e-6)
    g_env = np.where(
        pred_dist <= pred_radius, 1.0,
        np.clip((max_radius - pred_dist) / np.maximum(max_radius - pred_radius, 1e-6), 0.0, 1.0),
    ).astype(np.float32)
    return g_env
from kgnn.metrics import compute_osr_extended_metrics
from kgnn.utils import _configure_torch_determinism, _resolve_device, _select_protocol, _select_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-sample SCE gate analysis")
    parser.add_argument("--run", action="append", default=None)
    parser.add_argument("--output-dir", default="results/gate_analysis")
    parser.add_argument("--sce-quantile", type=float, default=0.90)
    parser.add_argument("--sce-max-mult", type=float, default=1.75)
    parser.add_argument("--sce-weight-low", type=float, default=0.50)
    parser.add_argument("--sce-weight-high", type=float, default=0.95)
    parser.add_argument("--source-frr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize embeddings to unit hypersphere")
    args = parser.parse_args()

    if args.run is None:
        args.run = [
            "M3T1|configs/manytx_owen_v0.yaml|MTX_RX6-6_TX20-20|1|resnet1d|5|30|128",
            "M3S1|configs/manysig_soda4.yaml|RX9-3_TX2-4|1|tiny|5|100|64",
        ]

    device = _resolve_device(args.device)
    _configure_torch_determinism()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for text in args.run:
        spec = _parse_run(text)
        rows = _analyze_one(spec, args, device, output_dir)
        all_rows.extend(rows)

    per_sample = output_dir / "per_sample_gate.csv"
    _write_csv(per_sample, all_rows)
    print(f"Wrote {per_sample} ({len(all_rows)} samples)")

    # Print stratified summary
    _print_summary(all_rows)
    return 0


def _parse_run(text: str) -> dict:
    parts = text.split("|")
    return {
        "run_id": parts[0], "config": parts[1], "protocol": parts[2],
        "split_id": int(parts[3]), "model": parts[4], "epochs": int(parts[5]),
        "max_samples_per_record": int(parts[6]), "embedding_dim": int(parts[7]),
    }


def _analyze_one(spec: dict, args: argparse.Namespace, device: str, output_dir: Path) -> list[dict]:
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
    source_records = [r for r in records if r["split_name"] == "source_train"]
    eval_records = [r for r in records if r["split_name"] != "source_train"]
    source = materialize_records(dataset, records=source_records, signal_equalized=1,
                                  max_samples_per_record=spec["max_samples_per_record"])
    eval_batch = materialize_records(dataset, records=eval_records, signal_equalized=1,
                                      max_samples_per_record=spec["max_samples_per_record"])

    train_result = train_sourceonly(
        x=source.x, y=source.known_label, num_classes=len(split["known_txs"]),
        epochs=spec["epochs"], batch_size=args.batch_size, seed=args.seed,
        embedding_dim=spec["embedding_dim"], model_name=spec["model"], device=device,
    )
    train_idx = np.asarray(train_result.train_indices, dtype=np.int64)
    val_idx = np.asarray(train_result.val_indices, dtype=np.int64)

    perturb = PerturbationEngine(config=PerturbationConfig(),
                                  seed=args.seed + int(spec["split_id"]) * 7919)
    safety_results = classify_perturbation_safety(
        encoder=train_result.model, source_x=source.x[val_idx],
        source_labels=source.known_label[val_idx], perturbation_engine=perturb,
        specs=default_perturbation_specs(), safe_accuracy=0.90, destructive_accuracy=0.50,
        threshold_mode="absolute", max_samples_per_class=25, batch_size=args.batch_size,
        device=device, seed=args.seed,
    )
    safe_specs = select_specs(safety_results, "safe")
    destructive_specs = select_specs(safety_results, "destructive")

    kgnn_model, _info = build_kgnn_model(
        encoder=train_result.model, source_x=source.x, source_labels=source.known_label,
        train_indices=train_idx, val_indices=val_idx,
        num_classes=len(split["known_txs"]), safe_specs=safe_specs,
        destructive_specs=destructive_specs, perturbation_engine=perturb,
        augment_count=4, sigma_mult=0.3, max_support_per_class=1000,
        max_destructive_bank=5000, destructive_per_sample=1,
        batch_size=args.batch_size, device=device, distance="cosine",
        score_mode="ratio", support_k=1, destructive_k=1, class_norm_alpha=1.0,
        destructive_balance="none", score_calibration="none",
        frr=float(args.source_frr), gate_support=True, seed=args.seed,
    )

    _sl, source_embeddings = infer_logits_embeddings(train_result.model, source.x,
                                                       batch_size=args.batch_size, device=device)
    _el, eval_embeddings = infer_logits_embeddings(train_result.model, eval_batch.x,
                                                     batch_size=args.batch_size, device=device)
    if args.normalize:
        source_embeddings = source_embeddings / (np.linalg.norm(source_embeddings, axis=1, keepdims=True) + 1e-12)
        eval_embeddings = eval_embeddings / (np.linalg.norm(eval_embeddings, axis=1, keepdims=True) + 1e-12)
    train_embeddings = source_embeddings[train_idx]
    train_labels = np.asarray(source.known_label[train_idx], dtype=np.int64)
    val_embeddings = source_embeddings[val_idx]

    knn_result = knn_unknown_score(train_embeddings, train_labels, eval_embeddings, k=5, metric="cosine")

    # Compute per-sample SCE gate (envelope-only, matching paper Eq. 8-9)
    g_env = _envelope_gate(
        train_embeddings=train_embeddings, train_labels=train_labels,
        query_embeddings=eval_embeddings, predicted_label=knn_result.predicted_label,
        num_classes=len(split["known_txs"]),
        quantile=float(args.sce_quantile), max_mult=float(args.sce_max_mult),
    )

    support_pred, rejected, score = predict_kgnn(kgnn_model, eval_embeddings)
    pred = knn_result.predicted_label

    rows = []
    for i in range(eval_batch.x.shape[0]):
        rows.append({
            "run_id": spec["run_id"],
            "dataset": manifest["dataset"]["name"],
            "protocol": protocol["name"],
            "split_id": int(split["split_id"]),
            "g_env": round(float(g_env[i]), 6),
            "is_known": bool(eval_batch.is_known[i]),
            "is_shifted_known": bool(eval_batch.is_shifted_known[i]),
            "true_label": int(eval_batch.known_label[i]),
            "predicted_label": int(pred[i]),
            "rejected": bool(rejected[i]),
            "score": float(score[i]),
            "knn_distance": float(knn_result.scores[i]),
        })

    metrics = compute_osr_extended_metrics(
        rejected=rejected, predicted_label=pred,
        true_label=eval_batch.known_label, is_known=eval_batch.is_known,
        is_shifted_known=eval_batch.is_shifted_known, unknown_score=score,
    )
    print(f"  {spec['run_id']} {protocol['name']}: {len(rows)} eval samples, "
          f"g_env mean={float(np.mean(g_env)):.4f}, "
          f"H={float(metrics['sample_open_set_h_score']):.4f}, "
          f"ACC={float(metrics['paper_a_open_set_accuracy']):.4f}, "
          f"Unk.Rej={float(metrics['true_unknown_rejection_rate']):.4f}, "
          f"FRR={float(metrics['shifted_known_false_rejection_rate']):.4f}, "
          f"AUROC={float(metrics['paper_a_auroc_shifted_known_vs_unknown']):.4f}, "
          f"OSCR={float(metrics['auosc_shifted_known_vs_unknown']):.4f}, "
          f"safe={len(safe_specs)}, destr={len(destructive_specs)}")
    return rows


def _print_summary(rows: list[dict]) -> None:
    bins = [(0.0, 0.33), (0.33, 0.67), (0.67, 1.01)]
    datasets = sorted(set(r["dataset"] for r in rows))
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        print(f"\n{'='*60}")
        print(f"  {ds}  (N={len(ds_rows)} eval samples)")
        print(f"  {'g_env bin':<16} {'#Samples':>8}  {'Known-Correct':>13}  {'Known-Wrong':>12}  {'Unk-Accepted':>13}  {'Unk-Rejected':>13}")
        print(f"  {'-'*16} {'-'*8}  {'-'*13}  {'-'*12}  {'-'*13}  {'-'*13}")
        for lo, hi in bins:
            bin_rows = [r for r in ds_rows if lo <= r["g_env"] < hi]
            n = len(bin_rows)
            if n == 0:
                continue
            known = [r for r in bin_rows if r["is_known"]]
            unknown = [r for r in bin_rows if not r["is_known"]]
            known_correct = sum(1 for r in known if not r["rejected"] and r["predicted_label"] == r["true_label"])
            known_wrong = sum(1 for r in known if not r["rejected"] and r["predicted_label"] != r["true_label"])
            unk_accepted = sum(1 for r in unknown if not r["rejected"])
            unk_rejected = sum(1 for r in unknown if r["rejected"])
            print(f"  [{lo:.2f}, {hi:.2f})   {n:>8}  "
                  f"{known_correct:>4} ({_pct(known_correct, len(known))})  "
                  f"{known_wrong:>4} ({_pct(known_wrong, len(known))})  "
                  f"{unk_accepted:>4} ({_pct(unk_accepted, len(unknown))})  "
                  f"{unk_rejected:>4} ({_pct(unk_rejected, len(unknown))})")
        # Unknown ratio by bin
        n_unknown = sum(1 for r in ds_rows if not r["is_known"])
        print(f"\n  Unknown sample distribution across g_env bins:")
        for lo, hi in bins:
            bin_rows = [r for r in ds_rows if lo <= r["g_env"] < hi]
            bin_unknown = sum(1 for r in bin_rows if not r["is_known"])
            print(f"    [{lo:.2f}, {hi:.2f}): {bin_unknown}/{len(bin_rows)} = {_pct(bin_unknown, len(bin_rows))} unknown")


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  N/A"
    return f"{100.0 * num / denom:5.1f}%"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
