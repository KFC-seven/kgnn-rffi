from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.compact import load_compact_dataset
from diagnostic.config import load_config
from diagnostic.datasets import materialize_records
from diagnostic.osr import reject_by_threshold
from diagnostic.sourceonly import infer_logits_embeddings, train_sourceonly
from diagnostic.splits import build_manifest, build_split_records
from kgnn.baselines.posthoc import knn_unknown_score
from kgnn import (
    build_ip_gate_model,
    classify_perturbation_safety,
    default_perturbation_specs,
    predict_ip_gate,
    select_specs,
)
from kgnn.envelope import (
    IpGateModel,
    calibrate_threshold,
    distance_matrix,
    prototype_scores,
)
from kgnn.metrics import (
    EXTENDED_OSR_METRIC_KEYS,
    compute_osr_extended_metrics,
)
from kgnn import PerturbationConfig, PerturbationEngine
from kgnn.utils import (
    _configure_torch_determinism,
    _resolve_device,
    _select_protocol,
    _select_split,
)


DEV_RUNS = [
    "V46S1a|configs/manysig_soda4.yaml|RX9-3_TX2-4|1|tiny|5|100|64",
    "V46S1b|configs/manysig_soda4.yaml|RX9-3_TX2-4|2|tiny|5|100|64",
    "V46T1a|configs/manytx_owen_v0.yaml|MTX_RX9-3_TX20-20|1|resnet1d|5|30|128",
    "V46T1b|configs/manytx_owen_v0.yaml|MTX_RX9-3_TX20-20|2|resnet1d|5|30|128",
]

BASE_FRR = 0.03
FRR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
AUX_METHODS = ["prototype_euclidean", "knn_cosine_k5", "knn_euclidean_k20"]
ALPHA_GRID = [0.25, 0.50, 0.75, 0.90]
IP_GATE_SUPPORT1_ID_KEY = "ip_gate_v41_support1_id"


def _parse_run(text: str) -> dict:
    parts = text.split("|")
    if len(parts) != 8:
        raise ValueError(
            "Run spec must be run_id|config|protocol|split_id|model|epochs|max_samples_per_record|embedding_dim"
        )
    return {
        "run_id": parts[0],
        "config": parts[1],
        "protocol": parts[2],
        "split_id": int(parts[3]),
        "model": parts[4],
        "epochs": int(parts[5]),
        "max_samples_per_record": _parse_sample_cap(parts[6]),
        "embedding_dim": int(parts[7]),
    }


def _parse_sample_cap(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().lower()
    if text in {"", "full", "none", "uncapped"}:
        return None
    return int(text)


def _format_sample_cap(value: int | None) -> str | int:
    return "full" if value is None else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Paper A v4.6 M0-M2 source-only trade-off dev gates.")
    parser.add_argument("--run", action="append", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal-equalized", type=int, default=1)
    parser.add_argument("--sample-mode", default="head", choices=["head", "random"])
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--source-frr", type=float, default=BASE_FRR)
    parser.add_argument("--frr-grid", default=",".join(str(value) for value in FRR_GRID))
    parser.add_argument("--aux-methods", default=",".join(AUX_METHODS))
    parser.add_argument("--alpha-grid", default=",".join(str(value) for value in ALPHA_GRID))
    parser.add_argument("--enable-v50-variants", action="store_true")
    parser.add_argument("--v50-alpha-grid", default="0.50,0.75,0.90")
    parser.add_argument("--v50-reliability-alpha-low", type=float, default=0.50)
    parser.add_argument("--v50-reliability-alpha-high", type=float, default=0.90)
    parser.add_argument("--enable-v51-variants", action="store_true")
    parser.add_argument("--v51-alpha-grid", default="0.50,0.75,0.90,0.95")
    parser.add_argument("--v51-alpha-low", type=float, default=0.50)
    parser.add_argument("--v51-alpha-high", type=float, default=0.95)
    parser.add_argument("--v51-switch-threshold", type=float, default=0.55)
    parser.add_argument("--v51-envelope-quantile", type=float, default=0.90)
    parser.add_argument("--v51-envelope-max-mult", type=float, default=1.75)
    parser.add_argument("--v51-ip-z-low", type=float, default=0.0)
    parser.add_argument("--v51-ip-z-high", type=float, default=1.0)
    parser.add_argument("--enable-v51-component-ablations", action="store_true")
    parser.add_argument("--enable-v51-sensitivity-grid", action="store_true")
    parser.add_argument("--v51-envelope-only-sensitivity", action="store_true")
    parser.add_argument("--v51-envelope-quantile-grid", default="0.80,0.85,0.90,0.95,0.975")
    parser.add_argument("--v51-envelope-max-mult-grid", default="1.25,1.50,1.75,2.00")
    parser.add_argument("--v51-alpha-pair-grid", default="0.50:0.90,0.50:0.95,0.50:1.00,0.60:0.95,0.40:0.95")
    parser.add_argument("--v51-knn-sensitivity-grid", default="cosine:3,cosine:5,cosine:10,euclidean:5,euclidean:20")
    parser.add_argument("--v51-source-frr-grid", default="0.01,0.02,0.03,0.05")
    parser.add_argument("--v51-ip-guard-grid", default="0.0:1.0,0.0:1.5,-0.5:1.0")
    parser.add_argument("--v51-switch-threshold-grid", default="0.45,0.55,0.65")
    parser.add_argument("--knn-k-list", default="5,20")
    parser.add_argument("--knn-metrics", default="cosine,euclidean")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--max-samples-per-record-override", default=None)
    parser.add_argument("--summary-title", default="Paper A v4.6 Trade-Off Bridge Results")
    parser.add_argument("--scope-note", default="Scope: M0-M2 dev gates only. No full M2 was run.")
    args = parser.parse_args()
    frr_grid = _parse_float_list(args.frr_grid)
    aux_methods = _parse_str_list(args.aux_methods)
    alpha_grid = _parse_float_list(args.alpha_grid)
    v50_alpha_grid = _parse_float_list(args.v50_alpha_grid)
    v51_alpha_grid = _parse_float_list(args.v51_alpha_grid)
    v51_envelope_quantile_grid = _parse_float_list(args.v51_envelope_quantile_grid)
    v51_envelope_max_mult_grid = _parse_float_list(args.v51_envelope_max_mult_grid)
    v51_alpha_pair_grid = _parse_float_pairs(args.v51_alpha_pair_grid)
    v51_knn_sensitivity_grid = _parse_knn_specs(args.v51_knn_sensitivity_grid)
    v51_source_frr_grid = _parse_float_list(args.v51_source_frr_grid)
    v51_ip_guard_grid = _parse_float_pairs(args.v51_ip_guard_grid)
    v51_switch_threshold_grid = _parse_float_list(args.v51_switch_threshold_grid)
    knn_k_list = _parse_int_list(args.knn_k_list)
    knn_metrics = _parse_str_list(args.knn_metrics)

    device = _resolve_device(args.device)
    _configure_torch_determinism()
    output_dir = Path(args.output_dir)
    score_dir = output_dir / "score_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    blend_rows: list[dict] = []
    id_head_rows: list[dict] = []
    v50_rows: list[dict] = []
    v51_rows: list[dict] = []
    selector_rows: list[dict] = []
    cache_manifest: list[dict] = []
    run_payloads: list[dict] = []

    run_texts = args.run or DEV_RUNS
    for spec in [_parse_run(text) for text in run_texts]:
        if args.epochs_override is not None:
            spec["epochs"] = int(args.epochs_override)
        if args.max_samples_per_record_override is not None:
            spec["max_samples_per_record"] = _parse_sample_cap(args.max_samples_per_record_override)
        payload = _run_one(
            spec=spec,
            args=args,
            device=device,
            score_dir=score_dir,
            frr_grid=frr_grid,
            aux_methods=aux_methods,
            alpha_grid=alpha_grid,
            v50_alpha_grid=v50_alpha_grid,
            v51_alpha_grid=v51_alpha_grid,
            v51_envelope_quantile_grid=v51_envelope_quantile_grid,
            v51_envelope_max_mult_grid=v51_envelope_max_mult_grid,
            v51_alpha_pair_grid=v51_alpha_pair_grid,
            v51_knn_sensitivity_grid=v51_knn_sensitivity_grid,
            v51_source_frr_grid=v51_source_frr_grid,
            v51_ip_guard_grid=v51_ip_guard_grid,
            v51_switch_threshold_grid=v51_switch_threshold_grid,
            knn_k_list=knn_k_list,
            knn_metrics=knn_metrics,
        )
        run_payloads.append(payload["run_payload"])
        metric_rows.extend(payload["metric_rows"])
        threshold_rows.extend(payload["threshold_rows"])
        blend_rows.extend(payload["blend_rows"])
        id_head_rows.extend(payload["id_head_rows"])
        v50_rows.extend(payload["v50_rows"])
        v51_rows.extend(payload["v51_rows"])
        selector_rows.extend(payload["selector_rows"])
        cache_manifest.append(payload["cache_manifest"])

    all_rows = metric_rows + threshold_rows + blend_rows + id_head_rows + v50_rows + v51_rows
    summary = _summarize(all_rows)
    gate_summary = _gate_summary(summary)
    threshold_gate = _threshold_gate_summary(summary)
    selector_summary = _selector_summary(selector_rows)
    refined_selector_summary = _refined_selector_summary(selector_rows)

    _write_csv(output_dir / "per_run.csv", all_rows)
    _write_csv(output_dir / "m0_reproduction_rows.csv", metric_rows)
    _write_csv(output_dir / "m1_threshold_grid.csv", threshold_rows)
    _write_csv(output_dir / "m2_blend_dev.csv", blend_rows)
    _write_csv(output_dir / "m3_id_head_swap.csv", id_head_rows)
    _write_csv(output_dir / "v50_blend_nnid_rows.csv", v50_rows)
    _write_csv(output_dir / "v51_adaptive_nnid_rows.csv", v51_rows)
    _write_csv(output_dir / "source_selector.csv", selector_rows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "gate_summary.csv", gate_summary)
    _write_csv(output_dir / "threshold_gate_summary.csv", threshold_gate)
    _write_csv(output_dir / "source_selector_summary.csv", selector_summary)
    _write_csv(output_dir / "refined_selector_summary.csv", refined_selector_summary)
    (output_dir / "score_cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "v46_tradeoff_results.json").write_text(
        json.dumps(
            {
                "runs": run_payloads,
                "summary": summary,
                "gate_summary": gate_summary,
                "threshold_gate_summary": threshold_gate,
                "source_selector_summary": selector_summary,
                "refined_selector_summary": refined_selector_summary,
                "id_head_swap_rows": id_head_rows,
                "v50_blend_nnid_rows": v50_rows,
                "v51_adaptive_nnid_rows": v51_rows,
                "score_cache_manifest": cache_manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            summary,
            gate_summary,
            threshold_gate,
            selector_summary,
            refined_selector_summary,
            title=args.summary_title,
            scope_note=args.scope_note,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}")
    return 0


def _run_one(
    *,
    spec: dict,
    args: argparse.Namespace,
    device: str,
    score_dir: Path,
    frr_grid: list[float],
    aux_methods: list[str],
    alpha_grid: list[float],
    v50_alpha_grid: list[float],
    v51_alpha_grid: list[float],
    v51_envelope_quantile_grid: list[float],
    v51_envelope_max_mult_grid: list[float],
    v51_alpha_pair_grid: list[tuple[float, float]],
    v51_knn_sensitivity_grid: list[tuple[str, int]],
    v51_source_frr_grid: list[float],
    v51_ip_guard_grid: list[tuple[float, float]],
    v51_switch_threshold_grid: list[float],
    knn_k_list: list[int],
    knn_metrics: list[str],
) -> dict:
    config = load_config(spec["config"])
    manifest = build_manifest(config)
    dataset = load_compact_dataset(config["dataset"]["path"])
    protocol = _select_protocol(manifest, spec["protocol"])
    split = _select_split(protocol, spec["split_id"])
    records = build_split_records(
        known_txs=split["known_txs"],
        unknown_txs=split["unknown_txs"],
        source_rxs=protocol["source_rxs"],
        drift_rxs=protocol["drift_rxs"],
        source_date=manifest["dates"]["source"],
        day_shift_date=manifest["dates"]["day_shift"],
    )
    source_records = [r for r in records if r["split_name"] == "source_train"]
    eval_records = [r for r in records if r["split_name"] != "source_train"]
    source = materialize_records(
        dataset=dataset,
        records=source_records,
        signal_equalized=int(args.signal_equalized),
        max_samples_per_record=spec["max_samples_per_record"],
        sample_mode=args.sample_mode,
        sample_seed=int(args.seed if args.sample_seed is None else args.sample_seed),
    )
    eval_batch = materialize_records(
        dataset=dataset,
        records=eval_records,
        signal_equalized=int(args.signal_equalized),
        max_samples_per_record=spec["max_samples_per_record"],
        sample_mode=args.sample_mode,
        sample_seed=int(args.seed if args.sample_seed is None else args.sample_seed),
    )

    train_result = train_sourceonly(
        x=source.x,
        y=source.known_label,
        num_classes=len(split["known_txs"]),
        epochs=spec["epochs"],
        batch_size=args.batch_size,
        seed=args.seed,
        embedding_dim=spec["embedding_dim"],
        model_name=spec["model"],
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        device=device,
    )
    source_logits, source_embeddings = infer_logits_embeddings(
        train_result.model,
        source.x,
        batch_size=args.batch_size,
        device=device,
    )
    eval_logits, eval_embeddings = infer_logits_embeddings(
        train_result.model,
        eval_batch.x,
        batch_size=args.batch_size,
        device=device,
    )
    train_idx = np.asarray(train_result.train_indices, dtype=np.int64)
    val_idx = np.asarray(train_result.val_indices, dtype=np.int64)
    train_labels = np.asarray(source.known_label[train_idx], dtype=np.int64)
    val_labels = np.asarray(source.known_label[val_idx], dtype=np.int64)
    num_classes = len(split["known_txs"])

    perturb = PerturbationEngine(
        config=PerturbationConfig(),
        seed=args.seed + int(spec["split_id"]) * 7919,
    )
    safety_results = classify_perturbation_safety(
        encoder=train_result.model,
        source_x=source.x[val_idx],
        source_labels=source.known_label[val_idx],
        perturbation_engine=perturb,
        specs=default_perturbation_specs(),
        safe_accuracy=0.90,
        destructive_accuracy=0.50,
        threshold_mode="absolute",
        max_samples_per_class=25,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        use_distance_fallback=True,
        fallback_min_safe_specs=2,
        fallback_clean_accuracy=0.85,
        safe_distance_mult=1.5,
        destructive_distance_mult=3.0,
    )
    safe_specs = select_specs(safety_results, "safe")
    destructive_specs = select_specs(safety_results, "destructive")
    ip_model, ip_info = build_ip_gate_model(
        encoder=train_result.model,
        source_x=source.x,
        source_labels=source.known_label,
        train_indices=train_idx,
        val_indices=val_idx,
        num_classes=num_classes,
        safe_specs=safe_specs,
        destructive_specs=destructive_specs,
        perturbation_engine=perturb,
        augment_count=4,
        sigma_mult=0.3,
        max_support_per_class=1000,
        max_destructive_bank=5000,
        destructive_per_sample=1,
        batch_size=args.batch_size,
        device=device,
        distance="cosine",
        score_mode="ratio",
        support_k=1,
        destructive_k=1,
        class_norm_alpha=1.0,
        destructive_balance="none",
        score_calibration="none",
        frr=float(args.source_frr),
        gate_support=True,
        seed=args.seed,
    )

    val_embeddings = source_embeddings[val_idx]
    ip_val_support_pred, _ip_val_rej, ip_val_score = predict_ip_gate(ip_model, val_embeddings)
    ip_eval_support_pred, _ip_eval_rej, ip_eval_score = predict_ip_gate(ip_model, eval_embeddings)
    logit_val_pred = np.argmax(source_logits[val_idx], axis=1).astype(np.int64)
    logit_eval_pred = np.argmax(eval_logits, axis=1).astype(np.int64)

    score_bank = _score_bank(
        source_embeddings=source_embeddings,
        eval_embeddings=eval_embeddings,
        source_labels=source.known_label,
        train_idx=train_idx,
        val_idx=val_idx,
        num_classes=num_classes,
        ip_val_score=ip_val_score,
        ip_eval_score=ip_eval_score,
        logit_val_pred=logit_val_pred,
        logit_eval_pred=logit_eval_pred,
        ip_val_support_pred=ip_val_support_pred,
        ip_eval_support_pred=ip_eval_support_pred,
        knn_k_list=knn_k_list,
        knn_metrics=knn_metrics,
    )
    deploy_score_bank = {
        name: payload
        for name, payload in score_bank.items()
        # This M3-ID entry shares IP-GATE v4.1 scores and changes only predicted_label.
        if name != IP_GATE_SUPPORT1_ID_KEY
    }
    common = {
        "run_id": spec["run_id"],
        "dataset": manifest["dataset"]["name"],
        "protocol": protocol["name"],
        "split_id": int(split["split_id"]),
        "model": spec["model"],
        "epochs": int(spec["epochs"]),
        "max_samples_per_record": _format_sample_cap(spec["max_samples_per_record"]),
        "embedding_dim": int(spec["embedding_dim"]),
        "signal_equalized": int(args.signal_equalized),
        "sample_mode": args.sample_mode,
        "sample_seed": int(args.seed if args.sample_seed is None else args.sample_seed),
        "source_val_accuracy": float(train_result.best_val_accuracy),
        "best_epoch": int(train_result.best_epoch),
        "trained_epochs": int(train_result.trained_epochs),
        "stopped_epoch": int(train_result.stopped_epoch),
        "early_stopped": bool(train_result.early_stopped),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "source_samples": int(source.x.shape[0]),
        "source_val_samples": int(val_idx.shape[0]),
        "eval_samples": int(eval_batch.x.shape[0]),
        "device": device,
        "safe_regimes": int(len(safe_specs)),
        "destructive_regimes": int(len(destructive_specs)),
        "destructive_bank_size": int(ip_model.destructive_embeddings.shape[0]),
        "support_samples": int(ip_model.support_embeddings.shape[0]),
        "ip_gate_threshold_v41": float(ip_model.threshold),
        "ip_gate_info_threshold": float(ip_info.get("threshold", 0.0)),
        "deployable": True,
        "uses_target_labels": False,
    }

    metric_rows = [
        _score_metric_row(
            common,
            stage="M0",
            method=method,
            val_score=payload["val_score"],
            eval_score=payload["eval_score"],
            pred=payload["eval_pred"],
            eval_batch=eval_batch,
            source_frr=float(args.source_frr),
            family=payload["family"],
        )
        for method, payload in deploy_score_bank.items()
    ]
    threshold_rows = [
        _score_metric_row(
            common,
            stage="M1",
            method=f"ip_gate_v41_source_frr_{_frr_tag(frr)}",
            val_score=score_bank["ip_gate_v41"]["val_score"],
            eval_score=score_bank["ip_gate_v41"]["eval_score"],
            pred=score_bank["ip_gate_v41"]["eval_pred"],
            eval_batch=eval_batch,
            source_frr=frr,
            family="threshold_sweep",
        )
        for frr in frr_grid
    ]
    blend_rows = []
    blend_payloads = _blend_bank(deploy_score_bank, aux_methods=aux_methods, alpha_grid=alpha_grid)
    for method, payload in blend_payloads.items():
        blend_rows.append(
            _score_metric_row(
                common,
                stage="M2",
                method=method,
                val_score=payload["val_score"],
                eval_score=payload["eval_score"],
                pred=payload["eval_pred"],
                eval_batch=eval_batch,
                source_frr=float(args.source_frr),
                family=payload["family"],
                alpha=payload["alpha"],
                aux_method=payload["aux_method"],
            )
        )
    id_head_rows = []
    for method, payload in _ip_gate_id_head_bank(score_bank).items():
        id_head_rows.append(
            _score_metric_row(
                common,
                stage="M3-ID",
                method=method,
                val_score=payload["val_score"],
                eval_score=payload["eval_score"],
                pred=payload["eval_pred"],
                eval_batch=eval_batch,
                source_frr=float(args.source_frr),
                family=payload["family"],
                aux_method=payload["aux_method"],
            )
        )
    v50_rows = []
    if bool(getattr(args, "enable_v50_variants", False)):
        for method, payload in _v50_blend_nnid_bank(
            score_bank,
            train_embeddings=source_embeddings[train_idx],
            train_labels=source.known_label[train_idx],
            val_embeddings=val_embeddings,
            eval_embeddings=eval_embeddings,
            alpha_grid=v50_alpha_grid,
            reliability_alpha_low=float(args.v50_reliability_alpha_low),
            reliability_alpha_high=float(args.v50_reliability_alpha_high),
        ).items():
            v50_rows.append(
                _score_metric_row(
                    common,
                    stage="V50",
                    method=method,
                    val_score=payload["val_score"],
                    eval_score=payload["eval_score"],
                    pred=payload["eval_pred"],
                    eval_batch=eval_batch,
                    source_frr=float(args.source_frr),
                    family=payload["family"],
                    alpha=payload.get("alpha", ""),
                    aux_method=payload.get("aux_method", ""),
                )
            )
    v51_rows = []
    if bool(getattr(args, "enable_v51_variants", False)):
        for method, payload in _v51_adaptive_nnid_bank(
            score_bank,
            ip_model=ip_model,
            source_embeddings=source_embeddings,
            source_labels=source.known_label,
            train_idx=train_idx,
            val_idx=val_idx,
            num_classes=num_classes,
            train_embeddings=source_embeddings[train_idx],
            train_labels=source.known_label[train_idx],
            val_embeddings=val_embeddings,
            eval_embeddings=eval_embeddings,
            alpha_grid=v51_alpha_grid,
            alpha_low=float(args.v51_alpha_low),
            alpha_high=float(args.v51_alpha_high),
            switch_threshold=float(args.v51_switch_threshold),
            envelope_quantile=float(args.v51_envelope_quantile),
            envelope_max_mult=float(args.v51_envelope_max_mult),
            ip_z_low=float(args.v51_ip_z_low),
            ip_z_high=float(args.v51_ip_z_high),
            source_frr=float(args.source_frr),
            component_ablations=bool(getattr(args, "enable_v51_component_ablations", False)),
            sensitivity_grid=bool(getattr(args, "enable_v51_sensitivity_grid", False)),
            envelope_only_sensitivity=bool(getattr(args, "v51_envelope_only_sensitivity", False)),
            envelope_quantile_grid=v51_envelope_quantile_grid,
            envelope_max_mult_grid=v51_envelope_max_mult_grid,
            alpha_pair_grid=v51_alpha_pair_grid,
            knn_sensitivity_grid=v51_knn_sensitivity_grid,
            source_frr_grid=v51_source_frr_grid,
            ip_guard_grid=v51_ip_guard_grid,
            switch_threshold_grid=v51_switch_threshold_grid,
        ).items():
            row_source_frr = float(payload.get("source_frr", args.source_frr))
            row = _score_metric_row(
                common,
                stage="V51",
                method=method,
                val_score=payload["val_score"],
                eval_score=payload["eval_score"],
                pred=payload["eval_pred"],
                eval_batch=eval_batch,
                source_frr=row_source_frr,
                family=payload["family"],
                alpha=payload.get("alpha", ""),
                aux_method=payload.get("aux_method", ""),
            )
            for extra_key in [
                "val_reliability_mean",
                "eval_reliability_mean",
                "selected_alpha",
                "source_selector_objective",
                "source_selector_metric",
                "v51_variant_role",
                "v51_sensitivity_axis",
                "v51_envelope_quantile",
                "v51_envelope_max_mult",
                "v51_alpha_low",
                "v51_alpha_high",
                "v51_knn_metric",
                "v51_knn_k",
                "v51_source_frr",
                "v51_ip_z_low",
                "v51_ip_z_high",
                "v51_switch_threshold",
                "v51_id_head",
            ]:
                if extra_key in payload:
                    row[extra_key] = payload[extra_key]
            v51_rows.append(row)

    selector_rows = _source_selector_rows(
        common=common,
        score_bank={**deploy_score_bank, **blend_payloads},
        ip_model=ip_model,
        source_embeddings=source_embeddings,
        source_labels=source.known_label,
        train_idx=train_idx,
        val_idx=val_idx,
        num_classes=num_classes,
        source_frr=float(args.source_frr),
    )
    cache_path = score_dir / f"{spec['run_id']}_{protocol['name']}_split{split['split_id']}.npz"
    np.savez_compressed(
        cache_path,
        val_labels=val_labels,
        val_is_known=np.ones(val_idx.shape[0], dtype=bool),
        eval_labels=eval_batch.known_label,
        eval_is_known=eval_batch.is_known,
        eval_is_shifted_known=eval_batch.is_shifted_known,
        **{f"val_{method}": payload["val_score"] for method, payload in score_bank.items()},
        **{f"eval_{method}": payload["eval_score"] for method, payload in score_bank.items()},
    )
    run_payload = {
        **common,
        "cache_path": str(cache_path),
        "methods_cached": sorted(score_bank.keys()),
        "frr_grid": frr_grid,
        "aux_methods": aux_methods,
        "alpha_grid": alpha_grid,
        "v50_enabled": bool(getattr(args, "enable_v50_variants", False)),
        "v50_alpha_grid": v50_alpha_grid,
        "v50_reliability_alpha_low": float(args.v50_reliability_alpha_low),
        "v50_reliability_alpha_high": float(args.v50_reliability_alpha_high),
        "v51_enabled": bool(getattr(args, "enable_v51_variants", False)),
        "v51_alpha_grid": v51_alpha_grid,
        "v51_alpha_low": float(args.v51_alpha_low),
        "v51_alpha_high": float(args.v51_alpha_high),
        "v51_switch_threshold": float(args.v51_switch_threshold),
        "v51_envelope_quantile": float(args.v51_envelope_quantile),
        "v51_envelope_max_mult": float(args.v51_envelope_max_mult),
        "v51_component_ablations": bool(getattr(args, "enable_v51_component_ablations", False)),
        "v51_sensitivity_grid": bool(getattr(args, "enable_v51_sensitivity_grid", False)),
        "v51_envelope_only_sensitivity": bool(getattr(args, "v51_envelope_only_sensitivity", False)),
        "v51_envelope_quantile_grid": v51_envelope_quantile_grid,
        "v51_envelope_max_mult_grid": v51_envelope_max_mult_grid,
        "v51_alpha_pair_grid": v51_alpha_pair_grid,
        "v51_knn_sensitivity_grid": v51_knn_sensitivity_grid,
        "v51_source_frr_grid": v51_source_frr_grid,
        "v51_ip_guard_grid": v51_ip_guard_grid,
        "v51_switch_threshold_grid": v51_switch_threshold_grid,
        "knn_k_list": knn_k_list,
        "knn_metrics": knn_metrics,
    }
    print(
        "{run_id} {dataset} {protocol}: IP AUC={auc:.4f} AUOSC={auosc:.4f} H={h:.4f} FRR={frr:.4f}".format(
            run_id=spec["run_id"],
            dataset=manifest["dataset"]["name"],
            protocol=protocol["name"],
            auc=float(metric_rows[0]["auc_shifted_known_vs_unknown"]),
            auosc=float(metric_rows[0]["auosc_shifted_known_vs_unknown"]),
            h=float(metric_rows[0]["sample_open_set_h_score"]),
            frr=float(metric_rows[0]["shifted_known_false_rejection_rate"]),
        )
    )
    return {
        "run_payload": run_payload,
        "metric_rows": metric_rows,
        "threshold_rows": threshold_rows,
        "blend_rows": blend_rows,
        "id_head_rows": id_head_rows,
        "v50_rows": v50_rows,
        "v51_rows": v51_rows,
        "selector_rows": selector_rows,
        "cache_manifest": {
            "run_id": spec["run_id"],
            "dataset": manifest["dataset"]["name"],
            "protocol": protocol["name"],
            "split_id": int(split["split_id"]),
            "cache_path": str(cache_path),
            "methods": sorted(score_bank.keys()),
        },
    }


def _score_bank(
    *,
    source_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    source_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    ip_val_score: np.ndarray,
    ip_eval_score: np.ndarray,
    logit_val_pred: np.ndarray,
    logit_eval_pred: np.ndarray,
    ip_val_support_pred: np.ndarray,
    ip_eval_support_pred: np.ndarray,
    knn_k_list: list[int],
    knn_metrics: list[str],
) -> dict[str, dict]:
    train_embeddings = source_embeddings[train_idx]
    train_labels = np.asarray(source_labels[train_idx], dtype=np.int64)
    val_embeddings = source_embeddings[val_idx]
    bank: dict[str, dict] = {
        "ip_gate_v41": {
            "val_score": np.asarray(ip_val_score, dtype=np.float32),
            "eval_score": np.asarray(ip_eval_score, dtype=np.float32),
            "val_pred": logit_val_pred,
            "eval_pred": logit_eval_pred,
            "family": "ip_gate",
        }
    }
    # Keep this key in sync with the deploy_score_bank exclusion above.
    bank[IP_GATE_SUPPORT1_ID_KEY] = {
        "val_score": np.asarray(ip_val_score, dtype=np.float32),
        "eval_score": np.asarray(ip_eval_score, dtype=np.float32),
        "val_pred": np.asarray(ip_val_support_pred, dtype=np.int64),
        "eval_pred": np.asarray(ip_eval_support_pred, dtype=np.int64),
        "family": "ip_gate_id_head",
    }
    prototypes = np.stack(
        [np.mean(train_embeddings[train_labels == cls], axis=0) for cls in range(int(num_classes))],
        axis=0,
    ).astype(np.float32)
    proto_val_score, proto_val_pred = prototype_scores(val_embeddings, prototypes, distance="euclidean")
    proto_eval_score, proto_eval_pred = prototype_scores(eval_embeddings, prototypes, distance="euclidean")
    bank["prototype_euclidean"] = {
        "val_score": proto_val_score,
        "eval_score": proto_eval_score,
        "val_pred": proto_val_pred,
        "eval_pred": proto_eval_pred,
        "family": "auxiliary",
    }
    for metric in knn_metrics:
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported kNN metric={metric!r}")
        for k in knn_k_list:
            if int(k) < 1:
                raise ValueError(f"kNN k must be positive, got {k!r}")
            val = knn_unknown_score(train_embeddings, train_labels, val_embeddings, k=int(k), metric=metric)
            ev = knn_unknown_score(train_embeddings, train_labels, eval_embeddings, k=int(k), metric=metric)
            bank[f"knn_{metric}_k{int(k)}"] = {
                "val_score": val.scores,
                "eval_score": ev.scores,
                "val_pred": val.predicted_label,
                "eval_pred": ev.predicted_label,
                "family": "auxiliary",
            }
    return bank


def _ip_gate_id_head_bank(score_bank: dict[str, dict]) -> dict[str, dict]:
    ip = score_bank["ip_gate_v41"]
    candidates = {
        IP_GATE_SUPPORT1_ID_KEY: score_bank[IP_GATE_SUPPORT1_ID_KEY],
        "prototype_euclidean": score_bank["prototype_euclidean"],
    }
    for name in sorted(score_bank):
        if name.startswith("knn_"):
            candidates[name] = score_bank[name]
    out: dict[str, dict] = {}
    for name, payload in candidates.items():
        method = "ip_gate_v41_score_" + name.replace("ip_gate_v41_", "")
        out[method] = {
            "val_score": ip["val_score"],
            "eval_score": ip["eval_score"],
            "val_pred": payload["val_pred"],
            "eval_pred": payload["eval_pred"],
            "family": "ip_gate_id_head_swap",
            "aux_method": name,
        }
    return out


def _blend_bank(score_bank: dict[str, dict], *, aux_methods: list[str], alpha_grid: list[float]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ip = score_bank["ip_gate_v41"]
    ip_val_z, ip_eval_z = _source_z(ip["val_score"], ip["eval_score"])
    for aux_name in aux_methods:
        aux = score_bank[aux_name]
        aux_val_z, aux_eval_z = _source_z(aux["val_score"], aux["eval_score"])
        for alpha in alpha_grid:
            method = f"blend_ip_{aux_name}_alpha{alpha:.2f}".replace(".", "p")
            out[method] = {
                "val_score": (float(alpha) * ip_val_z + (1.0 - float(alpha)) * aux_val_z).astype(np.float32),
                "eval_score": (float(alpha) * ip_eval_z + (1.0 - float(alpha)) * aux_eval_z).astype(np.float32),
                "val_pred": ip["val_pred"],
                "eval_pred": ip["eval_pred"],
                "family": "blend",
                "alpha": float(alpha),
                "aux_method": aux_name,
            }
    return out


def _v50_blend_nnid_bank(
    score_bank: dict[str, dict],
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    alpha_grid: list[float],
    reliability_alpha_low: float,
    reliability_alpha_high: float,
) -> dict[str, dict]:
    if "knn_cosine_k5" not in score_bank:
        return {}
    out: dict[str, dict] = {}
    ip = score_bank["ip_gate_v41"]
    knn = score_bank["knn_cosine_k5"]
    ip_val_z, ip_eval_z = _source_z(ip["val_score"], ip["eval_score"])
    knn_val_z, knn_eval_z = _source_z(knn["val_score"], knn["eval_score"])
    for alpha in alpha_grid:
        tag = _alpha_tag(float(alpha))
        method = f"v50_blend_ip_knn_cosine_k5_alpha{tag}_nnid"
        out[method] = {
            "val_score": (float(alpha) * ip_val_z + (1.0 - float(alpha)) * knn_val_z).astype(np.float32),
            "eval_score": (float(alpha) * ip_eval_z + (1.0 - float(alpha)) * knn_eval_z).astype(np.float32),
            "val_pred": knn["val_pred"],
            "eval_pred": knn["eval_pred"],
            "family": "v50_blend_nnid",
            "alpha": float(alpha),
            "aux_method": "knn_cosine_k5",
        }
    val_reliability = _knn_vote_reliability(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=val_embeddings,
        k=5,
        metric="cosine",
    )
    eval_reliability = _knn_vote_reliability(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=eval_embeddings,
        k=5,
        metric="cosine",
    )
    alpha_low = float(reliability_alpha_low)
    alpha_high = float(reliability_alpha_high)
    val_alpha = (alpha_high - (alpha_high - alpha_low) * val_reliability).astype(np.float32)
    eval_alpha = (alpha_high - (alpha_high - alpha_low) * eval_reliability).astype(np.float32)
    method = f"v50_reliability_blend_ip_knn_cosine_k5_alpha{_alpha_tag(alpha_low)}_{_alpha_tag(alpha_high)}_nnid"
    out[method] = {
        "val_score": (val_alpha * ip_val_z + (1.0 - val_alpha) * knn_val_z).astype(np.float32),
        "eval_score": (eval_alpha * ip_eval_z + (1.0 - eval_alpha) * knn_eval_z).astype(np.float32),
        "val_pred": knn["val_pred"],
        "eval_pred": knn["eval_pred"],
        "family": "v50_reliability_blend_nnid",
        "alpha": f"{alpha_low:.2f}-{alpha_high:.2f}",
        "aux_method": "knn_cosine_k5",
        "val_reliability_mean": float(np.mean(val_reliability)),
        "eval_reliability_mean": float(np.mean(eval_reliability)),
    }
    return out


def _knn_vote_reliability(
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    n_neighbors = max(1, min(int(k), train_embeddings.shape[0]))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(train_embeddings)
    _distances, indices = nn.kneighbors(query_embeddings, return_distance=True)
    neighbor_labels = train_labels[indices]
    reliabilities = np.zeros(neighbor_labels.shape[0], dtype=np.float32)
    for i, labels in enumerate(neighbor_labels):
        _values, counts = np.unique(labels, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        top = float(sorted_counts[0]) if sorted_counts.size else 0.0
        second = float(sorted_counts[1]) if sorted_counts.size > 1 else 0.0
        purity = top / float(n_neighbors)
        margin = (top - second) / float(n_neighbors)
        reliabilities[i] = math.sqrt(max(0.0, purity) * max(0.0, margin))
    return np.clip(reliabilities, 0.0, 1.0).astype(np.float32)


def _v51_adaptive_nnid_bank(
    score_bank: dict[str, dict],
    *,
    ip_model: IpGateModel,
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    alpha_grid: list[float],
    alpha_low: float,
    alpha_high: float,
    switch_threshold: float,
    envelope_quantile: float,
    envelope_max_mult: float,
    ip_z_low: float,
    ip_z_high: float,
    source_frr: float,
    component_ablations: bool,
    sensitivity_grid: bool,
    envelope_only_sensitivity: bool,
    envelope_quantile_grid: list[float],
    envelope_max_mult_grid: list[float],
    alpha_pair_grid: list[tuple[float, float]],
    knn_sensitivity_grid: list[tuple[str, int]],
    source_frr_grid: list[float],
    ip_guard_grid: list[tuple[float, float]],
    switch_threshold_grid: list[float],
) -> dict[str, dict]:
    if "knn_cosine_k5" not in score_bank:
        return {}
    out: dict[str, dict] = {}
    ip = score_bank["ip_gate_v41"]
    knn = score_bank["knn_cosine_k5"]
    ip_val_z, ip_eval_z = _source_z(ip["val_score"], ip["eval_score"])
    knn_val_z, knn_eval_z = _source_z(knn["val_score"], knn["eval_score"])
    val_vote = _knn_vote_reliability(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=val_embeddings,
        k=5,
        metric="cosine",
    )
    eval_vote = _knn_vote_reliability(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=eval_embeddings,
        k=5,
        metric="cosine",
    )
    val_envelope, val_center_margin = _class_envelope_and_margin_reliability(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=val_embeddings,
        predicted_label=knn["val_pred"],
        num_classes=num_classes,
        quantile=envelope_quantile,
        max_mult=envelope_max_mult,
    )
    eval_envelope, eval_center_margin = _class_envelope_and_margin_reliability(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=eval_embeddings,
        predicted_label=knn["eval_pred"],
        num_classes=num_classes,
        quantile=envelope_quantile,
        max_mult=envelope_max_mult,
    )
    val_margin_envelope = np.clip(np.sqrt(val_vote * val_center_margin) * val_envelope, 0.0, 1.0).astype(np.float32)
    eval_margin_envelope = np.clip(np.sqrt(eval_vote * eval_center_margin) * eval_envelope, 0.0, 1.0).astype(np.float32)
    val_ip_known = _ip_knownness_from_z(ip_val_z, low=ip_z_low, high=ip_z_high)
    eval_ip_known = _ip_knownness_from_z(ip_eval_z, low=ip_z_low, high=ip_z_high)
    val_guarded = np.clip(val_margin_envelope * val_ip_known, 0.0, 1.0).astype(np.float32)
    eval_guarded = np.clip(eval_margin_envelope * eval_ip_known, 0.0, 1.0).astype(np.float32)

    def add_adaptive(
        name: str,
        val_rel: np.ndarray,
        eval_rel: np.ndarray,
        family: str,
        *,
        low: float | None = None,
        high: float | None = None,
        val_pred: np.ndarray | None = None,
        eval_pred: np.ndarray | None = None,
        row_source_frr: float | None = None,
        extras: dict | None = None,
    ) -> None:
        row_low = float(alpha_low if low is None else low)
        row_high = float(alpha_high if high is None else high)
        val_alpha = (row_high - (row_high - row_low) * val_rel).astype(np.float32)
        eval_alpha = (row_high - (row_high - row_low) * eval_rel).astype(np.float32)
        payload = {
            "val_score": (val_alpha * ip_val_z + (1.0 - val_alpha) * knn_val_z).astype(np.float32),
            "eval_score": (eval_alpha * ip_eval_z + (1.0 - eval_alpha) * knn_eval_z).astype(np.float32),
            "val_pred": np.asarray(knn["val_pred"] if val_pred is None else val_pred, dtype=np.int64),
            "eval_pred": np.asarray(knn["eval_pred"] if eval_pred is None else eval_pred, dtype=np.int64),
            "family": family,
            "alpha": f"{row_low:.2f}-{row_high:.2f}",
            "aux_method": "knn_cosine_k5",
            "source_frr": float(source_frr if row_source_frr is None else row_source_frr),
            "v51_alpha_low": row_low,
            "v51_alpha_high": row_high,
            "v51_source_frr": float(source_frr if row_source_frr is None else row_source_frr),
            "v51_envelope_quantile": float(envelope_quantile),
            "v51_envelope_max_mult": float(envelope_max_mult),
            "v51_knn_metric": "cosine",
            "v51_knn_k": 5,
            "val_reliability_mean": float(np.mean(val_rel)),
            "eval_reliability_mean": float(np.mean(eval_rel)),
        }
        if extras:
            payload.update(extras)
        out[name] = payload

    prefix = f"alpha{_alpha_tag(alpha_low)}_{_alpha_tag(alpha_high)}"
    add_adaptive(
        f"v51_margin_envelope_blend_ip_knn_cosine_k5_{prefix}_nnid",
        val_margin_envelope,
        eval_margin_envelope,
        "v51_margin_envelope_blend_nnid",
    )
    add_adaptive(
        f"v51_ipguard_margin_envelope_blend_ip_knn_cosine_k5_{prefix}_nnid",
        val_guarded,
        eval_guarded,
        "v51_ipguard_margin_envelope_blend_nnid",
    )
    val_switch_alpha = np.where(val_guarded >= float(switch_threshold), float(alpha_low), float(alpha_high)).astype(np.float32)
    eval_switch_alpha = np.where(eval_guarded >= float(switch_threshold), float(alpha_low), float(alpha_high)).astype(np.float32)
    out[f"v51_hardswitch_ipguard_blend_ip_knn_cosine_k5_{prefix}_tau{_alpha_tag(switch_threshold)}_nnid"] = {
        "val_score": (val_switch_alpha * ip_val_z + (1.0 - val_switch_alpha) * knn_val_z).astype(np.float32),
        "eval_score": (eval_switch_alpha * ip_eval_z + (1.0 - eval_switch_alpha) * knn_eval_z).astype(np.float32),
        "val_pred": knn["val_pred"],
        "eval_pred": knn["eval_pred"],
        "family": "v51_hardswitch_ipguard_blend_nnid",
        "alpha": f"{alpha_low:.2f}-{alpha_high:.2f}",
        "aux_method": "knn_cosine_k5",
        "val_reliability_mean": float(np.mean(val_guarded)),
        "eval_reliability_mean": float(np.mean(eval_guarded)),
    }
    if component_ablations:
        component_rows = [
            (
                "no_vote",
                np.clip(np.sqrt(val_center_margin) * val_envelope, 0.0, 1.0).astype(np.float32),
                np.clip(np.sqrt(eval_center_margin) * eval_envelope, 0.0, 1.0).astype(np.float32),
                "v51_no_vote_blend_nnid",
            ),
            (
                "no_center_margin",
                np.clip(np.sqrt(val_vote) * val_envelope, 0.0, 1.0).astype(np.float32),
                np.clip(np.sqrt(eval_vote) * eval_envelope, 0.0, 1.0).astype(np.float32),
                "v51_no_center_margin_blend_nnid",
            ),
            (
                "no_vote_no_center_margin",
                np.clip(val_envelope, 0.0, 1.0).astype(np.float32),
                np.clip(eval_envelope, 0.0, 1.0).astype(np.float32),
                "v51_no_vote_no_center_margin_blend_nnid",
            ),
            (
                "no_class_envelope",
                np.clip(np.sqrt(val_vote * val_center_margin), 0.0, 1.0).astype(np.float32),
                np.clip(np.sqrt(eval_vote * eval_center_margin), 0.0, 1.0).astype(np.float32),
                "v51_no_class_envelope_blend_nnid",
            ),
            (
                "vote_only",
                np.clip(val_vote, 0.0, 1.0).astype(np.float32),
                np.clip(eval_vote, 0.0, 1.0).astype(np.float32),
                "v51_vote_only_blend_nnid",
            ),
        ]
        for mode, val_rel, eval_rel, family in component_rows:
            add_adaptive(
                f"v51_{mode}_blend_ip_knn_cosine_k5_{prefix}_nnid",
                val_rel,
                eval_rel,
                family,
                extras={"v51_variant_role": f"component_{mode}"},
            )
        add_adaptive(
            f"v51_no_vote_no_center_margin_blend_ip_knn_cosine_k5_{prefix}_no_nnid",
            np.clip(val_envelope, 0.0, 1.0).astype(np.float32),
            np.clip(eval_envelope, 0.0, 1.0).astype(np.float32),
            "v51_no_vote_no_center_margin_blend_no_nnid",
            val_pred=ip["val_pred"],
            eval_pred=ip["eval_pred"],
            extras={
                "v51_variant_role": "component_no_nnid",
                "v51_id_head": "source_classifier_logit",
            },
        )
    if envelope_only_sensitivity:
        for q in sorted(set(float(value) for value in envelope_quantile_grid)):
            q_val_envelope, _q_val_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                predicted_label=knn["val_pred"],
                num_classes=num_classes,
                quantile=q,
                max_mult=envelope_max_mult,
            )
            q_eval_envelope, _q_eval_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=eval_embeddings,
                predicted_label=knn["eval_pred"],
                num_classes=num_classes,
                quantile=q,
                max_mult=envelope_max_mult,
            )
            add_adaptive(
                f"v51_envonly_sens_q{_alpha_tag(q)}_blend_ip_knn_cosine_k5_{prefix}_nnid",
                np.clip(q_val_envelope, 0.0, 1.0).astype(np.float32),
                np.clip(q_eval_envelope, 0.0, 1.0).astype(np.float32),
                "v51_envelope_only_sensitivity_q",
                extras={
                    "v51_variant_role": "envelope_only_sensitivity",
                    "v51_sensitivity_axis": "class_envelope_quantile",
                    "v51_envelope_quantile": float(q),
                },
            )
        for max_mult_value in sorted(set(float(value) for value in envelope_max_mult_grid)):
            m_val_envelope, _m_val_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                predicted_label=knn["val_pred"],
                num_classes=num_classes,
                quantile=envelope_quantile,
                max_mult=max_mult_value,
            )
            m_eval_envelope, _m_eval_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=eval_embeddings,
                predicted_label=knn["eval_pred"],
                num_classes=num_classes,
                quantile=envelope_quantile,
                max_mult=max_mult_value,
            )
            add_adaptive(
                f"v51_envonly_sens_envmax{_alpha_tag(max_mult_value)}_blend_ip_knn_cosine_k5_{prefix}_nnid",
                np.clip(m_val_envelope, 0.0, 1.0).astype(np.float32),
                np.clip(m_eval_envelope, 0.0, 1.0).astype(np.float32),
                "v51_envelope_only_sensitivity_envelope_max_mult",
                extras={
                    "v51_variant_role": "envelope_only_sensitivity",
                    "v51_sensitivity_axis": "class_envelope_max_mult",
                    "v51_envelope_max_mult": float(max_mult_value),
                },
            )
    if sensitivity_grid:
        for q in sorted(set(float(value) for value in envelope_quantile_grid)):
            q_val_envelope, q_val_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                predicted_label=knn["val_pred"],
                num_classes=num_classes,
                quantile=q,
                max_mult=envelope_max_mult,
            )
            q_eval_envelope, q_eval_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=eval_embeddings,
                predicted_label=knn["eval_pred"],
                num_classes=num_classes,
                quantile=q,
                max_mult=envelope_max_mult,
            )
            q_val_rel = np.clip(np.sqrt(val_vote * q_val_margin) * q_val_envelope, 0.0, 1.0).astype(np.float32)
            q_eval_rel = np.clip(np.sqrt(eval_vote * q_eval_margin) * q_eval_envelope, 0.0, 1.0).astype(np.float32)
            add_adaptive(
                f"v51_sens_q{_alpha_tag(q)}_margin_envelope_blend_ip_knn_cosine_k5_{prefix}_nnid",
                q_val_rel,
                q_eval_rel,
                "v51_sensitivity_q",
                extras={
                    "v51_variant_role": "sensitivity",
                    "v51_sensitivity_axis": "envelope_quantile",
                    "v51_envelope_quantile": float(q),
                },
            )
        for max_mult_value in sorted(set(float(value) for value in envelope_max_mult_grid)):
            m_val_envelope, m_val_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                predicted_label=knn["val_pred"],
                num_classes=num_classes,
                quantile=envelope_quantile,
                max_mult=max_mult_value,
            )
            m_eval_envelope, m_eval_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=eval_embeddings,
                predicted_label=knn["eval_pred"],
                num_classes=num_classes,
                quantile=envelope_quantile,
                max_mult=max_mult_value,
            )
            m_val_rel = np.clip(np.sqrt(val_vote * m_val_margin) * m_val_envelope, 0.0, 1.0).astype(np.float32)
            m_eval_rel = np.clip(np.sqrt(eval_vote * m_eval_margin) * m_eval_envelope, 0.0, 1.0).astype(np.float32)
            add_adaptive(
                f"v51_sens_envmax{_alpha_tag(max_mult_value)}_margin_envelope_blend_ip_knn_cosine_k5_{prefix}_nnid",
                m_val_rel,
                m_eval_rel,
                "v51_sensitivity_envelope_max_mult",
                extras={
                    "v51_variant_role": "sensitivity",
                    "v51_sensitivity_axis": "envelope_max_mult",
                    "v51_envelope_max_mult": float(max_mult_value),
                },
            )
        for low, high in sorted(set((float(low), float(high)) for low, high in alpha_pair_grid)):
            add_adaptive(
                f"v51_sens_alpha{_alpha_tag(low)}_{_alpha_tag(high)}_margin_envelope_blend_ip_knn_cosine_k5_nnid",
                val_margin_envelope,
                eval_margin_envelope,
                "v51_sensitivity_alpha_endpoints",
                low=low,
                high=high,
                extras={
                    "v51_variant_role": "sensitivity",
                    "v51_sensitivity_axis": "alpha_endpoints",
                },
            )
        for row_frr in sorted(set(float(value) for value in source_frr_grid)):
            add_adaptive(
                f"v51_sens_frr{_frr_tag(row_frr)}_margin_envelope_blend_ip_knn_cosine_k5_{prefix}_nnid",
                val_margin_envelope,
                eval_margin_envelope,
                "v51_sensitivity_source_frr",
                row_source_frr=row_frr,
                extras={
                    "v51_variant_role": "sensitivity",
                    "v51_sensitivity_axis": "source_frr",
                },
            )
        for metric, k in sorted(set((str(metric), int(k)) for metric, k in knn_sensitivity_grid)):
            method = f"knn_{metric}_k{int(k)}"
            if method not in score_bank:
                continue
            sens_knn = score_bank[method]
            sens_knn_val_z, sens_knn_eval_z = _source_z(sens_knn["val_score"], sens_knn["eval_score"])
            sens_val_vote = _knn_vote_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                k=int(k),
                metric=metric,
            )
            sens_eval_vote = _knn_vote_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=eval_embeddings,
                k=int(k),
                metric=metric,
            )
            sens_val_envelope, sens_val_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=val_embeddings,
                predicted_label=sens_knn["val_pred"],
                num_classes=num_classes,
                quantile=envelope_quantile,
                max_mult=envelope_max_mult,
            )
            sens_eval_envelope, sens_eval_margin = _class_envelope_and_margin_reliability(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                query_embeddings=eval_embeddings,
                predicted_label=sens_knn["eval_pred"],
                num_classes=num_classes,
                quantile=envelope_quantile,
                max_mult=envelope_max_mult,
            )
            sens_val_rel = np.clip(np.sqrt(sens_val_vote * sens_val_margin) * sens_val_envelope, 0.0, 1.0).astype(np.float32)
            sens_eval_rel = np.clip(np.sqrt(sens_eval_vote * sens_eval_margin) * sens_eval_envelope, 0.0, 1.0).astype(np.float32)
            sens_low = float(alpha_low)
            sens_high = float(alpha_high)
            sens_val_alpha = (sens_high - (sens_high - sens_low) * sens_val_rel).astype(np.float32)
            sens_eval_alpha = (sens_high - (sens_high - sens_low) * sens_eval_rel).astype(np.float32)
            out[f"v51_sens_knn_{metric}_k{int(k)}_margin_envelope_blend_ip_{method}_{prefix}_nnid"] = {
                "val_score": (sens_val_alpha * ip_val_z + (1.0 - sens_val_alpha) * sens_knn_val_z).astype(np.float32),
                "eval_score": (sens_eval_alpha * ip_eval_z + (1.0 - sens_eval_alpha) * sens_knn_eval_z).astype(np.float32),
                "val_pred": sens_knn["val_pred"],
                "eval_pred": sens_knn["eval_pred"],
                "family": "v51_sensitivity_knn",
                "alpha": f"{sens_low:.2f}-{sens_high:.2f}",
                "aux_method": method,
                "source_frr": float(source_frr),
                "v51_variant_role": "sensitivity",
                "v51_sensitivity_axis": "knn",
                "v51_alpha_low": sens_low,
                "v51_alpha_high": sens_high,
                "v51_source_frr": float(source_frr),
                "v51_envelope_quantile": float(envelope_quantile),
                "v51_envelope_max_mult": float(envelope_max_mult),
                "v51_knn_metric": metric,
                "v51_knn_k": int(k),
                "val_reliability_mean": float(np.mean(sens_val_rel)),
                "eval_reliability_mean": float(np.mean(sens_eval_rel)),
            }
        for low_z, high_z in sorted(set((float(low), float(high)) for low, high in ip_guard_grid)):
            val_known = _ip_knownness_from_z(ip_val_z, low=low_z, high=high_z)
            eval_known = _ip_knownness_from_z(ip_eval_z, low=low_z, high=high_z)
            sens_val_guarded = np.clip(val_margin_envelope * val_known, 0.0, 1.0).astype(np.float32)
            sens_eval_guarded = np.clip(eval_margin_envelope * eval_known, 0.0, 1.0).astype(np.float32)
            add_adaptive(
                f"v51_sens_ipguard{_alpha_tag(low_z)}_{_alpha_tag(high_z)}_blend_ip_knn_cosine_k5_{prefix}_nnid",
                sens_val_guarded,
                sens_eval_guarded,
                "v51_sensitivity_ip_guard",
                extras={
                    "v51_variant_role": "sensitivity",
                    "v51_sensitivity_axis": "ip_guard",
                    "v51_ip_z_low": float(low_z),
                    "v51_ip_z_high": float(high_z),
                },
            )
        for tau in sorted(set(float(value) for value in switch_threshold_grid)):
            sens_val_switch_alpha = np.where(val_guarded >= tau, float(alpha_low), float(alpha_high)).astype(np.float32)
            sens_eval_switch_alpha = np.where(eval_guarded >= tau, float(alpha_low), float(alpha_high)).astype(np.float32)
            out[f"v51_sens_tau{_alpha_tag(tau)}_hardswitch_ipguard_blend_ip_knn_cosine_k5_{prefix}_nnid"] = {
                "val_score": (sens_val_switch_alpha * ip_val_z + (1.0 - sens_val_switch_alpha) * knn_val_z).astype(np.float32),
                "eval_score": (sens_eval_switch_alpha * ip_eval_z + (1.0 - sens_eval_switch_alpha) * knn_eval_z).astype(np.float32),
                "val_pred": knn["val_pred"],
                "eval_pred": knn["eval_pred"],
                "family": "v51_sensitivity_hard_switch_tau",
                "alpha": f"{alpha_low:.2f}-{alpha_high:.2f}",
                "aux_method": "knn_cosine_k5",
                "source_frr": float(source_frr),
                "v51_variant_role": "sensitivity",
                "v51_sensitivity_axis": "hard_switch_tau",
                "v51_alpha_low": float(alpha_low),
                "v51_alpha_high": float(alpha_high),
                "v51_source_frr": float(source_frr),
                "v51_envelope_quantile": float(envelope_quantile),
                "v51_envelope_max_mult": float(envelope_max_mult),
                "v51_knn_metric": "cosine",
                "v51_knn_k": 5,
                "v51_switch_threshold": float(tau),
                "val_reliability_mean": float(np.mean(val_guarded)),
                "eval_reliability_mean": float(np.mean(eval_guarded)),
            }
    selected = _v51_source_select_alpha_payload(
        score_bank,
        ip_model=ip_model,
        source_embeddings=source_embeddings,
        source_labels=source_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        num_classes=num_classes,
        alpha_grid=alpha_grid,
        source_frr=source_frr,
    )
    if selected:
        out["v51_source_selector_alpha_nnid"] = selected
    return out


def _class_envelope_and_margin_reliability(
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    predicted_label: np.ndarray,
    num_classes: int,
    quantile: float,
    max_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    predicted_label = np.asarray(predicted_label, dtype=np.int64).reshape(-1)
    centers = []
    radii = []
    for cls in range(int(num_classes)):
        cls_embeddings = train_embeddings[train_labels == cls]
        if cls_embeddings.size == 0:
            centers.append(np.zeros(train_embeddings.shape[1], dtype=np.float32))
            radii.append(1.0)
            continue
        center = np.mean(cls_embeddings, axis=0).astype(np.float32)
        centers.append(center)
        cls_dist = _cosine_distance_matrix(cls_embeddings, center[None, :])[:, 0]
        radii.append(max(float(np.quantile(cls_dist, float(quantile))), 1e-6))
    centers = np.stack(centers, axis=0).astype(np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    dist = _cosine_distance_matrix(query_embeddings, centers)
    pred = np.clip(predicted_label, 0, int(num_classes) - 1)
    pred_dist = dist[np.arange(dist.shape[0]), pred]
    competitor = dist.copy()
    competitor[np.arange(competitor.shape[0]), pred] = np.inf
    second_dist = np.min(competitor, axis=1)
    pred_radius = np.maximum(radii[pred], 1e-6)
    margin = np.clip((second_dist - pred_dist) / pred_radius, 0.0, 1.0).astype(np.float32)
    max_radius = np.maximum(float(max_mult) * pred_radius, pred_radius + 1e-6)
    envelope = np.where(
        pred_dist <= pred_radius,
        1.0,
        np.clip((max_radius - pred_dist) / np.maximum(max_radius - pred_radius, 1e-6), 0.0, 1.0),
    ).astype(np.float32)
    return envelope, margin


def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
    return np.clip(1.0 - a_norm @ b_norm.T, 0.0, 2.0).astype(np.float32)


def _ip_knownness_from_z(z: np.ndarray, *, low: float, high: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    denom = max(float(high) - float(low), 1e-6)
    return np.clip((float(high) - z) / denom, 0.0, 1.0).astype(np.float32)


def _v51_source_select_alpha_payload(
    score_bank: dict[str, dict],
    *,
    ip_model: IpGateModel,
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    alpha_grid: list[float],
    source_frr: float,
) -> dict:
    if "knn_cosine_k5" not in score_bank:
        return {}
    ip = score_bank["ip_gate_v41"]
    knn = score_bank["knn_cosine_k5"]
    ip_val_z, ip_eval_z = _source_z(ip["val_score"], ip["eval_score"])
    knn_val_z, knn_eval_z = _source_z(knn["val_score"], knn["eval_score"])
    train_embeddings = source_embeddings[train_idx]
    train_labels = np.asarray(source_labels[train_idx], dtype=np.int64)
    val_embeddings = source_embeddings[val_idx]
    val_labels = np.asarray(source_labels[val_idx], dtype=np.int64)
    ip_pseudo = []
    knn_pseudo = []
    for cls in range(int(num_classes)):
        mask = val_labels == cls
        if not np.any(mask):
            continue
        query = val_embeddings[mask]
        ip_pseudo.append(
            _loo_score(
                method="ip_gate_v41",
                query_embeddings=query,
                holdout_class=cls,
                ip_model=ip_model,
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                num_classes=num_classes,
            )
        )
        knn_pseudo.append(
            _loo_score(
                method="knn_cosine_k5",
                query_embeddings=query,
                holdout_class=cls,
                ip_model=ip_model,
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                num_classes=num_classes,
            )
        )
    if not ip_pseudo or not knn_pseudo:
        return {}
    ip_unk = np.concatenate(ip_pseudo, axis=0).astype(np.float32)
    knn_unk = np.concatenate(knn_pseudo, axis=0).astype(np.float32)
    ip_stats = _stats(ip["val_score"])
    knn_stats = _stats(knn["val_score"])
    ip_unk_z = _z_with_stats(ip_unk, ip_stats)
    knn_unk_z = _z_with_stats(knn_unk, knn_stats)
    best = None
    for alpha in sorted(set(float(a) for a in [*alpha_grid, 1.0])):
        known_score = (alpha * ip_val_z + (1.0 - alpha) * knn_val_z).astype(np.float32)
        unknown_score = (alpha * ip_unk_z + (1.0 - alpha) * knn_unk_z).astype(np.float32)
        threshold = calibrate_threshold(known_score, frr=source_frr)
        known_reject = known_score > float(threshold)
        unknown_reject = unknown_score > float(threshold)
        pseudo_metrics = _source_pseudo_metrics(
            known_score=known_score,
            known_pred=np.asarray(knn["val_pred"], dtype=np.int64),
            known_label=val_labels,
            unknown_score=unknown_score,
            threshold=float(threshold),
        )
        pseudo_open_h = _hmean(1.0 - _mean(known_reject), _mean(unknown_reject))
        objective = float(pseudo_open_h + 0.25 * pseudo_metrics["source_pseudo_auosc"])
        candidate = {
            "alpha": float(alpha),
            "objective": objective,
            "source_pseudo_open_h": float(pseudo_open_h),
            "source_pseudo_auosc": float(pseudo_metrics["source_pseudo_auosc"]),
            "known_score": known_score,
            "eval_score": (alpha * ip_eval_z + (1.0 - alpha) * knn_eval_z).astype(np.float32),
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    if best is None:
        return {}
    return {
        "val_score": best["known_score"],
        "eval_score": best["eval_score"],
        "val_pred": knn["val_pred"],
        "eval_pred": knn["eval_pred"],
        "family": "v51_source_selector_alpha_nnid",
        "alpha": float(best["alpha"]),
        "aux_method": "knn_cosine_k5",
        "selected_alpha": float(best["alpha"]),
        "source_selector_objective": float(best["objective"]),
        "source_selector_metric": f"open_h={best['source_pseudo_open_h']:.4f};auosc={best['source_pseudo_auosc']:.4f}",
    }


def _source_selector_rows(
    *,
    common: dict,
    score_bank: dict[str, dict],
    ip_model: IpGateModel,
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    source_frr: float,
) -> list[dict]:
    pseudo_unknown: dict[str, list[np.ndarray]] = {method: [] for method in score_bank}
    train_embeddings = source_embeddings[train_idx]
    train_labels = np.asarray(source_labels[train_idx], dtype=np.int64)
    val_embeddings = source_embeddings[val_idx]
    val_labels = np.asarray(source_labels[val_idx], dtype=np.int64)
    base_methods = [method for method in score_bank if not method.startswith("blend_")]
    for cls in range(int(num_classes)):
        mask = val_labels == cls
        if not np.any(mask):
            continue
        query = val_embeddings[mask]
        for method in base_methods:
            pseudo_unknown[method].append(
                _loo_score(
                    method=method,
                    query_embeddings=query,
                    holdout_class=cls,
                    ip_model=ip_model,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    num_classes=num_classes,
                )
            )
    for method in base_methods:
        if not pseudo_unknown[method]:
            pseudo_unknown[method].append(np.empty(0, dtype=np.float32))
    for method, payload in score_bank.items():
        if not method.startswith("blend_"):
            continue
        alpha = float(payload["alpha"])
        aux_name = str(payload["aux_method"])
        ip_unk = np.concatenate(pseudo_unknown["ip_gate_v41"], axis=0)
        aux_unk = np.concatenate(pseudo_unknown[aux_name], axis=0)
        ip_stats = _stats(score_bank["ip_gate_v41"]["val_score"])
        aux_stats = _stats(score_bank[aux_name]["val_score"])
        pseudo_unknown[method].append(
            (
                alpha * _z_with_stats(ip_unk, ip_stats)
                + (1.0 - alpha) * _z_with_stats(aux_unk, aux_stats)
            ).astype(np.float32)
        )
    rows = []
    for method, payload in score_bank.items():
        known_score = np.asarray(payload["val_score"], dtype=np.float32)
        known_pred = np.asarray(payload["val_pred"], dtype=np.int64)
        unknown_score = np.concatenate(pseudo_unknown[method], axis=0).astype(np.float32)
        threshold = calibrate_threshold(known_score, frr=source_frr)
        known_reject = known_score > float(threshold)
        unknown_reject = unknown_score > float(threshold)
        pseudo_known_accept = 1.0 - _mean(known_reject)
        pseudo_unknown_rej = _mean(unknown_reject)
        y = np.concatenate(
            [np.zeros(known_score.shape[0], dtype=np.int64), np.ones(unknown_score.shape[0], dtype=np.int64)]
        )
        score = np.concatenate([known_score, unknown_score])
        pseudo_metrics = _source_pseudo_metrics(
            known_score=known_score,
            known_pred=known_pred,
            known_label=val_labels,
            unknown_score=unknown_score,
            threshold=threshold,
        )
        rows.append(
            {
                **common,
                "stage": "M2",
                "method": method,
                "source_selector_threshold": float(threshold),
                "source_known_frr": float(_mean(known_reject)),
                "source_pseudo_unknown_rejection": float(pseudo_unknown_rej),
                "source_pseudo_open_h": float(_hmean(pseudo_known_accept, pseudo_unknown_rej)),
                "source_pseudo_open_auc": _safe_auc(y, score),
                "source_pseudo_open_aupr_unknown": _safe_aupr(y, score),
                **pseudo_metrics,
                "source_pseudo_unknown_samples": int(unknown_score.shape[0]),
                "source_known_samples": int(known_score.shape[0]),
                "family": payload.get("family", ""),
                "alpha": payload.get("alpha", ""),
                "aux_method": payload.get("aux_method", ""),
            }
        )
    return rows


def _source_pseudo_metrics(
    *,
    known_score: np.ndarray,
    known_pred: np.ndarray,
    known_label: np.ndarray,
    unknown_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    known_score = np.asarray(known_score, dtype=np.float32)
    unknown_score = np.asarray(unknown_score, dtype=np.float32)
    known_n = known_score.shape[0]
    unknown_n = unknown_score.shape[0]
    score = np.concatenate([known_score, unknown_score]).astype(np.float32)
    pred = np.concatenate([known_pred.astype(np.int64), np.full(unknown_n, -1, dtype=np.int64)])
    labels = np.concatenate([known_label.astype(np.int64), np.full(unknown_n, -1, dtype=np.int64)])
    is_known = np.concatenate([np.ones(known_n, dtype=bool), np.zeros(unknown_n, dtype=bool)])
    is_shifted = np.concatenate([np.ones(known_n, dtype=bool), np.zeros(unknown_n, dtype=bool)])
    rejected = score > float(threshold)
    metrics = compute_osr_extended_metrics(
        rejected=rejected,
        predicted_label=pred,
        true_label=labels,
        is_known=is_known,
        is_shifted_known=is_shifted,
        unknown_score=score,
    )
    return {
        "source_pseudo_auosc": float(metrics["auosc_shifted_known_vs_unknown"]),
        "source_pseudo_ccr5": float(metrics["shifted_known_ccr_at_unknown_fpr_05"]),
        "source_pseudo_ccr10": float(metrics["shifted_known_ccr_at_unknown_fpr_10"]),
        "source_pseudo_openacc": float(metrics["paper_a_open_set_accuracy"]),
        "source_pseudo_known_correct_id": float(metrics["shifted_known_correct_id_rate"]),
    }


def _loo_score(
    *,
    method: str,
    query_embeddings: np.ndarray,
    holdout_class: int,
    ip_model: IpGateModel,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    if method == "ip_gate_v41":
        return _ip_gate_loo_score(ip_model, query_embeddings, holdout_class)
    if method == "prototype_euclidean":
        keep_classes = [cls for cls in range(int(num_classes)) if cls != int(holdout_class)]
        prototypes = np.stack(
            [np.mean(train_embeddings[train_labels == cls], axis=0) for cls in keep_classes],
            axis=0,
        ).astype(np.float32)
        score, _ = prototype_scores(query_embeddings, prototypes, distance="euclidean")
        return score
    if method.startswith("knn_"):
        parts = method.split("_")
        metric = parts[1]
        k = int(parts[2].lstrip("k"))
        keep = train_labels != int(holdout_class)
        return knn_unknown_score(
            train_embeddings[keep],
            train_labels[keep],
            query_embeddings,
            k=k,
            metric=metric,
        ).scores
    raise KeyError(method)


def _ip_gate_loo_score(model: IpGateModel, embeddings: np.ndarray, holdout_class: int) -> np.ndarray:
    support_keep = model.support_labels != int(holdout_class)
    support = model.support_embeddings[support_keep]
    if support.shape[0] == 0:
        return np.full(embeddings.shape[0], np.inf, dtype=np.float32)
    nearest_support = NearestNeighbors(n_neighbors=1, metric=model.distance)
    nearest_support.fit(support)
    known_dist, _ = nearest_support.kneighbors(np.asarray(embeddings, dtype=np.float32), n_neighbors=1)
    known = known_dist[:, 0].astype(np.float32)
    if model.nearest_destructive is None or model.destructive_embeddings.shape[0] == 0:
        return known
    destructive_dist, _ = model.nearest_destructive.kneighbors(np.asarray(embeddings, dtype=np.float32), n_neighbors=1)
    destructive = destructive_dist[:, 0].astype(np.float32)
    return (known / np.maximum(destructive, 1e-6)).astype(np.float32)


def _score_metric_row(
    common: dict,
    *,
    stage: str,
    method: str,
    val_score: np.ndarray,
    eval_score: np.ndarray,
    pred: np.ndarray,
    eval_batch,
    source_frr: float,
    family: str,
    alpha: float | str = "",
    aux_method: str = "",
) -> dict:
    threshold = calibrate_threshold(val_score, frr=source_frr)
    rejected = reject_by_threshold(eval_score, threshold)
    metrics = compute_osr_extended_metrics(
        rejected=rejected,
        predicted_label=pred,
        true_label=eval_batch.known_label,
        is_known=eval_batch.is_known,
        is_shifted_known=eval_batch.is_shifted_known,
        unknown_score=eval_score,
    )
    return {
        **common,
        "stage": stage,
        "method": method,
        "family": family,
        "alpha": alpha,
        "aux_method": aux_method,
        "source_frr": float(source_frr),
        "threshold": float(threshold),
        "deployable": True,
        "uses_target_labels": False,
        "auc_shifted_known_vs_unknown": _safe_auc(~eval_batch.is_known, eval_score),
        **metrics,
    }


def _gate_summary(summary: list[dict]) -> list[dict]:
    by_dataset_method = {(row["dataset"], row["method"]): row for row in summary}
    methods = sorted({row["method"] for row in summary if str(row["method"]).startswith("blend_")})
    datasets = sorted({row["dataset"] for row in summary})
    out = []
    for method in methods:
        deltas = []
        row = {"method": method}
        for dataset in datasets:
            base = by_dataset_method.get((dataset, "ip_gate_v41"))
            cand = by_dataset_method.get((dataset, method))
            if not base or not cand:
                continue
            ds = _dataset_short(dataset)
            delta = _delta_dict(cand, base)
            deltas.append((dataset, delta))
            for key, value in delta.items():
                row[f"{ds}_{key}"] = value
        if deltas:
            mean_delta = {
                key: float(np.mean([item[1][key] for item in deltas]))
                for key in [
                    "d_auc",
                    "d_aupr_u",
                    "d_auosc",
                    "d_ccr5",
                    "d_ccr10",
                    "d_h",
                    "d_frr",
                ]
            }
            row.update({f"mean_{key}": value for key, value in mean_delta.items()})
            row["gate_auc_aupr"] = bool(
                all(item[1]["d_auc"] >= -0.005 and item[1]["d_aupr_u"] >= -0.005 for item in deltas)
            )
            row["gate_auosc"] = bool(mean_delta["d_auosc"] >= 0.010 and all(item[1]["d_auosc"] >= 0.0 for item in deltas))
            row["gate_ccr"] = bool(all(item[1]["d_ccr5"] >= -0.005 and item[1]["d_ccr10"] >= -0.005 for item in deltas))
            row["gate_manytx_h"] = bool(
                all(item[1]["d_h"] >= -0.020 for item in deltas if _dataset_short(item[0]) == "ManyTx")
            )
            row["gate_frr"] = bool(mean_delta["d_frr"] <= 0.0)
            row["gate_pass_strong"] = bool(
                row["gate_auc_aupr"]
                and row["gate_auosc"]
                and row["gate_ccr"]
                and row["gate_manytx_h"]
                and row["gate_frr"]
            )
        out.append(row)
    return out


def _threshold_gate_summary(summary: list[dict]) -> list[dict]:
    by_dataset_method = {(row["dataset"], row["method"]): row for row in summary}
    methods = sorted({row["method"] for row in summary if str(row["method"]).startswith("ip_gate_v41_source_frr_")})
    datasets = sorted({row["dataset"] for row in summary})
    out = []
    for method in methods:
        deltas = []
        row = {"method": method}
        for dataset in datasets:
            base = by_dataset_method.get((dataset, "ip_gate_v41"))
            cand = by_dataset_method.get((dataset, method))
            if not base or not cand:
                continue
            ds = _dataset_short(dataset)
            delta = _delta_dict(cand, base)
            deltas.append((dataset, delta))
            for key, value in delta.items():
                row[f"{ds}_{key}"] = value
        if deltas:
            row["mean_d_h"] = float(np.mean([item[1]["d_h"] for item in deltas]))
            row["mean_d_frr"] = float(np.mean([item[1]["d_frr"] for item in deltas]))
            row["mean_d_u_rej"] = float(np.mean([item[1]["d_u_rej"] for item in deltas]))
            row["gate_frr_recovery"] = bool(row["mean_d_frr"] <= -0.030 and row["mean_d_h"] >= -0.020)
        out.append(row)
    return out


def _selector_summary(selector_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in selector_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    out = []
    for method, rows in sorted(grouped.items()):
        item = {
            "method": method,
            "runs": len(rows),
            "family": rows[0].get("family", ""),
            "alpha": rows[0].get("alpha", ""),
            "aux_method": rows[0].get("aux_method", ""),
        }
        for key in [
            "source_known_frr",
            "source_pseudo_unknown_rejection",
            "source_pseudo_open_h",
            "source_pseudo_open_auc",
            "source_pseudo_open_aupr_unknown",
            "source_pseudo_auosc",
            "source_pseudo_ccr5",
            "source_pseudo_ccr10",
            "source_pseudo_openacc",
            "source_pseudo_known_correct_id",
        ]:
            item[f"{key}_mean"] = float(np.mean([float(row[key]) for row in rows]))
            item[f"{key}_std"] = float(np.std([float(row[key]) for row in rows]))
        out.append(item)
    return out


def _refined_selector_summary(selector_rows: list[dict]) -> list[dict]:
    rows = _selector_summary(selector_rows)
    for row in rows:
        row["source_auosc_h_product"] = float(row["source_pseudo_auosc_mean"]) * float(
            row["source_pseudo_open_h_mean"]
        )
    product_sorted = sorted(rows, key=lambda row: float(row["source_auosc_h_product"]), reverse=True)
    for rank, row in enumerate(product_sorted, start=1):
        row["source_auosc_h_rank"] = rank
    profile_metrics = [
        ("source_pseudo_open_auc_mean", True),
        ("source_pseudo_open_aupr_unknown_mean", True),
        ("source_pseudo_auosc_mean", True),
        ("source_pseudo_ccr5_mean", True),
        ("source_pseudo_ccr10_mean", True),
        ("source_pseudo_open_h_mean", True),
        ("source_known_frr_mean", False),
    ]
    rank_maps = {
        key: _rank_map(rows, key=key, descending=descending)
        for key, descending in profile_metrics
    }
    for row in rows:
        ranks = [rank_maps[key][row["method"]] for key, _descending in profile_metrics]
        row["source_profile_mean_rank"] = float(np.mean(ranks))
    profile_sorted = sorted(rows, key=lambda row: float(row["source_profile_mean_rank"]))
    for rank, row in enumerate(profile_sorted, start=1):
        row["source_profile_rank"] = rank
    return sorted(rows, key=lambda row: (int(row["source_auosc_h_rank"]), str(row["method"])))


def _rank_map(rows: list[dict], *, key: str, descending: bool) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=descending)
    return {str(row["method"]): rank for rank, row in enumerate(ordered, start=1)}


def _delta_dict(cand: dict, base: dict) -> dict[str, float]:
    return {
        "d_auc": float(cand["auc_shifted_known_vs_unknown_mean"]) - float(base["auc_shifted_known_vs_unknown_mean"]),
        "d_aupr_u": float(cand["aupr_unknown_mean"]) - float(base["aupr_unknown_mean"]),
        "d_auosc": float(cand["auosc_shifted_known_vs_unknown_mean"]) - float(base["auosc_shifted_known_vs_unknown_mean"]),
        "d_ccr5": float(cand["shifted_known_ccr_at_unknown_fpr_05_mean"])
        - float(base["shifted_known_ccr_at_unknown_fpr_05_mean"]),
        "d_ccr10": float(cand["shifted_known_ccr_at_unknown_fpr_10_mean"])
        - float(base["shifted_known_ccr_at_unknown_fpr_10_mean"]),
        "d_h": float(cand["sample_open_set_h_score_mean"]) - float(base["sample_open_set_h_score_mean"]),
        "d_frr": float(cand["shifted_known_false_rejection_rate_mean"])
        - float(base["shifted_known_false_rejection_rate_mean"]),
        "d_u_rej": float(cand["true_unknown_rejection_rate_mean"]) - float(base["true_unknown_rejection_rate_mean"]),
    }


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)
    metric_keys = [
        "source_val_accuracy",
        "auc_shifted_known_vs_unknown",
        "shifted_known_false_rejection_rate",
        "shifted_known_correct_id_rate",
        "true_unknown_rejection_rate",
        "sample_open_set_h_score",
        "paper_a_open_set_accuracy",
        "auosc_shifted_known_vs_unknown",
        "aupr_unknown",
        "shifted_known_ccr_at_unknown_fpr_05",
        "shifted_known_ccr_at_unknown_fpr_10",
        "safe_regimes",
        "destructive_regimes",
        "destructive_bank_size",
        "support_samples",
        *EXTENDED_OSR_METRIC_KEYS,
    ]
    summary = []
    for (dataset, method), items in sorted(grouped.items()):
        row = {
            "dataset": dataset,
            "method": method,
            "stage": items[0].get("stage", ""),
            "family": items[0].get("family", ""),
            "alpha": items[0].get("alpha", ""),
            "aux_method": items[0].get("aux_method", ""),
            "runs": len(items),
            "deployable": bool(all(bool(item.get("deployable", True)) for item in items)),
            "uses_target_labels": bool(any(bool(item.get("uses_target_labels", False)) for item in items)),
        }
        for key in metric_keys:
            values = [float(item.get(key, 0.0)) for item in items]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        summary.append(row)
    return summary


def _summary_markdown(
    summary: list[dict],
    gate_summary: list[dict],
    threshold_gate: list[dict],
    selector_summary: list[dict],
    refined_selector_summary: list[dict],
    title: str,
    scope_note: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        scope_note,
        "",
        "## Summary Metrics",
        "",
        "| dataset | stage | method | runs | AUC | AUPR-U | AUOSC | CCR5 | CCR10 | H | FRR | U-Rej | OpenAcc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {stage} | {method} | {runs} | {auc:.4f} | {aupr:.4f} | {auosc:.4f} | {ccr5:.4f} | {ccr10:.4f} | {h:.4f} | {frr:.4f} | {urej:.4f} | {openacc:.4f} |".format(
                dataset=row["dataset"],
                stage=row["stage"],
                method=row["method"],
                runs=int(row["runs"]),
                auc=float(row["auc_shifted_known_vs_unknown_mean"]),
                aupr=float(row["aupr_unknown_mean"]),
                auosc=float(row["auosc_shifted_known_vs_unknown_mean"]),
                ccr5=float(row["shifted_known_ccr_at_unknown_fpr_05_mean"]),
                ccr10=float(row["shifted_known_ccr_at_unknown_fpr_10_mean"]),
                h=float(row["sample_open_set_h_score_mean"]),
                frr=float(row["shifted_known_false_rejection_rate_mean"]),
                urej=float(row["true_unknown_rejection_rate_mean"]),
                openacc=float(row["paper_a_open_set_accuracy_mean"]),
            )
        )
    lines.extend(["", "## M2 Blend Gate", ""])
    if gate_summary:
        lines.append("| method | mean dAUC | mean dAUPR-U | mean dAUOSC | mean dCCR5 | mean dCCR10 | mean dH | mean dFRR | pass |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in gate_summary:
            lines.append(
                "| {method} | {dauc:.4f} | {daupr:.4f} | {dauosc:.4f} | {dccr5:.4f} | {dccr10:.4f} | {dh:.4f} | {dfrr:.4f} | {passed} |".format(
                    method=row["method"],
                    dauc=float(row.get("mean_d_auc", 0.0)),
                    daupr=float(row.get("mean_d_aupr_u", 0.0)),
                    dauosc=float(row.get("mean_d_auosc", 0.0)),
                    dccr5=float(row.get("mean_d_ccr5", 0.0)),
                    dccr10=float(row.get("mean_d_ccr10", 0.0)),
                    dh=float(row.get("mean_d_h", 0.0)),
                    dfrr=float(row.get("mean_d_frr", 0.0)),
                    passed=str(row.get("gate_pass_strong", False)),
                )
            )
    lines.extend(["", "## M1 Threshold Gate", ""])
    if threshold_gate:
        lines.append("| method | mean dH | mean dFRR | mean dU-Rej | FRR recovery gate |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in threshold_gate:
            lines.append(
                "| {method} | {dh:.4f} | {dfrr:.4f} | {durej:.4f} | {passed} |".format(
                    method=row["method"],
                    dh=float(row.get("mean_d_h", 0.0)),
                    dfrr=float(row.get("mean_d_frr", 0.0)),
                    durej=float(row.get("mean_d_u_rej", 0.0)),
                    passed=str(row.get("gate_frr_recovery", False)),
                )
            )
    m3_rows = [row for row in summary if row.get("stage") == "M3-ID"]
    lines.extend(["", "## M3-ID Head Swap", ""])
    if m3_rows:
        lines.append(
            "| dataset | method | runs | AUOSC | H | OpenAcc | C_native | ID_given_accept | CCR5 | CCR10 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(
            m3_rows,
            key=lambda r: (r["dataset"], -float(r.get("sample_open_set_h_score_mean", 0.0)), r["method"]),
        ):
            lines.append(
                "| {dataset} | {method} | {runs} | {auosc:.4f} | {h:.4f} | {openacc:.4f} | {cnat:.4f} | {idacc:.4f} | {ccr5:.4f} | {ccr10:.4f} |".format(
                    dataset=row["dataset"],
                    method=row["method"],
                    runs=int(row["runs"]),
                    auosc=float(row["auosc_shifted_known_vs_unknown_mean"]),
                    h=float(row["sample_open_set_h_score_mean"]),
                    openacc=float(row["paper_a_open_set_accuracy_mean"]),
                    cnat=float(row["shifted_known_correct_id_rate_mean"]),
                    idacc=float(row["accepted_shifted_known_id_accuracy_mean"]),
                    ccr5=float(row["shifted_known_ccr_at_unknown_fpr_05_mean"]),
                    ccr10=float(row["shifted_known_ccr_at_unknown_fpr_10_mean"]),
                )
            )
    lines.extend(["", "## Source-Only Selector Summary", ""])
    if selector_summary:
        lines.append("| method | pseudo AUC | pseudo AUPR-U | pseudo AUOSC | pseudo CCR5 | pseudo CCR10 | pseudo H | source FRR | pseudo unknown rejection |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in selector_summary:
            lines.append(
                "| {method} | {auc:.4f} | {aupr:.4f} | {auosc:.4f} | {ccr5:.4f} | {ccr10:.4f} | {h:.4f} | {frr:.4f} | {urej:.4f} |".format(
                    method=row["method"],
                    auc=float(row["source_pseudo_open_auc_mean"]),
                    aupr=float(row["source_pseudo_open_aupr_unknown_mean"]),
                    auosc=float(row["source_pseudo_auosc_mean"]),
                    ccr5=float(row["source_pseudo_ccr5_mean"]),
                    ccr10=float(row["source_pseudo_ccr10_mean"]),
                    h=float(row["source_pseudo_open_h_mean"]),
                    frr=float(row["source_known_frr_mean"]),
                    urej=float(row["source_pseudo_unknown_rejection_mean"]),
                )
            )
    lines.extend(["", "## Refined Source-Only Selector", ""])
    if refined_selector_summary:
        lines.append("| AUOSC-H rank | profile rank | method | AUOSC-H product | pseudo AUOSC | pseudo H | pseudo CCR5 | pseudo CCR10 |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
        for row in refined_selector_summary:
            lines.append(
                "| {rank} | {profile_rank} | {method} | {product:.4f} | {auosc:.4f} | {h:.4f} | {ccr5:.4f} | {ccr10:.4f} |".format(
                    rank=int(row["source_auosc_h_rank"]),
                    profile_rank=int(row["source_profile_rank"]),
                    method=row["method"],
                    product=float(row["source_auosc_h_product"]),
                    auosc=float(row["source_pseudo_auosc_mean"]),
                    h=float(row["source_pseudo_open_h_mean"]),
                    ccr5=float(row["source_pseudo_ccr5_mean"]),
                    ccr10=float(row["source_pseudo_ccr10_mean"]),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_z(val_score: np.ndarray, eval_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stats = _stats(val_score)
    return _z_with_stats(val_score, stats), _z_with_stats(eval_score, stats)


def _stats(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)), max(float(np.std(arr)), 1e-6)


def _z_with_stats(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    mean, std = stats
    return ((np.asarray(values, dtype=np.float32) - float(mean)) / float(std)).astype(np.float32)


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if np.unique(y).size < 2:
        return 0.0
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float32).reshape(-1)))


def _safe_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if np.unique(y).size < 2:
        return 0.0
    return float(average_precision_score(y, np.asarray(score, dtype=np.float32).reshape(-1)))


def _mean(values: np.ndarray) -> float:
    arr = np.asarray(values)
    return float(np.mean(arr)) if arr.size else 0.0


def _hmean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float(2.0 * a * b / (a + b))


def _dataset_short(dataset: str) -> str:
    if "ManyTx" in dataset:
        return "ManyTx"
    if "ManySig" in dataset:
        return "ManySig"
    return dataset.replace("-", "_")


def _frr_tag(frr: float) -> str:
    return f"{float(frr):.3f}".replace(".", "p")


def _alpha_tag(alpha: float) -> str:
    return f"{float(alpha):.2f}".replace(".", "p")


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _parse_float_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
        elif "-" in item[1:]:
            idx = item[1:].index("-") + 1
            left, right = item[:idx], item[idx + 1 :]
        else:
            raise ValueError(f"Expected float pair like 0.50:0.95, got {item!r}")
        pairs.append((float(left.strip()), float(right.strip())))
    if not pairs:
        raise ValueError("Expected at least one float pair")
    return pairs


def _parse_knn_specs(text: str) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected kNN spec like cosine:5, got {item!r}")
        metric, k_text = item.split(":", 1)
        specs.append((metric.strip(), int(k_text.strip())))
    if not specs:
        raise ValueError("Expected at least one kNN sensitivity spec")
    return specs


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_str_list(text: str) -> list[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one method name")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
