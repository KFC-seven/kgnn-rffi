"""Compare Pure Ratio vs Full DPRNN-RFFI vs NoAug on representative protocols."""
from __future__ import annotations

import sys, numpy as np
from pathlib import Path

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


def cos_dist_matrix(query, bank):
    q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    b_n = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    return 1.0 - q_n @ b_n.T


def main():
    device = _resolve_device("auto")
    _configure_torch_determinism()

    runs = [
        ("ManySig_TX42", "configs/manysig_soda4.yaml", "RX9-3_TX4-2", 1,
         "tiny", 5, 100, 64),
        ("ManyTx_TX2020", "configs/manytx_owen_v0.yaml", "MTX_RX6-6_TX20-20", 1,
         "resnet1d", 5, 30, 128),
    ]

    for label, cfg, proto, split_id, model_name, epochs, max_samp, emb_dim in runs:
        config = load_config(cfg)
        manifest = build_manifest(config)
        dataset = load_compact_dataset(config["dataset"]["path"])
        protocol = _select_protocol(manifest, proto)
        sp = _select_split(protocol, split_id)
        records = build_split_records(
            known_txs=sp["known_txs"], unknown_txs=sp["unknown_txs"],
            source_rxs=protocol["source_rxs"], drift_rxs=protocol["drift_rxs"],
            source_date=manifest["dates"]["source"], day_shift_date=manifest["dates"]["day_shift"],
        )
        source_recs = [r for r in records if r["split_name"] == "source_train"]
        eval_recs = [r for r in records if r["split_name"] != "source_train"]
        source = materialize_records(dataset, records=source_recs, signal_equalized=1,
                                      max_samples_per_record=max_samp)
        eval_batch = materialize_records(dataset, records=eval_recs, signal_equalized=1,
                                          max_samples_per_record=max_samp)

        tr = train_sourceonly(
            x=source.x, y=source.known_label, num_classes=len(sp["known_txs"]),
            epochs=epochs, batch_size=256, seed=42, embedding_dim=emb_dim,
            model_name=model_name, device=device,
        )
        train_idx = np.asarray(tr.train_indices, dtype=np.int64)
        val_idx = np.asarray(tr.val_indices, dtype=np.int64)
        num_classes = len(sp["known_txs"])

        perturb = PerturbationEngine(config=PerturbationConfig(), seed=42 + split_id * 7919)
        safety = classify_perturbation_safety(
            encoder=tr.model, source_x=source.x[val_idx],
            source_labels=source.known_label[val_idx], perturbation_engine=perturb,
            specs=default_perturbation_specs(), safe_accuracy=0.90, destructive_accuracy=0.50,
            threshold_mode="absolute", max_samples_per_class=25, batch_size=256,
            device=device, seed=42,
        )
        safe_specs = select_specs(safety, "safe")
        destructive_specs = select_specs(safety, "destructive")

        full_model, _ = build_kgnn_model(
            encoder=tr.model, source_x=source.x, source_labels=source.known_label,
            train_indices=train_idx, val_indices=val_idx, num_classes=num_classes,
            safe_specs=safe_specs, destructive_specs=destructive_specs,
            perturbation_engine=perturb, augment_count=4, sigma_mult=0.3,
            max_support_per_class=1000, max_destructive_bank=5000, destructive_per_sample=1,
            batch_size=256, device=device, distance="cosine", score_mode="ratio",
            support_k=1, destructive_k=1, class_norm_alpha=1.0, destructive_balance="none",
            score_calibration="none", frr=0.03, gate_support=True, seed=42,
        )

        _sl, src_emb = infer_logits_embeddings(tr.model, source.x, batch_size=256, device=device)
        _el, eval_emb = infer_logits_embeddings(tr.model, eval_batch.x, batch_size=256, device=device)

        rk = full_model.support_embeddings
        rd = full_model.destructive_embeddings

        is_known = np.asarray(eval_batch.is_known, dtype=bool)
        is_shifted = np.asarray(eval_batch.is_shifted_known, dtype=bool)

        # Source validation ratio for threshold
        src_val_emb = src_emb[val_idx]
        dk_val = cos_dist_matrix(src_val_emb, rk).min(axis=1)
        dd_val = cos_dist_matrix(src_val_emb, rd).min(axis=1)
        ratio_val = dk_val / np.maximum(dd_val, 1e-8)

        # Eval ratio
        dk_eval = cos_dist_matrix(eval_emb, rk).min(axis=1)
        dd_eval = cos_dist_matrix(eval_emb, rd).min(axis=1)
        ratio_eval = dk_eval / np.maximum(dd_eval, 1e-8)

        # Pure ratio: threshold = source val 97th percentile
        thresh_ratio = np.percentile(ratio_val, 97)
        rejected_ratio = ratio_eval > thresh_ratio

        # kNN identity
        src_train_emb = src_emb[train_idx]
        src_train_lbl = source.known_label[train_idx]
        knn = knn_unknown_score(src_train_emb, src_train_lbl, eval_emb, k=5, metric="cosine")
        pred = knn.predicted_label

        # NoAug model
        noaug_model, _ = build_kgnn_model(
            encoder=tr.model, source_x=source.x, source_labels=source.known_label,
            train_indices=train_idx, val_indices=val_idx, num_classes=num_classes,
            safe_specs=[], destructive_specs=[],
            perturbation_engine=perturb, augment_count=0, sigma_mult=0.3,
            max_support_per_class=1000, max_destructive_bank=0, destructive_per_sample=0,
            batch_size=256, device=device, distance="cosine", score_mode="known",
            support_k=1, destructive_k=1, class_norm_alpha=1.0, destructive_balance="none",
            score_calibration="none", frr=0.03, gate_support=True, seed=42,
        )

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  Pure ratio threshold (97pct src val): {thresh_ratio:.4f}")
        print(f"  {'Variant':<22s} {'H-score':>8s}  {'ACC':>8s}  {'Unk.Rej':>8s}  {'FRR':>8s}")
        print(f"  {'-'*54}")

        for vname, rejected, score in [
            ("Pure ratio", rejected_ratio, ratio_eval),
        ]:
            m = compute_osr_extended_metrics(
                rejected=rejected, predicted_label=pred,
                true_label=eval_batch.known_label, is_known=is_known,
                is_shifted_known=is_shifted, unknown_score=score,
            )
            h = float(m.get("sample_open_set_h_score", 0))
            acc = float(m.get("paper_a_open_set_accuracy", 0))
            unk = float(m.get("true_unknown_rejection_rate", 0))
            frr = float(m.get("shifted_known_false_rejection_rate", 0))
            print(f"  {vname:<22s} {h:8.4f}  {acc:8.4f}  {unk:8.4f}  {frr:8.4f}")

        for vname, model in [("Full DPRNN-RFFI", full_model), ("NoAug (d_K only)", noaug_model)]:
            _, rejected, score = predict_kgnn(model, eval_emb)
            m = compute_osr_extended_metrics(
                rejected=rejected, predicted_label=pred,
                true_label=eval_batch.known_label, is_known=is_known,
                is_shifted_known=is_shifted, unknown_score=score,
            )
            h = float(m.get("sample_open_set_h_score", 0))
            acc = float(m.get("paper_a_open_set_accuracy", 0))
            unk = float(m.get("true_unknown_rejection_rate", 0))
            frr = float(m.get("shifted_known_false_rejection_rate", 0))
            print(f"  {vname:<22s} {h:8.4f}  {acc:8.4f}  {unk:8.4f}  {frr:8.4f}")


if __name__ == "__main__":
    main()
