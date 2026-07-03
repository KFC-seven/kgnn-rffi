"""Visualize DPR (Dual Perturbation Reference) mechanism via t-SNE.

Generates for each protocol:
  (a) d_K vs d_D scatter, colored by shifted_known / unknown
  (b) d_K/d_D ratio histogram
  (c) t-SNE: source + R_K + R_D, colored by device class
  (d) t-SNE: R_K + R_D + shifted known, colored by device class
  (e) t-SNE: R_K + R_D + unknown (single color)

Reproduce: python scripts/visualize_dpr.py
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from diagnostic.compact import load_compact_dataset
from diagnostic.config import load_config
from diagnostic.datasets import materialize_records
from diagnostic.sourceonly import infer_logits_embeddings, train_sourceonly
from diagnostic.splits import build_manifest, build_split_records
from kgnn import (
    build_kgnn_model, classify_perturbation_safety, default_perturbation_specs,
    select_specs, PerturbationConfig, PerturbationEngine,
)
from kgnn.utils import _configure_torch_determinism, _resolve_device, _select_protocol, _select_split


def cos_dist_matrix(query, bank):
    q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    b_n = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    return 1.0 - q_n @ b_n.T


# Distinct colormap for up to 10 device classes
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

UNKNOWN_COLOR = "#555555"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/visualize_dpr")
    parser.add_argument("--source-frr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tsne", type=int, default=3000)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    _configure_torch_determinism()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("ManySig_RX93_TX42",  "configs/manysig_soda4.yaml",   "RX9-3_TX4-2",    1,
         "tiny",     5, 100, 64),
    ]

    for label, cfg, proto, split_id, model_name, epochs, max_samp, emb_dim in run_specs:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")

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
        num_classes = len(split["known_txs"])

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
        print(f"  safe_regimes={len(safe_specs)}  destructive_regimes={len(destructive_specs)}")

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

        _sl, source_emb = infer_logits_embeddings(train_result.model, source.x,
                                                   batch_size=args.batch_size, device=device)
        _el, eval_emb = infer_logits_embeddings(train_result.model, eval_batch.x,
                                                 batch_size=args.batch_size, device=device)

        rk_emb = full_model.support_embeddings
        rk_labels = full_model.support_labels
        rd_emb = full_model.destructive_embeddings
        src_labels = source.known_label

        print(f"  |source|={source_emb.shape[0]}  |R_K|={rk_emb.shape[0]}  |R_D|={rd_emb.shape[0]}  |eval|={eval_emb.shape[0]}")

        # Compute d_K and d_D for eval embeddings
        dk = cos_dist_matrix(eval_emb, rk_emb).min(axis=1)
        dd = cos_dist_matrix(eval_emb, rd_emb).min(axis=1) if rd_emb.shape[0] > 0 else np.full(len(eval_emb), np.nan)
        ratio = dk / np.maximum(dd, 1e-8)

        sk_mask = np.asarray(eval_batch.is_known, dtype=bool) & np.asarray(eval_batch.is_shifted_known, dtype=bool)
        unk_mask = ~np.asarray(eval_batch.is_known, dtype=bool)
        eval_true_labels = eval_batch.known_label  # -1 for unknown, 0..K-1 for known

        print(f"  N shifted_known={sk_mask.sum()}  N unknown={unk_mask.sum()}")

        # ---- (a) d_K vs d_D scatter ----
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(dk[sk_mask], dd[sk_mask], s=3, alpha=0.5, c="#2196F3",
                   label="Shifted known", rasterized=True)
        ax.scatter(dk[unk_mask], dd[unk_mask], s=3, alpha=0.5, c="#F44336",
                   label="Unknown", rasterized=True)
        lims = [0, max(dk.max(), dd.max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("d_K (distance to R_K)")
        ax.set_ylabel("d_D (distance to R_D)")
        ax.set_title(f"{label}: d_K vs d_D")
        ax.legend(loc="upper left", markerscale=3, framealpha=0.9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_scatter_dk_vs_dd.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{label}_scatter_dk_vs_dd.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- (b) ratio histogram ----
        fig, ax = plt.subplots(figsize=(6, 4))
        valid_ratio = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]
        bins = np.linspace(0, np.percentile(valid_ratio, 98), 50)
        ax.hist(ratio[sk_mask & ~np.isnan(ratio) & ~np.isinf(ratio)], bins=bins,
                alpha=0.6, color="#2196F3", label="Shifted known", density=True)
        ax.hist(ratio[unk_mask & ~np.isnan(ratio) & ~np.isinf(ratio)], bins=bins,
                alpha=0.6, color="#F44336", label="Unknown", density=True)
        ax.set_xlabel("d_K / d_D ratio")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}: DPR ratio distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_hist_ratio.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{label}_hist_ratio.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- t-SNE with class colors ----
        rng = np.random.default_rng(args.seed)
        n_max = args.max_tsne

        # Sample source (keep all or cap)
        n_src = min(n_max, len(source_emb))
        idx_src = rng.choice(len(source_emb), size=n_src, replace=False)
        src_lbl_sample = src_labels[idx_src]

        # Sample R_K
        n_rk = min(n_max, len(rk_emb))
        idx_rk = rng.choice(len(rk_emb), size=n_rk, replace=False)
        rk_lbl_sample = rk_labels[idx_rk]

        # Sample R_D (no labels)
        n_rd = min(n_max, rd_emb.shape[0]) if rd_emb.shape[0] > 0 else 0
        idx_rd = rng.choice(len(rd_emb), size=n_rd, replace=False) if n_rd > 0 else np.array([], dtype=int)

        # Sample shifted known
        sk_indices = np.where(sk_mask)[0]
        n_sk = min(n_max // 2, len(sk_indices))
        idx_sk = rng.choice(sk_indices, size=n_sk, replace=False)
        sk_lbl_sample = eval_true_labels[idx_sk]

        # Sample unknown
        uk_indices = np.where(unk_mask)[0]
        n_uk = min(n_max // 2, len(uk_indices))
        idx_uk = rng.choice(uk_indices, size=n_uk, replace=False)

        all_parts = [source_emb[idx_src], rk_emb[idx_rk]]
        all_categories = [f"src_c{int(l)}" for l in src_lbl_sample] + [f"rk_c{int(l)}" for l in rk_lbl_sample]
        offsets = [0]
        offsets.append(offsets[-1] + n_src)
        offsets.append(offsets[-1] + n_rk)
        if n_rd > 0:
            all_parts.append(rd_emb[idx_rd])
            all_categories += ["R_D"] * n_rd
            offsets.append(offsets[-1] + n_rd)
        all_parts.append(eval_emb[idx_sk])
        all_categories += [f"sk_c{int(l)}" for l in sk_lbl_sample]
        offsets.append(offsets[-1] + n_sk)
        all_parts.append(eval_emb[idx_uk])
        all_categories += ["Unknown"] * n_uk
        offsets.append(offsets[-1] + n_uk)

        all_emb = np.vstack(all_parts)
        print(f"  t-SNE: {all_emb.shape[0]} samples (src={n_src}, R_K={n_rk}, R_D={n_rd}, SK={n_sk}, UK={n_uk})")

        tsne = TSNE(n_components=2, perplexity=min(30, all_emb.shape[0] - 1),
                    random_state=args.seed, max_iter=1000, verbose=1)
        tsne_xy = tsne.fit_transform(all_emb)

        # Extract coordinates by group
        ofs = 0
        xy_src = tsne_xy[ofs:ofs+n_src]; ofs += n_src
        xy_rk = tsne_xy[ofs:ofs+n_rk]; ofs += n_rk
        xy_rd = tsne_xy[ofs:ofs+n_rd] if n_rd > 0 else np.zeros((0, 2)); ofs += n_rd
        xy_sk = tsne_xy[ofs:ofs+n_sk]; ofs += n_sk
        xy_uk = tsne_xy[ofs:ofs+n_uk]

        # Helper: assign color by class label
        def class_color(lbl, alpha=0.7):
            if lbl < 0:
                return UNKNOWN_COLOR
            return CLASS_COLORS[int(lbl) % len(CLASS_COLORS)]

        # ---- (c) t-SNE overview: source + R_K + R_D, colored by class ----
        fig, ax = plt.subplots(figsize=(7, 5.5))
        # Source: colored by class
        for cls in np.unique(src_lbl_sample):
            mask = src_lbl_sample == cls
            ax.scatter(xy_src[mask, 0], xy_src[mask, 1], s=3, alpha=0.6,
                       c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                       label=f"Source TX{int(cls)}", rasterized=True)
        # R_K: same class colors, lighter marker
        for cls in np.unique(rk_lbl_sample):
            mask = rk_lbl_sample == cls
            ax.scatter(xy_rk[mask, 0], xy_rk[mask, 1], s=1, alpha=0.25,
                       c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                       rasterized=True)
        # R_D: dark grey, no class distinction
        if n_rd > 0:
            ax.scatter(xy_rd[:, 0], xy_rd[:, 1], s=1, alpha=0.25,
                       c="#795548", label="R_D (identity-disrupting)", rasterized=True)
        ax.set_title(f"{label}: Source, R_K (light), and R_D")
        ax.legend(loc="upper right", markerscale=3, framealpha=0.9, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_tsne_references.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{label}_tsne_references.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- (d) t-SNE: R_K + R_D + shifted known, class-colored ----
        fig, ax = plt.subplots(figsize=(7, 5.5))
        # R_K as light background
        for cls in np.unique(rk_lbl_sample):
            mask = rk_lbl_sample == cls
            ax.scatter(xy_rk[mask, 0], xy_rk[mask, 1], s=1, alpha=0.15,
                       c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)], rasterized=True)
        # R_D as light background
        if n_rd > 0:
            ax.scatter(xy_rd[:, 0], xy_rd[:, 1], s=1, alpha=0.12,
                       c="#795548", rasterized=True)
        # Shifted known: colored by true class
        for cls in np.unique(sk_lbl_sample):
            mask = sk_lbl_sample == cls
            ax.scatter(xy_sk[mask, 0], xy_sk[mask, 1], s=10, alpha=0.75,
                       c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                       edgecolors="k", linewidth=0.3,
                       label=f"Shifted known TX{int(cls)}", rasterized=True)
        ax.set_title(f"{label}: Shifted known vs references")
        ax.legend(loc="upper right", markerscale=3, framealpha=0.9, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_tsne_shifted_known.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{label}_tsne_shifted_known.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- (e) t-SNE: R_K + R_D + unknown (grey) ----
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for cls in np.unique(rk_lbl_sample):
            mask = rk_lbl_sample == cls
            ax.scatter(xy_rk[mask, 0], xy_rk[mask, 1], s=1, alpha=0.15,
                       c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)], rasterized=True)
        if n_rd > 0:
            ax.scatter(xy_rd[:, 0], xy_rd[:, 1], s=1, alpha=0.12,
                       c="#795548", rasterized=True)
        ax.scatter(xy_uk[:, 0], xy_uk[:, 1], s=8, alpha=0.6,
                   c="#555555", label="Unknown", edgecolors="#333333",
                   linewidth=0.2, rasterized=True)
        # Highlight shifted known in light blue for contrast
        for cls in np.unique(sk_lbl_sample):
            mask = sk_lbl_sample == cls
            ax.scatter(xy_sk[mask, 0], xy_sk[mask, 1], s=8, alpha=0.5,
                       c=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                       edgecolors="k", linewidth=0.3, rasterized=True)
        ax.set_title(f"{label}: Unknown (dark grey) vs references and shifted known (colored)")
        ax.legend(loc="upper right", markerscale=3, framealpha=0.9, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_tsne_unknown.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{label}_tsne_unknown.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
