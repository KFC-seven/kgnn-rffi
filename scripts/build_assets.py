#!/usr/bin/env python3
"""Build paper-facing experimental-evaluation tables and protocol plots.

The manuscript must only expose short paper-facing method names. Internal
variant keys are retained in provenance CSV/JSON files for auditability.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Paths to experiment result CSVs — update these to point to your run outputs.
V49_PER_RUN = ROOT / "results" / "v49_baselines" / "combined_all_per_run.csv"
V51_M2_PER_RUN = ROOT / "results" / "v51_main" / "per_run.csv"
V51_MISSING_PER_RUN = ROOT / "results" / "v51_ablation" / "per_run.csv"
EFFICIENCY_CSV = ROOT / "results" / "efficiency" / "efficiency_profile_summary.csv"
COMPONENT_SUMMARY_CSV = ROOT / "results" / "v51_component_summary_by_dataset.csv"
SIMPLIFIED_MAIN_CSV = ROOT / "results" / "v51_simplified_main_comparison_by_dataset.csv"
ENVELOPE_SENSITIVITY_CSV = ROOT / "results" / "v51_envelope_only_sensitivity_summary_by_axis_dataset.csv"
K_SENSITIVITY_CSV = ROOT / "results" / "v47_full_ablation_ksens_kgrid_by_dataset.csv"

OUT_DIR = ROOT / "results" / "assets"
DATA_DIR = OUT_DIR / "data"
TABLE_DIR = OUT_DIR / "tables"


METHOD_ORDER = [
    "KGNN-RFFI",
    "Energy",
    "kNN",
    "OSSEI",
    "HyperRSI",
    "MeDAE",
    "OpenSVDD",
    "OpenMax",
]

TABLE_METHOD_LABELS = {
    "KGNN-RFFI": "KGNN-RFFI",
    "Energy": "Energy~\\cite{Liu2020EnergyOOD}",
    "kNN": "kNN~\\cite{Cover1967NearestNeighbor}",
    "OSSEI": "OSSEI~\\cite{OSSEI2025ClassIrrelevant}",
    "HyperRSI": "HyperRSI~\\cite{HyperRSI2024}",
    "MeDAE": "MeDAE~\\cite{Huang2025MeDAE}",
    "OpenSVDD": "OpenSVDD~\\cite{Wu2025OpenSVDD}",
    "OpenMax": "OpenMax~\\cite{Bendale2016OpenMax}",
}

BASELINE_KEYS = {
    "Energy": "submission_pack/energy",
    "kNN": "submission_pack/knn_cosine_k5",
    "OSSEI": "ossei2025/ossei2025_zscore_fused",
    "HyperRSI": "hyperrsi_full/hyperrsi_full_hme_gpd_evt",
    "MeDAE": "medae_near_full/medae_near_full_center_3sigma",
    "OpenSVDD": "opensvdd_full/opensvdd_full_arpl_rbf_ocsvm",
    "OpenMax": "metric_full/openmax_cl_full_evt",
}

V51_CLASS_ENVELOPE_ONLY_KEY = "v51_no_vote_no_center_margin_blend_ip_knn_cosine_k5_alpha0p50_0p95_nnid"

METRICS = {
    "AUROC": "auc_shifted_known_vs_unknown",
    "OSCR": "auosc_shifted_known_vs_unknown",
    "H-score": "sample_open_set_h_score",
    "ACC": "paper_a_open_set_accuracy",
    "Unk. Rej.": "true_unknown_rejection_rate",
    "FRR": "shifted_known_false_rejection_rate",
}

MECHANISM_TABLE_METRICS = [
    ("AUROC", "AUROC$\\uparrow$", "auc_shifted_known_vs_unknown"),
    ("OSCR", "OSCR$\\uparrow$", "auosc_shifted_known_vs_unknown"),
    ("H-score", "H-score$\\uparrow$", "sample_open_set_h_score"),
    ("ACC", "ACC$\\uparrow$", "paper_a_open_set_accuracy"),
    ("Unk. Rej.", "Unk. Rej.$\\uparrow$", "true_unknown_rejection_rate"),
    ("FRR", "FRR$\\downarrow$", "shifted_known_false_rejection_rate"),
]
MECHANISM_TABLE_METRIC_SOURCE_COLUMNS = [source_col for _, _, source_col in MECHANISM_TABLE_METRICS]

PERFORMANCE_TABLES = [
    {
        "filename": "table_c_open_set_recognition.tex",
        "caption": "Open-set recognition results under source-only domain-shift protocols.",
        "label": "tab:open_set_recognition",
        "columns": [("H-score", "H-score$\\uparrow$", True), ("OSCR", "OSCR$\\uparrow$", True)],
    },
    {
        "filename": "table_d_unknown_detection.tex",
        "caption": "Unknown-device detection results under source-only domain-shift protocols.",
        "label": "tab:unknown_detection",
        "columns": [("AUROC", "AUROC$\\uparrow$", True), ("Unk. Rej.", "Unk. Rej.$\\uparrow$", True)],
    },
    {
        "filename": "table_e_open_set_accuracy.tex",
        "caption": "Overall open-set accuracy and known-device false rejection under source-only domain-shift protocols.",
        "label": "tab:open_set_accuracy",
        "columns": [("ACC", "ACC$\\uparrow$", True), ("FRR", "FRR$\\downarrow$", False)],
    },
]

PROTOCOL_ORDER = {
    "ManySig": [
        "RX6-6_TX3-3",
        "RX9-3_TX2-4",
        "RX9-3_TX4-2",
        "RX3-9_TX2-4",
    ],
    "ManyTx": [
        "MTX_RX9-3_TX20-20",
        "MTX_RX9-3_TX20-40",
        "MTX_RX9-3_TX40-40",
        "MTX_RX6-6_TX20-20",
        "MTX_RX6-6_TX20-40",
        "MTX_RX6-6_TX40-40",
        "MTX_RX3-9_TX20-80",
    ],
}

STYLE = {
    "KGNN-RFFI": {"color": "#0F4D92", "marker": "o", "ls": "-", "lw": 2.0, "zorder": 10, "ms": 7.0, "alpha": 1.0},
    "Energy": {"color": "#B64342", "marker": "s", "ls": (0, (4, 3)), "lw": 1.4, "zorder": 8, "ms": 5.5, "alpha": 0.95},
    "kNN": {"color": "#42949E", "marker": "D", "ls": (0, (2, 2)), "lw": 1.2, "zorder": 7, "ms": 5.0, "alpha": 0.9},
    "OSSEI": {"color": "#4C9A2A", "marker": "^", "ls": (0, (2, 2)), "lw": 1.2, "zorder": 6, "ms": 5.5, "alpha": 0.9},
    "HyperRSI": {"color": "#9A4D8E", "marker": "*", "ls": (0, (1.5, 2.5)), "lw": 1.2, "zorder": 5, "ms": 7.0, "alpha": 0.9},
    "MeDAE": {"color": "#E69F00", "marker": "P", "ls": (0, (5, 2)), "lw": 1.2, "zorder": 4, "ms": 5.5, "alpha": 0.9},
    "OpenSVDD": {"color": "#D55E00", "marker": "v", "ls": (0, (3, 2, 1, 2)), "lw": 1.2, "zorder": 3, "ms": 5.5, "alpha": 0.9},
    "OpenMax": {"color": "#767676", "marker": "x", "ls": (0, (2, 2)), "lw": 1.2, "zorder": 2, "ms": 5.5, "alpha": 0.9},
}


def dataset_short(value: str) -> str:
    if str(value).startswith("ManySig"):
        return "ManySig"
    if str(value).startswith("ManyTx"):
        return "ManyTx"
    raise ValueError(f"Unknown dataset name: {value}")


def ensure_dirs() -> None:
    for path in [OUT_DIR, DATA_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_canonical_per_run() -> pd.DataFrame:
    baseline_df = pd.read_csv(V49_PER_RUN)
    key_to_method = {v: k for k, v in BASELINE_KEYS.items()}
    baseline_df = baseline_df[baseline_df["method_key"].isin(BASELINE_KEYS.values())].copy()
    baseline_df["paper_method"] = baseline_df["method_key"].map(key_to_method)
    baseline_df["internal_method_key"] = baseline_df["method_key"]
    baseline_df["source_table"] = str(V49_PER_RUN.relative_to(ROOT))

    v51_df = pd.concat([pd.read_csv(V51_M2_PER_RUN), pd.read_csv(V51_MISSING_PER_RUN)], ignore_index=True)
    v51_df = v51_df[v51_df["method"] == V51_CLASS_ENVELOPE_ONLY_KEY].copy()
    v51_df["paper_method"] = "KGNN-RFFI"
    v51_df["method_key"] = V51_CLASS_ENVELOPE_ONLY_KEY
    v51_df["internal_method_key"] = V51_CLASS_ENVELOPE_ONLY_KEY
    v51_df["source_table"] = f"{V51_M2_PER_RUN.relative_to(ROOT)}; {V51_MISSING_PER_RUN.relative_to(ROOT)}"

    metric_cols = list(METRICS.values())
    shared_cols = [
        "run_id",
        "dataset",
        "protocol",
        "split_id",
        "paper_method",
        "method_key",
        "internal_method_key",
        "source_table",
    ] + metric_cols
    combined = pd.concat([baseline_df[shared_cols], v51_df[shared_cols]], ignore_index=True)
    combined["dataset_short"] = combined["dataset"].map(dataset_short)
    combined["method_order"] = combined["paper_method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    combined["split_id"] = combined["split_id"].astype(int)
    for col in metric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="raise")

    combined = combined.sort_values(["method_order", "dataset_short", "protocol", "split_id"]).reset_index(drop=True)
    check_coverage(combined)
    return combined


def check_coverage(df: pd.DataFrame) -> None:
    expected = {"ManySig": 12, "ManyTx": 21}
    grouped = df.groupby(["paper_method", "dataset_short"]).size()
    missing = []
    for method in METHOD_ORDER:
        for ds, n in expected.items():
            got = int(grouped.get((method, ds), 0))
            if got != n:
                missing.append(f"{method}/{ds}: expected {n}, got {got}")
    if missing:
        raise RuntimeError("Coverage check failed: " + "; ".join(missing))

    dup_subset = ["paper_method", "run_id", "dataset_short", "protocol", "split_id"]
    dup = df[df.duplicated(dup_subset, keep=False)]
    if not dup.empty:
        sample = dup[dup_subset].head(10).to_dict("records")
        raise RuntimeError(f"Duplicate per-run rows detected: {sample}")


def summarize_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for ds in ["ManySig", "ManyTx"]:
            subset = df[(df["paper_method"] == method) & (df["dataset_short"] == ds)]
            row = {"paper_method": method, "dataset": ds, "runs": int(len(subset))}
            for label, col in METRICS.items():
                row[f"{label}_mean"] = float(subset[col].mean())
                row[f"{label}_std"] = float(subset[col].std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_protocol(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dataset_short", "protocol", "paper_method"], as_index=False)
        .agg(
            h_score_mean=("sample_open_set_h_score", "mean"),
            h_score_std=("sample_open_set_h_score", "std"),
            oscr_mean=("auosc_shifted_known_vs_unknown", "mean"),
            oscr_std=("auosc_shifted_known_vs_unknown", "std"),
            runs=("sample_open_set_h_score", "size"),
        )
    )
    grouped["method_order"] = grouped["paper_method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    grouped["protocol_order"] = grouped.apply(
        lambda r: protocol_sort_key(str(r["dataset_short"]), str(r["protocol"])), axis=1
    )
    return grouped.sort_values(["dataset_short", "protocol_order", "method_order"]).reset_index(drop=True)


def protocol_sort_key(ds: str, protocol: str) -> int:
    order = PROTOCOL_ORDER.get(ds, [])
    if protocol in order:
        return order.index(protocol)
    return len(order) + sorted(order + [protocol]).index(protocol)


def formatted_value(mean: float, std: float) -> str:
    return f"{mean:.3f}$\\pm${std:.3f}"


def formatted_delta(value: float, delta: float | None) -> str:
    if delta is None or pd.isna(delta):
        return f"{value:.3f}"
    if abs(delta) < 0.0005:
        delta = 0.0
    return f"{value:.3f} ({delta:+.3f})"


def rank_marks(summary: pd.DataFrame, dataset: str, metric: str, higher_is_better: bool) -> dict[str, str]:
    mean_col = f"{metric}_mean"
    subset = summary[summary["dataset"] == dataset].copy()
    subset["method_order"] = subset["paper_method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    subset = subset.sort_values(
        [mean_col, "method_order"],
        ascending=[not higher_is_better, True],
    )
    marks = {}
    if len(subset) >= 1:
        marks[str(subset.iloc[0]["paper_method"])] = "best"
    if len(subset) >= 2:
        marks[str(subset.iloc[1]["paper_method"])] = "second"
    return marks


def rank_marks_by_setting(rows: pd.DataFrame, dataset: str, metric: str, higher_is_better: bool) -> dict[str, str]:
    value_col = f"{dataset}_{metric}"
    subset = rows[["setting", value_col]].copy()
    subset["row_order"] = range(len(subset))
    subset = subset.sort_values(
        [value_col, "row_order"],
        ascending=[not higher_is_better, True],
    )
    marks = {}
    if len(subset) >= 1:
        marks[str(subset.iloc[0]["setting"])] = "best"
    if len(subset) >= 2:
        marks[str(subset.iloc[1]["setting"])] = "second"
    return marks


def apply_mark(text: str, mark: str | None) -> str:
    if mark == "best":
        return f"\\textbf{{{text}}}"
    if mark == "second":
        return f"\\underline{{{text}}}"
    return text


def table_method_label(method: str) -> str:
    return TABLE_METHOD_LABELS.get(method, method)


def latex_performance_table(summary: pd.DataFrame, config: dict) -> str:
    columns = config["columns"]
    marks = {
        (ds, label): rank_marks(summary, ds, label, hib)
        for ds in ["ManySig", "ManyTx"]
        for label, _, hib in columns
    }

    lines = [
        "\\begin{table}[h]",
        f"\\caption{{{config['caption']}}}",
        f"\\label{{{config['label']}}}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Method & \\multicolumn{2}{c}{ManySig} & \\multicolumn{2}{c}{ManyTx} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        " & " + " & ".join([c[1] for c in columns] * 2) + " \\\\",
        "\\midrule",
    ]

    for method in METHOD_ORDER:
        cells = [table_method_label(method)]
        for ds in ["ManySig", "ManyTx"]:
            row = summary[(summary["paper_method"] == method) & (summary["dataset"] == ds)].iloc[0]
            for label, _, _ in columns:
                value = formatted_value(row[f"{label}_mean"], row[f"{label}_std"])
                cells.append(apply_mark(value, marks[(ds, label)].get(method)))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_performance_tables(summary: pd.DataFrame) -> None:
    for config in PERFORMANCE_TABLES:
        out_path = TABLE_DIR / config["filename"]
        out_path.write_text(latex_performance_table(summary, config), encoding="utf-8")


def load_efficiency_table() -> pd.DataFrame:
    df = pd.read_csv(EFFICIENCY_CSV)
    df = df[
        (df["dataset"] == "ManyTx")
        & (df["profile_scope"] == "full_decision")
        & (df["paper_method"].isin(METHOD_ORDER))
        & (df["batch_size"].isin([1, 256]))
    ].copy()

    rows = []
    for method in METHOD_ORDER:
        sub = df[df["paper_method"] == method]
        if set(sub["batch_size"].astype(int)) != {1, 256}:
            raise RuntimeError(f"Missing batch-1 or batch-256 efficiency row for {method}")
        b1 = sub[sub["batch_size"].astype(int) == 1].iloc[0]
        b256 = sub[sub["batch_size"].astype(int) == 256].iloc[0]
        rows.append(
            {
                "paper_method": method,
                "params_m": float(b1["params_m"]),
                "model_size_mb": float(b1["model_size_mb"]),
                "source_memory_mb": float(b1["source_memory_mb"]),
                "setup_time_s": float(b1["setup_time_s"]),
                "b1_latency_ms": float(b1["latency_per_sample_ms_mean"]),
                "b256_throughput": float(b256["throughput_from_mean_elapsed_samples_s"]),
            }
        )
    return pd.DataFrame(rows)


def latex_efficiency_table(eff: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[h]",
        "\\caption{Computational efficiency on ManyTx.}",
        "\\label{tab:computational_efficiency_manytx}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Method & Params (M) & Model MB & Source Mem. MB & Setup (s) & Batch-1 Lat. (ms) & Batch-256 Thru. (samples/s) \\\\",
        "\\midrule",
    ]
    for _, row in eff.iterrows():
        cells = [
            table_method_label(str(row["paper_method"])),
            f"{row['params_m']:.3f}",
            f"{row['model_size_mb']:.3f}",
            f"{row['source_memory_mb']:.3f}",
            f"{row['setup_time_s']:.2f}",
            f"{row['b1_latency_ms']:.3f}",
            f"{row['b256_throughput']:.1f}",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "\\vspace{2pt}",
            "\\parbox{\\textwidth}{\\footnotesize \\textit{Note:} Measurements were taken on an RTX 4080 with CUDA synchronization. Reported latency and throughput cover the complete inference pipeline after source-side preparation. Setup time covers source-side memory construction, reference-set building, and threshold calibration. Data loading time is excluded.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_efficiency_table() -> pd.DataFrame:
    eff = load_efficiency_table()
    eff.to_csv(DATA_DIR / "table_g_efficiency_manytx_source.csv", index=False)
    (TABLE_DIR / "table_g_efficiency_manytx.tex").write_text(latex_efficiency_table(eff), encoding="utf-8")
    return eff


def load_mechanism_rows() -> pd.DataFrame:
    component = pd.read_csv(COMPONENT_SUMMARY_CSV)
    component = component[component["dataset_short"].isin(["ManySig", "ManyTx"])].copy()
    simplified = pd.read_csv(SIMPLIFIED_MAIN_CSV)
    simplified = simplified[simplified["dataset"].isin(["ManySig", "ManyTx"])].copy()
    default_key = V51_CLASS_ENVELOPE_ONLY_KEY
    simpler_key = "IP-GATE-NN-ID"
    no_envelope_key = "v51_no_class_envelope_blend_ip_knn_cosine_k5_alpha0p50_0p95_nnid"
    fixed_blend_key = "v51_source_selector_alpha_nnid"
    hard_switch_key = "v51_hardswitch_ipguard_blend_ip_knn_cosine_k5_alpha0p50_0p95_tau0p55_nnid"

    component_rows = []
    ablation_specs = [
        ("KGNN-RFFI", "component", default_key),
        ("w/o NN Knownness Branch", "simplified", simpler_key),
        ("w/o Source Class Envelope", "component", no_envelope_key),
        ("w/o Sample-wise Gate", "component", fixed_blend_key),
        ("w/o Soft Gate", "component", hard_switch_key),
    ]
    for setting, source_kind, method_key in ablation_specs:
        if source_kind == "simplified":
            source_path = SIMPLIFIED_MAIN_CSV
        else:
            source_path = COMPONENT_SUMMARY_CSV
        row = {
            "block": "Mechanism ablation (33 runs)",
            "setting": setting,
            "source": str(source_path.relative_to(ROOT)),
        }
        for ds in ["ManySig", "ManyTx"]:
            if source_kind == "simplified":
                default_ds = simplified[
                    (simplified["dataset"] == ds) & (simplified["method"] == "V51 class-envelope-only")
                ].iloc[0]
                current_ds = simplified[
                    (simplified["dataset"] == ds) & (simplified["method"] == method_key)
                ].iloc[0]
            else:
                default_ds = component[
                    (component["dataset_short"] == ds) & (component["method"] == default_key)
                ].iloc[0]
                current_ds = component[
                    (component["dataset_short"] == ds) & (component["method"] == method_key)
                ].iloc[0]
            for metric_name, _, source_col in MECHANISM_TABLE_METRICS:
                value = float(current_ds[source_col])
                delta = None if setting == "KGNN-RFFI" else value - float(default_ds[source_col])
                row[f"{ds}_{metric_name}"] = value
                row[f"d_{ds}_{metric_name}"] = delta
        component_rows.append(row)

    return pd.DataFrame(component_rows)


def latex_mechanism_table(rows: pd.DataFrame) -> str:
    metric_headers = [display for _, display, _ in MECHANISM_TABLE_METRICS]
    metric_names = [name for name, _, _ in MECHANISM_TABLE_METRICS]
    marks = {
        (ds, metric): rank_marks_by_setting(rows, ds, metric, metric != "FRR")
        for ds in ["ManySig", "ManyTx"]
        for metric in metric_names
    }
    lines = [
        "\\begin{table*}[h]",
        "\\caption{Mechanism ablation evidence on ManySig and ManyTx.}",
        "\\label{tab:source_envelope_mechanism}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccccccccc@{}}",
        "\\toprule",
        "Setting & \\multicolumn{6}{c}{ManySig} & \\multicolumn{6}{c}{ManyTx} \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
        " & "
        + " & ".join(metric_headers)
        + " & "
        + " & ".join(metric_headers)
        + " \\\\",
        "\\midrule",
    ]
    for _, row in rows.iterrows():
        setting = str(row["setting"])
        cells = [setting]
        for ds in ["ManySig", "ManyTx"]:
            for metric in metric_names:
                value = f"{float(row[f'{ds}_{metric}']):.3f}"
                cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}%",
            "\\vspace{2pt}",
            "\\parbox{\\textwidth}{\\footnotesize \\textit{Note:} Each variant removes one component from \\method{}. For the soft-gate variant, the soft gate is replaced by a hard switch.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_mechanism_table() -> pd.DataFrame:
    rows = load_mechanism_rows()
    rows.to_csv(DATA_DIR / "table_f_source_envelope_mechanism_source.csv", index=False)
    (TABLE_DIR / "table_f_source_envelope_mechanism.tex").write_text(
        latex_mechanism_table(rows), encoding="utf-8"
    )
    return rows


def write_mechanism_claim_delta_summary(mechanism: pd.DataFrame) -> None:
    mechanism.to_csv(DATA_DIR / "table_f_claim_delta_summary.csv", index=False)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.0,
            "figure.dpi": 300,
            "savefig.dpi": 1200,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 6.5,
        }
    )


def protocol_label(protocol: str) -> str:
    return protocol.replace("MTX_", "")


def protocols_for_metric(
    protocol_summary: pd.DataFrame,
    dataset: str,
    metric_mean_col: str,
    sort_descending_by_kgnn: bool = False,
) -> list[str]:
    dataset_protocols = set(protocol_summary[protocol_summary["dataset_short"] == dataset]["protocol"])
    if not sort_descending_by_kgnn:
        return [p for p in PROTOCOL_ORDER[dataset] if p in dataset_protocols]

    kgnn = protocol_summary[
        (protocol_summary["dataset_short"] == dataset)
        & (protocol_summary["paper_method"] == "KGNN-RFFI")
    ].copy()
    kgnn = kgnn.sort_values([metric_mean_col, "protocol_order"], ascending=[False, True])
    return [str(p) for p in kgnn["protocol"].tolist()]


def draw_protocol_metric(
    ax: plt.Axes,
    protocol_summary: pd.DataFrame,
    dataset: str,
    protocols: list[str],
    metric_mean_col: str,
    ylabel: str,
    ylim: tuple[float, float],
    title: str | None = None,
    force_solid_lines: bool = False,
    show_grid: bool = True,
) -> None:
    x = np.arange(len(protocols))
    for method in METHOD_ORDER:
        sub = protocol_summary[
            (protocol_summary["dataset_short"] == dataset) & (protocol_summary["paper_method"] == method)
        ].set_index("protocol")
        vals = [float(sub.loc[p, metric_mean_col]) for p in protocols]
        s = STYLE[method]
        mec = "white" if method == "KGNN-RFFI" else s["color"]
        mew = 0.8 if method == "KGNN-RFFI" else 0.4
        ax.plot(
            x,
            vals,
            label=method,
            color=s["color"],
            marker=s["marker"],
            linestyle="-" if force_solid_lines else s["ls"],
            linewidth=s["lw"],
            markersize=s["ms"],
            zorder=s["zorder"],
            alpha=s["alpha"],
            markeredgewidth=mew,
            markeredgecolor=mec,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([protocol_label(p) for p in protocols], rotation=24 if len(protocols) > 4 else 0, ha="right" if len(protocols) > 4 else "center")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, 0.1))
    if title:
        ax.set_title(title)
    if show_grid:
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)


def plot_protocol_metric(
    protocol_summary: pd.DataFrame,
    dataset: str,
    figsize: tuple[float, float],
    metric_mean_col: str,
    ylabel: str,
    ylim: tuple[float, float],
    sort_descending_by_kgnn: bool = False,
    force_solid_lines: bool = False,
    show_grid: bool = True,
) -> plt.Figure:
    protocols = protocols_for_metric(protocol_summary, dataset, metric_mean_col, sort_descending_by_kgnn)
    fig, ax = plt.subplots(figsize=figsize)
    draw_protocol_metric(
        ax,
        protocol_summary,
        dataset,
        protocols,
        metric_mean_col,
        ylabel,
        ylim,
        force_solid_lines=force_solid_lines,
        show_grid=show_grid,
    )
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        handlelength=1.3,
        handletextpad=0.45,
        borderpad=0.35,
        labelspacing=0.25,
        columnspacing=0.75,
        framealpha=0.90,
        edgecolor="#cccccc",
        facecolor="white",
        fancybox=False,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout(pad=0.6)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ["pdf", "png", "svg", "eps"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", format=ext if ext == "eps" else None)


def parse_setting_float(setting: str, prefix: str) -> float:
    if not setting.startswith(prefix):
        raise ValueError(f"Expected setting prefix {prefix!r}, got {setting!r}")
    return float(setting[len(prefix):])


def load_sensitivity_plot_rows() -> pd.DataFrame:
    rows = []

    k_grid = pd.read_csv(K_SENSITIVITY_CSV)
    k_grid = k_grid[
        (k_grid["family"] == "knn_id")
        & (k_grid["metric"] == "cosine")
        & (k_grid["dataset"].isin(["ManySig-SODA4", "ManyTx-OWEN-v0"]))
    ].copy()
    k_grid["dataset_short"] = k_grid["dataset"].map(dataset_short)
    k_grid["k"] = pd.to_numeric(k_grid["k"], errors="raise").astype(int)
    k_grid["H"] = pd.to_numeric(k_grid["H"], errors="raise")
    for ds in ["ManySig", "ManyTx"]:
        sub = k_grid[k_grid["dataset_short"] == ds]
        default_h = float(sub[sub["k"] == 5]["H"].iloc[0])
        for _, row in sub.sort_values("k").iterrows():
            value = float(row["H"])
            rows.append(
                {
                    "axis": "k",
                    "dataset": ds,
                    "setting": str(int(row["k"])),
                    "x_value": float(row["k"]),
                    "h_score": value,
                    "delta_h_score": value - default_h,
                    "is_default": int(row["k"] == 5),
                    "runs": int(row["runs"]),
                    "source": str(K_SENSITIVITY_CSV.relative_to(ROOT)),
                    "evidence_scope": "33-run ID-head k-grid audit",
                }
            )

    envelope = pd.read_csv(ENVELOPE_SENSITIVITY_CSV)
    envelope = envelope[envelope["dataset_short"].isin(["ManySig", "ManyTx"])].copy()
    envelope["sample_open_set_h_score"] = pd.to_numeric(envelope["sample_open_set_h_score"], errors="raise")
    for axis, paper_axis, prefix, default_value, scope in [
        ("class_envelope_quantile", "q", "q=", 0.90, "12-run envelope-only sensitivity"),
        ("class_envelope_max_mult", "lambda", "max_mult=", 1.75, "12-run envelope-only sensitivity"),
    ]:
        axis_rows = envelope[envelope["axis"] == axis].copy()
        axis_rows["x_value"] = axis_rows["setting"].map(lambda s: parse_setting_float(str(s), prefix))
        for ds in ["ManySig", "ManyTx"]:
            sub = axis_rows[axis_rows["dataset_short"] == ds]
            default_h = float(sub[np.isclose(sub["x_value"], default_value)]["sample_open_set_h_score"].iloc[0])
            for _, row in sub.sort_values("x_value").iterrows():
                value = float(row["sample_open_set_h_score"])
                rows.append(
                    {
                        "axis": paper_axis,
                        "dataset": ds,
                        "setting": str(row["setting"]),
                        "x_value": float(row["x_value"]),
                        "h_score": value,
                        "delta_h_score": value - default_h,
                        "is_default": int(np.isclose(float(row["x_value"]), default_value)),
                        "runs": int(row["runs"]),
                        "source": str(ENVELOPE_SENSITIVITY_CSV.relative_to(ROOT)),
                        "evidence_scope": scope,
                    }
                )

    return pd.DataFrame(rows)


def build_sensitivity_plot() -> pd.DataFrame:
    configure_matplotlib()
    sens = load_sensitivity_plot_rows()
    sens.to_csv(DATA_DIR / "fig_f_sensitivity_hscore_source.csv", index=False)

    axis_configs = [
        ("k", r"$k$", ["1", "3", "5", "10", "20"], "fig_f_sensitivity_k_hscore"),
        ("q", r"$q$", ["0.80", "0.85", "0.90", "0.95", "0.975"], "fig_f_sensitivity_q_hscore"),
        ("lambda", r"$\lambda$", ["1.25", "1.50", "1.75", "2.00"], "fig_f_sensitivity_lambda_hscore"),
    ]
    dataset_styles = {
        "ManySig": {"color": "#0F4D92", "marker": "o"},
        "ManyTx": {"color": "#D55E00", "marker": "s"},
    }
    for axis_name, xlabel, tick_labels, stem in axis_configs:
        fig, ax = plt.subplots(figsize=(2.35, 1.85))
        axis_rows = sens[sens["axis"] == axis_name].copy()
        axis_rows = axis_rows.sort_values("x_value")
        x_values = sorted(axis_rows["x_value"].unique())
        x_positions = {v: i for i, v in enumerate(x_values)}
        for dataset in ["ManySig", "ManyTx"]:
            sub = axis_rows[axis_rows["dataset"] == dataset].sort_values("x_value")
            xs = [x_positions[float(v)] for v in sub["x_value"]]
            ys = [100.0 * float(v) for v in sub["delta_h_score"]]
            style = dataset_styles[dataset]
            ax.plot(
                xs,
                ys,
                label=dataset,
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.15,
                markersize=4.0,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        default_rows = axis_rows[axis_rows["is_default"] == 1].copy()
        for _, default_row in default_rows.iterrows():
            ax.scatter(
                [x_positions[float(default_row["x_value"])]],
                [0.0],
                marker="D",
                s=22,
                facecolors="white",
                edgecolors="#333333",
                linewidths=0.8,
                zorder=6,
            )
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(xlabel, fontsize=8.2, labelpad=1.0)
        ax.tick_params(axis="both", labelsize=7.2)
        ax.set_ylabel(r"$\Delta$H-score (pp)", fontsize=8.2, labelpad=1.0)
        ax.set_ylim(-2.1, 1.2)
        ax.set_yticks([-2.0, -1.0, 0.0, 1.0])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            handlelength=1.0,
            handletextpad=0.45,
            borderpad=0.25,
            labelspacing=0.25,
            columnspacing=0.7,
            fontsize=7.0,
            framealpha=0.90,
            edgecolor="#cccccc",
            facecolor="white",
            fancybox=False,
        )
        fig.tight_layout(pad=0.45, rect=(0, 0, 1, 0.90))
        save_figure(fig, stem)
        plt.close(fig)
    return sens


def tradeoff_axis_limits(values: pd.Series, *, lower_bound: float = 0.0) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="raise")
    vmin = float(vals.min())
    vmax = float(vals.max())
    span = max(vmax - vmin, 0.05)
    lo = max(lower_bound, vmin - 0.12 * span)
    hi = min(1.0, vmax + 0.12 * span)
    return lo, hi


def draw_tradeoff(
    ax: plt.Axes,
    summary: pd.DataFrame,
    dataset: str,
    *,
    title: str | None = None,
    legend: bool = True,
) -> None:
    subset = summary[summary["dataset"] == dataset].set_index("paper_method")
    for method in METHOD_ORDER:
        row = subset.loc[method]
        s = STYLE[method]
        size = 8.2 if method == "KGNN-RFFI" else 6.2
        mec = "white" if method == "KGNN-RFFI" else s["color"]
        mew = 0.8 if method == "KGNN-RFFI" else 0.5
        ax.plot(
            float(row["ACC_mean"]),
            float(row["Unk. Rej._mean"]),
            label=method,
            marker=s["marker"],
            color=s["color"],
            linestyle="",
            markersize=size,
            markeredgecolor=mec,
            markeredgewidth=mew,
            alpha=s["alpha"],
            zorder=s["zorder"],
        )

    ax.set_xlabel("ACC$\\uparrow$")
    ax.set_ylabel("Unk. Rej.$\\uparrow$")
    if title:
        ax.set_title(title)
    ax.set_xlim(*tradeoff_axis_limits(subset["ACC_mean"]))
    ax.set_ylim(*tradeoff_axis_limits(subset["Unk. Rej._mean"]))
    ax.tick_params(axis="both", labelsize=9)
    if legend:
        leg = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=3,
            handlelength=1.0,
            handletextpad=0.45,
            borderpad=0.35,
            labelspacing=0.25,
            columnspacing=0.75,
            framealpha=0.90,
            edgecolor="#cccccc",
            facecolor="white",
            fancybox=False,
        )
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)


def build_tradeoff_plots(summary: pd.DataFrame) -> None:
    configure_matplotlib()
    for ds, figsize in [("ManySig", (4.9, 3.2)), ("ManyTx", (4.9, 3.2))]:
        fig, ax = plt.subplots(figsize=figsize)
        draw_tradeoff(ax, summary, ds, legend=True)
        fig.tight_layout(pad=0.55)
        save_figure(fig, f"fig_de_tradeoff_{ds.lower()}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, ds in zip(axes, ["ManySig", "ManyTx"]):
        draw_tradeoff(ax, summary, ds, title=ds, legend=False)
        if ds == "ManyTx":
            ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        handlelength=1.0,
        handletextpad=0.45,
        borderpad=0.35,
        labelspacing=0.25,
        columnspacing=0.65,
        framealpha=0.90,
        edgecolor="#cccccc",
        facecolor="white",
        fancybox=False,
    )
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.94))
    save_figure(fig, "fig_de_tradeoff_acc_unknown_rej")
    plt.close(fig)


def build_combined_protocol_metric(
    protocol_summary: pd.DataFrame,
    metric_mean_col: str,
    ylabel: str,
    ylim: tuple[float, float],
    stem: str,
    sort_descending_by_kgnn: bool = False,
) -> None:
    protocols_by_ds = {
        ds: protocols_for_metric(protocol_summary, ds, metric_mean_col, sort_descending_by_kgnn)
        for ds in ["ManySig", "ManyTx"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, ds in zip(axes, ["ManySig", "ManyTx"]):
        draw_protocol_metric(
            ax,
            protocol_summary,
            ds,
            protocols_by_ds[ds],
            metric_mean_col,
            ylabel,
            ylim,
            title=ds,
        )
        if ds == "ManyTx":
            ax.set_ylabel("")
        if len(protocols_by_ds[ds]) > 4:
            ax.set_xticklabels([protocol_label(p) for p in protocols_by_ds[ds]], rotation=28, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=5,
        handlelength=1.2,
        handletextpad=0.45,
        borderpad=0.35,
        labelspacing=0.25,
        columnspacing=0.65,
        framealpha=0.90,
        edgecolor="#cccccc",
        facecolor="white",
        fancybox=False,
    )
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.95))
    save_figure(fig, stem)
    plt.close(fig)


def build_protocol_plots(protocol_summary: pd.DataFrame) -> None:
    configure_matplotlib()
    fig = plot_protocol_metric(
        protocol_summary,
        "ManySig",
        (5.2, 3.2),
        metric_mean_col="h_score_mean",
        ylabel="H-score",
        ylim=(0.0, 0.68),
    )
    save_figure(fig, "fig_c_hscore_protocols_manysig")
    plt.close(fig)

    fig = plot_protocol_metric(
        protocol_summary,
        "ManyTx",
        (6.2, 3.35),
        metric_mean_col="h_score_mean",
        ylabel="H-score",
        ylim=(0.0, 0.4),
        sort_descending_by_kgnn=True,
        show_grid=False,
    )
    save_figure(fig, "fig_c_hscore_protocols_manytx")
    plt.close(fig)

    build_combined_protocol_metric(
        protocol_summary,
        metric_mean_col="h_score_mean",
        ylabel="H-score",
        ylim=(0.0, 0.68),
        stem="fig_c_hscore_protocols",
    )

    fig = plot_protocol_metric(
        protocol_summary,
        "ManySig",
        (5.2, 3.2),
        metric_mean_col="oscr_mean",
        ylabel="OSCR",
        ylim=(0.0, 0.75),
        sort_descending_by_kgnn=True,
    )
    save_figure(fig, "fig_c_oscr_protocols_manysig")
    plt.close(fig)

    fig = plot_protocol_metric(
        protocol_summary,
        "ManyTx",
        (6.2, 3.35),
        metric_mean_col="oscr_mean",
        ylabel="OSCR",
        ylim=(0.0, 0.75),
        sort_descending_by_kgnn=True,
    )
    save_figure(fig, "fig_c_oscr_protocols_manytx")
    plt.close(fig)

    build_combined_protocol_metric(
        protocol_summary,
        metric_mean_col="oscr_mean",
        ylabel="OSCR",
        ylim=(0.0, 0.75),
        stem="fig_c_oscr_protocols",
        sort_descending_by_kgnn=True,
    )


def write_provenance() -> None:
    provenance = {
        "source_files": {
            "baseline_per_run": str(V49_PER_RUN.relative_to(ROOT)),
            "kgnn_manysig_m2_per_run": str(V51_M2_PER_RUN.relative_to(ROOT)),
            "kgnn_missing_per_run": str(V51_MISSING_PER_RUN.relative_to(ROOT)),
            "efficiency": str(EFFICIENCY_CSV.relative_to(ROOT)),
            "source_envelope_component_summary": str(COMPONENT_SUMMARY_CSV.relative_to(ROOT)),
            "mechanism_simplified_main_comparison": str(SIMPLIFIED_MAIN_CSV.relative_to(ROOT)),
            "source_envelope_sensitivity": str(ENVELOPE_SENSITIVITY_CSV.relative_to(ROOT)),
            "k_sensitivity": str(K_SENSITIVITY_CSV.relative_to(ROOT)),
        },
        "paper_method_order": METHOD_ORDER,
        "baseline_keys": BASELINE_KEYS,
        "kgnn_internal_key": V51_CLASS_ENVELOPE_ONLY_KEY,
        "metric_mapping": METRICS,
        "acc_definition": "ACC maps to paper_a_open_set_accuracy, i.e., overall open-set accuracy over target known and target unknown samples.",
        "h_score_definition": "H-score maps to sample_open_set_h_score and is the harmonic mean of target-known CCR and unknown-device rejection, not a harmonic mean of ACC.",
        "dataset_reporting": "ManySig and ManyTx are emitted as separate LaTeX table groups.",
    }
    (DATA_DIR / "asset_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    canonical = load_canonical_per_run()
    summary = summarize_by_dataset(canonical)
    protocol_summary = summarize_by_protocol(canonical)

    canonical.to_csv(DATA_DIR / "main_comparison_canonical_per_run.csv", index=False)
    summary.to_csv(DATA_DIR / "main_comparison_by_dataset_source.csv", index=False)
    protocol_summary.to_csv(DATA_DIR / "protocol_hscore_by_dataset_source.csv", index=False)
    protocol_summary.to_csv(DATA_DIR / "protocol_metrics_by_dataset_source.csv", index=False)

    build_performance_tables(summary)
    eff = build_efficiency_table()
    mechanism = build_mechanism_table()
    write_mechanism_claim_delta_summary(mechanism)
    build_protocol_plots(protocol_summary)
    build_tradeoff_plots(summary)
    sensitivity = build_sensitivity_plot()
    write_provenance()

    print(f"Wrote canonical per-run rows: {len(canonical)}")
    print(f"Wrote dataset summary rows: {len(summary)}")
    print(f"Wrote protocol summary rows: {len(protocol_summary)}")
    print(f"Wrote efficiency rows: {len(eff)}")
    print(f"Wrote mechanism rows: {len(mechanism)}")
    print(f"Wrote sensitivity rows: {len(sensitivity)}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
