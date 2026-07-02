# KGNN-RFFI

**Knownness-Gated Nearest Neighbor RF Fingerprint Identification under Domain Shift**

Official code release for the paper submitted to IEEE Internet of Things Journal.

## Overview

KGNN-RFFI is a source-only open-set RF fingerprint identification framework. Under receiver, session, or channel domain shift, it separates unknown-device rejection from enrolled-device identity assignment, using a **source class envelope (SCE)** gate to control when source-memory nearest-neighbor evidence may support acceptance.

## Requirements

Python 3.10+, PyTorch 2.0+, and dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data

Experiments use the public [WiSig](https://cores.ee.ucla.edu/downloads/) dataset (Hanna et al., 2022). Download the dataset and place it under a `data/` directory. The configuration files in `configs/` specify dataset paths — update them to match your local setup.

## Quick Start

Reproduce the main KGNN-RFFI result on a single ManyTx protocol:

```bash
python scripts/run.py \
  --run "demo|configs/manytx_owen_v0.yaml|MTX_RX9-3_TX20-20|1|resnet1d|5|30|128" \
  --enable-kgnn \
  --sce-sensitivity \
  --output-dir results/demo
```

Key hyperparameters and their defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sce-quantile` | 0.90 | SCE source quantile $q$ |
| `--sce-max-mult` | 1.75 | SCE expansion factor $\lambda$ |
| `--sce-weight-low` | 0.50 | Gate weight $w_{\mathrm{low}}$ (inside envelope) |
| `--sce-weight-high` | 0.95 | Gate weight $w_{\mathrm{high}}$ (outside envelope) |
| `--source-frr` | 0.03 | Source-validation false-rejection budget $\rho_s$ |
| `--safe-accuracy` | 0.90 | Perturbation safety screening threshold |
| `--destructive-accuracy` | 0.50 | Perturbation destructive screening threshold |

All hyperparameters are fixed before target evaluation and were confirmed by source-validation-only sensitivity sweeps. No target labels, target-domain statistics, or unlabeled target batches were used for parameter selection.

## Repository Structure

```
kgnn-rffi/
├── kgnn/                    # Core method package
│   ├── envelope.py          # SCE gate + adaptive rejection scoring
│   ├── perturbation.py      # RF perturbation specs + classifier-based screening
│   ├── phantom.py           # Physics perturbation engine (phase, CFO, timing, etc.)
│   ├── metrics.py           # Open-set recognition metrics (H-score, AUROC, OSCR, etc.)
│   ├── supcon_loss.py       # Supervised contrastive loss (for OSSEI baseline)
│   ├── utils.py             # Device resolution, determinism, protocol helpers
│   └── baselines/           # Baseline method implementations
│       ├── posthoc.py       # Energy (OOD score) + kNN
│       ├── ossei2025.py     # OSSEI: class-irrelevant representation learning
│       ├── hyperrsi.py      # HyperRSI: hypersphere prototype modeling
│       ├── medae.py         # MeDAE: metric denoising autoencoder
│       ├── opensvdd.py      # OpenSVDD: class-conditioned open boundaries
│       └── metric_learning.py  # Metric-learning baseline
├── diagnostic/              # Data pipeline
│   ├── sourceonly.py        # CNN/ResNet encoder training (source-only)
│   ├── compact.py           # Compact dataset loading
│   ├── datasets.py          # Record materialization
│   ├── splits.py            # Protocol/split construction
│   ├── config.py            # YAML config loading
│   └── osr.py               # Threshold-based rejection
├── scripts/
│   ├── run.py               # Main experiment runner (KGNN-RFFI with SCE gate)
│   └── build_assets.py      # Table/figure builder from result CSVs
├── configs/
│   ├── manysig_soda4.yaml   # ManySig dataset config (6 TX, 12 RX)
│   └── manytx_owen_v0.yaml  # ManyTx dataset config (100 TX, 12 RX)
├── results/                 # Output directory for experiment results
├── requirements.txt
└── LICENSE
```

## Method Components

### Source-Calibrated Rejection Evidence ($u_{\mathrm{p}}$)

A ratio-based unknownness score computed from source-only embedding reference sets. A source-support reference set $\mathcal{R}_K$ (original + identity-preserving perturbed samples) and an optional identity-disrupting reference set $\mathcal{R}_D$ together define the score $u_{\mathrm{p}} = d_K(\mathbf{z}) / \max(d_D(\mathbf{z}), \epsilon)$.

The perturbation screening uses 52 RF perturbation specifications across 7 families (phase, CFO, timing, amplitude, IQ imbalance, noise, multipath), each at 5–6 severity levels. Specs are classified as **safe** (classifier accuracy ≥ 0.90 on perturbed source-validation samples) or **destructive** (accuracy < 0.50) using only source validation data.

### SCE Knownness Gate ($g_{\mathrm{env}}$)

For each enrolled device class $c$, a source class envelope is defined by a centroid $\mathbf{m}_c$ and a source-quantile radius $R_c$. The piecewise-linear gate $g_{\mathrm{env}}(\mathbf{x})$ evaluates whether a query lies within the predicted class's envelope, with a linear transition zone controlled by the expansion factor $\lambda$.

### Adaptive Score Composition

The final unknownness score combines the standardized source-calibrated rejection score $\tilde{u}_{\mathrm{p}}$ and the standardized kNN distance score $\tilde{u}_{\mathrm{nn}}$:
$$s_U(\mathbf{x}) = w_{\mathrm{p}}(\mathbf{x}) \tilde{u}_{\mathrm{p}}(\mathbf{x}) + (1 - w_{\mathrm{p}}(\mathbf{x})) \tilde{u}_{\mathrm{nn}}(\mathbf{x})$$
where $w_{\mathrm{p}}(\mathbf{x}) = w_{\mathrm{high}} - (w_{\mathrm{high}} - w_{\mathrm{low}}) g_{\mathrm{env}}(\mathbf{x})$.

### Inference

Samples with $s_U(\mathbf{x}) > \theta_s$ are rejected as unknown. Accepted samples receive the enrolled-device identity from source-memory kNN ($k=5$, cosine distance, majority voting).

## Reproducing Paper Results

The full 33-run evaluation (12 ManySig + 21 ManyTx runs across 11 protocols × 3 splits) requires substantial compute. The experiment runner `scripts/run.py` supports:

- `--enable-kgnn`: Run KGNN-RFFI (class-envelope-only adaptive gate)
- `--sce-sensitivity`: Run envelope-only q/λ sensitivity sweep
- `--enable-kgnn-ablations`: Run component ablation variants
- `--enable-kgnn-sensitivity`: Run full parameter sensitivity grid

### Baseline Methods

All baselines (Energy, kNN, OSSEI, HyperRSI, MeDAE, OpenSVDD, OpenMax) share the same source-only protocol and source-FRR operating-point calibration. Each baseline calibrates its own score threshold on source validation data rather than sharing a numeric threshold across incompatible score spaces.

## Citation

```bibtex
@article{zhong2026kgnn,
  title={KGNN-RFFI: Knownness-Gated Nearest Neighbor RF Fingerprint Identification under Domain Shift},
  author={Zhong, Yuan and Li, Dongming},
  journal={IEEE Internet of Things Journal},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
