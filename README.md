# DPR-RFFI

**Dual Perturbation References for Open-Set RF Fingerprint Identification under Domain Shift**

Code release accompanying the DPR-RFFI manuscript.

## Overview

DPR-RFFI addresses open-set radio frequency fingerprint identification under domain shift. It must identify signals from enrolled devices while rejecting signals from devices that were not represented during training.

The method contains two complementary components:

1. **Dual Perturbation Reference (DPR)** screens RF perturbations according to their effect on source-domain classification, constructs low-impact and high-impact reference sets, and uses their distance ratio as relative rejection evidence.
2. **Class-Consistency Assessment (CCA)** evaluates a received sample against the source region of its nearest-neighbor class and regulates the contribution of source-memory evidence to the final decision.

Samples accepted by the resulting score are assigned to enrolled devices through nearest-neighbor voting. All model training, perturbation screening, score normalization, and threshold calibration use source-domain data only.

## Implementation Naming

The implementation was developed before the paper-facing DPR-RFFI terminology was finalized. To preserve the submitted experiment commands and configuration compatibility, the Python package name `kgnn` and legacy command-line flags such as `--enable-kgnn` and `--sce-*` remain unchanged.

The corresponding manuscript terms are:

| Implementation name | Manuscript term |
|---|---|
| safe or identity-preserving perturbation | low-impact perturbation |
| destructive or identity-disrupting perturbation | high-impact perturbation |
| source class envelope or SCE gate | Class-Consistency Assessment (CCA) |
| ratio score `u_p` | DPR score |

These are naming differences and do not change the implemented inference procedure.

## Requirements

Python 3.10+, PyTorch 2.0+, and the dependencies in `requirements.txt` are required.

```bash
pip install -r requirements.txt
```

## Data

The experiments use the public [WiSig](https://cores.ee.ucla.edu/downloads/) dataset. Download the dataset and place it under a `data/` directory, then update the paths in `configs/` for the local environment.

## Quick Start

The following command runs DPR-RFFI on one ManyTx protocol. The legacy flag names are retained for reproducibility.

```bash
python scripts/run.py \
  --run "demo|configs/manytx_owen_v0.yaml|MTX_RX9-3_TX20-20|1|resnet1d|5|30|128" \
  --enable-kgnn \
  --sce-sensitivity \
  --output-dir results/demo
```

Key parameters are listed below.

| Parameter | Default | Description |
|---|---:|---|
| `--sce-quantile` | 0.90 | CCA source-class quantile $q$ |
| `--sce-max-mult` | 1.75 | CCA expansion factor $\lambda$ |
| `--sce-weight-low` | 0.50 | DPR weight inside the source class region |
| `--sce-weight-high` | 0.95 | DPR weight outside the expanded source class region |
| `--source-frr` | 0.03 | Source-validation false-rejection budget $\rho_s$ |
| `--safe-accuracy` | 0.90 | Low-impact perturbation screening threshold |
| `--destructive-accuracy` | 0.50 | High-impact perturbation screening threshold |

All parameter settings are fixed before target evaluation. Target labels, target-domain statistics, and unlabeled target batches are not used for parameter selection.

## Repository Structure

```text
DPR-RFFI/
├── kgnn/                    # Core DPR-RFFI implementation
│   ├── envelope.py          # CCA and adaptive score composition
│   ├── perturbation.py      # RF perturbations and source-based screening
│   ├── phantom.py           # RF perturbation engine
│   ├── metrics.py           # Open-set recognition metrics
│   └── baselines/           # Baseline implementations
├── diagnostic/              # Dataset and source-only training pipeline
├── scripts/
│   ├── run.py               # Main experiment runner
│   └── build_assets.py      # Table and figure generation
├── configs/                 # ManySig and ManyTx configurations
├── results/                 # Experiment outputs
├── requirements.txt
└── LICENSE
```

## Reproducing the Paper Evaluation

The complete evaluation contains 12 ManySig runs and 21 ManyTx runs across 11 protocols and three splits. The experiment runner supports the main method, component ablations, parameter sensitivity, and the paper baselines.

The included baselines are Energy, kNN, NNDR, OSSEI, HyperRSI, MeDAE, OpenSVDD, and OpenMax. All methods use the same protocol splits and source-validation operating-point calibration. Each method calibrates its own score threshold because their score spaces are not numerically interchangeable.

## Citation

```bibtex
@article{zhong2026dprrffi,
  title={DPR-RFFI: Dual Perturbation References for Open-Set RF Fingerprint Identification under Domain Shift},
  author={Zhong, Yuan and Li, Dongming},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
