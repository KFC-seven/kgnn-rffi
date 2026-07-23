from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MANY_SIG = [
    "RX9-3_TX2-4",
    "RX9-3_TX4-2",
    "RX6-6_TX3-3",
    "RX3-9_TX2-4",
]
MANY_TX = [
    "MTX_RX9-3_TX20-20",
    "MTX_RX9-3_TX20-40",
    "MTX_RX9-3_TX40-40",
    "MTX_RX6-6_TX20-20",
    "MTX_RX6-6_TX20-40",
    "MTX_RX6-6_TX40-40",
    "MTX_RX3-9_TX20-80",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 33 DPR-RFFI paper protocols.")
    parser.add_argument("--manysig-data", required=True)
    parser.add_argument("--manytx-data", required=True)
    parser.add_argument("--output-dir", default="outputs/paper")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for dataset, protocols, config, data, architecture, dimension, record_limit in [
        (
            "manysig",
            MANY_SIG,
            "configs/manysig.yaml",
            args.manysig_data,
            "tiny",
            64,
            100,
        ),
        (
            "manytx",
            MANY_TX,
            "configs/manytx.yaml",
            args.manytx_data,
            "resnet1d",
            128,
            30,
        ),
    ]:
        for protocol in protocols:
            for split in (1, 2, 3):
                run_id = f"{dataset}_{protocol}_split{split}"
                command = [
                    sys.executable,
                    "scripts/run_protocol.py",
                    "--config",
                    config,
                    "--data",
                    data,
                    "--protocol",
                    protocol,
                    "--split",
                    str(split),
                    "--architecture",
                    architecture,
                    "--embedding-dim",
                    str(dimension),
                    "--epochs",
                    str(args.epochs),
                    "--max-samples-per-record",
                    str(record_limit),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--output",
                    str(output_dir / f"{run_id}.json"),
                ]
                print(f"[DPR-RFFI] {run_id}", flush=True)
                subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
