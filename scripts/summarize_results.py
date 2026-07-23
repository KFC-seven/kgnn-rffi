from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate protocol JSON files by dataset.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in payload:
            continue
        rows.append(
            {
                "dataset": payload["dataset"],
                "protocol": payload["protocol"],
                "split": payload["split"],
                **payload["metrics"],
            }
        )
    if not rows:
        raise RuntimeError("No DPR-RFFI protocol result files were found.")
    frame = pd.DataFrame(rows)
    metrics = [
        "h_score",
        "oscr",
        "auroc",
        "unknown_rejection_rate",
        "accuracy",
        "false_rejection_rate",
    ]
    summary = frame.groupby("dataset")[metrics].agg(["mean", "std"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
