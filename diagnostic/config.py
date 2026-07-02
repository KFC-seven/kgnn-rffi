from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must be JSON-compatible YAML for the M0 tools. "
            "Use a JSON object even if the file extension is .yaml."
        ) from exc
