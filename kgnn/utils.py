"""Utility functions extracted from run_method_gate_v4.py for the KGNN-RFFI release."""

import torch


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested


def _configure_torch_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select_protocol(manifest: dict, name: str) -> dict:
    for protocol in manifest["protocols"]:
        if protocol["name"] == name:
            return protocol
    raise KeyError(name)


def _select_split(protocol: dict, split_id: int) -> dict:
    for split in protocol["tx_splits"]:
        if int(split["split_id"]) == int(split_id):
            return split
    raise KeyError(split_id)
