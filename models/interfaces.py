from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class InterfaceSpec:
    name: str
    top_k: int | None


def parse_interface(name: str) -> InterfaceSpec:
    if name == "argmax":
        return InterfaceSpec(name="argmax", top_k=1)
    if name == "topk":
        return InterfaceSpec(name="topk", top_k=5)
    if name.startswith("top") and name[3:].isdigit():
        return InterfaceSpec(name=name, top_k=int(name[3:]))
    if name == "probs":
        return InterfaceSpec(name="probs", top_k=None)
    raise ValueError(f"Unsupported interface: {name}")


def compress_distribution(probs: np.ndarray, interface: InterfaceSpec) -> Dict[str, np.ndarray | int]:
    if interface.name == "argmax":
        return {"token": int(np.argmax(probs))}

    if interface.name == "probs":
        return {"probs": probs.astype(np.float64, copy=True)}

    if interface.top_k is None:
        raise ValueError("top_k cannot be None for top-k interfaces")

    k = min(interface.top_k, probs.shape[-1])
    idx = np.argpartition(-probs, k - 1)[:k]
    sorted_idx = idx[np.argsort(probs[idx])[::-1]]
    top_probs = probs[sorted_idx]
    top_probs = top_probs / np.clip(top_probs.sum(), 1e-12, None)
    return {"tokens": sorted_idx.astype(np.int64), "probs": top_probs.astype(np.float64)}
