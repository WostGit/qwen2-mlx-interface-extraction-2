from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class InterfaceObservation:
    interface: str
    argmax_token: int | None = None
    topk_tokens: np.ndarray | None = None
    topk_probs: np.ndarray | None = None
    probs: np.ndarray | None = None


@dataclass
class BucketedExtractor:
    """Simple frequency-based extractor keyed by context buckets."""

    vocab_size: int
    smoothing: float = 1e-5
    table: Dict[int, np.ndarray] = field(default_factory=dict)

    def _row(self, bucket: int) -> np.ndarray:
        if bucket not in self.table:
            self.table[bucket] = np.full(self.vocab_size, self.smoothing, dtype=np.float64)
        return self.table[bucket]

    def update(self, bucket: int, obs: InterfaceObservation) -> None:
        row = self._row(bucket)
        if obs.interface == "argmax":
            row[obs.argmax_token] += 1.0
        elif obs.interface.startswith("top"):
            row[obs.topk_tokens] += obs.topk_probs
        elif obs.interface == "probs":
            row += obs.probs
        else:
            raise ValueError(f"Unknown interface: {obs.interface}")

    def predict_probs(self, bucket: int) -> np.ndarray:
        row = self._row(bucket)
        return row / row.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def evaluate(
    extractor: BucketedExtractor,
    eval_items: Iterable[Tuple[int, np.ndarray]],
) -> Tuple[float, float]:
    agreements: List[float] = []
    kls: List[float] = []
    for bucket, victim_probs in eval_items:
        student_probs = extractor.predict_probs(bucket)
        agreements.append(float(np.argmax(student_probs) == np.argmax(victim_probs)))
        kls.append(kl_divergence(victim_probs, student_probs))
    return float(np.mean(agreements)), float(np.mean(kls))
