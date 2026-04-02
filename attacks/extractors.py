from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from models.victim_api import VictimResponse


@dataclass
class ContextEstimate:
    seen: int = 0
    winner: int | None = None
    token_mass: Dict[int, float] = field(default_factory=dict)
    probs_mean: np.ndarray | None = None


class StudentExtractor:
    def __init__(self, interface: str, vocab_size: int, smoothing: float = 1e-6) -> None:
        self.interface = interface
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ctx: Dict[int, ContextEstimate] = {}
        self.global_mass = np.ones(vocab_size, dtype=np.float64)

    def _get(self, context_id: int) -> ContextEstimate:
        if context_id not in self.ctx:
            self.ctx[context_id] = ContextEstimate()
        return self.ctx[context_id]

    def observe(self, context_id: int, response: VictimResponse) -> None:
        slot = self._get(context_id)
        slot.seen += 1
        if self.interface == "argmax":
            assert response.argmax_id is not None
            slot.winner = response.argmax_id
            self.global_mass[response.argmax_id] += 1.0
        elif self.interface == "topk":
            assert response.topk_ids is not None and response.topk_probs is not None
            for idx, p in zip(response.topk_ids, response.topk_probs):
                slot.token_mass[int(idx)] = slot.token_mass.get(int(idx), 0.0) + float(p)
                self.global_mass[int(idx)] += float(p)
        elif self.interface == "probs":
            assert response.probs is not None
            vec = response.probs.astype(np.float64)
            self.global_mass += vec
            if slot.probs_mean is None:
                slot.probs_mean = vec.copy()
            else:
                alpha = 1.0 / slot.seen
                slot.probs_mean = (1.0 - alpha) * slot.probs_mean + alpha * vec
        else:
            raise ValueError(self.interface)

    def predict_dist(self, context_id: int) -> np.ndarray:
        slot = self._get(context_id)
        prior = self.global_mass / np.sum(self.global_mass)
        if self.interface == "argmax":
            dist = np.full(self.vocab_size, self.smoothing, dtype=np.float64)
            if slot.winner is None:
                return prior
            dist[slot.winner] = 1.0
            dist /= np.sum(dist)
            return dist
        if self.interface == "topk":
            dist = prior * 0.25
            if slot.token_mass:
                total = sum(slot.token_mass.values())
                for idx, mass in slot.token_mass.items():
                    dist[idx] += 0.75 * (mass / total)
            dist /= np.sum(dist)
            return dist
        if self.interface == "probs":
            if slot.probs_mean is None:
                return prior
            dist = slot.probs_mean + self.smoothing
            dist /= np.sum(dist)
            return dist
        raise ValueError(self.interface)


def top1_agreement(p: np.ndarray, q: np.ndarray) -> float:
    return float(int(np.argmax(p) == np.argmax(q)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p2 = p + eps
    q2 = q + eps
    return float(np.sum(p2 * (np.log(p2) - np.log(q2))))
