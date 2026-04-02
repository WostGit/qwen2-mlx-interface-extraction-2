from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NgramExtractor:
    vocab_size: int
    smoothing: float = 1e-3

    def __post_init__(self) -> None:
        self.global_counts = np.full(self.vocab_size, self.smoothing, dtype=np.float64)
        self.by_context: dict[int, np.ndarray] = {}

    def _ctx_counts(self, context_id: int) -> np.ndarray:
        if context_id not in self.by_context:
            self.by_context[context_id] = np.full(self.vocab_size, self.smoothing, dtype=np.float64)
        return self.by_context[context_id]

    def update_from_argmax(self, context_id: int, token_id: int) -> None:
        self._ctx_counts(context_id)[token_id] += 1.0
        self.global_counts[token_id] += 1.0

    def update_from_sparse(self, context_id: int, token_ids: np.ndarray, probs: np.ndarray) -> None:
        ctx = self._ctx_counts(context_id)
        ctx[token_ids] += probs
        self.global_counts[token_ids] += probs

    def update_from_probs(self, context_id: int, probs: np.ndarray) -> None:
        self._ctx_counts(context_id)[:] += probs
        self.global_counts[:] += probs

    def predict(self, context_id: int) -> np.ndarray:
        if context_id in self.by_context:
            dist = self.by_context[context_id]
        else:
            dist = self.global_counts
        return dist / np.clip(dist.sum(), 1e-12, None)
