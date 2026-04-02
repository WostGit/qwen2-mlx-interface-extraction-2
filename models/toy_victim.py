from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attacks.extractors import InterfaceObservation


@dataclass
class ToyVictim:
    vocab_size: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        logits = rng.normal(0.0, 1.0, size=(self.vocab_size, self.vocab_size))
        self.transition_probs = _softmax(logits, axis=-1)

    def bucket_from_query(self, query_token: int) -> int:
        return int(query_token)

    def probs(self, query_token: int) -> np.ndarray:
        return self.transition_probs[self.bucket_from_query(query_token)]

    def query(self, query_token: int, interface: str, topk: int = 2) -> InterfaceObservation:
        p = self.probs(query_token)
        if interface == "argmax":
            return InterfaceObservation(interface=interface, argmax_token=int(np.argmax(p)))
        if interface.startswith("top"):
            k = topk
            idx = np.argpartition(-p, k)[:k]
            idx = idx[np.argsort(-p[idx])]
            top_probs = p[idx]
            top_probs = top_probs / top_probs.sum()
            return InterfaceObservation(
                interface=interface,
                topk_tokens=idx.astype(int),
                topk_probs=top_probs,
            )
        if interface == "probs":
            return InterfaceObservation(interface=interface, probs=p.copy())
        raise ValueError(interface)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=axis, keepdims=True)
