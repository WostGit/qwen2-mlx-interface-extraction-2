from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import numpy as np
from mlx_lm import load

from attacks.extractors import InterfaceObservation


@dataclass
class QwenVictim:
    model_id: str = "Qwen/Qwen2-0.5B"
    vocab_cap: int = 32768

    def __post_init__(self) -> None:
        self.model, self.tokenizer = load(self.model_id)

    def _encode(self, text: str) -> np.ndarray:
        if hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text)
        else:
            ids = self.tokenizer(text)["input_ids"]
        if len(ids) == 0:
            ids = [self.tokenizer.eos_token_id]
        return np.asarray(ids, dtype=np.int32)

    def _forward_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        x = mx.array(np.asarray(token_ids, dtype=np.int32)[None, :])
        out = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        logits_np = np.array(logits)
        next_logits = logits_np[0, -1]
        if self.vocab_cap and self.vocab_cap < next_logits.shape[0]:
            next_logits = next_logits[: self.vocab_cap]
        return next_logits

    def bucket_from_query(self, text: str) -> int:
        token_ids = self._encode(text)
        return int(token_ids[-1] % self.vocab_cap)

    def probs(self, text: str) -> np.ndarray:
        logits = self._forward_logits(self._encode(text))
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def query(self, text: str, interface: str, topk: int = 2) -> InterfaceObservation:
        p = self.probs(text)
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
            return InterfaceObservation(interface=interface, probs=p)
        raise ValueError(interface)
