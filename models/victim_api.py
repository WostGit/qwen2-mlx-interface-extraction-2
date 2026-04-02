from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.sum(exp)


@dataclass
class VictimResponse:
    interface: str
    argmax_id: int | None = None
    topk_ids: np.ndarray | None = None
    topk_probs: np.ndarray | None = None
    probs: np.ndarray | None = None


class ToyVictim:
    def __init__(self, vocab_size: int = 20, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.context_logits: Dict[int, np.ndarray] = {}

    def _get_logits(self, context_id: int) -> np.ndarray:
        if context_id not in self.context_logits:
            base = self.rng.normal(0.0, 1.0, self.vocab_size)
            base[(context_id * 7) % self.vocab_size] += 2.25
            self.context_logits[context_id] = base
        return self.context_logits[context_id]

    def full_probs(self, context_id: int) -> np.ndarray:
        return _softmax_np(self._get_logits(context_id))

    def query(self, context_id: int, interface: str, topk: int = 2) -> VictimResponse:
        probs = self.full_probs(context_id)
        if interface == "argmax":
            return VictimResponse(interface=interface, argmax_id=int(np.argmax(probs)))
        if interface == "topk":
            idx = np.argsort(-probs)[:topk]
            return VictimResponse(interface=interface, topk_ids=idx, topk_probs=probs[idx])
        if interface == "probs":
            return VictimResponse(interface=interface, probs=probs)
        raise ValueError(f"Unsupported interface: {interface}")


class QwenVictim:
    def __init__(self, model_name: str = "mlx-community/Qwen2-0.5B-Instruct-4bit") -> None:
        from mlx_lm import load  # lazy import keeps module import cheap

        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

    def _logits_for_prompt(self, prompt: str) -> np.ndarray:
        import mlx.core as mx

        token_ids: Sequence[int] = self.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = self.tokenizer.encode(" ")
        x = mx.array([list(token_ids)], dtype=mx.int32)
        logits = self.model(x)
        last = logits[0, -1, :]
        probs = mx.softmax(last, axis=-1)
        return np.array(probs)

    def full_probs(self, prompt: str) -> np.ndarray:
        return self._logits_for_prompt(prompt)

    def query(self, prompt: str, interface: str, topk: int = 2) -> VictimResponse:
        probs = self.full_probs(prompt)
        if interface == "argmax":
            return VictimResponse(interface=interface, argmax_id=int(np.argmax(probs)))
        if interface == "topk":
            idx = np.argsort(-probs)[:topk]
            return VictimResponse(interface=interface, topk_ids=idx, topk_probs=probs[idx])
        if interface == "probs":
            return VictimResponse(interface=interface, probs=probs)
        raise ValueError(f"Unsupported interface: {interface}")


def build_qwen_context_pool(n: int = 64) -> List[str]:
    stems = [
        "The capital of France is",
        "In one sentence, explain gravity:",
        "Python lists are useful because",
        "A healthy breakfast might include",
        "The opposite of hot is",
        "When debugging code, first",
        "The largest planet in our solar system is",
        "Machine learning models generalize when",
    ]
    tails = [
        " today",
        " for students",
        " in simple words",
        " briefly",
        " with an example",
        " and why",
        " in two words",
        " right now",
    ]
    prompts: List[str] = []
    i = 0
    while len(prompts) < n:
        prompts.append(f"{stems[i % len(stems)]}{tails[(i // len(stems)) % len(tails)]}")
        i += 1
    return prompts
