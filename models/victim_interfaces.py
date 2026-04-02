from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from attacks.metrics import stable_softmax


@dataclass(frozen=True)
class InterfaceSpec:
    name: str
    top_k: int | None = None


class ToyVictim:
    def __init__(self, vocab_size: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(vocab_size, vocab_size))
        self.transition_probs = np.apply_along_axis(stable_softmax, 1, raw)
        self.vocab_size = vocab_size

    def source_count(self) -> int:
        return self.vocab_size

    def true_distribution(self, source_id: int) -> np.ndarray:
        return self.transition_probs[source_id]

    def query(self, source_id: int, interface: InterfaceSpec) -> Dict[str, np.ndarray | int]:
        probs = self.true_distribution(source_id)
        if interface.name == "argmax":
            return {"token": int(np.argmax(probs))}
        if interface.name == "topk":
            if interface.top_k is None:
                raise ValueError("topk interface requires top_k")
            top_ids = np.argsort(-probs)[: interface.top_k]
            return {"token_ids": top_ids.astype(int), "probs": probs[top_ids]}
        if interface.name == "probs":
            return {"probs": probs.copy()}
        raise ValueError(f"Unsupported interface: {interface.name}")


class QwenVictim:
    def __init__(self, model_name: str, prompts: Sequence[str], candidate_size: int = 128):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_name)
        self.prompts = list(prompts)
        self.candidate_ids = self._build_candidate_set(candidate_size)
        self.true_probs_matrix = np.vstack([self._next_token_probs(prompt) for prompt in self.prompts])

    def source_count(self) -> int:
        return len(self.prompts)

    def _next_token_logits(self, prompt: str) -> np.ndarray:
        import mlx.core as mx

        token_ids = self.tokenizer.encode(prompt)
        x = mx.array([token_ids])
        logits = self.model(x)
        next_logits = np.array(logits[0, -1, :])
        return next_logits

    def _build_candidate_set(self, candidate_size: int) -> np.ndarray:
        all_ids: List[int] = []
        per_prompt_topk = max(8, candidate_size // max(1, len(self.prompts)))
        for prompt in self.prompts:
            logits = self._next_token_logits(prompt)
            top_ids = np.argsort(-logits)[:per_prompt_topk]
            all_ids.extend([int(t) for t in top_ids])
        unique = np.array(sorted(set(all_ids)), dtype=int)
        if len(unique) > candidate_size:
            unique = unique[:candidate_size]
        return unique

    def _next_token_probs(self, prompt: str) -> np.ndarray:
        logits = self._next_token_logits(prompt)[self.candidate_ids]
        return stable_softmax(logits)

    def true_distribution(self, source_id: int) -> np.ndarray:
        return self.true_probs_matrix[source_id]

    def query(self, source_id: int, interface: InterfaceSpec) -> Dict[str, np.ndarray | int]:
        probs = self.true_distribution(source_id)
        if interface.name == "argmax":
            return {"token": int(np.argmax(probs))}
        if interface.name == "topk":
            if interface.top_k is None:
                raise ValueError("topk interface requires top_k")
            top_ids = np.argsort(-probs)[: interface.top_k]
            return {"token_ids": top_ids.astype(int), "probs": probs[top_ids]}
        if interface.name == "probs":
            return {"probs": probs.copy()}
        raise ValueError(f"Unsupported interface: {interface.name}")


def default_qwen_prompts() -> List[str]:
    return [
        "The capital of France is",
        "In machine learning, overfitting happens when",
        "A recipe for pancakes usually starts with",
        "Python is a programming language that",
        "The Pacific Ocean is",
        "To improve battery life on a laptop,",
        "The theory of evolution explains",
        "When writing tests, it is important to",
        "A healthy breakfast can include",
        "The purpose of encryption is to",
        "In astronomy, a supernova is",
        "For effective teamwork, communication should",
        "Climate change policies often focus on",
        "Database indexing helps by",
        "In economics, inflation refers to",
        "The process of photosynthesis requires",
    ]
