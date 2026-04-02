from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load

from experiments.common import evaluate_extractor, train_extractor
from models.interfaces import parse_interface


def _build_prompt_pool() -> list[str]:
    subjects = ["physics", "math", "biology", "history", "economics", "music", "coding", "ethics"]
    styles = ["one-word", "short", "concise", "direct"]
    pool = []
    for s in subjects:
        for st in styles:
            pool.append(f"Topic: {s}. Provide a {st} continuation:")
            pool.append(f"In {s}, the key idea is")
            pool.append(f"Question about {s}: the answer starts with")
    return pool


@dataclass
class QwenVictim:
    model_name: str = "mlx-community/Qwen2-0.5B-Instruct-4bit"

    def __post_init__(self):
        self.model, self.tokenizer = load(self.model_name)
        self.prompt_pool = _build_prompt_pool()
        self.vocab_size = int(self.tokenizer.vocab_size)

    def _context_id(self, prompt: str) -> int:
        ids = self.tokenizer.encode(prompt)
        return int(ids[-1]) if ids else 0

    def sample_train_prompt(self, rng: np.random.Generator):
        prompt = self.prompt_pool[int(rng.integers(0, len(self.prompt_pool)))]
        return self._context_id(prompt), prompt

    def sample_eval_prompt(self, rng: np.random.Generator):
        prompt = self.prompt_pool[int(rng.integers(0, len(self.prompt_pool)))]
        return self._context_id(prompt), prompt

    def query_probs(self, prompt: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(prompt)
        arr = mx.array([input_ids])
        logits = self.model(arr)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_logits = np.array(logits[0, -1, :], dtype=np.float64)
        probs = np.exp(next_logits - np.max(next_logits))
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        return probs


def run_qwen_budget_sweep(budgets, seeds, interfaces=("argmax", "topk", "probs"), model_name: str = "mlx-community/Qwen2-0.5B-Instruct-4bit"):
    rows = []
    victim = QwenVictim(model_name=model_name)
    for interface_name in interfaces:
        interface = parse_interface(interface_name)
        for budget in budgets:
            for seed in seeds:
                extractor = train_extractor(victim=victim, interface=interface, budget=budget, seed=seed)
                metrics = evaluate_extractor(victim=victim, extractor=extractor, seed=seed, eval_size=96)
                rows.append(
                    {
                        "source": "qwen2-0.5b",
                        "interface": interface_name,
                        "budget": int(budget),
                        "seed": int(seed),
                        "agreement": metrics.agreement,
                        "kl_divergence": metrics.kl_divergence,
                    }
                )
    return pd.DataFrame(rows)


def run_qwen_topk_sweep(fixed_budget: int, seeds, interfaces=("argmax", "top2", "top3", "top5", "probs"), model_name: str = "mlx-community/Qwen2-0.5B-Instruct-4bit"):
    rows = []
    victim = QwenVictim(model_name=model_name)
    for interface_name in interfaces:
        interface = parse_interface(interface_name)
        for seed in seeds:
            extractor = train_extractor(victim=victim, interface=interface, budget=fixed_budget, seed=seed)
            metrics = evaluate_extractor(victim=victim, extractor=extractor, seed=seed, eval_size=96)
            rows.append(
                {
                    "source": "qwen2-0.5b-topk-sweep",
                    "interface": interface_name,
                    "budget": int(fixed_budget),
                    "seed": int(seed),
                    "agreement": metrics.agreement,
                    "kl_divergence": metrics.kl_divergence,
                }
            )
    return pd.DataFrame(rows)
