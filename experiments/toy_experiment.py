from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import evaluate_extractor, train_extractor
from models.interfaces import parse_interface


class ToyVictim:
    def __init__(self, vocab_size: int = 32, n_contexts: int = 16, seed: int = 7):
        self.vocab_size = vocab_size
        self.n_contexts = n_contexts
        rng = np.random.default_rng(seed)
        self.table = rng.dirichlet(np.ones(vocab_size), size=n_contexts)

    def sample_train_prompt(self, rng: np.random.Generator):
        context_id = int(rng.integers(0, self.n_contexts))
        return context_id, f"toy_context_{context_id}"

    def sample_eval_prompt(self, rng: np.random.Generator):
        context_id = int(rng.integers(0, self.n_contexts))
        return context_id, f"toy_eval_context_{context_id}"

    def query_probs(self, prompt: str) -> np.ndarray:
        context_id = int(prompt.split("_")[-1])
        return self.table[context_id]


def run_toy_budget_sweep(budgets, seeds, interfaces=("argmax", "topk", "probs")) -> pd.DataFrame:
    rows = []
    victim = ToyVictim()
    for interface_name in interfaces:
        interface = parse_interface(interface_name)
        for budget in budgets:
            for seed in seeds:
                extractor = train_extractor(victim=victim, interface=interface, budget=budget, seed=seed)
                metrics = evaluate_extractor(victim=victim, extractor=extractor, seed=seed)
                rows.append(
                    {
                        "source": "toy",
                        "interface": interface_name,
                        "budget": int(budget),
                        "seed": int(seed),
                        "agreement": metrics.agreement,
                        "kl_divergence": metrics.kl_divergence,
                    }
                )
    return pd.DataFrame(rows)
