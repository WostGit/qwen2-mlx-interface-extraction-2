from __future__ import annotations

from typing import Iterable, List

import numpy as np

from attacks.extractors import BucketedExtractor, evaluate
from experiments.common import parse_interface, set_seed
from models.toy_victim import ToyVictim


def run_toy(
    budgets: Iterable[int],
    interfaces: Iterable[str],
    seeds: Iterable[int],
    vocab_size: int = 32,
    eval_samples: int = 128,
) -> List[dict]:
    rows: List[dict] = []
    for seed in seeds:
        set_seed(seed)
        victim = ToyVictim(vocab_size=vocab_size, seed=seed)
        rng = np.random.default_rng(seed + 11)
        eval_tokens = rng.integers(0, vocab_size, size=eval_samples)
        eval_items = [(int(t), victim.probs(int(t))) for t in eval_tokens]

        for interface in interfaces:
            _, k = parse_interface(interface)
            for budget in budgets:
                extractor = BucketedExtractor(vocab_size=vocab_size)
                query_tokens = rng.integers(0, vocab_size, size=budget)
                for token in query_tokens:
                    obs = victim.query(int(token), interface=interface, topk=k)
                    extractor.update(victim.bucket_from_query(int(token)), obs)

                agreement, kl = evaluate(extractor, eval_items)
                rows.append(
                    {
                        "source": "toy",
                        "interface": interface,
                        "budget": int(budget),
                        "seed": int(seed),
                        "agreement": agreement,
                        "kl_divergence": kl,
                    }
                )
    return rows
