from __future__ import annotations

from typing import Iterable, List

import numpy as np

from attacks.extractors import BucketedExtractor, evaluate
from experiments.common import parse_interface, set_seed
from models.qwen_victim import QwenVictim

PROMPTS = [
    "The capital of France is",
    "A quick brown fox jumps",
    "Machine learning is",
    "In one sentence, explain gravity:",
    "The most important thing about testing code is",
    "Python developers often use",
    "The weather today feels",
    "To make tea, first",
    "Open source software enables",
    "My favorite algorithm for sorting is",
]


def run_qwen(
    budgets: Iterable[int],
    interfaces: Iterable[str],
    seeds: Iterable[int],
    eval_samples: int = 64,
    model_id: str = "Qwen/Qwen2-0.5B",
    vocab_cap: int = 32768,
) -> List[dict]:
    rows: List[dict] = []
    victim = QwenVictim(model_id=model_id, vocab_cap=vocab_cap)

    for seed in seeds:
        set_seed(seed)
        rng = np.random.default_rng(seed + 101)
        eval_prompts = [PROMPTS[i] for i in rng.integers(0, len(PROMPTS), size=eval_samples)]
        eval_items = [
            (victim.bucket_from_query(p), victim.probs(p))
            for p in eval_prompts
        ]

        for interface in interfaces:
            _, k = parse_interface(interface)
            for budget in budgets:
                extractor = BucketedExtractor(vocab_size=vocab_cap)
                query_prompts = [PROMPTS[i] for i in rng.integers(0, len(PROMPTS), size=budget)]
                for prompt in query_prompts:
                    bucket = victim.bucket_from_query(prompt)
                    obs = victim.query(prompt, interface=interface, topk=k)
                    extractor.update(bucket, obs)

                agreement, kl = evaluate(extractor, eval_items)
                rows.append(
                    {
                        "source": "qwen2-0.5b",
                        "interface": interface,
                        "budget": int(budget),
                        "seed": int(seed),
                        "agreement": agreement,
                        "kl_divergence": kl,
                    }
                )
    return rows
