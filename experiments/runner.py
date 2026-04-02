from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from attacks.extractors import FrequencyExtractor
from attacks.metrics import mean_kl, top1_agreement
from models.victim_interfaces import InterfaceSpec, QwenVictim, ToyVictim, default_qwen_prompts


@dataclass
class RunConfig:
    budgets: List[int]
    seeds: List[int]
    qwen_model: str
    qwen_prompt_limit: int
    default_topk: int
    fixed_topk_budget: int
    fixed_topk_interfaces: List[str]
    output_dir: Path


def parse_interface(name: str, default_topk: int) -> InterfaceSpec:
    if name == "argmax":
        return InterfaceSpec(name="argmax")
    if name == "probs":
        return InterfaceSpec(name="probs")
    if name == "topk":
        return InterfaceSpec(name="topk", top_k=default_topk)
    if name.startswith("top"):
        return InterfaceSpec(name="topk", top_k=int(name.replace("top", "")))
    raise ValueError(f"Unknown interface: {name}")


def run_single(victim, interface: InterfaceSpec, budget: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    num_sources = victim.source_count()
    vocab_size = len(victim.true_distribution(0))
    extractor = FrequencyExtractor(num_sources=num_sources, vocab_size=vocab_size)

    for _ in range(budget):
        source_id = int(rng.integers(0, num_sources))
        response = victim.query(source_id, interface)
        extractor.ingest(source_id, response, interface)

    est = extractor.estimate().estimated_probs
    truth = np.vstack([victim.true_distribution(i) for i in range(num_sources)])
    return top1_agreement(truth, est), mean_kl(truth, est)


def run_budget_sweep(source_name: str, victim_factory, interfaces: Iterable[str], budgets: List[int], seeds: List[int], default_topk: int) -> pd.DataFrame:
    rows = []
    for interface_name in interfaces:
        interface = parse_interface(interface_name, default_topk=default_topk)
        for budget in budgets:
            for seed in seeds:
                victim = victim_factory(seed)
                agreement, kl = run_single(victim=victim, interface=interface, budget=budget, seed=seed)
                rows.append(
                    {
                        "source": source_name,
                        "interface": interface_name,
                        "budget": budget,
                        "seed": seed,
                        "agreement": agreement,
                        "kl_divergence": kl,
                    }
                )
    return pd.DataFrame(rows)


def run_day1(config: RunConfig) -> dict[str, pd.DataFrame]:
    prompts = default_qwen_prompts()[: config.qwen_prompt_limit]

    toy_df = run_budget_sweep(
        source_name="toy",
        victim_factory=lambda seed: ToyVictim(vocab_size=32, seed=seed),
        interfaces=["argmax", "topk", "probs"],
        budgets=config.budgets,
        seeds=config.seeds,
        default_topk=config.default_topk,
    )

    qwen_df = run_budget_sweep(
        source_name="qwen2-0.5b",
        victim_factory=lambda _seed: QwenVictim(model_name=config.qwen_model, prompts=prompts, candidate_size=128),
        interfaces=["argmax", "topk", "probs"],
        budgets=config.budgets,
        seeds=config.seeds,
        default_topk=config.default_topk,
    )

    qwen_topk_df = run_budget_sweep(
        source_name="qwen2-0.5b",
        victim_factory=lambda _seed: QwenVictim(model_name=config.qwen_model, prompts=prompts, candidate_size=128),
        interfaces=config.fixed_topk_interfaces,
        budgets=[config.fixed_topk_budget],
        seeds=config.seeds,
        default_topk=config.default_topk,
    )

    return {
        "budget_sweep": pd.concat([toy_df, qwen_df], ignore_index=True),
        "qwen_topk_sweep": qwen_topk_df,
    }
