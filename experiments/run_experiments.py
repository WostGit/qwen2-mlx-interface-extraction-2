from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from attacks.extractors import StudentExtractor, kl_divergence, top1_agreement
from models.victim_api import QwenVictim, ToyVictim, build_qwen_context_pool


@dataclass
class RunRecord:
    source: str
    interface: str
    budget: int
    seed: int
    agreement: float
    kl_divergence: float


def _contexts_to_ids(contexts: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(contexts)}


def _evaluate_student(
    extractor: StudentExtractor,
    contexts: List[str],
    ctx_ids: Dict[str, int],
    victim_full,
) -> tuple[float, float]:
    agreements = []
    kls = []
    for c in contexts:
        p = victim_full(c)
        q = extractor.predict_dist(ctx_ids[c])
        agreements.append(top1_agreement(p, q))
        kls.append(kl_divergence(p, q))
    return float(np.mean(agreements)), float(np.mean(kls))


def run_toy(config: dict, seed: int, budgets: List[int], out_rows: List[RunRecord]) -> None:
    toy_cfg = config["toy"]
    rng = np.random.default_rng(seed)
    context_pool = [f"toy_ctx_{i}" for i in range(toy_cfg["context_pool_size"])]
    eval_contexts = context_pool[: toy_cfg["eval_size"]]
    ctx_ids = _contexts_to_ids(context_pool)
    victim = ToyVictim(vocab_size=toy_cfg["vocab_size"], seed=seed)

    interface_specs = [("argmax", None), ("topk", int(toy_cfg["train_topk"])), ("probs", None)]
    for interface, topk in interface_specs:
        extractor = StudentExtractor(interface=interface, vocab_size=toy_cfg["vocab_size"])
        max_budget = max(budgets)
        for step in range(1, max_budget + 1):
            c = context_pool[int(rng.integers(0, len(context_pool)))]
            cid = ctx_ids[c]
            response = victim.query(context_id=cid, interface=interface, topk=topk or 2)
            extractor.observe(cid, response)
            if step in budgets:
                agreement, kl = _evaluate_student(
                    extractor,
                    eval_contexts,
                    ctx_ids,
                    lambda x: victim.full_probs(ctx_ids[x]),
                )
                out_rows.append(
                    RunRecord(
                        source="toy",
                        interface=interface,
                        budget=step,
                        seed=seed,
                        agreement=agreement,
                        kl_divergence=kl,
                    )
                )


def run_qwen(config: dict, seed: int, budgets: List[int], out_rows: List[RunRecord]) -> None:
    qcfg = config["qwen"]
    rng = np.random.default_rng(seed)
    contexts = build_qwen_context_pool(qcfg["context_pool_size"])
    eval_contexts = contexts[: qcfg["eval_size"]]
    ctx_ids = _contexts_to_ids(contexts)
    victim = QwenVictim(model_name=qcfg["model_name"])

    sample_probs = victim.full_probs(eval_contexts[0])
    vocab_size = int(sample_probs.shape[0])
    interface_specs = [("argmax", None), ("topk", int(qcfg["train_topk"])), ("probs", None)]
    for interface, topk in interface_specs:
        extractor = StudentExtractor(interface=interface, vocab_size=vocab_size)
        max_budget = max(budgets)
        for step in range(1, max_budget + 1):
            c = contexts[int(rng.integers(0, len(contexts)))]
            cid = ctx_ids[c]
            response = victim.query(prompt=c, interface=interface, topk=topk or 2)
            extractor.observe(cid, response)
            if step in budgets:
                agreement, kl = _evaluate_student(extractor, eval_contexts, ctx_ids, victim.full_probs)
                out_rows.append(
                    RunRecord(
                        source="qwen2-0.5b",
                        interface=interface,
                        budget=step,
                        seed=seed,
                        agreement=agreement,
                        kl_divergence=kl,
                    )
                )


def run_qwen_topk_sweep(config: dict, seed: int, out_rows: List[RunRecord]) -> None:
    qcfg = config["qwen"]
    rng = np.random.default_rng(seed)
    contexts = build_qwen_context_pool(qcfg["context_pool_size"])
    eval_contexts = contexts[: qcfg["eval_size"]]
    ctx_ids = _contexts_to_ids(contexts)
    victim = QwenVictim(model_name=qcfg["model_name"])

    vocab_size = int(victim.full_probs(eval_contexts[0]).shape[0])
    fixed_budget = int(qcfg["fixed_budget"])
    sweep = [("argmax", None)] + [(f"top{k}", k) for k in qcfg["topk_sweep"]] + [("probs", None)]

    for label, k in sweep:
        interface = "topk" if label.startswith("top") else label
        extractor = StudentExtractor(interface=interface, vocab_size=vocab_size)
        for _ in range(fixed_budget):
            c = contexts[int(rng.integers(0, len(contexts)))]
            cid = ctx_ids[c]
            response = victim.query(prompt=c, interface=interface, topk=k or 2)
            extractor.observe(cid, response)
        agreement, kl = _evaluate_student(extractor, eval_contexts, ctx_ids, victim.full_probs)
        out_rows.append(
            RunRecord(
                source="qwen2-0.5b-topk-sweep",
                interface=label,
                budget=fixed_budget,
                seed=seed,
                agreement=agreement,
                kl_divergence=kl,
            )
        )


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_all(config_path: str | Path, output_dir: str | Path) -> dict:
    cfg = load_config(config_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in cfg["budgets"]]

    rows: List[RunRecord] = []
    for seed in cfg["seed_list"]:
        run_toy(cfg, int(seed), budgets, rows)
        run_qwen(cfg, int(seed), budgets, rows)
        run_qwen_topk_sweep(cfg, int(seed), rows)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(Path(output_dir) / "results_raw.csv", index=False)

    day1 = df[df["source"].isin(["toy", "qwen2-0.5b"])].copy()
    summary = (
        day1.groupby(["source", "interface", "budget"], as_index=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
        )
        .fillna(0.0)
    )
    summary.to_csv(Path(output_dir) / "results_summary.csv", index=False)

    topk = df[df["source"] == "qwen2-0.5b-topk-sweep"].copy()
    topk_summary = (
        topk.groupby(["interface", "budget"], as_index=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
        )
        .fillna(0.0)
    )
    topk_summary.to_csv(Path(output_dir) / "qwen_topk_sweep_summary.csv", index=False)

    return {
        "raw": str(Path(output_dir) / "results_raw.csv"),
        "summary": str(Path(output_dir) / "results_summary.csv"),
        "topk_summary": str(Path(output_dir) / "qwen_topk_sweep_summary.csv"),
    }
