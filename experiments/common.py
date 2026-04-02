from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attacks.ngram_extractor import NgramExtractor
from models.interfaces import InterfaceSpec, compress_distribution


@dataclass
class EvalResult:
    agreement: float
    kl_divergence: float


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def train_extractor(victim, interface: InterfaceSpec, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    extractor = NgramExtractor(vocab_size=victim.vocab_size)
    for _ in range(budget):
        context_id, prompt = victim.sample_train_prompt(rng)
        full_probs = victim.query_probs(prompt)
        payload = compress_distribution(full_probs, interface)
        if interface.name == "argmax":
            extractor.update_from_argmax(context_id=context_id, token_id=payload["token"])
        elif interface.name == "probs":
            extractor.update_from_probs(context_id=context_id, probs=payload["probs"])
        else:
            extractor.update_from_sparse(context_id=context_id, token_ids=payload["tokens"], probs=payload["probs"])
    return extractor


def evaluate_extractor(victim, extractor, seed: int, eval_size: int = 128) -> EvalResult:
    rng = np.random.default_rng(seed + 10000)
    agreements = []
    kls = []
    for _ in range(eval_size):
        context_id, prompt = victim.sample_eval_prompt(rng)
        true_probs = victim.query_probs(prompt)
        pred_probs = extractor.predict(context_id)
        agreements.append(int(np.argmax(true_probs) == np.argmax(pred_probs)))
        kls.append(kl_divergence(true_probs, pred_probs))
    return EvalResult(agreement=float(np.mean(agreements)), kl_divergence=float(np.mean(kls)))
