from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def mean_kl(true_probs: np.ndarray, est_probs: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean([kl_divergence(t, e, eps=eps) for t, e in zip(true_probs, est_probs)]))


def top1_agreement(true_probs: np.ndarray, est_probs: np.ndarray) -> float:
    true_argmax = np.argmax(true_probs, axis=1)
    est_argmax = np.argmax(est_probs, axis=1)
    return float(np.mean(true_argmax == est_argmax))
