from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.victim_interfaces import InterfaceSpec


@dataclass
class ExtractionResult:
    estimated_probs: np.ndarray


class FrequencyExtractor:
    def __init__(self, num_sources: int, vocab_size: int):
        self.counts = np.zeros((num_sources, vocab_size), dtype=float)

    def ingest(self, source_id: int, response: dict, interface: InterfaceSpec) -> None:
        if interface.name == "argmax":
            self.counts[source_id, response["token"]] += 1.0
            return
        if interface.name == "topk":
            token_ids = response["token_ids"]
            probs = response["probs"]
            self.counts[source_id, token_ids] += probs
            return
        if interface.name == "probs":
            self.counts[source_id] += response["probs"]
            return
        raise ValueError(f"Unsupported interface: {interface.name}")

    def estimate(self) -> ExtractionResult:
        row_sums = self.counts.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0.0, 1.0, row_sums)
        est = self.counts / safe
        uniform_rows = (row_sums[:, 0] == 0.0)
        if np.any(uniform_rows):
            est[uniform_rows] = 1.0 / est.shape[1]
        return ExtractionResult(estimated_probs=est)
