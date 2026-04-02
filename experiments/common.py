from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_interface(interface: str) -> tuple[str, int]:
    if interface == "argmax":
        return interface, 1
    if interface == "probs":
        return interface, 0
    if interface.startswith("top"):
        return interface, int(interface.replace("top", ""))
    raise ValueError(interface)


def rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "source",
        "interface",
        "budget",
        "seed",
        "agreement",
        "kl_divergence",
    ])
