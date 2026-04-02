from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["source", "interface", "budget"], as_index=False)
        .agg(agreement_mean=("agreement", "mean"), agreement_std=("agreement", "std"), kl_mean=("kl_divergence", "mean"), kl_std=("kl_divergence", "std"))
        .fillna(0.0)
    )


def make_plots(df: pd.DataFrame, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    agg = _agg(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), group in agg.groupby(["source", "interface"]):
        group = group.sort_values("budget")
        label = f"{source}:{interface}"
        ax.plot(group["budget"], group["agreement_mean"], marker="o", label=label)
    ax.set_xlabel("Query budget")
    ax.set_ylabel("Top-1 agreement")
    ax.set_title("Agreement vs Budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "agreement_vs_budget.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), group in agg.groupby(["source", "interface"]):
        group = group.sort_values("budget")
        label = f"{source}:{interface}"
        ax.plot(group["budget"], group["kl_mean"], marker="o", label=label)
    ax.set_xlabel("Query budget")
    ax.set_ylabel("KL divergence")
    ax.set_title("KL vs Budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "kl_vs_budget.png", dpi=200)
    plt.close(fig)
