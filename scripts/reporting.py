from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["source", "interface", "budget"], as_index=False).agg(
        agreement_mean=("agreement", "mean"),
        agreement_std=("agreement", "std"),
        kl_divergence_mean=("kl_divergence", "mean"),
        kl_divergence_std=("kl_divergence", "std"),
    )
    return grp.fillna(0.0)


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename in [
        ("agreement", "agreement_vs_budget.png"),
        ("kl_divergence", "kl_vs_budget.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (source, interface), sub in df.groupby(["source", "interface"]):
            sub = sub.sort_values("budget")
            ax.plot(sub["budget"], sub[metric], marker="o", label=f"{source}:{interface}")
        ax.set_xlabel("Query budget")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs budget")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
