from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    grouped = (
        df.groupby(["source", "interface", "budget"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    for (source, interface), sub in grouped.groupby(["source", "interface"]):
        sub = sub.sort_values("budget")
        label = f"{source}:{interface}"
        plt.plot(sub["budget"], sub["mean"], marker="o", label=label)
        plt.fill_between(sub["budget"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.15)

    plt.xscale("log", base=2)
    plt.xlabel("Query budget")
    plt.ylabel(metric)
    plt.title(f"{metric} vs budget")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def make_plots(budget_df: pd.DataFrame, output_dir: Path) -> None:
    _plot(budget_df, metric="agreement", out_path=output_dir / "agreement_vs_budget.png")
    _plot(budget_df, metric="kl_divergence", out_path=output_dir / "kl_vs_budget.png")
