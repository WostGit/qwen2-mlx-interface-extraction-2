from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(summary_csv: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), part in df.groupby(["source", "interface"]):
        part = part.sort_values("budget")
        ax.plot(part["budget"], part["agreement_mean"], marker="o", label=f"{source}:{interface}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Query budget")
    ax.set_ylabel("Top-1 agreement")
    ax.set_title("Agreement vs Budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "agreement_vs_budget.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), part in df.groupby(["source", "interface"]):
        part = part.sort_values("budget")
        ax.plot(part["budget"], part["kl_mean"], marker="o", label=f"{source}:{interface}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Query budget")
    ax.set_ylabel("KL divergence")
    ax.set_title("KL vs Budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "kl_vs_budget.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    make_plots("results/results_summary.csv", "results")
