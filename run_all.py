from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.qwen_experiment import run_qwen_budget_sweep, run_qwen_topk_sweep
from experiments.toy_experiment import run_toy_budget_sweep
from scripts.plot_results import make_plots


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
        )
        .fillna(0.0)
    )


def main() -> None:
    budgets = [64, 128, 256, 512, 1024]
    seeds = [0, 1, 2]
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    toy_df = run_toy_budget_sweep(budgets=budgets, seeds=seeds)
    qwen_df = run_qwen_budget_sweep(budgets=budgets, seeds=seeds)
    budget_df = pd.concat([toy_df, qwen_df], ignore_index=True)
    budget_df.to_csv(output_dir / "budget_sweep_raw.csv", index=False)

    budget_summary = summarize(budget_df, ["source", "interface", "budget"])
    budget_summary.to_csv(output_dir / "budget_sweep_summary.csv", index=False)

    qwen_topk_df = run_qwen_topk_sweep(fixed_budget=256, seeds=seeds)
    qwen_topk_df.to_csv(output_dir / "qwen_topk_sweep_raw.csv", index=False)

    qwen_topk_summary = summarize(qwen_topk_df, ["source", "interface", "budget"])
    qwen_topk_summary.to_csv(output_dir / "qwen_topk_sweep_summary.csv", index=False)

    multi_seed_summary = summarize(
        pd.concat([budget_df, qwen_topk_df], ignore_index=True),
        ["source", "interface", "budget"],
    )
    multi_seed_summary.to_csv(output_dir / "multi_seed_mean_std.csv", index=False)

    make_plots(budget_df, output_dir)


if __name__ == "__main__":
    main()
