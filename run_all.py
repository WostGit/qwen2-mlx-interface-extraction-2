from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from experiments.runner import RunConfig, run_day1
from scripts.plot_results import make_plots


def load_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig(
        budgets=raw["budgets"],
        seeds=raw["seeds"],
        qwen_model=raw["qwen_model"],
        qwen_prompt_limit=raw["qwen_prompt_limit"],
        default_topk=raw["default_topk"],
        fixed_topk_budget=raw["fixed_topk_budget"],
        fixed_topk_interfaces=raw["fixed_topk_interfaces"],
        output_dir=Path(raw["output_dir"]),
    )


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        df.groupby(by, as_index=False)[["agreement", "kl_divergence"]]
        .agg(["mean", "std"])
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction experiments")
    parser.add_argument("--config", default="experiments/day1.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = run_day1(cfg)
    budget_df = outputs["budget_sweep"]
    topk_df = outputs["qwen_topk_sweep"]

    budget_csv = cfg.output_dir / "budget_sweep_raw.csv"
    topk_csv = cfg.output_dir / "qwen_topk_raw.csv"
    budget_df.to_csv(budget_csv, index=False)
    topk_df.to_csv(topk_csv, index=False)

    budget_summary = summarize(budget_df, by=["source", "interface", "budget"])
    topk_summary = summarize(topk_df, by=["source", "interface", "budget"])
    budget_summary.to_csv(cfg.output_dir / "budget_sweep_summary.csv", index=False)
    topk_summary.to_csv(cfg.output_dir / "qwen_topk_summary.csv", index=False)

    make_plots(budget_df, cfg.output_dir)

    print(f"Wrote outputs under {cfg.output_dir}")


if __name__ == "__main__":
    main()
