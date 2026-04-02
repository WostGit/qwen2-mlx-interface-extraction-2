from __future__ import annotations

from pathlib import Path

from experiments.common import rows_to_df
from experiments.qwen_experiment import run_qwen
from experiments.toy_experiment import run_toy
from scripts.reporting import make_plots, summarize


BUDGETS = [64, 128, 256, 512, 1024]
DAY1_INTERFACES = ["argmax", "top2", "probs"]
QWEN_TOPK_SWEEP = ["argmax", "top2", "top3", "top5", "probs"]


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    day1_rows = []
    day1_rows.extend(run_toy(BUDGETS, DAY1_INTERFACES, seeds=[0]))
    day1_rows.extend(run_qwen(BUDGETS, DAY1_INTERFACES, seeds=[0]))
    day1_df = rows_to_df(day1_rows)
    day1_df.to_csv(out_dir / "day1_budget_sweep.csv", index=False)

    qwen_topk_rows = run_qwen([256], QWEN_TOPK_SWEEP, seeds=[0])
    qwen_topk_df = rows_to_df(qwen_topk_rows)
    qwen_topk_df.to_csv(out_dir / "qwen_fixed_budget_topk_sweep.csv", index=False)

    qwen_multiseed_rows = run_qwen([256], QWEN_TOPK_SWEEP, seeds=[0, 1, 2])
    qwen_multiseed_df = rows_to_df(qwen_multiseed_rows)
    qwen_multiseed_df.to_csv(out_dir / "qwen_multiseed_raw.csv", index=False)
    qwen_summary_df = summarize(qwen_multiseed_df)
    qwen_summary_df.to_csv(out_dir / "qwen_multiseed_summary.csv", index=False)

    make_plots(day1_df, out_dir)

    print("Wrote:")
    for p in sorted(out_dir.glob("*")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
