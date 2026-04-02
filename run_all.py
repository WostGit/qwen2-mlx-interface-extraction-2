from __future__ import annotations

import argparse
from pathlib import Path

from experiments.run_experiments import run_all
from scripts.plot_results import make_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Day 1 extraction experiments")
    p.add_argument("--config", default="experiments/day1.yaml")
    p.add_argument("--output", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_all(args.config, args.output)
    make_plots(outputs["summary"], args.output)
    print("Wrote:")
    for key, path in outputs.items():
        print(f"  {key}: {Path(path).resolve()}")
    print(f"  agreement_plot: {(Path(args.output) / 'agreement_vs_budget.png').resolve()}")
    print(f"  kl_plot: {(Path(args.output) / 'kl_vs_budget.png').resolve()}")


if __name__ == "__main__":
    main()
