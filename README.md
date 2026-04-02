# qwen2-mlx-interface-extraction

Research repo for black-box model extraction experiments on GitHub Actions macOS runners using Python + MLX.

## Scope

This repo implements two experiment families:

1. **Toy extraction baseline** (`source=toy`).
2. **Tiny-LLM next-token imitation** with **Qwen2-0.5B via MLX** (`source=qwen2-0.5b`).

Victim API interfaces:
- `argmax`
- `topk` (default `top5` in budget sweeps)
- `probs`

Additional fixed-budget Qwen top-k sweep:
- `argmax`, `top2`, `top3`, `top5`, `probs`

## Layout

- `models/` interface compression utilities.
- `attacks/` extraction learner.
- `experiments/` toy and Qwen experiments.
- `scripts/` plotting helpers.
- `results/` CSV outputs and plots.
- `.github/workflows/` CI definitions.

## Minimal dependencies

Pinned by policy to:
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `mlx`
- `mlx-lm`
- `huggingface_hub`

## Run locally

```bash
pip install -r requirements.txt
python run_all.py
```

Outputs:
- `results/budget_sweep_raw.csv`
- `results/budget_sweep_summary.csv`
- `results/qwen_topk_sweep_raw.csv`
- `results/qwen_topk_sweep_summary.csv`
- `results/multi_seed_mean_std.csv`
- `results/agreement_vs_budget.png`
- `results/kl_vs_budget.png`

## Day 1 defaults

- Budget sweep: `64, 128, 256, 512, 1024`
- Seeds: `0, 1, 2`
- Logged columns: `source, interface, budget, seed, agreement, kl_divergence`

The CI uploads all CSVs and plots as workflow artifacts.
