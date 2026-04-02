# qwen2-mlx-interface-extraction

Black-box extraction experiments designed to run on **GitHub Actions macOS runners** using **Python + MLX**.

## Experiment families

1. **Toy extraction baseline** (`source=toy`)
2. **Tiny-LLM next-token imitation** with `Qwen/Qwen2-0.5B` via `mlx-lm` (`source=qwen2-0.5b`)

Victim API interfaces compared:
- `argmax`
- `topk` (Day 1 uses `top2`, and fixed-budget sweep includes `top2/top3/top5`)
- `probs`

## Minimal dependencies

Only these packages are used:
- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

## Repo layout

- `models/`
- `attacks/`
- `experiments/`
- `scripts/`
- `results/`
- `.github/workflows/`

## Run all experiments

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Outputs in `results/`:
- `day1_budget_sweep.csv`
- `qwen_fixed_budget_topk_sweep.csv`
- `qwen_multiseed_raw.csv`
- `qwen_multiseed_summary.csv`
- `agreement_vs_budget.png`
- `kl_vs_budget.png`
