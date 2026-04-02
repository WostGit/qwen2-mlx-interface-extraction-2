# qwen2-mlx-interface-extraction

Research repo for black-box model extraction experiments on GitHub Actions macOS runners using Python + MLX.

## Layout
- `models/`: victim model interfaces (toy and Qwen2-0.5B via MLX)
- `attacks/`: extraction estimators and metrics
- `experiments/`: configs and experiment orchestration
- `scripts/`: plotting helpers
- `results/`: generated CSVs and plots
- `.github/workflows/`: CI workflows

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python run_all.py --config experiments/day1.yaml
```

Outputs include:
- `budget_sweep_raw.csv`
- `budget_sweep_summary.csv`
- `qwen_topk_raw.csv`
- `qwen_topk_summary.csv`
- `agreement_vs_budget.png`
- `kl_vs_budget.png`
