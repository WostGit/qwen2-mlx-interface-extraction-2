# qwen2-mlx-interface-extraction

Research repo for black-box model extraction experiments on GitHub Actions macOS runners using Python + MLX.

## What this runs
- **Toy extraction baseline**
- **Tiny-LLM next-token imitation** with **Qwen2-0.5B (MLX)**

It compares victim API interfaces:
- `argmax`
- `topk` (e.g. top-2/top-3/top-5)
- `probs` (full probabilities)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --config experiments/day1.yaml
```

Outputs are written to `results/` as CSV summaries and plots.
