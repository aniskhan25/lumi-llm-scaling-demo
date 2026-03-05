# Execution Checklist

## Minimum deliverables

- `logs/train_1gpu_rank0.jsonl`
- `logs/train_4gpu_rank0.jsonl`
- `logs/train_8gpu_rank0.jsonl`
- `artifacts/scaling_summary.md`
- `artifacts/scaling_summary.csv`
- `artifacts/scaling_plot.png`
- Optional: `artifacts/before_outputs.jsonl` and `artifacts/after_outputs.jsonl`

## Done criteria

- Environment setup works from `docs/ENVIRONMENT.md`.
- All three launchers run from repo root.
- Scaling summary comes from real run logs.
- 4-GPU live run reaches regular step logs and GPU activity.
