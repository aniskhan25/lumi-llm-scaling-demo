# Handoff Plan and Ownership

## Workstream split

## Dev A: Environment + launch

- Validate module + container stack from `docs/ENVIRONMENT.md`.
- Confirm `run-scripts/run_1gpu.sh`, `run-scripts/run_4gpu.sh`, and `run-scripts/run_2node_8gpu.sbatch` execute on target partition.
- Deliverables:
  - Verified SLURM logs in `logs/`
  - Finalized account/partition settings
  - One smoke run per launcher

## Dev B: Training + scaling metrics

- Own `scripts/train_lora_ddp.py` and `scripts/parse_logs.py`.
- Run matrix:
  - 1 GPU baseline
  - 4 GPU single node
  - 8 GPU (2 nodes x 4 GPUs)
- Deliverables:
  - `logs/train_1gpu_rank0.jsonl`
  - `logs/train_4gpu_rank0.jsonl`
  - `logs/train_8gpu_rank0.jsonl`
  - `artifacts/scaling_summary.csv`
  - `artifacts/scaling_summary.md`
  - `artifacts/scaling_plot.png`

## Dev C: Inference + domain quality

- Curate production-safe support/policy subset in `data/`.
- Own `scripts/infer_before_after.py` and `prompts/demo_prompts.jsonl`.
- Produce `adapter_demo` from pre-event run.
- Deliverables:
  - `artifacts/before_outputs.jsonl`
  - `artifacts/after_outputs.jsonl`
  - `artifacts/adapters/adapter_demo`

## Presenter support

- Rehearse command order from `docs/DEMO_RUNBOOK.md`.
- Capture fallback materials:
  - Multi-GPU `rocm-smi` screenshot(s)
  - Scaling plot screenshot
  - Optional 2-3 minute recording

## Definition of done checklist

- [ ] Environment reproducible from clean allocation using `ENVIRONMENT.md`.
- [ ] Before/after prompts show clear qualitative improvement.
- [ ] Live 4-GPU run reaches logging within 30-60s from launch.
- [ ] Scaling artifacts generated from real LUMI logs (not synthetic).
- [ ] Fallback ladder validated end-to-end.

