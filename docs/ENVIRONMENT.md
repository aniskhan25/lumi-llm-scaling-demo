# Environment (LUMI ROCm)

## 1) Base setup

```bash
cd /scratch/project_462000131/$USER/lumi-llm-scaling-demo
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export VENV_ACTIVATE=/project/project_462000131/$USER/venvs/myvenv/bin/activate
```

If container paths change:

```bash
ls -1 /appl/local/laifs/containers
```

## 2) Create venv (once)

Use LUMI-recommended system site packages:

```bash
singularity run "$SIF_IMAGE" bash -lc '
python -m venv --system-site-packages /project/project_462000131/$USER/venvs/myvenv
source /project/project_462000131/$USER/venvs/myvenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
'
```

## 3) Verify inside GPU allocation

```bash
srun --ntasks=1 --gpus=1 singularity exec "$SIF_IMAGE" bash -lc '
source "$VENV_ACTIVATE"
python -c "import torch; print(\"torch\", torch.__version__); print(\"gpus\", torch.cuda.device_count())"
python -c "import torch; print(\"nccl available\", torch.distributed.is_nccl_available())"
'
```

## 4) Multi-node defaults (8-GPU run)

Only needed for `run_2node_8gpu.sh`:

```bash
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_PROVIDER=cxi
```

## 5) Quick checklist

- `torch.cuda.device_count()` matches allocated GPUs.
- `torch.distributed.is_nccl_available()` is `True`.
- `rocm-smi` shows allocated GPUs.

## References

- LUMI PyTorch docs: <https://docs.lumi-supercomputer.eu/software/packages/pytorch/>
- LUMI AI docs: <https://docs.lumi-supercomputer.eu/ai/>
