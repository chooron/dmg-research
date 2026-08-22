# Flex-MOPEX training instructions

## Scope

These instructions describe the model-modified CAMELS-671 formal protocol. The
formal result set contains 671 fixed basins and must be trained only on a CUDA
GPU. CPU-only training is not supported for the formal run.

## Environment

Remote formal workspace:

```text
/root/flexmopex_formal_671/project/flexmopex
```

Python executable:

```text
/root/miniconda3/bin/python
```

Before starting any job:

```bash
nvidia-smi
/root/miniconda3/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

`GPU_IDS` must use the indices visible inside the process. If only one GPU is
visible, use `GPU_IDS=0`; do not assume the physical host index is visible.

## Formal protocol

- Fixed basin set: `data/gage_id.npy` (671 basins)
- Maximum epochs: 100
- Minimum epochs: 50
- Early-stopping patience: 10
- Early-stopping `min_delta`: 0.0001
- Structure encoder: `35 -> 128 -> 128 -> 8`
- Ordinary/main jobs: explicitly pass `--nmul 1`
- DFlex: excluded from the formal queue because tracked formal provenance and
  configuration are not available

`nmul` expands the physical parameter/forcing/state copy dimension. The four
process weights remain one weight vector per basin and are broadcast over the
nmul dimension; nmul does not create independent copies of `w`.

## Staged formal launcher

From the project root:

```bash
cd /root/flexmopex_formal_671/project/flexmopex

GPU_IDS=0 \
PROCESSES_PER_GPU=5 \
MAX_PARALLEL=5 \
NMUL_SMALL_PROCESSES_PER_GPU=5 \
NMUL16_PROCESSES_PER_GPU=3 \
NMUL32_PROCESSES_PER_GPU=2 \
SCHEDULER_MODE=per_gpu \
PYTHON_BIN=/root/miniconda3/bin/python \
RESULTS_ROOT=results/formal_671_unified_nmul1_tail3 \
RESUME_INCOMPLETE=1 \
./scripts/run_formal_671_staged.sh unified \
  >> logs/formal_671_unified_nmul1_tail3.log 2>&1
```

The unified queue runs in this order:

1. Main/non-sensitivity jobs, `nmul=1`, capacity 5 per visible GPU;
2. Small sensitivity jobs, `nmul=1/4/8`, capacity 5;
3. `nmul=16`, capacity 3;
4. `nmul=32`, capacity 2.

The later sensitivity phases wait for the earlier phase to finish. A completed
run is skipped. An incomplete run with `model/last_checkpoint.pt` is resumed;
an incomplete run without a checkpoint is retried from scratch. Existing output
is never overwritten by the launcher.

## Lightweight validation and monitoring

Do not start a full training queue until a one-epoch smoke test and the visible
GPU check pass. Monitor with:

```bash
pgrep -af run_formal_671_staged
nvidia-smi

tail -f logs/formal_671_unified_nmul1_tail3.log
```

A formal run is complete only when the log contains:

```text
STAGE COMPLETE: unified
```

and every run directory has `early_stopping.json`,
`model/last_checkpoint.pt`, and either `sim/metrics.json` or a `test*/metrics.json`
file.

## Resource rules

- Never start formal training when no CUDA device is visible.
- Use one visible GPU and conservative concurrency when memory is uncertain.
- Avoid stacking all temporal predictions across jobs; this was the source of a
  previous high-`nmul` memory failure.
- Do not run multiple full training queues concurrently.
- Preserve incomplete results and resume rather than overwrite.
