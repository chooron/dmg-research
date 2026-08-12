# CN+XAJ 531-basin IC-CMA-ES training standard (canonical)

This document records the verified canonical method used for the formal
531-basin IC-CMA-ES calibration of the CemaNeige + XAJ models.  It was
recovered from the production deployment that produced
`results/xaj_base_cmaes_531_batched_paired_v2/` and
`results/xaj_cn_cmaes_531_batched_paired_v2/`
(remote snapshot `archive/remote_runtime_snapshots/20260730/ssh_53700_hydro_deployment/`
and the live queue script `experiment/run_xaj_base_cn_cmaes_531_queue.sh`).

**All future 531 IC-CMA-ES training must use this method** (not ad-hoc chunk /
parallelism choices).

## Protocol

| Item | Value |
| --- | --- |
| Basins | 531 (CAMELS-US subset, `data/531sub_id.txt`, deterministic order) |
| Forcing/data | `camels_dataset` (pickle tuple), `gage_id.npy`, `camels_dates.npy` |
| Periods | warmup 1980-10-01..1981-09-30 (365 d); train 1981-10-01..1995-09-30; test 1995-10-01..2010-09-30 |
| Model variant | `lite` (compact streamflow-only path), float32 forward, float64 metric, CUDA |
| Objective | KGE(Q), maximize, **train period only**; report test KGE for best candidate |
| Starts | 10 independent starts per basin |
| Population | `max(12, round(25 * dimension / 17))`; XAJ-class (17 dims) = 25 |
| Generations | 300 (hard stop; no early-stopping) |
| Chunking | `--chunk-basins 100` (6 chunks per generation; 25,000 units/chunk at pop 25) |
| Checkpoint | every 5 generations (`--checkpoint-interval 5`), atomic `.pt` in `checkpoints/` |
| Execution | **serial, one model at a time**; resumable from checkpoint |
| Seeds | `sha256(PROTOCOL:model:basin_id:start)` (PHASE0-IC-CMAES-v1 for the five controlled models) |
| Threads | `OMP/MKL/OPENBLAS/NUMEXPR/TORCHINDUCTOR_COMPILE_THREADS=1` |
| torch.compile | always on, `fullgraph=True`, per-step kernels (Dynamo cache limits raised) |
| CUDA memory | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recommended |

## Launcher

`training/ic/run_tgd2_batched_cmaes_531.py`:

```bash
python training/ic/run_tgd2_batched_cmaes_531.py \
  --model <MODEL> --output <results_dir>/<MODEL> \
  --starts 10 --population 25 --generations 300 \
  --chunk-basins 100 --checkpoint-interval 5
```

Supported controlled models: `N`, `D_E`, `G_E`, `D_R`, `G_R`
(dimensions 17/16/17/15/16).  The launcher:
- loads `ablation/configs/ic_foundation_531_v1.json` (data paths) and the 531 bundle;
- validates the checkpoint identity (`model`, `structure_version`, `basin_ids`,
  `starts`, `population`, `generations`) and resumes if a checkpoint exists
  (`chunk_basins` is not part of the resume contract);
- writes `manifest.json`, `checkpoints/<model>_batched.pt`, per-basin/start
  records under `raw/<model>/<basin>_startNN.json` (theta_normalized, physical
  parameters, train/test KGE, seed, status), `summaries/`, and `DONE.json`.

## Canonical serial chain (template)

```bash
run_one() {
  local M="$1"; local out="$OUT/$M"
  [[ -f "$out/DONE.json" ]] && return
  python training/ic/run_tgd2_batched_cmaes_531.py \
    --model "$M" --output "$out" --starts 10 --population 25 \
    --generations 300 --chunk-basins 100 --checkpoint-interval 5
}
run_one N; run_one D_E; run_one G_E; run_one D_R; run_one G_R
```

Run in background with `setsid nohup bash <script> > log 2>&1 < /dev/null &`.

## Performance notes

- On the production host that produced the paired_v2 results: ~13 s/generation
  (300 generations ≈ 65 min per model, 39,825,000 candidate evaluations).
- Measured on a slower single-GPU host: ~107 s/generation at chunk=100,
  ~200 s/generation at chunk=40 with two concurrent models.
- A single GPU can hold one chunk-100 job (~10.6 GiB on a 12 GiB GPU); do not
  run multiple chunk-100 jobs concurrently on one GPU.
- Container CPU-RAM limit: check `memory.limit_in_bytes` (cgroup v1) or
  `memory.max` (cgroup v2); the training process needs > 2 GiB RSS.
