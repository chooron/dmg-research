# Dual-Evidence Hydrologic Benchmark

This module initializes the benchmark framework for later dual-evidence model recommendation and parameter recommendation.

Stage 1 focuses on independent calibration. For each CAMELS basin, each selected `dmotpy` hydrologic model, and each objective (`nse`, `log_nse`), the runner optimizes many random initial parameter sets in parallel and keeps every optimized start, not only the best result.

## Structure

- `conf/benchmark.yaml` - benchmark configuration, data paths, splits, models, objectives, and optimizer settings.
- `benchmark/` - reusable Python package for CAMELS data access, dmotpy model construction, objectives, metrics, task generation, and evidence merging.
- `generate_tasks.py` - creates a basin-model-objective task table.
- `run_independent_calibration.py` - runs one basin-model-objective calibration task.
- `scripts/run_independent_calibration_parallel.sh` - generic parallel launcher for a task table.
- `collect_evidence.py` - merges per-task `results.csv` files into one evidence table.
- `benchmark/parameter_learning.py` - reserved stage-2 interface for MLP differentiable parameter learning.

## Generate Tasks

Small smoke task table:

```bash
python project/benchmark/generate_tasks.py \
  --config project/benchmark/conf/benchmark.yaml \
  --output project/benchmark/outputs/tasks/smoke_tasks.csv \
  --limit-basins 1
```

Full task table uses every basin in `data/531sub_id.txt`, every model in `dmotpy.models.registry.PARAM_INFO`, and both configured objectives:

```bash
python project/benchmark/generate_tasks.py \
  --config project/benchmark/conf/benchmark.yaml \
  --output project/benchmark/outputs/tasks/independent_calibration_tasks.csv
```

## Run One Calibration Task

```bash
python project/benchmark/run_independent_calibration.py \
  --config project/benchmark/conf/benchmark.yaml \
  --basin-id 1022500 \
  --model-id hbv96 \
  --objective nse
```

For a quick local check:

```bash
python project/benchmark/run_independent_calibration.py \
  --config project/benchmark/conf/benchmark.yaml \
  --basin-id 1022500 \
  --model-id hbv96 \
  --objective nse \
  --num-starts 2 \
  --epochs 1
```

Each task writes:

- `results.csv` - one row per random start, with parameters and split metrics.
- `simulations.npz` - train/validation/test prediction and observation arrays.
- `metadata.json` - task metadata and parameter names.

## Run Tasks In Parallel

Use environment variables:

```bash
JOBS=4 \
CONFIG=project/benchmark/conf/benchmark.yaml \
TASK_TABLE=project/benchmark/outputs/tasks/independent_calibration_tasks.csv \
OUTPUT_DIR=project/benchmark/outputs \
LOG_DIR=project/benchmark/logs \
bash project/benchmark/scripts/run_independent_calibration_parallel.sh
```

Or command-line arguments:

```bash
bash project/benchmark/scripts/run_independent_calibration_parallel.sh \
  --jobs 4 \
  --config project/benchmark/conf/benchmark.yaml \
  --task-table project/benchmark/outputs/tasks/independent_calibration_tasks.csv \
  --output-dir project/benchmark/outputs \
  --log-dir project/benchmark/logs
```

## Evidence Table

Merge all independent calibration task outputs:

```bash
python project/benchmark/collect_evidence.py \
  --root project/benchmark/outputs/independent_calibration \
  --output project/benchmark/outputs/evidence/independent_calibration_evidence.csv
```

The independent calibration rows include `basin_id`, `model_id`, `objective`, `random_start_id`, optimized physical and normalized parameters, train/validation/test NSE and logNSE, KGE and components, high/low-flow diagnostics, boundary flag, optimization success flag, runtime, and final loss.

## Stage 2 Placeholder

`benchmark/parameter_learning.py` reserves the differentiable parameter learning entry point. The intended next stage is an MLP that maps CAMELS basin attributes to normalized `dmotpy` model parameters and trains through `HydrologyModel` with NSE or logNSE losses. Outputs should align with the evidence-table conventions so independent calibration evidence and learned-parameter evidence can be merged.
