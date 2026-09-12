# PRIMARY 8 basin-held-out OOB runner

This directory implements the local experiment contract from
`oob_primary8_5fold_20260902.yaml`.

## Correctness boundary

- One deterministic assignment is generated with the mechanics of
  `KFold(n_splits=5, shuffle=True, random_state=20260902)`.
- A fresh parameterizer, hydrological model, and AdamW optimizer are created
  for every model/fold job.
- Training data, window catalogs, loss, plateau counter, and `best.pt`
  selection receive only the fold's training basins.
- Attribute log shifts and z-score moments are fitted on training basins and
  then applied unchanged to held-out basins.
- Held-out test discharge is not loaded until after training ends, the exact
  `best.pt` is reloaded, and `Phase.EVAL` is entered. The access policy and
  phase gate fail fast on accidental cross-boundary access.
- `mopex4` is passed through the same calendar-forcing adapter used by the
  canonical benchmark (`add_calendar_forcing`); ordinary models retain three
  forcing channels.

The `bettermodel` directory was inspected; it contains the generic trainer and
not a basin-held-out OOB implementation. The actual reusable spatial-CV
pattern is the benchmark's explicit five-fold parameter-pretraining script,
while dPL training/evaluation uses the canonical benchmark model registry and
KGE objective.

## Local commands

```bash
.venv/bin/python project/benchmark/scripts/oob/make_oob_folds.py

.venv/bin/python project/benchmark/scripts/oob/oob_preflight.py
.venv/bin/python -m pytest project/benchmark/tests/test_oob_spatial_generalization.py
.venv/bin/python project/benchmark/scripts/oob/prepare_oob_data.py
.venv/bin/python project/benchmark/scripts/oob/run_oob_job.py \
  --model alpine2 --fold 0 --smoke --device cuda:0 \
  --epochs 2 --steps-per-epoch 1 --batch-size 4
.venv/bin/python project/benchmark/scripts/oob/prepare_oob_deployment.py
.venv/bin/python project/benchmark/scripts/oob/run_oob_queue.py
.venv/bin/python project/benchmark/scripts/oob/oob_status.py
.venv/bin/python project/benchmark/scripts/oob/summarize_oob.py
```

The queue has four worker slots and may intentionally map all four slots to one physical GPU (for example, `--devices cuda:0,cuda:0,cuda:0,cuda:0`). Each child is isolated with `CUDA_VISIBLE_DEVICES` and runs as `cuda:0` inside its namespace; completed jobs immediately release a slot for the next job.

For a remote conda environment, activate the environment before starting the queue so child jobs inherit its interpreter, for example:
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python project/benchmark/scripts/oob/run_oob_queue.py
```

The real data directory must contain the canonical
`data/caravan_671_attributes.npy`; the runner intentionally does not silently
fall back to non-Caravan attributes. The smoke output is marked
`NON-SCIENTIFIC_SMOKE_ONLY`.
