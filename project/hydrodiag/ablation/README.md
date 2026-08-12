# IC Ablation workspace

This directory contains the independent 531-basin IC foundation and is reserved
for future optimizer adapters. It does not implement a complete optimizer and
does not modify the legacy production runner.

- `project_audit/`: read-only audit helpers and audit notes; temporary files go
  under `project_audit/tmp/`.
- `configs/`: future ablation-only resolved configurations.
- `manifests/`: future run manifests and environment fingerprints.
- `runners/`: future optimizer adapters/runners.
- `analysis/`: future result aggregation and diagnostics.
- `tests/`: future ablation-specific tests.
- `utils/`: future shared ablation utilities.

The current foundation is under `ic_core/`, with inspection and smoke entry
points under `runners/`, protocol documentation under `docs/`, and focused
tests under `tests/`. Its effective daily protocol is inherited from the dPL
CAMELS-531 configuration: warmup `1980-10-01..1981-09-30`, train
`1981-10-01..1995-09-30`, and test `1995-10-01..2010-09-30`.

The IC ablation basin protocol is fixed to 531 basins: read the ordered IDs from
`/home/jingxin/code/dmg-research/data/531sub_id.txt` and select them by ID from
`/home/jingxin/code/dmg-research/data/camels_dataset`. The legacy 559 NPZ/ID
path is not an allowed training input for ablation.

Future ablation results must be written under a descriptive
`results/archive/ablation/<run_id>/` directory for exploratory work or
`results/<run_id>/` for a preregistered formal run. The old
`outputs/ic_ablation/` path is a compatibility link into archived result data
and must not receive new physical output.

The Lite validation entry point is
`python -m ablation.runners.run_lite_gpu_validation`. It maps every current IC
model key to its existing native `*Lite` class, uses one selected 531 basin and
a small candidate batch by default, limits host Torch threads to one, and runs
the model/objective path on CUDA. This is only a start-up validation: it does
not start XNES or any optimizer generations.
