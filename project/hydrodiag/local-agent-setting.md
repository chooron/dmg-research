# Codex project rules — hydrodiag

This file applies to the entire project tree. Read it before creating,
moving, or modifying files.

## Canonical project layout

- `models/`: active hydrological equations, compositions, parameter specs,
  and model registries. Do not place experiments or result files here.
- `training/dpl/`: active dPL data/window logic, trainers, launchers, and
  checked-in templates.
- `training/ic/`: active independent-calibration data and training entry
  points.
- `ablation/`: reusable IC adapters, optimizer ablations, and their tests.
- `optimization/`: reusable optimizer implementations.
- `scripts/`: maintained analysis, validation, migration, and reporting tools.
- `configs/`: reusable human-authored configurations. Generated resolved
  configs belong inside their run directory.
- `tests/`: active regression and scientific-invariant tests.
- `docs/`: maintained design and protocol documentation.
- `manuscript/`: manuscript source and reproducible supplement tooling.
- `results/`: the only canonical physical root for run products.
- `archive/`: inactive code/config/script snapshots and provenance only.

Do not create scratch Python files, patch scripts, logs, checkpoints, CSVs, or
new experiment directories at the project root.

## Where generated files go

Formal runs must write to:

```text
results/<descriptive_run_id>/
├── manifest.json
├── config.json or configs/
├── environment.json
├── logs/
├── checkpoints/
├── raw/
├── summaries/
└── DONE.json or COMPLETE
```

Smoke, interrupted, superseded, and exploratory outputs go under an explicit
subdirectory of `results/archive/`, for example:

```text
results/archive/smoke_runs/<run_id>/
results/archive/interrupted_runs/<run_id>/
results/archive/legacy_outputs_<date>/<run_id>/
```

Never create a new physical `outputs/` or `experiment/results/` tree. Existing
paths are compatibility symlinks whose data lives under `results/`.

Temporary files should use a task-specific directory created by `mktemp -d`.
If temporary material must be retained for audit, move it to
`archive/<topic>_<YYYYMMDD>/` and record it in a migration manifest.

## Archive policy

Archive code is immutable historical evidence and must not be imported by
active code. Use these categories:

- `archive/tgd_legacy_<date>/`: retired model mechanisms and their scripts.
- `archive/remote_runtime_snapshots/<date>/`: exact SSH-deployed source,
  configs, commands, frozen manifests, and hashes.
- `archive/project_cleanup_<date>/`: one-off scripts, duplicate environments,
  caches, logs, and completed runners removed from active paths.

Historical result data must remain physically below `results/archive/`, not
inside `archive/`. Compatibility symlinks from old paths are allowed.

## Model and parameter changes

- Define physical bounds once in `models/parameter_specs.py`.
- Update model exports, IC/dPL registries, parameter order, checkpoint
  metadata, and tests together.
- Do not silently change Base, GD/PD, CN, forcing preprocessing, periods,
  objectives, or metric definitions when adding a model.
- TGD2 is a temperature-dependent generic precipitation-memory module, not a
  snow accumulation/melt model.
- Keep dPL and IC protocols separate. dPL Lite-v2 uses random contiguous
  730-day samples (365 warm-up + 365 scored days) and a sigmoid parameter
  network. IC uses its own full-period calibration and normalized optimizer
  adapter. Shared physical equations do not authorize sharing window logic.

## Experiment discipline

- Freeze basin list, periods, objective, parameter order/bounds, seed scheme,
  code version, and environment in the run manifest before production.
- Choose starts or checkpoints only using the declared training/validation
  rule; never select using test performance.
- Preserve failed starts, runtime, seeds, and failure reasons.
- Use atomic checkpoints and resumable basin/model/start units for long runs.
- Do not overwrite or mutate completed result directories. Derived analyses
  must write to a new `analysis/` or `summaries/` subdirectory.
- Never modify original data under the repository-level `data/` directory.

## SSH provenance

Before changing or retiring code used by a live SSH task:

1. record host, port, process command, working directory, run ID, config, and
   checkpoint/result paths;
2. snapshot every executed source/config/frozen-manifest file under
   `archive/remote_runtime_snapshots/<date>/`;
3. generate and verify `SHA256SUMS`;
4. do not copy, move, or modify a live remote checkpoint tree unless the user
   explicitly requests it.

## Verification before handoff

- Run the smallest relevant tests plus affected registry/interface tests.
- Confirm active processes still point to existing files and their latest
  checkpoint/log advances.
- Confirm every formal result has its declared record count and completion
  marker.
- Search for stale physical result roots and broken symlinks.
- Update `results/README.md`, `archive/README.md`, and the cleanup migration
  manifest after moving material.

