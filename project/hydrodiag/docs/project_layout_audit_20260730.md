# Project layout audit — 2026-07-30

## Decision

The project now has one active source tree, one physical result root, and one
inactive code/provenance archive. No historical material was deleted.

## Necessary active directories

| Directory | Status | Reason |
| --- | --- | --- |
| `models/` | necessary | Active equations, compositions, bounds, and model exports |
| `training/` | necessary | Active dPL run and IC data/training entry points |
| `ablation/` | necessary | IC adapters, normalized parameter mapping, runners, and tests |
| `optimization/` | necessary | Reusable optimizer implementations |
| `scripts/` | necessary | Maintained validation and analysis tools |
| `tests/` | necessary | Scientific invariants and regression gates |
| `configs/` | necessary | Reusable and historical-reference configurations used by maintained analyses |
| `docs/` | necessary | Protocol and source-of-truth documentation |
| `manuscript/` | necessary | Manuscript source and reproducible supplement scripts |
| `evotorch/` | necessary vendored dependency | Referenced by optimizer/ablation paths |
| `results/` | necessary | Sole physical run-product root |
| `archive/` | necessary provenance | Inactive code and exact remote runtime snapshots |
| `experiment/` | compatibility only | README plus result-path compatibility symlink |
| `experiments` | compatibility only | Link to retired XNES/old-TGD sources in the cleanup archive |

## Material removed from active paths

- Root one-off fix/patch/report/watch/sync scripts.
- Completed TGD2/CMA-ES runners and logs.
- Retired `experiments/ic_tgd_cmaes` and `experiments/ic_xnes` sources.
- Duplicate project-local `.venv` and `venv`; the repository-level
  `/home/jingxin/code/dmg-research/.venv` remains active.
- Nested `models/archive`.
- Inactive generated multi-model dPL configs.
- Python and pytest caches.
- Root historical log files.

All of the above is recoverable from
`archive/project_cleanup_20260730/`; exact moves are listed in its
`migration_manifest.csv`.

## Result consolidation

Physical result data formerly under `outputs/`, `experiment/results/`,
`archive/{outputs,results,validation_results}`, and
`manuscript/supplement/results` now resides under `results/`. Relative
compatibility symlinks preserve historical scripts and manuscript tools.

Formal XAJ, XAJ-CN, and XAJ-TGD2 CMA-ES runs remain top-level result entries.
Smoke, interrupted, superseded, old-TGD, XNES, and legacy outputs are
classified under `results/archive/`. The active three-seed XAJ-TGD2 dPL run
remains at its canonical result path while it is running.

## Remote provenance

The active SSH 36-model CMA-ES deployment was observed on port 53700 and its
executed `dmotpy` code, experiment source, configs, frozen manifests, logs,
small metadata, and runtime records were copied to
`archive/remote_runtime_snapshots/20260730/`. Large live checkpoints and
benchmarks were not modified or copied. The snapshot contains a verified
905-file `SHA256SUMS` inventory.

The remote hydro deployment used by the completed paired XAJ CMA-ES work was
also copied as a source-only snapshot; its completed result data was already
verified under local `results/`.

The active local three-seed XAJ-TGD2 dPL source, resolved configs, and focused
tests were independently frozen under
`archive/local_runtime_snapshots/20260730_xaj_tgd2_dpl/` with a verified
38-file SHA-256 inventory.

## Compatibility and exceptions

- `outputs` and `experiment/results` are links, not physical result roots.
- Paths named `results` or `outputs` inside the immutable SSH source snapshot
  are part of the captured remote tree and are not active local result roots.
- `training/dpl/logs/` remains active while the three-seed TGD2 dPL process is
  running. On completion, the logs should be copied into that run's `logs/`
  directory and the training log location may be retained as a compatibility
  link.
