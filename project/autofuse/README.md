# AutoFuse experiment layer

This directory owns CAMELS protocol/data access, SCE and structure-conditioned
dPL orchestration, metrics, run metadata, and later regret/rank/equivalence
analysis.  It calls `dfuse.simulate` through `UnifiedEvaluator`; it does not
contain a second hydrological implementation.

`configs/phase1.yaml` is a locked plan for the recovered 544-catchment subset and 78 structures. The paper scripts use a 559-row list; the explicit 544 membership comes from HydroShare `CatchmentBoundaries_544` and is frozen at `manifests/camels_544.json`. SCE and dPL are disabled in this round. For the reference oracle, first build with `build_reference.sh`, then run bounded fidelity checks with:

```bash
.venv/bin/python -m project.autofuse.smoke_test

FUSE_TOOLCHAIN_ROOT=/tmp/autofuse-reference-toolchain/root FUSE_REFERENCE_BUILD_DIR=/tmp/autofuse-reference-toolchain/repro-build bash project/autofuse/build_reference.sh

export FUSE_REFERENCE_EXE=/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe
export FUSE_REFERENCE_LIB_DIR=/tmp/autofuse-reference-toolchain/root/lib:/tmp/autofuse-reference-toolchain/root/lib/hdf5/serial
FUSE_REFERENCE_EXE="$FUSE_REFERENCE_EXE" .venv/bin/python -m project.autofuse.run_fidelity --executable "$FUSE_REFERENCE_EXE" --mode mothers
FUSE_REFERENCE_EXE="$FUSE_REFERENCE_EXE" .venv/bin/python -m project.autofuse.run_fidelity --executable "$FUSE_REFERENCE_EXE" --mode explicit --substeps 1 2 4 8 12 24 48 96 192 --output project/autofuse/docs/explicit_solver_scan.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 .venv/bin/python -m project.autofuse.compile_validation --executable "$FUSE_REFERENCE_EXE" --output project/autofuse/docs/compile_validation.json

```

The runtime-step prototype is GPU-first and deliberately bounded to the four mother structures `(2, 108, 178, 210)` with fixed candidate order `S1` and `n_substeps=1`.  It compiles one structure at a time, keeps the time loop in Python, and uses a persistent Inductor cache under `project/autofuse/.cache/` (ignored by Git):

```bash
mkdir -p project/autofuse/.cache/runtime-validation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR="$PWD/project/autofuse/.cache/runtime-validation" \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
.venv/bin/python -m project.autofuse.runtime_validation \
  --output project/autofuse/docs/runtime_step_validation.json
```


The completed full-catalog audit uses the same fixed S1 step but isolates each signature in a serial worker process.  This is required to prevent PyTorch/Inductor native in-process RSS accumulation; workers are never concurrent and share only the persistent cache:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR="$PWD/project/autofuse/.cache/runtime-validation-78-safe" \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
.venv/bin/python -m project.autofuse.runtime_validation_78_serial \
  --cache-dir project/autofuse/.cache/runtime-validation-78-safe \
  --output project/autofuse/docs/runtime_step_validation_78.json

# Resume from a saved partial after a targeted repair recheck:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_CACHE_DIR="$PWD/project/autofuse/.cache/runtime-validation-78-safe" \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
.venv/bin/python -m project.autofuse.runtime_validation_78_serial \
  --partial project/autofuse/docs/runtime_step_validation_78.partial.json \
  --recheck project/autofuse/docs/runtime_step_validation_78_oflow_recheck.json \
  --cache-dir project/autofuse/.cache/runtime-validation-78-safe \
  --output project/autofuse/docs/runtime_step_validation_78.json
```

The validator reports generated-eager parity against the current sequential implementation, compiled parity, active/inactive gradients, guard/recompile audit, CUDA forward/backward/total timings, and peak GPU memory.  It does not start SCE/dPL and the documented default does not run the full 78/1248 structures.  After the four mothers pass, the optional `--extra-smoke` run records four additional structures in `docs/runtime_step_validation_extra_smoke.json`.  Cross-process cache probes use `--mode cache-probe` with the same `TORCHINDUCTOR_CACHE_DIR`; PyTorch 2.9.1 also exposes `torch.compiler.save_cache_artifacts()`/`load_cache_artifacts()`, but this prototype validates the persistent Inductor/FX/AOT cache path directly.

The sequential process-order selection experiment is independent of the completed S1 runtime/compiler artifact.  It verifies the locked 18-model development set over S1--S4, freezes one order, then evaluates only that order on the remaining 60 structures.  It uses the pinned Fortran reference with the same forcing/parameter/initialization protocol and one fresh serial GPU worker per order/model case:

```bash
FUSE_REFERENCE_LIB_DIR=/tmp/autofuse-reference-toolchain/root/lib \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
.venv/bin/python -m project.autofuse.sequential_order_validation \
  --cache-dir project/autofuse/.cache/sequential-order-validation \
  --output project/autofuse/docs/sequential_order_validation.json

# Resume from the saved partial artifact:
FUSE_REFERENCE_LIB_DIR=/tmp/autofuse-reference-toolchain/root/lib \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
.venv/bin/python -m project.autofuse.sequential_order_validation \
  --partial project/autofuse/docs/sequential_order_validation.partial.json \
  --cache-dir project/autofuse/.cache/sequential-order-validation \
  --output project/autofuse/docs/sequential_order_validation.json
```

The machine-readable result records Stage 0 plumbing checks, per-order development fidelity and CUDA benchmarks, the frozen `best_order`, held-out validation, topology group summaries, compile/resource/cache audits, and the explicit `S5_triggered` decision.  This command does not start SCE, dPL, formal training, implicit/Newton/IFT code, or outer-loop compilation.
