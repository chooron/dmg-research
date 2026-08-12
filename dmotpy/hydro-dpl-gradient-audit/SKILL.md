---
name: hydro-dpl-gradient-audit
description: Audit, diagnose, and harden PyTorch hydrological models for differentiable parameter learning (dPL). Use when Codex must inspect an existing conceptual or hybrid hydrological model repository; verify gradient connectivity and numerical usability in the real training dtype; diagnose NaN, Inf, zero, vanishing, detached, or sanitized gradients; test parameter mappings, recurrent states, snow/routing modules, warm-up, masking, and losses; preserve forward physics and mass balance while repairing autograd failures; add regression tests; or decide whether model-comparison results are valid. Trigger for HBV, XAJ/Xinanjiang, GR4J, SAC-SMA, FLEX, MARRMoT-family, CemaNeige, unit-hydrograph routing, and similar torch implementations.
---

# Hydrological Model Differentiability Audit

Treat forward correctness, autograd connectivity, numerical gradient usability, and practical learnability as separate properties. Do not infer one from another.

## Operating rules

1. Inspect the repository and real training path before editing code.
2. Use the actual model forward, parameter mapper, warm-up, masking, loss, dtype, device, and representative forcing whenever available.
3. Preserve equations and forward behavior by default. Do not rewrite hydrological logic until a failure is reproduced and localized.
4. Never hide non-finite gradients with `nan_to_num`, replacement by zero, broad exception handling, or silent sample deletion.
5. Change one cause at a time. Re-run the smallest failing audit after each change.
6. Treat static findings as risks, not proof. Treat one successful backward pass as necessary, not sufficient.
7. Allow physically inactive parameters to have zero gradients in non-applicable cases. Require reachability across a declared realistic-plus-excitation audit matrix.
8. Keep all generated audit code, reports, and tests inside the target repository unless the user specifies another location.
9. Use the bundled scripts with the execution environment available to Codex. When this Skill is executed in an environment that requires `container` for code, use `container` for every script and test command.

## Required outputs

Create or update an audit directory such as `audit/differentiability/<model_name>/` containing:

- `audit_manifest.md`: model entrypoint, training entrypoint, parameter path, dtype/device, data slice, loss, warm-up, and audit matrix.
- `static_risk_scan.json` and `static_risk_scan.md`.
- `gradient_adapter.py`: repository-specific adapter following `references/adapter-contract.md`.
- `gradient_audit.json` and `gradient_audit.md`.
- `learning_audit.json` and `learning_audit.md` when short training is available.
- `forward_parity.md`: pre/post-fix hydrograph, state, mass-balance, and routing checks.
- `patch_notes.md`: evidence, root cause, minimal fix, residual risks, and rerun scope.
- regression tests under the repository test tree.

Do not claim completion when a required item is unavailable. Mark it `NOT RUN`, explain why, and state the consequence.

## Workflow

### 1. Establish the real execution chain

Trace and record:

- raw trainables or encoder outputs;
- transformation into physical parameters and parameter bounds;
- model states, forcing, warm-up, routing buffers, and recurrent loop;
- simulated discharge and auxiliary fluxes;
- observation mask and loss;
- optimizer, AMP/autocast, gradient clipping, gradient accumulation, and sanitization;
- state detachment or truncated backpropagation behavior.

Search for alternate model implementations and confirm which one the training command imports. Do not audit an unused class.

Read `references/codex-execution-protocol.md` before modifying a repository.

### 2. Run a static risk scan

Run:

```bash
python <skill>/scripts/scan_torch_grad_risks.py \
  --root <repo> \
  --json-out <audit_dir>/static_risk_scan.json \
  --md-out <audit_dir>/static_risk_scan.md
```

Review every high-severity result in the active forward/training path. Prioritize:

- gradient sanitization;
- `.detach()`, `.data`, `.item()`, NumPy conversion, Python scalar conversion, and tensor re-wrapping;
- `no_grad` or inference mode in the training path;
- integer casting, indexing, rounding, hard selection, and parameter-dependent control flow;
- fractional powers, square roots, logarithms, division, exponentials, clamps, and branch boundaries;
- in-place state updates and accidental graph truncation.

Consult `references/failure-patterns.md`. Do not mechanically replace all flagged operations.

### 3. Define an audit matrix

Use at least:

1. one representative real-data batch in the actual training dtype;
2. one boundary-stress case covering dry storage, zero precipitation, low flow, and parameter bounds where physically valid;
3. one targeted excitation case for each conditionally active process, such as snow, interception, fast flow, groundwater, or routing;
4. float64 analytical-versus-numerical checking on a small smooth case when feasible;
5. CPU/GPU and AMP checks when those execution modes are used in production.

Declare per-target applicability. A snow parameter may be `not applicable` for a snow-free case, but it must be reachable in a snow-excitation case. Do not excuse zero gradients without an explicit process-based applicability argument.

Follow `references/audit-matrix-and-verdicts.md` for gates and thresholds.

### 4. Build the repository adapter

Create `gradient_adapter.py` with `build_case(...)` as specified in `references/adapter-contract.md`.

Expose both levels when applicable:

- raw trainables: logits, basin-specific parameters, or neural-network weights;
- transformed physical parameters: call `retain_grad()` through the adapter targets;
- optional initial states or forcing tensors when input sensitivity matters.

Use the real loss. Return predictions and named forward checks so the runner can reject non-finite forward output before backward.

### 5. Run the runtime gradient audit

Run a single case first, then the full matrix:

```bash
python <skill>/scripts/run_gradient_audit.py \
  --adapter <audit_dir>/gradient_adapter.py \
  --matrix <audit_dir>/audit_matrix.json \
  --output-dir <audit_dir>
```

Interpret per-target results, not only global gradient norms. Inspect:

- missing gradients;
- NaN/Inf counts;
- exact-zero and nonzero fractions;
- mean, maximum, and L2 magnitude;
- normalized sensitivity relative to parameter range and loss scale;
- consistency across seeds, dtypes, devices, and cases.

A float64 `gradcheck` is supplemental evidence for local Jacobian correctness on smooth inputs. It does not replace float32 real-path auditing.

### 6. Localize any failure

For a reproducible failing case:

1. enable anomaly detection only for the smallest failing sequence;
2. shorten or bisect the time horizon to find the first failing step;
3. retain gradients on physical parameters and selected intermediate states/fluxes;
4. add temporary hooks or finite checks around suspected operations;
5. compare float32 and float64 values at the failing operation;
6. sweep only the relevant epsilon or boundary treatment;
7. distinguish graph disconnection, expected process inactivity, numerical singularity, saturation, loss failure, and recurrent accumulation.

Do not accept “anomaly detection did not raise” as evidence of health when target gradients are non-finite.

### 7. Repair minimally and preserve physics

Prefer a source-level, mathematically justified repair. Examples include evaluating a fractional power on a strictly positive safe base while preserving the intended zero-limit, stabilizing a loss denominator, replacing a graph-breaking scalar conversion, or separating a diagnostic detach from the training path.

Reject patches that merely:

- zero or clip non-finite gradients after backward;
- add a large epsilon without sensitivity evidence;
- remove difficult samples;
- detach recurrent state unintentionally;
- alter parameter bounds to hide the failure;
- replace a physical branch with an unrelated smooth approximation without documenting the changed model.

After each patch, rerun the failing case, then the matrix.

### 8. Verify forward parity and hydrological invariants

Compare pre-fix and post-fix outputs on identical inputs and parameters. Record:

- maximum and mean absolute discharge difference;
- state and flux differences;
- water-balance residual;
- unit-hydrograph normalization, retained tail mass, and causal alignment;
- warm-up and window-boundary behavior;
- dtype/device differences.

Use a reference implementation when available. A gradient-safe patch that materially changes the hydrograph is a model change, not a numerical fix, and requires explicit approval.

### 9. Audit practical learning

When a short training route exists, implement `run_short_training(...)` in the adapter and run:

```bash
python <skill>/scripts/run_learning_audit.py \
  --adapter <audit_dir>/gradient_adapter.py \
  --matrix <audit_dir>/learning_matrix.json \
  --output-dir <audit_dir>
```

Evaluate per parameter:

- displacement from initialization, normalized by its physical range;
- boundary occupancy;
- cross-basin or cross-sample variance;
- loss reduction and finite optimizer state;
- repeatability across seeds.

Separate “differentiable but weakly identifiable” from “not receiving usable gradients.” Do not diagnose identifiability before connectivity and numerical stability pass.

### 10. Add regression gates

Add tests that fail loudly when:

- any applicable target has missing or non-finite gradients;
- every applicable excitation case gives an exact-zero gradient;
- the real training path sanitizes gradients;
- the repaired forward exceeds the declared parity tolerance;
- mass balance or routing invariants regress.

Keep tests small enough for routine CI. Keep larger real-data matrices as explicit pre-experiment gates.

### 11. Issue a verdict and rerun scope

Use only these top-level verdicts:

- `PASS`: all required applicable targets are reachable with finite usable gradients; invariants pass.
- `PASS_WITH_CAVEAT`: gradients are usable, but a documented limitation remains, such as weak sensitivity, intentional truncated BPTT, or untested accelerator mode.
- `FAIL_AUTOGRAD`: graph disconnection or unsupported discrete operation blocks learning.
- `FAIL_NUMERICAL`: non-finite or unstable gradients occur in an applicable case.
- `FAIL_TRAINABILITY`: backward passes, but one or more applicable parameters remain effectively unmoved or saturated under the short-learning audit.
- `NOT_EVALUATED`: required data, execution path, or process excitation is missing.

State which prior experiments are invalidated. Invalidate only results that depend on the failed model/path/configuration, but include all downstream comparisons and conclusions derived from them.

## Evidence discipline

- Cite exact file paths, symbols, and line numbers from the audited repository.
- Record commands and environment versions.
- Distinguish observed facts, root-cause inference, and untested hypotheses.
- Preserve failing fixtures when licensing and data constraints allow.
- Never describe a model as differentiable solely because `loss.backward()` returns without an exception.
