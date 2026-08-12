# Codex repository execution protocol

## 1. Inspect before editing

Record:

- repository status and current commit;
- package/environment files;
- model and training entrypoints;
- existing tests and numerical reference fixtures;
- dataset access constraints;
- existing audit or debug utilities.

Do not overwrite unrelated uncommitted work. Do not reformat broad files during a numerical patch.

## 2. Reproduce the actual command

Find the exact configuration used for training. Confirm imports at runtime when duplicate names exist. Capture dtype, device, AMP, seeds, batch dimensions, sequence length, warm-up, and observation masking.

## 3. Create a small audit surface

Add a repository-local adapter and small fixtures. Prefer a real short data slice plus synthetic excitation cases. Keep sensitive or large data outside the Skill package.

## 4. Preserve a pre-fix baseline

Before changing model code, save:

- forward outputs and selected states/fluxes;
- mass-balance summary;
- failing gradient report;
- exact command and environment.

Use a temporary test, fixture, or serialized tensor set that can be rerun after the patch.

## 5. Patch minimally

Edit the smallest responsible function. Keep equations recognizable. Add comments only where the gradient-safe form is non-obvious. Avoid optimizer or global epsilon changes unless the root cause is there.

## 6. Validate in increasing scope

1. smallest failing case;
2. all targeted excitation cases;
3. representative real-data batch;
4. float64 derivative check;
5. CPU/GPU/AMP production modes;
6. existing unit and integration tests;
7. short learning audit.

Stop and report if a broader test invalidates the proposed root cause.

## 7. Deliver evidence, not only code

Report:

- observed failure;
- exact source location;
- mechanism;
- patch;
- pre/post gradient evidence;
- forward and mass-balance parity;
- tests run and not run;
- affected historical experiments and required reruns.
