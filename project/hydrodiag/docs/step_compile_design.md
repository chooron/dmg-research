# Step-Kernel Compilation Design

## 1. Why Compile Only the Step Kernel

Compiling only the single-step kernel (not the full forward or time loop) has several
advantages:

1. **Deterministic compilation**: A pure tensor function with no `self` access, no Python
   object mutations, and no data-dependent control flow can always compile with
   `fullgraph=True`.

2. **Flexible time loop**: The Python `for` loop in `_step_loop` can easily accommodate
   warmup periods, dynamic parameter time-dependence, and other runtime logic that
   would be hard to express in a compiled graph.

3. **Incremental adoption**: Each model can be tested independently. A compilation
   failure in one step kernel does not affect others.

4. **Debuggability**: The eager Python loop makes it easy to insert logging, checkpointing,
   or inspection at any timestep.

## 2. Step Kernel Summary

| Model | Step Kernel Name | Key Inputs | Key Outputs |
|-------|-----------------|------------|-------------|
| HBV | `_hbv_step` | precip_t, temp_t, pet_t, 5 states, 12 params | q_t, 5 new states |
| GR4J | `_gr4j_step` | precip_t, pet_t, s_prod, s_route, uh1/uh2 bufs, 3 params (x1,x2,x3), pre-computed UH ords | q_t, 4 new states |
| XAJ | `_xaj_step` | precip_t, pet_t, 8 states, 15 params | q_t, 15 outputs |
| CemaNeige | `_cemaneige_step` | precip_t, temp_t, 2 states, 2 params, G threshold | outflow_t, 2 new states, SCA, rain_t, melt_t |
| CemaNeigeHyst | `_cemaneige_hyst_step` | precip_t, temp_t, 4 states, 4 params, psol_annual | outflow_t, 4 new states, rain_t, melt_t |

### GR4J Note

The GR4J step kernel does NOT take x4 directly. Instead, UH ordinates are
pre-computed outside the step loop using a differentiable S-curve difference
method (`models/unit_hydro.py`). This keeps x4 differentiable while avoiding
`.item()` and Python control flow in the compiled kernel. The sequential UH
buffer updates occur within the compiled step kernel using the pre-computed
ordinates.

### XAJ Note

XAJ core processes are faithfully implemented from the hydromodel reference.
Muskingum routing is not included — the reference treats routing as a configurable
external module. Channel lag routing is used in this implementation. See
`docs/model_source_notes.md` for details.

## 3. Step Kernel Input/Output Convention

Every step kernel:

- Receives current-timestep forcing scalars (shape `[batch]`)
- Receives current state tensors (shape `[batch]` or `[batch, max_len]`)
- Receives physical parameter tensors (shape `[batch]`)
- Receives necessary constants (shape `[]`)
- Returns output flow at timestep (shape `[batch]`)
- Returns updated state tensors

The step kernel must NOT:
- Access `self`
- Modify Python object state
- Use numpy
- Execute data-dependent control flow that varies across the batch

## 4. `_step_loop` Design

Each model's `_step_loop`:

```python
def _step_loop(self, ...,):
    qsim = torch.zeros(batch, nsteps, ...)
    for t in range(nsteps):
        qsim[:, t], state1, state2, ... = self._step(
            forcing[:, t], state1, state2, ...
        )
    return qsim, (state1, state2, ...)
```

The loop iterates sequentially because hydrological processes are inherently
time-sequential. This is the correct design for time-stepping simulation.

## 5. Why Silent Fallback Is Forbidden

Silent fallback (catching compile exceptions and falling back to eager mode) is
forbidden because:

1. It masks compilation bugs that may cause incorrect graph traces.
2. It makes debugging impossible — the developer never knows if the compiled
   or eager code path is running.
3. It can lead to silent performance degradation.
4. Subtle numerical differences between compiled and eager paths may go unnoticed.

If a step kernel cannot compile with `fullgraph=True`, the code must be fixed.

## 6. Current Compilation Status

All four step kernels compile successfully with `torch.compile(..., fullgraph=True)`:

- [x] `_hbv_step` — compiles and matches eager
- [x] `_gr4j_step` — compiles and matches eager
- [x] `_xaj_step` — compiles and matches eager
- [x] `_cemaneige_step` — compiles and matches eager
- [x] `_cemaneige_hyst_step` — compiles and matches eager
