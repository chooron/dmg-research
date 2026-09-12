# dfuse

Independent tensor-first FUSE implementation area.  The structural catalogue
is extracted from the paper repository into `specs/structures_78.json`; the
single conversion authority is `dfuse.spec` (`get_structure(model_id)`).

The runnable kernel is a fixed-step, one-lumped-elevation-band implementation with differentiable Newton implicit-Euler steps. It exposes active states, union-parameter masks, selected fluxes, state topology, routed Q, and water/snow balance residuals. It does not import upstream Fortran. The reference oracle and fidelity harness live under `project/autofuse`; full multi-band dFUSE snow parity remains gated.

```python
import torch
from dfuse import simulate

forcing = torch.tensor([[5.0, 2.0, 10.0]])  # ppt, pet, temp
result = simulate(84, forcing)
```

The implicit Newton-16 path remains the default. For solver scans, a separate fixed-substep explicit Euler path is available without changing the default:

```python
result = simulate(84, forcing, solver="explicit", n_substeps=24)
```

Each substep evaluates the shared tensor derivative at the current state and advances with `dt_sub = dt_days / n_substeps`; the outer daily loop and daily routing/output convention are unchanged.

The optional `compile_inner=True` path compiles only the fixed-shape tensor `flux/RHS` kernel used inside Newton. `simulate`'s time loop, Newton iterations, Jacobian/autograd, linear solve, and line search remain eager; the default is always eager. The union-state/mask context is shared so structure changes do not create one compiled function per model:

```python
result = simulate(
    84, forcing, implicit_iterations=16,
    compile_inner=True, compile_inner_backend="inductor", compile_inner_fullgraph=True,
    )
```


## dPL output modes

`simulate_coupled_rk2_batched(..., output_mode="lite")` is the training/calibration contract. It executes the same float64 coupled RK2/Heun and FIX_STATES recurrence, retains routed `q` with shape `[B, T]`, and exposes the final active state through `result.final_states` with shape `[B, S]`; it does not retain state, flux, balance, snow, or coupled-diagnostic histories. The optional `q_instantaneous` and `snow` fields are `None` in Lite. `output_mode="full"` retains the complete `[B, T+1, S]` state history and all existing diagnostic histories for validation and scientific analysis. The legacy spelling `output_mode="q_only"` is accepted as an alias for `"lite"`.
Use `compile_diagnostics()` and `reset_compile_diagnostics()` to audit compile, graph, and fallback behavior. `evaluate_implicit_residual(..., compile_residual=True)` is a separate one-residual Level-B probe and is never part of the Newton loop.

The bounded runtime-builder prototype uses the fixed candidate `S1` sequential explicit-Euler order.  `StructureSpec -> GraphSignature -> RuntimeStepBuilder` selects architecture and process composition before compilation; the generated function contains only the selected process paths.  The packed union interface remains `[9 hydro states + snow + 500 routing bins]`, with union `theta` and fixed-shape topographic/routing tensors as inputs.  The outer time loop remains an ordinary Python `for`:

```python
from dfuse import simulate_sequential
result = simulate_sequential(
    210, forcing, order="S1", n_substeps=1,
    compile_step=True, compile_backend="inductor", compile_fullgraph=True,
)
```

`GeneratedStepRegistry` caches eager generated functions by canonical `GraphSignature`; `CompiledStepRegistry` adds device, dtype, fixed input shapes, backend, and `fullgraph` to its key.  Runtime generation uses `compile()`/`exec()` to give different signatures independent Python code objects.  `compile_diagnostics()` reports the runtime records.  Validation is deliberately bounded to mother models 2/108/178/210, with optional smoke models in `project.autofuse.runtime_validation`; it does not start SCE/dPL or a 78/1248 sweep.  `S5` symmetric splitting is intentionally not implemented.
