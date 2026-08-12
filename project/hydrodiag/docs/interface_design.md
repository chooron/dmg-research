# Interface Design

## 1. Uniform Interface

All models share the same call signature:

```python
qsim, aux = model(
    forcings=forcings,
    params=params,
    initial_states=None,
    return_states=False,
)
```

### `forcings`: Dict[str, Tensor]

Required keys:
- `precip`: `[batch, time]` — precipitation (mm/day)
- `pet`: `[batch, time]` — potential evapotranspiration (mm/day)
- `temp`: `[batch, time]` — temperature (degC)

All tensors must share the same device and dtype.

### `params`: Dict[str, Tensor]

- Key is the parameter name (e.g., `"x1"`, `"parBETA"`, `"cn_ctg"`)
- Value is a physical-scale tensor of shape `[batch]`
- Parameters use their natural physical units (mm, degC, dimensionless, etc.)
- No normalization or log-transformation inside the model

### `initial_states`: Optional[Dict[str, Tensor]]

- Key is the state variable name (e.g., `"SNOWPACK"`, `"s_prod"`)
- Value is a tensor of shape `[batch]` (or `[batch, N]` for vector states)
- If `None`, the model initializes sensible defaults (e.g., 50% of capacity)

### Return Value

- `qsim`: `[batch, time]` — simulated streamflow (mm/day)
- `aux`: dict with intermediate variables and optionally final states

## 2. CemaNeige + Runoff Model Composition

### Pattern

```
Raw Forcings (precip, temp, pet)
        │
        ▼
   CemaNeige (snow module)
        │
        ▼
Effective Precip (rain + melt), PET, Temp
        │
        ▼
   GR4J / XAJ (runoff module)
        │
        ▼
   Streamflow (qsim)
```

### Parameter Namespacing

To avoid conflicts, parameters are prefixed:

| Module | Prefix | Example |
|--------|--------|---------|
| CemaNeige | `cn_` | `cn_ctg`, `cn_kf` |
| GR4J | `gr4j_` | `gr4j_x1`, `gr4j_x2` |
| XAJ | `xaj_` | `xaj_k`, `xaj_b` |
| XAJ-CemaNeige | `cn_` + `xaj_` | `cn_ctg`, `xaj_k` |

### XAJ Model Note

This is a XinAnJiang (XAJ) model implementing the core runoff-generation structure.
It does not include Muskingum routing — the hydromodel reference treats routing
as a configurable external module. Channel lag routing is used in this implementation.

### Step Kernel Isolation

CemaNeige and the runoff model each have their own compiled step kernel.
They run sequentially (CemaNeige first over the full time series, then
the runoff model), not interleaved in a single compiled graph.

## 3. How IC/dPL Will Call Models

### IC (Iterative Calibration)

```python
for iteration in range(max_iter):
    params = propose_new_params()
    qsim, _ = model(forcings=data.forcings, params=params)
    loss = compute_objective(qsim, data.obs)
    update_params(params, loss)
```

### dPL (Differentiable Parameter Learning)

```python
params = nn.ParameterDict({name: nn.Parameter(val) for name, val in init_params.items()})
optimizer = torch.optim.Adam(params.parameters())

for epoch in range(epochs):
    qsim, _ = model(forcings=data.forcings, params=params)
    loss = loss_fn(qsim, data.obs)
    loss.backward()
    optimizer.step()
```

Both patterns are supported by the current interface.

## 4. Physical-Scale Parameter Convention

Parameters are always passed in their natural physical scale. No sigmoid,
log, or normalization transform is applied inside the model. This keeps the
interface clean and the model code focused on the hydrological equations.

The calling code (IC/dPL training loop) is responsible for:
- Parameter normalization within bounds
- Applying transformations (log, logit, etc.)
- Clamping to valid ranges
