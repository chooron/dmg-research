# `McMlpModel` Interface Contract

This document describes how `McMlpModel` must interact with:

- `implements/causal_dpl_model.py`
- `models/hbv_static.py`

## Required tensor shapes

### Input to `McMlpModel`

`CausalDplModel.forward()` calls:

```python
parameters = self.nn_model(data_dict["xc_nn_norm"])
```

So, for the causal dPL pipeline, `McMlpModel` must accept:

- `xc_nn_norm` with shape `[T, B, nx]`
  - `T`: time steps
  - `B`: basins / batch items
  - `nx`: number of static attributes

`McMlpModel` also supports direct 2D input `[B, nx]` for isolated testing.

### Output from `McMlpModel`

`HbvStatic._unpack()` expects:

```python
parameters: [T, B, N_PHY * nmul + N_ROUTE]
```

Therefore `McMlpModel` must return:

- `[T, B, ny]` for 3D input
- `[B, ny]` for 2D input

where:

- `ny = phy_model.learnable_param_count`
- for `HbvStatic`, `ny = N_PHY * nmul + N_ROUTE`

With the current static-HBV setup:

- `N_PHY = 12`
- `N_ROUTE = 2`
- `nmul = 1`
- so `ny = 14`

## Why time broadcasting is required

`HbvStatic` uses only the final timestep of the parameter tensor:

```python
p = parameters[-1, :, :self.N_PHY * self.nmul]
r = parameters[-1, :, self.N_PHY * self.nmul:]
```

Even though the parameters are basin-static, the tensor still has to be shaped
as `[T, B, ny]` so that it can pass through the existing dPL interface without
special-casing `CausalDplModel` or `HbvStatic`.

`McMlpModel` therefore:

1. pools the static input across time (`last` by default),
2. predicts one basin-level parameter vector `[B, ny]`,
3. repeats it across `T` to produce `[T, B, ny]`.

## Configuration rules for the `HbvStatic` pathway

To keep the interface correct, the following settings are required:

```yaml
model:
  phy:
    nmul: 1
  nn:
    name: McMlpModel
    forcings: []
    output_activation: sigmoid
```

### Why `forcings: []`

`McMlpModel` is used here only for static basin attributes. Time-varying
forcings belong to `x_phy`, which is consumed by `HbvStatic`, not by the
parameter MLP.

### Why `output_activation: sigmoid`

`HbvStatic` maps NN outputs into physical parameter ranges through
`change_param_range(...)`, which assumes normalized inputs in `[0, 1]`.
For this pathway, `sigmoid` is the compatible activation.

## Tests

The interface is covered by:

- `tests/test_mc_mlp_interface.py`

Run with:

```bash
PYTHONPATH=/workspace/autoresearch /workspace/autoresearch/.venv/bin/python -m unittest tests.test_mc_mlp_interface
```
