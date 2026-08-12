# Adapter contract

## Purpose

Use a thin repository-specific adapter to expose the real hydrological training graph to the bundled generic audit runners without forcing a universal model API.

## Required function: `build_case`

```python
from __future__ import annotations

from typing import Any, Mapping
import torch


def build_case(
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build one complete forward-and-loss graph for an audit case."""
    ...
```

Return a mapping with these keys:

| Key | Required | Type | Meaning |
|---|---:|---|---|
| `loss` | yes | scalar `torch.Tensor` | The actual differentiable training loss. |
| `targets` | yes | mapping `name -> Tensor` | Raw trainables, transformed physical parameters, states, or inputs whose gradients must be audited. |
| `predictions` | no | Tensor or mapping of Tensors | Outputs that must be finite before backward. |
| `applicability` | no | mapping `name -> bool` | Whether a target is physically expected to be active in this case. Default: `True`. |
| `parameter_ranges` | no | mapping `name -> [lower, upper]` | Physical ranges used for normalized sensitivity. |
| `target_groups` | no | mapping `name -> string` | Examples: `raw_trainable`, `physical_parameter`, `initial_state`, `forcing`. |
| `forward_checks` | no | mapping `name -> Tensor/number/bool` | Named finite or invariant checks for reporting. |
| `metadata` | no | JSON-compatible mapping | Basin IDs, horizon, warm-up, loss name, process excitation, commit, etc. |

### Target requirements

- Return the exact tensor participating in the graph.
- Return leaf trainables directly.
- Call `retain_grad()` on non-leaf physical parameters before returning them, or allow the runner to call it when possible.
- Use stable, unique names. Prefix levels, for example `raw.encoder.weight`, `raw.logit_k`, and `physical.k`.
- Do not clone, detach, convert to NumPy, or reconstruct targets with `torch.tensor(existing_tensor)`.
- Do not call backward inside `build_case`.
- Do not sanitize gradients.

### Applicability

Mark a target false only when the process is physically inactive by construction for that case. Examples:

- snow parameters in a strictly snow-free warm-climate excitation;
- interception storage capacity when precipitation is identically zero;
- a routing branch disabled by the model configuration.

Do not mark a target false merely because its observed gradient is zero.

## Optional function: `run_short_training`

```python
def run_short_training(
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
    steps: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    ...
```

Return:

| Key | Required | Type |
|---|---:|---|
| `initial_parameters` | yes | mapping `name -> Tensor` |
| `final_parameters` | yes | mapping `name -> Tensor` |
| `parameter_ranges` | no | mapping `name -> [lower, upper]` |
| `loss_history` | no | sequence of finite numbers |
| `optimizer_state_finite` | no | bool |
| `applicability` | no | mapping `name -> bool` |
| `required_to_move` | no | mapping `name -> bool`; declare only parameters expected to move in this short case |
| `metadata` | no | JSON-compatible mapping |

Return physical parameters when possible. To audit raw and physical levels, use separate names.

## Matrix format

`run_gradient_audit.py` accepts:

```json
{
  "runs": [
    {
      "name": "real_float32_seed0",
      "device": "cpu",
      "dtype": "float32",
      "seed": 0,
      "config": {"case": "realistic", "basins": ["01013500"], "days": 730}
    },
    {
      "name": "snow_excitation",
      "device": "cpu",
      "dtype": "float32",
      "seed": 0,
      "config": {"case": "snow_excitation"}
    }
  ]
}
```

`run_learning_audit.py` uses the same structure and accepts a per-run `steps` field.

## Minimal example

```python
from pathlib import Path
from typing import Any, Mapping
import torch

# Import the repository model here.


def build_case(*, device: str, dtype: torch.dtype, seed: int,
               config: Mapping[str, Any]):
    torch.manual_seed(seed)

    raw = torch.nn.Parameter(torch.zeros(4, device=device, dtype=dtype))
    physical = torch.sigmoid(raw)
    physical.retain_grad()

    forcing = make_forcing(config).to(device=device, dtype=dtype)
    obs = make_observation(config).to(device=device, dtype=dtype)
    sim = real_model_forward(forcing=forcing, parameters=physical)
    loss = real_training_loss(sim, obs)

    return {
        "loss": loss,
        "targets": {
            "raw.logits": raw,
            "physical.parameters": physical,
        },
        "predictions": {"discharge": sim},
        "applicability": {
            "raw.logits": True,
            "physical.parameters": True,
        },
        "parameter_ranges": {
            "physical.parameters": [0.0, 1.0],
        },
        "target_groups": {
            "raw.logits": "raw_trainable",
            "physical.parameters": "physical_parameter",
        },
        "metadata": {
            "case": config.get("case", "unknown"),
            "source": str(Path(__file__).resolve()),
        },
    }
```
