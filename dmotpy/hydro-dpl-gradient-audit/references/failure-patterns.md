# PyTorch hydrological gradient failure patterns

Static presence of an operation is not proof of failure. Reproduce the issue in the active path and inspect values at the failing state.

## 1. Graph disconnection

### Tensor to Python or NumPy

Risky forms:

```python
x.item()
float(x)
int(x)
x.detach().cpu().numpy()
torch.tensor(existing_tensor)
```

These can remove a parameter-dependent value from the graph. Keep control values as tensors when gradients are required. Use detached conversions only for logging or diagnostics outside the loss path.

### Intentional versus accidental detach

`state = state.detach()` can be intentional truncated BPTT. Record the truncation boundary and verify it matches the experiment design. A detach inside every hydrological time step usually destroys temporal gradients.

### No-grad contexts

`torch.no_grad()` and `torch.inference_mode()` are valid for evaluation but invalid around the training forward. Verify call-site scope rather than deleting them globally.

## 2. Discrete and parameter-dependent structure

Operations such as `argmax`, integer casting, hard indexing, rounding, and using a learned value as a loop length are non-differentiable with respect to the selected structure. Use a differentiable relaxation only when it represents the intended model, or classify the quantity as non-learnable.

Boolean masks independent of learnable parameters are normally safe. Masks produced from learnable parameters create piecewise-constant selection and require explicit design justification.

## 3. Fractional powers near zero

For `y = x**a`, gradients with respect to `x` or `a` can be singular or undefined at `x=0`, especially for non-integer `a`.

Diagnose:

- log the minimum base before the power;
- count exact zeros in the training dtype;
- compare float32 and float64;
- identify whether zero is physically valid;
- inspect gradients with respect to both base and exponent.

A safe rewrite must preserve the intended zero-limit and water balance. Do not blindly replace `x` by `x + 1e-5` everywhere. Consider a branch with an analytically justified zero value and a strictly positive evaluation base only on the active branch.

## 4. Square root, logarithm, reciprocal, and division

`sqrt(0)`, `log(0)`, and division by zero or a tiny storage can produce infinite or unstable derivatives. Stabilize the mathematical expression at the physically valid limit, not only the final gradient.

For normalized losses such as NSE/KGE variants, inspect denominator variance and standard deviation. Mask constant-observation windows or define a documented stable loss policy.

## 5. `torch.where` and inactive branches

`torch.where(mask, a, b)` selects outputs, but both branches are evaluated before selection. A branch containing invalid values may contaminate backward depending on the operation and graph.

Make both branches numerically valid before selection. Do not assume an inactive branch can contain `log(negative)`, `0/0`, or a fractional power of a negative base.

## 6. Clamp, min/max, and saturation

`clamp`, `minimum`, `maximum`, and ReLU are piecewise differentiable. They are not automatically broken, but gradients can be zero on active bounds.

Audit:

- hit rate at each bound;
- duration of bound occupancy through time;
- whether the raw parameter mapper saturates;
- whether bounds are physical constraints or numerical bandages.

`sigmoid(logit).clamp(eps, 1-eps)` can create two saturation layers. Prefer a well-scaled transform and initialization; measure raw-logit and physical-parameter gradients separately.

## 7. Exponential overflow and underflow

`exp`, softplus, gamma kernels, and temperature-index formulas can overflow or underflow in float32. Inspect argument ranges at parameter bounds. Use algebraically stable forms and justified parameter limits.

## 8. In-place recurrent updates

In-place operations on tensors needed for backward can trigger version errors or silently make reasoning difficult. Distinguish safe buffer writes from mutation of graph values. Prefer explicit next-state tensors in model logic and test long horizons.

Preallocating an output tensor and assigning differentiable values can preserve a `CopySlices` graph, so do not classify it as a failure without runtime evidence.

## 9. Warm-up, masking, and loss reachability

A parameter can influence only warm-up states while the implementation detaches at the warm-up boundary. Observation masks can also remove all loss-bearing timesteps for a process. Audit gradient reachability across warm-up and evaluation periods separately.

## 10. Unit-hydrograph and routing kernels

Audit:

- differentiability of shape/scale parameters;
- kernel normalization;
- retained mass before truncation and renormalization;
- causal index alignment;
- integer lag or hard support selection;
- parameter-dependent kernel length;
- float32 behavior near zero support.

A normalized but heavily truncated kernel can be differentiable yet physically distorted.

## 11. Gradient sanitization and clipping

Gradient clipping is not a repair for NaN/Inf. Apply clipping only after proving gradients are finite and record the unclipped norm.

Any `nan_to_num` or non-finite replacement on gradients must be converted into a loud assertion or counted diagnostic during auditing. Training that continued after sanitization does not prove all parameters learned.

## 12. AMP and mixed precision

Autocast can lower precision for sensitive operations. Test full float32 and production AMP separately. Use selective autocast exclusion only for operations with demonstrated precision failure, and verify forward parity and throughput implications.
