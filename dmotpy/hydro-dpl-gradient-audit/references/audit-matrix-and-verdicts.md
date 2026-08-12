# Audit matrix and verdict rules

## 1. Evidence ladder

### Gate A — active code-path integrity

Pass only when the audited model class, parameter mapper, loss, and optimizer path are the ones used by the experiment command. Flag duplicate or legacy implementations.

### Gate B — forward finiteness

Pass only when discharge, loss-bearing states/fluxes, and the scalar loss are finite for all required cases. A non-finite forward is not a gradient problem; fix it first.

### Gate C — graph reachability

For each applicable target, require a gradient tensor in at least one declared excitation case. Missing gradients indicate disconnection, unsupported discrete logic, or an adapter error.

### Gate D — numerical gradient usability

For each applicable target across the matrix:

- fail on any NaN or Inf gradient in a required case;
- fail when every applicable case is exactly zero;
- allow zero gradients in individual non-exciting cases when applicability is justified;
- flag extremely small normalized sensitivity as a trainability warning, not automatically as non-differentiability.

Use normalized sensitivity when a physical range is available:

\[
S = \frac{\operatorname{mean}(|\partial L/\partial p|)\,(p_{max}-p_{min})}
         {\max(|L|, \epsilon)}.
\]

Interpret it comparatively across targets and cases. Do not impose a universal hydrological threshold without scale evidence.

### Gate E — local derivative agreement

Use `torch.autograd.gradcheck` or directional finite differences on a small float64 smooth case. Pass when analytical and numerical derivatives agree under justified tolerances. Do not run this at branch boundaries or treat it as evidence of float32 trainability.

### Gate F — forward and physical invariants

After a patch, require declared forward parity, water-balance tolerance, routing normalization, and causal alignment. Tolerances must reflect the repository's existing numerical fidelity standard.

### Gate G — practical learning

In a short representative run, require finite optimizer state and evidence that applicable parameters can move. Diagnose weak sensitivity, saturation, and non-identifiability separately.

## 2. Minimum case set

| Case | Purpose | Typical content |
|---|---|---|
| realistic mixed batch | real training behavior | real forcing, observation mask, warm-up, actual loss |
| dry/zero boundary | expose singularities | zero precipitation, dry storages, low flow |
| wet/high-flow boundary | expose overflow/saturation | high precipitation, near-capacity storages |
| process excitation | prove conditional reachability | snowmelt, interception, fast flow, groundwater, routing |
| smooth float64 microcase | derivative agreement | short horizon away from thresholds |
| production accelerator | execution parity | CUDA and/or AMP if used |

## 3. Classification of zero gradients

- `EXPECTED_INACTIVE`: process is explicitly not applicable in this case.
- `PIECEWISE_ZERO`: active branch has a legitimate zero derivative at this state; require another excitation case.
- `SATURATED_MAPPING`: raw-to-physical transform suppresses gradient near a bound.
- `GRAPH_BREAK`: target does not influence loss because the graph is severed.
- `LOSS_MASKED`: all target influence is removed by warm-up or observation masking.
- `NUMERICAL_UNDERFLOW`: values collapse below dtype resolution.
- `WEAK_SENSITIVITY`: finite nonzero gradients exist but are consistently tiny relative to scale.
- `UNRESOLVED`: evidence is insufficient.

## 4. Top-level verdicts

- `PASS`: Gates A–F pass; Gate G passes when required.
- `PASS_WITH_CAVEAT`: core gates pass with a bounded documented limitation.
- `FAIL_AUTOGRAD`: Gate C fails because the graph is disconnected or discrete.
- `FAIL_NUMERICAL`: Gate B or D fails because required values or gradients are non-finite/unstable.
- `FAIL_TRAINABILITY`: Gates B–F pass, but Gate G shows effective freezing or saturation.
- `NOT_EVALUATED`: a required gate could not be run.

## 5. Prior-result implications

- A graph or numerical failure invalidates training results from the affected model/configuration and all comparisons derived from those results.
- A process-specific excitation gap invalidates claims about that process, not necessarily the entire model.
- A production-only CUDA/AMP failure invalidates production runs but not verified CPU float32 evidence.
- A forward-changing repair creates a new model version; do not merge its results with the prior implementation without rerunning fidelity and experiments.
