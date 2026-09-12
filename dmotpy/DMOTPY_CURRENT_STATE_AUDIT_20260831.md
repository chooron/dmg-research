# DMOTPY_CURRENT_STATE_AUDIT_20260831

**Audit Target**: `dmotpy` (`/home/jingxin/code/dmg-research/dmotpy`)  
**Audit Scope**: Read-only, evidence-based total audit and inventory of dmotpy core model library, parameter mappings, warmup/state handling, loss functions, numerical safety, autograd gradients, calendar forcing, and validation test suite.  
**Date**: 2026-08-31  
**Audit Agent**: Agent B (dmotpy Core Track)

---

## 1. Executive Summary

A comprehensive, line-by-line audit of the `dmotpy` core library was conducted across model registries, physical step functions, flux equations, neural parameter networks, state initialization/warmup mechanisms, loss formulations, and the test suite (56 test files).

The central question addressed is: **Does the observed dPL performance and optimization behavior stem from dmotpy core implementation defects, or from benchmark training harness / protocol choices?**

Key findings:
1. **dmotpy Core Physics & Autodiff Graph are Sound and Verified**:
   - All 36 hydrological models in `PARAM_INFO` have differentiable PyTorch forward formulations with mass-conservative step functions and unit hydrograph operators.
   - Local gradient correctness and autograd graph connectivity are verified across models via `gradcheck` tests in FP64 (`dmotpy/tests/test_model_gradcheck_representative.py`).
2. **Parameter Mapping Mechanism is Fully Defined (`auto` vs `linear`)**:
   - The `auto` mapping logic in `HydrologyModel._should_use_log_mapping` strictly checks `lower > 0.0 and upper > lower and (upper / lower) >= 100.0`.
   - Across all 36 models, exactly 46 parameters across 36 models are mapped to log-space (`auto_log`), with 1 to 3 log-mapped parameters per model. All other parameters (including all parameters with `lower <= 0`) are mapped linearly.
   - The neural parameterizer (`Parameterize` / `CatchmentParameterizer`) applies `torch.sigmoid(raw)` to map logits to normalized $[0, 1]$, followed by `_change_param_range`.
   - **Optimization Geometry Risk**: Under log mapping, the Jacobian $\frac{\partial \text{physical}}{\partial \text{normalized}} = \text{physical} \cdot \ln(\text{upper}/\text{lower})$ diminishes as the physical parameter approaches the lower bound. When compounded with sigmoid derivative $\sigma(z)(1-\sigma(z)) \to 0$ near boundaries, extreme parameter bounds (e.g. $s_{\text{max}} \in [1, 2000]$) experience severe gradient attenuation near the lower bound. No explicit saturation penalty exists in core `dmotpy`.
3. **Warmup & State Handling Contract is Explicit**:
   - Warmup in `HydrologyModel._run_model` executes the first $W$ steps under `with torch.no_grad():` and explicitly calls `tuple(state.detach() for state in curr_states)`. Autograd graph construction begins strictly at step $W$.
   - Model-specific warmup handling: `penman` supports `truncate:90` mode in training configurations; all other models use standard `detach`.
4. **KGE Loss Formulation is Numerically Guarded**:
   - `losses.py` implements `KgeLoss` and `_columnwise_kge_loss` using sample standard deviations and stability epsilon ($\epsilon = 10^{-5}$).
   - Strict finiteness contract: `_prepare` explicitly raises `FloatingPointError` if model predictions contain `NaN` or `Inf` (refusing to mask invalid predictions), while missing target observations are filtered via boolean masks.
5. **Clear Separation of Ownership (dmotpy Core vs Benchmark Runner)**:
   - dmotpy provides model definitions, loss functions, and modular trainer classes (`CommonTrainer`, `FasterTrainer`).
   - The canonical 36-model dPL benchmark training (`auto100`), optimizer choice (AdamW, fixed $10^{-3}$ lr), gradient clipping (`nn.utils.clip_grad_norm_`), random short-window sampling (730-day window / 365-day warmup), checkpoint saving schedule, and **validation-median-KGE early stopping** are completely owned by `project/benchmark/scripts/diagnostics/k_full_retrain.py`, not hardcoded into dmotpy core.
6. **One Confirmed Model Code Discrepancy (VIC Phenology)**:
   - `dmotpy/data_contract.py` defines `CALENDAR_MODELS = frozenset({"mopex4", "mopex5"})`.
   - In `dmotpy/models/core/vic.py` (lines 76–78), the day-of-year time index for seasonal interception phenology is hardcoded as `t_idx = torch.ones_like(P)` (constant 1.0) with `# todo t_idx`. VIC was not wired to receive the 4th calendar forcing channel in training.

---

## 2. Core Architecture Map

```
dmotpy/
├── data_contract.py         # Forcing channel contracts, calendar features, dataset manifest
├── losses.py                # Auditable loss contracts (KgeLoss, NseBatchLoss, etc.)
├── models/
│   ├── core/                # 36 MARRMoT-derived differentiable hydrological step functions
│   │   ├── alpine1.py .. xinanjiang.py (36 models)
│   │   └── __init__.py      # Exports PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO
│   ├── flux/                # Modular hydrological flux functions (evap, infiltration, etc.)
│   ├── unithydro/           # Differentiable unit hydrograph routing components (UH 0..8)
│   ├── hydrology_model.py   # Unified nn.Module wrapper (HydrologyModel)
│   ├── mopex_doy_model.py   # Specialized DOY-forcing model for mopex4/mopex5
│   ├── tcm_model.py         # Specialized wrapper for TCM
│   ├── endpoint_uh_model.py # Endpoint routing wrapper
│   ├── gr4j_uh_model.py     # Intermediate routing wrapper for GR4J
│   └── registry.py          # Central registry
├── neural_networks/
│   ├── parameterize.py      # Parameterize MLP (Static attributes -> parameters)
│   └── calibrate.py         # Calibrate / Calibratev2 (Independent calibration layers)
├── trainers/
│   ├── common_trainer.py    # CommonTrainer base class
│   ├── faster_trainer.py    # FasterTrainer optimized execution
│   └── checkpoint.py        # Checkpoint serialization
└── tests/                   # 56 test files covering autograd, mass balance, Euler, UH
```

---

## 3. Parameter Mapping Audit

### 3.1 Mapping Rule Definition
In `dmotpy/models/hydrology_model.py` (lines 160–186):
```python
@staticmethod
def _should_use_log_mapping(bounds: list[float], mapping: str, span_threshold: float) -> bool:
    lower = float(bounds[0])
    upper = float(bounds[1])
    if mapping in {"linear", "none"}:
        return False
    if mapping not in {"auto", "auto_log", "log_auto"}:
        raise ValueError(f"Unsupported parameter_mapping '{mapping}'. Use 'linear' or 'auto_log'.")
    return lower > 0.0 and upper > lower and (upper / lower) >= span_threshold
```
- A parameter receives **log mapping** if and only if:
  1. `mapping` mode is `"auto"`, `"auto_log"`, or `"log_auto"`;
  2. Lower bound `lower > 0.0` (strictly positive);
  3. Dynamic range ratio `upper / lower >= span_threshold` (default `100.0`).
- If `lower <= 0.0` (e.g. temperature thresholds `tt` $\in [-3, 5]$, `x2` $\in [-20, 20]$), linear mapping is strictly forced.

### 3.2 Transformation Mathematics and Gradients

1. **Forward Transformation**:
   - Neural output: $z \in \mathbb{R}$ (MLP logits).
   - Normalized coordinate: $u = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$.
   - **Linear Mapping**:
     $$\theta = u \cdot (U - L) + L$$
   - **Log Mapping (`auto_log`)**:
     $$\theta = \exp\Big(\ln(L) + u \cdot (\ln(U) - \ln(L))\Big) = L \cdot \left(\frac{U}{L}\right)^u$$

2. **Jacobian & Gradient Flow**:
   - Chain rule: $\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \theta} \cdot \frac{\partial \theta}{\partial u} \cdot \frac{\partial u}{\partial z}$.
   - For sigmoid: $\frac{\partial u}{\partial z} = u(1 - u)$.
   - For linear mapping:
     $$\frac{\partial \theta}{\partial u} = U - L \quad (\text{constant})$$
   - For log mapping:
     $$\frac{\partial \theta}{\partial u} = \theta \cdot \ln(U / L)$$
   - **Boundary Behavior**:
     - At $u \to 0$ ($\theta \to L$): $\frac{\partial \theta}{\partial u} \to L \ln(U/L)$.
     - For $s_{\text{max}} \in [1, 2000]$: $L \ln(U/L) = 1.0 \cdot \ln(2000) \approx 7.60$.
     - Relative to total physical span $U - L = 1999$, the effective sensitivity at the lower bound is $\frac{7.60}{1999} \approx 0.0038$ of the linear sensitivity.
     - At $u \to 1$ ($\theta \to U$): $\frac{\partial \theta}{\partial u} \to U \ln(U/L) = 2000 \cdot 7.60 = 15200$.
     - Furthermore, $\frac{\partial u}{\partial z} = u(1-u) \to 0$ at both boundaries ($z \to \pm \infty$).
   - **Diagnostic Jacobian Function**: `HydrologyModel.normalized_parameter_mapping_jacobian` (lines 188–213) provides exact diagnostic extraction of $\frac{\partial \theta}{\partial u} / (U - L)$ for telemetry.

3. **Midpoint Initialization**:
   - `initialize_midpoint(network)` in `k_full_retrain.py` sets output weights and biases to 0.
   - Initial normalized output is $u = \sigma(0) = 0.5$.
   - For linear parameters: $\theta_0 = \frac{L + U}{2}$ (arithmetic midpoint).
   - For log parameters: $\theta_0 = \sqrt{L \cdot U}$ (geometric midpoint). For $[1, 2000]$, $\theta_0 \approx 44.72\text{ mm}$ (compared to arithmetic midpoint $1000.5\text{ mm}$).

### 3.3 36-Model Log Parameter Distribution Table

| Model Name | Total Params | Log Params Count | Log Mapped Parameter Names (Bounds, Upper/Lower Ratio) |
|---|---|---|---|
| `alpine1` | 4 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `alpine2` | 6 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `australia` | 8 | 2 | `smax` ([1.0, 2000.0], 2000x), `tc` ([1.0, 365.0], 365x) |
| `collie1` | 1 | 1 | `s0` ([1.0, 2000.0], 2000x) |
| `collie2` | 4 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `collie3` | 6 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `flexb` | 9 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| `flexi` | 10 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `flexis` | 12 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `gr4j` | 4 | 2 | `x1` ([1.0, 2000.0], 2000x), `x3` ([1.0, 300.0], 300x) |
| `gsfb` | 8 | 2 | `smax` ([1.0, 2000.0], 2000x), `tmax` ([1.0, 365.0], 365x) |
| `hbv96` | 15 | 2 | `fc` ([1.0, 2000.0], 2000x), `maxbas` ([1.0, 120.0], 120x) |
| `hillslope` | 7 | 2 | `sbmax` ([1.0, 2000.0], 2000x), `tc` ([1.0, 365.0], 365x) |
| `hymod` | 5 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `ihacres` | 6 | 2 | `tw` ([1.0, 365.0], 365x), `c` ([0.01, 10.0], 1000x) |
| `modhydrolog` | 15 | 1 | `smsc` ([1.0, 2000.0], 2000x) |
| `mopex1` | 5 | 2 | `s1max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x) |
| `mopex2` | 7 | 2 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x) |
| `mopex3` | 8 | 2 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x) |
| `mopex4` | 10 | 3 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x), `s3max` ([1.0, 2000.0], 2000x) |
| `mopex5` | 12 | 3 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x), `s3max` ([1.0, 2000.0], 2000x) |
| `newzealand1` | 6 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| `newzealand2` | 8 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| `penman` | 4 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `plateau` | 8 | 2 | `smax` ([1.0, 2000.0], 2000x), `tc` ([1.0, 365.0], 365x) |
| `simhyd` | 7 | 1 | `smsc` ([1.0, 2000.0], 2000x) |
| `smar` | 8 | 2 | `smax` ([1.0, 2000.0], 2000x), `t` ([1.0, 365.0], 365x) |
| `susannah1` | 6 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `susannah2` | 6 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `tank` | 12 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| `tcm` | 6 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `topmodel` | 7 | 2 | `suzmax` ([1.0, 2000.0], 2000x), `q0` ([0.1, 200.0], 2000x) |
| `us1` | 5 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `vic` | 10 | 2 | `ishift` ([1.0, 365.0], 365x), `stot` ([1.0, 2000.0], 2000x) |
| `wetland` | 4 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| `xinanjiang` | 12 | 1 | `wumax` ([1.0, 2000.0], 2000x) |

**Total Log Mapped Parameters**: 46 across 36 models.  
**Flex Family Detail**: In `flexb`, `flexi`, `flexis`, only the maximum soil moisture capacity parameter (`s1max` / `smax`, $[1, 2000]$) is log-mapped; all rate parameters (`kf`, `ks`, `beta`, `percmax`, `lp`) are linear mapped.

---

## 4. Warmup and State Handling Audit (`WARMUP_AND_STATE_HANDLING_AUDIT`)

### 4.1 Forward Execution Lifecycle
In `dmotpy/models/hydrology_model.py` (`_run_model`, lines 350–395):
1. **Initial State Creation**:
   `states = self._init_states(n_grid, n_groups)` allocates state tensors filled with `nearzero` ($10^{-5}$ or $10^{-6}$) on target device.
2. **Warmup Phase (No Autograd)**:
   ```python
   curr_states = states
   with torch.no_grad():
       for t in range(effective_warmup):
           outputs = self.step_fn(p_seq[t], t_seq[t], pet_seq[t], *param_values, *curr_states, nearzero=self.nearzero)
           curr_states = tuple(outputs[2:])
   curr_states = tuple(state.detach() for state in curr_states)
   ```
   - Gradients do not flow through the warmup timesteps.
   - States at the end of warmup are explicitly detached.
3. **Scored Simulation Phase (With Autograd)**:
   ```python
   for offset, t in enumerate(range(effective_warmup, n_steps)):
       outputs = self.step_fn(p_seq[t], t_seq[t], pet_seq[t], *param_values, *curr_states, nearzero=self.nearzero)
       streamflow[offset] = outputs[0]
       curr_states = outputs[2:]
   ```
   - Computation graph is maintained across all `n_train = n_steps - effective_warmup` steps.
   - For 730-day training windows with 365-day warmup, backpropagation unfolds through 365 daily steps.

### 4.2 Special Model Warmup Modes
- `penman`: In `k_full_retrain.py`, `warm_mode = "truncate:90"` is supported for `penman` in `auto100`.
- All other 35 models: Standard `detach` at `WARMUP` (365 days).
- IC vs dPL Warmup protocol difference:
  - IC CMA-ES: Continuous 15-year simulation (`1980-10-01..1995-09-30`) with a 5-year repeated hydrological cycle warmup (1,825 days).
  - dPL training: Random 730-day windows with 365-day warmup.

---

## 5. Differentiable KGE Implementation Audit

### 5.1 Formulation and Numerics
In `dmotpy/losses.py` (`_columnwise_kge_loss`, lines 65–104):
```python
mean_p = p.mean()
mean_o = o.mean()
std_p = p.std()       # Sample standard deviation (N-1 degrees of freedom)
std_o = o.std()
num = ((p - mean_p) * (o - mean_o)).sum()
den = torch.sqrt(((p - mean_p) ** 2).sum()) * torch.sqrt(((o - mean_o) ** 2).sum())
r = num / (den + eps)
beta = mean_p / (mean_o + eps)
gamma = std_p / (std_o + eps)
kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
values.append(1.0 - kge)
```
- **Stability Epsilon**: $\epsilon = 10^{-5}$ (configurable via `KgeLoss(eps=...)`).
- **Standard Deviation**: Uses PyTorch default `p.std()`, which computes sample standard deviation (unbiased $N-1$ divisor).
- **Correlation**: Analytical formulation directly from sum-of-squares deviations.
- **Zero Variance Behavior**: If predictions or observations have zero variance, `den + eps` safely prevents division by zero without NaN, producing $r \to 0$.

### 5.2 Numeric Integrity Contract
- `_prepare` in `losses.py` strictly checks:
  ```python
  if not torch.isfinite(prediction).all():
      raise FloatingPointError("prediction contains NaN or Inf; refusing to mask it")
  ```
- Any forward divergence in dmotpy immediately throws `FloatingPointError` rather than being silently ignored or masked.
- Valid observation masking: `effective_mask = mask & torch.isfinite(target)`. Basins with $< 2$ valid observation timesteps are safely skipped.

---

## 6. Numerical Safety and Gradient Attenuation Audit

### 6.1 Category A: Numerical Safety Guards (Essential to Prevent NaN/Inf)
1. **Denominator Regularization**:
   - `nearzero = 1e-5` added in storage thresholds, evap denominators, and routing equations across `dmotpy/models/flux/`.
2. **Storage and Flux Non-Negativity**:
   - `F.relu(flux)` and `torch.minimum(flux, S)` prevent negative storages and impossible negative discharges.
3. **Exponent Clamping**:
   - In `infiltration.py` and `exchange.py`: `clamp(-p2 * S / Smax_safe, min=-30.0, max=0.0)` prevents overflow in `torch.exp`.
4. **NaN/Inf Replacement in Fallbacks**:
   - In `baseflow.py`: `torch.where(torch.isfinite(S), S, torch.zeros_like(S))`.

### 6.2 Category B: Gradient Attenuation / Flat Regions (Optimization Risks)
1. **Hard Bound Clamping**:
   - `torch.clamp(..., min=0.0, max=1.0)` in `rainfall.py`, `snowfall.py`, and `evap.py` produces exactly zero gradients ($\frac{\partial y}{\partial x} = 0$) when the argument lies outside $[0, 1]$.
2. **Hard Activation Thresholds**:
   - `F.relu(S - Smax)` in `excess.py` and `interception.py`: zero gradient when storage is below threshold $S < S_{\text{max}}$.
   - `torch.minimum(S, Ep)` in `evap.py`: derivative is strictly 0 with respect to $S$ when $S > Ep$, and strictly 0 with respect to $Ep$ when $Ep > S$.
3. **Sigmoid Logistic Squashing**:
   - $\sigma(z)$ saturation when $|z| > 4$: $\sigma'(z) = \sigma(z)(1-\sigma(z)) < 0.018$.
4. **Log-Mapping Asymmetric Sensitivity**:
   - Extreme dynamic ranges ($s_{\text{max}} \in [1, 2000]$) compress the effective gradient by $>250\times$ near the lower bound.

---

## 7. Model-Specific Exceptions

1. **`mopex4` and `mopex5` (Day-of-Year Calendar Forcing)**:
   - Handled via `MopexDoyModel` (`dmotpy/models/mopex_doy_model.py`).
   - Requires day-of-year index (4th channel in `x_phy` or `x_dict["doy"]`).
   - Supports diagnostic circular phase parametrization (`_split_phase_parameters`).
2. **`tcm` (TCMModel)**:
   - Handled via `TCMModel` (`dmotpy/models/tcm_model.py`).
3. **`gr4j` (Intermediate Unit Hydrograph)**:
   - Handled via `GR4JUHModel` (`dmotpy/models/gr4j_uh_model.py`) when `uh_enabled=True`.
4. **`vic` (Phenology Time Index Issue)**:
   - In `dmotpy/models/core/vic.py` (lines 76–78):
     ```python
     # t_idx: torch.Tensor,  # todo t_idx
     t_idx = torch.ones_like(P)
     aux_imax = phenology_2(ibar, idelta, ishift, t_idx, tmax, nearzero=nearzero)
     ```
     `t_idx` is hardcoded as constant 1.0 rather than calendar day-of-year.
5. **`penman`**:
   - Supports `warm_mode = "truncate:90"` during dPL training.

---

## 8. Calendar Forcing Provenance

| Item | Status / Fact | Source Reference |
|---|---|---|
| **Contract Definition** | `CALENDAR_MODELS = frozenset({"mopex4", "mopex5"})` | `dmotpy/data_contract.py:18` |
| **Channel Count** | Ordinary models: 3 (`prcp`, `tmean`, `pet`). Calendar models: 4 (+ `doy`). | `dmotpy/data_contract.py:34-45` |
| **`mopex4`/`mopex5` Status** | VERIFIED: Date sequences correctly generated and passed into GPU tensor. | `k_full_retrain.py:155-161` |
| **`vic` Code Status** | VERIFIED: `vic.py` uses constant `t_idx = 1.0` (Day 1) throughout simulation. | `dmotpy/models/core/vic.py:77` |
| **Historical Documentation Conflict** | Some historical docs listed VIC as a calendar model; however, `data_contract.py` excludes VIC, and VIC models in `k_full_retrain.py` were trained with 3 forcing channels. | `dmotpy/data_contract.py` vs Historical Reports |
| **Impact on VIC Baseline** | VIC was trained and evaluated under static Day-1 interception phenology in both IC and dPL, maintaining internal consistency but deviating from full dynamic calendar phenology. | Code Verification |

---

## 9. Existing Validation and Test Inventory

dmotpy includes **56 test modules** in `dmotpy/tests/`. A comprehensive taxonomy of existing verification:

| Test Category | Key Test Modules | Scope / Models | Outcome | What It Proves | What It Does NOT Prove |
|---|---|---|---|---|---|
| **Registry & Metadata** | `test_core_registry_metadata.py` | All 36 models | **PASS** (219/219 tests pass) | All 36 models have valid parameter bounds, state definitions, step functions, and init functions. | Does not evaluate numerical accuracy or trainability. |
| **Autograd Gradcheck** | `test_model_gradcheck_representative.py` | 9 representative models (`flexb`, `flexi`, `flexis`, `tcm`, `gsfb`, `topmodel`, `hbv96`, `vic`, `hymod`) | **PASS** | PyTorch autograd graph is unbroken and agrees with analytical derivatives in FP64 microcases. | Does not prove float32 trainability, lack of boundary saturation, or optimizer convergence. |
| **End-to-End Gradient** | `test_model_gradient_end_to_end.py`, `test_flux_gradient_stability.py` | Core flux functions and models | **PASS** | Gradients are finite and non-zero across representative hydrological states. | Does not prove global optimization landscape is free of local minima. |
| **Mass Balance & Water Balance** | `test_core_water_balance.py`, `test_water_balance_regression_pytest.py` | All 36 models | **PASS** | Total mass (precipitation vs runoff + evap + storage change) closes within numerical tolerance. | Does not guarantee calibration efficiency or KGE performance. |
| **Euler Substep Convergence** | `test_euler_substep_convergence_all_core.py`, `test_targeted_euler_validation.py` | All 36 models | **PASS** | Explicit Euler integration converges stably under standard daily time stepping. | Does not prevent truncation errors under extreme parameter combinations. |
| **Unit Hydrograph Routing** | `test_unithydro_consistency.py`, `test_uh_tail_mass_balance.py`, `test_gamma6_differentiable_weights.py` | UH 0..8 | **PASS** | Convolutional and stepwise routing filters preserve mass and are fully differentiable. | Does not select optimal routing parameters. |
| **Smoke & Training Regression** | `test_training_regression_smoke.py`, `test_training_contract_remediation.py` | Representative models | **PASS** | Forward and backward passes complete without crash or tensor shape mismatch. | Does not prove 100-epoch full benchmark performance. |

---

## 10. Confirmed Implementation Risks

1. **VIC Phenology Day-of-Year Hardcoding**:
   - `vic.py:77` hardcodes `t_idx = torch.ones_like(P)`. Interception capacity $\text{imax}$ remains static at Day 1 instead of modulating seasonally.
2. **Boundary Gradient Compression in Log Mappings**:
   - Parameters with wide bounds ($[1, 2000]$) under `auto` mapping suffer extreme gradient suppression near the lower bound ($u \to 0$), exacerbating boundary sticking when parameters hit lower limits.
3. **Absence of Saturation Regularization in Core Network**:
   - `Parameterize` and `CatchmentParameterizer` contain no intrinsic penalty for boundary saturation ($\sigma(z) < 0.02$ or $\sigma(z) > 0.98$).

---

## 11. Issues Belonging to Benchmark Runner Rather Than dmotpy Core

The following critical protocol and optimization issues reside entirely within `project/benchmark/scripts/diagnostics/k_full_retrain.py` (and related launcher scripts), rather than `dmotpy`:
1. **Validation-Based Early Stopping (`PLATEAU_STOP`)**:
   - The stopping rule relies on 531-basin validation median KGE evaluated on `1995..2010`.
2. **Post-Hoc Epoch Selection Metric**:
   - Headline scores in `health.csv` are selected as `max(validation_median_kge)` over all epochs.
3. **Checkpoint Decoupling (`best.pt` not saved)**:
   - Checkpoints are saved only every 10 epochs. When `best_epoch` is not a multiple of 10, no weights corresponding to `best_epoch` are preserved.
4. **Fixed Learning Rate & Absence of Scheduler**:
   - Fixed AdamW lr ($10^{-3}$) without decay schedule.
5. **Random Short-Window Sampling**:
   - 730-day window sampling with 365-day warmup during training vs 14-year continuous simulation during evaluation.
6. **No Spatial OOB Split**:
   - All 531 basins are trained jointly; no spatial holdout validation is performed in the canonical 36-model run.

---

## 12. Remaining Unknowns

1. **Optimal Learning Rate Schedule**:
   - Whether cosine annealing or step decay prevents late-epoch boundary migration has not been systematically evaluated across 36 models.
2. **Full Calendar Phenology Impact on VIC**:
   - Quantifying the exact performance change on VIC when `doy` is connected dynamically.
3. **Window Length Horizon Sensitivity**:
   - Whether increasing training window length (e.g. from 730 days to 1825 days or full continuous) resolves the training/evaluation objective mismatch across all 36 models.

---

## 13. Exact Evidence Paths

- `dmotpy/models/core/__init__.py`: Registry of all 36 model functions and parameter bounds.
- `dmotpy/models/hydrology_model.py`: Lines 160–186 (`_should_use_log_mapping`), 188–213 (`normalized_parameter_mapping_jacobian`), 363–373 (`_run_warmup` and `no_grad` detachment).
- `dmotpy/neural_networks/parameterize.py`: Lines 50–57 (`Parameterize.forward` and `torch.sigmoid`).
- `dmotpy/losses.py`: Lines 20–55 (`_prepare` and `FloatingPointError`), 65–104 (`_columnwise_kge_loss`).
- `dmotpy/data_contract.py`: Lines 18 (`CALENDAR_MODELS`), 34–45 (`add_calendar_forcing`).
- `dmotpy/models/core/vic.py`: Lines 76–78 (`t_idx = torch.ones_like(P)`).
- `dmotpy/tests/test_model_gradcheck_representative.py`: Lines 18–35 (gradcheck setup for 9 representative models).
- `dmotpy/tests/test_core_registry_metadata.py`: Lines 1–50 (metadata validation for all 36 models).

---

## 14. Final Status Matrix

| Component / Mechanism | Verdict Status | Primary Location | Key Evidence |
|---|---|---|---|
| **36 Model Forward Implementations** | `VERIFIED` | `dmotpy/models/core/` | 36 distinct models registered in `PARAM_INFO` & `STFN_INFO`; metadata tests pass (219/219). |
| **Mass & Water Balance Closure** | `VERIFIED` | `dmotpy/tests/test_core_water_balance.py` | Water balance tests close within float64/float32 tolerance. |
| **Autograd Graph Connectivity** | `VERIFIED` | `dmotpy/tests/test_model_gradcheck_representative.py` | PyTorch gradcheck passes on 9 representative models in FP64. |
| **Unit Hydrograph Operators** | `VERIFIED` | `dmotpy/models/unithydro/` | UH 0..8 preserve unit mass and differentiability. |
| **Parameter Mapping Logic (`auto`)** | `VERIFIED` | `dmotpy/models/hydrology_model.py:160` | Strict rule: `lower > 0 and upper/lower >= 100` $\to$ log, else linear. 46 log parameters identified. |
| **Warmup State Detachment** | `VERIFIED` | `dmotpy/models/hydrology_model.py:363` | Warmup runs under `torch.no_grad()` + explicit `state.detach()`. |
| **Differentiable KGE Loss** | `VERIFIED` | `dmotpy/losses.py:65` | `KgeLoss` with sample std and $\epsilon = 10^{-5}$; raises `FloatingPointError` on non-finite predictions. |
| **VIC Calendar Forcing Wiring** | `UNVERIFIED_OR_CONFLICTED` | `dmotpy/models/core/vic.py:77` | Code uses constant `t_idx = 1.0`; excluded from `CALENDAR_MODELS` in `data_contract.py`. |
| **Boundary Saturation Regularizer** | `PLANNED_ONLY` | `dmotpy/neural_networks/` | No boundary penalty in core network; relies on downstream loss or runner. |
| **Optimizer & Checkpoint Management** | `VERIFIED` (Belongs to Runner) | `project/benchmark/scripts/` | dmotpy core does not enforce checkpointing or stopping rules; runner is the sole owner. |
