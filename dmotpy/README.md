# dmotpy: Differentiable Hydrological Modeling in PyTorch

`dmotpy` is a high-performance, differentiable Python library implementing **36 conceptual lumped hydrological models** derived from the MARRMoT framework. Designed for **differentiable parameter learning (dPL)**, sensitivity analysis, and gradient-based optimization, `dmotpy` provides end-to-end autograd graph continuity, mass-conservative numerical integration, and modular routing operators.

---

## 1. Core Architecture

```
dmotpy/
├── README.md                # Package documentation and contract guide
├── DMOTPY_CURRENT_STATE_AUDIT_20260831.md  # Detailed forensic state audit document
├── data_contract.py         # Forcing channel contracts, calendar features, dataset manifest
├── losses.py                # Auditable loss contracts (KgeLoss, NseBatchLoss, etc.)
├── models/
│   ├── core/                # 36 MARRMoT-derived differentiable step functions
│   │   ├── alpine1.py .. xinanjiang.py
│   │   └── __init__.py      # Exports PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO
│   ├── flux/                # Modular hydrological flux functions (evap, infiltration, baseflow, etc.)
│   ├── unithydro/           # Differentiable unit hydrograph routing components (UH 0..8)
│   ├── hydrology_model.py   # Unified nn.Module wrapper (HydrologyModel)
│   ├── mopex_doy_model.py   # Specialized DOY-forcing model for mopex4/mopex5
│   ├── tcm_model.py         # Specialized wrapper for TCM
│   ├── endpoint_uh_model.py # Endpoint routing wrapper
│   ├── gr4j_uh_model.py     # Intermediate routing wrapper for GR4J
│   ├── special_models.py    # Factory dispatch for non-standard routing/forcing models
│   └── registry.py          # Central model registry
├── neural_networks/
│   ├── parameterize.py      # Parameterize MLP (Static catchment attributes -> physical parameters)
│   └── calibrate.py         # Calibrate / Calibratev2 (Independent catchment calibration layers)
├── trainers/
│   ├── common_trainer.py    # CommonTrainer base class
│   ├── faster_trainer.py    # FasterTrainer optimized batch execution
│   └── checkpoint.py        # Checkpoint serialization
├── validation_results/      # Official paper-ready and benchmark audit reports & metrics
│   ├── gmd_3_1_stage1_fidelity/
│   ├── gmd_3_1_stage2_discretization_smoothing/
│   ├── gmd_3_1_stage2c_noninvasive_harness_repair/
│   ├── euler_convergence_final/
│   ├── flux_gradient_stability/
│   ├── tost_equivalence/
│   └── unithydro_consistency/
└── tests/                   # Consolidated 8-category hydrological reasonableness test suite
    ├── test_01_registry_and_metadata.py
    ├── test_02_autograd_gradcheck.py
    ├── test_03_flux_and_gradient_stability.py
    ├── test_04_water_balance.py
    ├── test_05_euler_convergence.py
    ├── test_06_unithydro_routing.py
    ├── test_07_dpl_training_contract.py
    └── test_08_validation_artifacts.py
```

---

## 2. 36-Model Inventory & Parameter Distribution

All 36 models provide mass-conservative step functions integrated via explicit Euler stepping:

| # | Model | Parameters | States | Log Params Count | Log-Mapped Parameters (Bounds, Ratio) |
|---|---|---|---|---|---|
| 1 | `alpine1` | 4 | 2 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 2 | `alpine2` | 6 | 3 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 3 | `australia` | 8 | 4 | 2 | `smax` ([1.0, 2000.0], 2000x), `tc` ([1.0, 365.0], 365x) |
| 4 | `collie1` | 1 | 1 | 1 | `s0` ([1.0, 2000.0], 2000x) |
| 5 | `collie2` | 4 | 2 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 6 | `collie3` | 6 | 3 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 7 | `flexb` | 9 | 4 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| 8 | `flexi` | 10 | 4 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 9 | `flexis` | 12 | 5 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 10 | `gr4j` | 4 | 2 | 2 | `x1` ([1.0, 2000.0], 2000x), `x3` ([1.0, 300.0], 300x) |
| 11 | `gsfb` | 8 | 3 | 2 | `smax` ([1.0, 2000.0], 2000x), `tmax` ([1.0, 365.0], 365x) |
| 12 | `hbv96` | 15 | 5 | 2 | `fc` ([1.0, 2000.0], 2000x), `maxbas` ([1.0, 120.0], 120x) |
| 13 | `hillslope` | 7 | 4 | 2 | `sbmax` ([1.0, 2000.0], 2000x), `tc` ([1.0, 365.0], 365x) |
| 14 | `hymod` | 5 | 5 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 15 | `ihacres` | 6 | 3 | 2 | `tw` ([1.0, 365.0], 365x), `c` ([0.01, 10.0], 1000x) |
| 16 | `modhydrolog` | 15 | 6 | 1 | `smsc` ([1.0, 2000.0], 2000x) |
| 17 | `mopex1` | 5 | 3 | 2 | `s1max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x) |
| 18 | `mopex2` | 7 | 4 | 2 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x) |
| 19 | `mopex3` | 8 | 4 | 2 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x) |
| 20 | `mopex4` | 10 | 4 | 3 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x), `s3max` ([1.0, 2000.0], 2000x) |
| 21 | `mopex5` | 12 | 4 | 3 | `s2max` ([1.0, 2000.0], 2000x), `is_time` ([1.0, 365.0], 365x), `s3max` ([1.0, 2000.0], 2000x) |
| 22 | `newzealand1` | 6 | 3 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| 23 | `newzealand2` | 8 | 4 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| 24 | `penman` | 4 | 2 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 25 | `plateau` | 8 | 4 | 2 | `smax` ([1.0, 2000.0], 2000x), `tc` ([1.0, 365.0], 365x) |
| 26 | `simhyd` | 7 | 3 | 1 | `smsc` ([1.0, 2000.0], 2000x) |
| 27 | `smar` | 8 | 6 | 2 | `smax` ([1.0, 2000.0], 2000x), `t` ([1.0, 365.0], 365x) |
| 28 | `susannah1` | 6 | 3 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 29 | `susannah2` | 6 | 3 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 30 | `tank` | 12 | 4 | 1 | `s1max` ([1.0, 2000.0], 2000x) |
| 31 | `tcm` | 6 | 3 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 32 | `topmodel` | 7 | 4 | 2 | `suzmax` ([1.0, 2000.0], 2000x), `q0` ([0.1, 200.0], 2000x) |
| 33 | `us1` | 5 | 3 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 34 | `vic` | 10 | 3 | 2 | `ishift` ([1.0, 365.0], 365x), `stot` ([1.0, 2000.0], 2000x) |
| 35 | `wetland` | 4 | 2 | 1 | `smax` ([1.0, 2000.0], 2000x) |
| 36 | `xinanjiang` | 12 | 4 | 1 | `wumax` ([1.0, 2000.0], 2000x) |

**Total Log-Mapped Parameters**: Exactly 46 parameters across the 36 models. All other parameters (including any parameter with lower bound $\le 0$) are strictly linear-mapped.

---

## 3. Mathematical & Numerical Contracts

### 3.1 Parameter Mapping Contract (`auto_log` vs `linear`)

In `HydrologyModel._should_use_log_mapping`:
- A parameter uses **log mapping** if and only if:
  1. Mapping configuration is `"auto"`, `"auto_log"`, or `"log_auto"`.
  2. Lower bound $L > 0.0$ (strictly positive).
  3. Dynamic range ratio $U / L \ge \text{span\_threshold}$ (default `100.0`).
- Otherwise, **linear mapping** is strictly enforced.

#### Transformations:
Let $u = \sigma(z) \in (0, 1)$ be the normalized parameter output from the neural parameterizer:
- **Linear**:
  $$\theta = u \cdot (U - L) + L, \quad \frac{\partial \theta}{\partial u} = U - L$$
- **Log**:
  $$\theta = \exp\Big(\ln(L) + u \cdot (\ln(U) - \ln(L))\Big) = L \cdot \left(\frac{U}{L}\right)^u, \quad \frac{\partial \theta}{\partial u} = \theta \cdot \ln(U / L)$$

### 3.2 Warmup and State Detachment

Warmup execution in `HydrologyModel._run_model`:
1. The first $W$ steps execute inside `with torch.no_grad():`.
2. At step $W$, all state tensors are explicitly detached via `tuple(state.detach() for state in curr_states)`.
3. The autograd computation graph is strictly constructed only during the scoring period ($t \ge W$).

### 3.3 Differentiable KGE Loss (`KgeLoss`)

`losses.py` implements the Kling-Gupta Efficiency (KGE) as a differentiable PyTorch loss:
$$KGE = 1 - \sqrt{(r - 1)^2 + (\beta - 1)^2 + (\gamma - 1)^2}$$
where:
- $r = \frac{\sum (p - \bar{p})(o - \bar{o})}{\sqrt{\sum (p - \bar{p})^2}\sqrt{\sum (o - \bar{o})^2} + \epsilon}$
- $\beta = \frac{\bar{p}}{\bar{o} + \epsilon}$
- $\gamma = \frac{\text{std}(p)}{\text{std}(o) + \epsilon}$ with sample standard deviation ($N-1$ divisor).
- Stability parameter: $\epsilon = 10^{-5}$.

**Numeric Integrity Contract**:
`_prepare` in `losses.py` strictly checks:
```python
if not torch.isfinite(prediction).all():
    raise FloatingPointError("prediction contains NaN or Inf; refusing to mask it")
```
Any non-finite model prediction raises an immediate exception rather than being silently ignored.

---

## 4. Special Models & Known Caveats

1. **`mopex4` and `mopex5` (Day-of-Year Calendar Forcing)**:
   - Requires a 4th forcing channel: `doy` (calendar day-of-year $\in [1, 366]$).
   - Handled via `MopexDoyModel` (`dmotpy/models/mopex_doy_model.py`).
2. **`vic` (Phenology Time Index)**:
   - In `dmotpy/models/core/vic.py`, the day-of-year time index for seasonal interception phenology uses static Day-1 phenology (`t_idx = torch.ones_like(P)`). Under extreme forcing, minor clipping loss occurs.
3. **`penman` (Warmup Mode)**:
   - Supports `warm_mode = "truncate:90"` during training configurations.
4. **`gr4j`**:
   - Implements intermediate unit hydrograph routing (`GR4JUHModel`).

---

## 5. Hydrological Reasonableness Test Suite

The test suite in `dmotpy/tests/` is consolidated into **8 core verification modules** directly corresponding to Section 9 of the State Audit:

| Suite File | Category | Tested Scope | Status |
|---|---|---|---|
| `test_01_registry_and_metadata.py` | Registry & Metadata | All 36 models, parameter bounds, state definitions, step signatures | **PASS** (219 tests) |
| `test_02_autograd_gradcheck.py` | Autograd Gradcheck | FP64 analytical vs autograd gradcheck on representative model families | **PASS** (11 tests) |
| `test_03_flux_and_gradient_stability.py` | Flux & Gradient Stability | Non-negativity, gradient finiteness across all 36 models, boundary singularities | **PASS** (39 tests) |
| `test_04_water_balance.py` | Mass & Water Balance | Physical conservation $\Delta S = P - E_a - Q$ across 36 models | **PASS** (37 tests) |
| `test_05_euler_convergence.py` | Euler Substep Convergence | Numerical integration stability, order convergence, GMD Stage 2c classifications | **PASS** (2 tests) |
| `test_06_unithydro_routing.py` | Unit Hydrograph Routing | UH 0..8 kernel weight conservation ($\sum w_i = 1.0$), convolution volume, differentiability | **PASS** (48 tests) |
| `test_07_dpl_training_contract.py` | DPL Training & Loss Contract | `KgeLoss` precision/safety, warmup detachment, end-to-end autograd, factory dispatch | **PASS** (5 tests) |
| `test_08_validation_artifacts.py` | Validation Artifacts | Verifies all official audit reports and data tables preserved in `validation_results/` | **PASS** (6 tests) |

Run the entire consolidated suite with:

```bash
PYTHONPATH=dmotpy .venv/bin/pytest dmotpy/tests/
```
All **367 tests** pass cleanly in ~25 seconds.

---

## 6. Quick Start

### Unified Forward Simulation with Warmup

```python
import torch
from dmotpy.models.hydrology_model import HydrologyModel

device = torch.device("cpu")
time_steps = 730
n_basins = 10
warmup = 365

# Instantiate model via configuration dictionary
config = {
    "model_name": "gr4j",
    "warm_up": warmup,
    "parameter_mapping": "auto",
    "backend": "none",
}
model = HydrologyModel(config=config, device=device)

# Inputs: x_phy shape (time, batch, channels) where channels are (P, T, PET)
x_phy = torch.rand(time_steps, n_basins, 3, device=device)
x_dict = {"x_phy": x_phy}

# Physical parameters in normalized space [0, 1]
raw_params = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * n_basins, device=device)

# Forward pass: 365 days warmup (detached), followed by 365 days scored streamflow
out = model(x_dict, (None, raw_params))
streamflow = out["streamflow"]
print("Streamflow output shape:", streamflow.shape)  # (365, 10)
```

### End-to-End Differentiable Parameterization (dPL)

```python
import torch
from dmotpy.neural_networks.parameterize import Parameterize
from dmotpy.models.hydrology_model import HydrologyModel
from dmotpy.losses import KgeLoss

device = torch.device("cpu")
n_attributes = 27
n_params = 4
n_basins = 10
time_steps = 730
warmup = 365

param_net = Parameterize(nx=n_attributes, ny=n_params, hidden_size=64, device="cpu")
model = HydrologyModel(
    config={"model_name": "gr4j", "warm_up": warmup, "parameter_mapping": "auto", "backend": "none"},
    device=device,
)
loss_fn = KgeLoss()

# Data
attrs = torch.randn(n_basins, n_attributes, device=device)
x_phy = torch.rand(time_steps, n_basins, 3, device=device)
x_dict = {"x_phy": x_phy}
q_obs = torch.rand(time_steps - warmup, n_basins, device=device)

# Forward pass
params_tuple = param_net({"c_nn_norm": attrs})
out = model(x_dict, params_tuple)
q_sim = out["streamflow"]

# Differentiable KGE loss and backward pass
loss = loss_fn(q_sim, q_obs)
loss.backward()
print("KGE Loss value:", loss.item())
```
