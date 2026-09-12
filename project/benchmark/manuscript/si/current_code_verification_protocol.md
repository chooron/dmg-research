# Differentiable Model Verification Protocol (Current Codebase)

**Audit Target**: `dmotpy/` and `project/benchmark/`  
**Git Commit Tested**: `3caca37a7d446903ee52aa791b5c4900a0dcfeea` (with working-tree dynamic DOY calendar updates)  
**Verification Date**: 2026-09-08

---

## 1. Verification Protocol Architecture & Test Suite Inventory

### 1.1 36-Model Registry & Parameter Bounds Contract
- **Source Files**: `dmotpy/models/core/__init__.py` (lines 1–436), `dmotpy/models/registry.py` (lines 1–15), `dmotpy/tests/core_model_registry.py` (lines 1–144).
- **Protocol**: Validates that all 36 canonical conceptual models are registered with non-empty step functions, physical parameter bounds matching MARRMoT calibration ranges, exact state variable counts, and that total parameter counts sum to 271.
- **Enabled Models (36)**: `alpine1`, `alpine2`, `australia`, `collie1`, `collie2`, `collie3`, `flexb`, `flexi`, `flexis`, `gr4j`, `gsfb`, `hbv96`, `hillslope`, `hymod`, `ihacres`, `modhydrolog`, `mopex1`, `mopex2`, `mopex3`, `mopex4`, `mopex5`, `newzealand1`, `newzealand2`, `penman`, `plateau`, `simhyd`, `smar`, `susannah1`, `susannah2`, `tank`, `tcm`, `topmodel`, `us1`, `vic`, `wetland`, `xinanjiang`.
- **Sign Overrides**: Handled per model contract in `dmotpy/tests/core_model_registry.py` (`ihacres`: `(-1.0,)`, `penman`: `(1.0, -1.0, 1.0)`, `tcm`: `(1.0, -1.0, 1.0, 1.0)`, `topmodel`: `(1.0, -1.0)`).

### 1.2 Water-Balance Closure Protocol
- **Source Files**: `dmotpy/tests/test_core_water_balance.py` (lines 1–42), `dmotpy/tests/core_water_balance_utils.py` (lines 1–785), `dmotpy/scripts/validate_core_water_balance.py`.
- **Forcing Cases**: 12 standard synthetic and historical regimes (`zero_zero_pet_short`, `zero_pos_pet_short`, `impulse_short`, `shifted_impulse_short`, `constant_medium`, `alternating_medium`, `random_medium`, `very_dry_medium`, `very_wet_medium`, `high_pet_medium`, `low_pet_long`, `random_long`) plus snow-specific regimes (`snow_cold_warm`, `snow_transition_batch`, `snow_mixed_short`).
- **Precision Modes**: `float64` CPU, `float32` CPU smoke, and `float32` CUDA smoke.
- **Tolerances**:
  - `NEARZERO = 1.0e-6`
  - Step-wise residual: $\max(1.0 \times 10^{-8}, N_{\text{states}} \times \text{NEARZERO} \times 1.1)$
  - Full-sequence absolute tolerance: $\max(1.0 \times 10^{-7}, N_{\text{states}} \times T \times \text{NEARZERO})$
  - Full-sequence relative tolerance: $1.0 \times 10^{-8}$

### 1.3 Forward Finite-Value & Numerical Stability Protocol
- **Source Files**: `dmotpy/tests/test_training_regression_smoke.py`, `dmotpy/tests/training_regression_utils.py`, `dmotpy/scripts/run_training_regression_after_validation.py`.
- **Protocol**: Checks that simulated streamflow, internal storages, and loss values remain finite without `NaN` or `Inf` across realistic and extreme meteorological forcing inputs.

### 1.4 End-to-End Loss Backpropagation & Autograd Protocol
- **Source Files**: `dmotpy/tests/test_model_gradient_end_to_end.py` (lines 1–353).
- **Protocol**: Executes continuous simulation over $T=5, N_{\text{basins}}=2$ under FP64 arithmetic, computes mean squared error loss against target hydrographs, and verifies that analytical loss gradients $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ exist, are finite, and contain valid backpropagated values for all 36 models.

### 1.5 Finite-Difference FP64 Gradcheck Protocol
- **Source Files**: `dmotpy/tests/test_model_gradcheck_representative.py` (lines 1–237), `dmotpy/scripts/stage1_gradcheck_all_models.py`.
- **Protocol**: Executes PyTorch finite-difference `torch.autograd.gradcheck` on representative models (`flexb`, `flexi`, `flexis`, `tcm`, `gsfb`, `topmodel`, `hbv96`, `vic`, `hymod`) under `torch.float64` on CPU with perturbation step $\epsilon = 1.0 \times 10^{-6}$, absolute tolerance $\text{atol} = 1.0 \times 10^{-4}$, and relative tolerance $\text{rtol} = 1.0 \times 10^{-3}$.

### 1.6 Gradient Sparsity & Inactive Process Classification
- **Source Files**: `dmotpy/scripts/standalone_validation_36.py` (lines 1–640), `dmotpy/tests/test_complete_flux_gradient_review_status.py`.
- **Protocol**: Evaluates gradient zero-fraction ($\text{zero\_frac}$) across all parameters under nominal dry/midpoint forcing. Distinguishes full active gradient flow (`median_zf = 0.00`) from physical process sparsity (`SPARSE_GRAD`, $\text{zero\_frac} = 1.00$) in models with conditionally inactive snowmelt or threshold parameters.

### 1.7 Routing & Unit Hydrograph Mass Conservation
- **Source Files**: `dmotpy/tests/test_unithydro_consistency.py`, `dmotpy/tests/test_uh_tail_mass_balance.py`.
- **Protocol**: Verifies non-negativity, unit area normalization ($\sum w_i = 1.0$), delay accuracy, and 1D causal convolution mass conservation across all 9 routing kernels (`uh_0` to `uh_8`).

### 1.8 Daily Time-Step & Euler Substep Convergence
- **Source Files**: `dmotpy/tests/test_euler_substep_convergence.py`, `dmotpy/scripts/standalone_validation_36.py`.
- **Protocol**: Measures empirical convergence rate $p = \log_2(e_{k-1}/e_k) / \log_2(h_{k-1}/h_k)$ across substeps $K \in \{1, 2, 4, 8\}$ against high-resolution reference $K_{\text{ref}} = 256$.
