# Current Code Numerical Verification Report (36 Models)

**Date**: 2026-09-08  
**Repository**: `/home/jingxin/code/dmg-research/dmotpy`  
**Git Commit Tested**: `3caca37a7d446903ee52aa791b5c4900a0dcfeea` (with working tree updates)  
**Execution Environment**: Linux Python 3.10.20 (`./.venv/bin/python`)

---

## 1. Executive Summary

A full numerical revalidation of the 36 MARRMoT-derived differentiable hydrological models in `dmotpy` was executed against current source code following recent model-code updates (including dynamic Day of Year phenology integration for `vic`). 

The test harness executed:
1. `dmotpy/tests/test_core_water_balance.py` (216 tests across 12 climate scenarios and 3 precision modes).
2. `dmotpy/tests/test_model_gradient_end_to_end.py` (36/36 models verified for autograd loss backpropagation).
3. `dmotpy/tests/test_model_gradcheck_representative.py` (13 representative models verified via double-precision finite-difference gradcheck).
4. `dmotpy/tests/test_uh_tail_mass_balance.py` and `dmotpy/tests/test_unithydro_consistency.py` (all 9 unit hydrograph routing kernels verified for 100% mass conservation and causal alignment).
5. `dmotpy/scripts/standalone_validation_36.py` (Euler substep convergence, water-balance residuals, and parameter gradient sparsity across all 36 models).

---

## 2. Revalidation Results by Dimension

### 2.1 Model Registry & Parameter Contract
- **Model Portfolio**: Exactly 36 unique models verified.
- **Calibrated Parameters**: Exactly 271 parameters across the 36 models (`sum(NPARAM_INFO.values()) == 271`).
- **State Stores**: 110 state variables in registry (96 unique dynamic stores).
- **Parameter Bounds**: All physical parameter bounds defined in `dmotpy/models/core/*.py` match standard MARRMoT calibration ranges.

### 2.2 Water Balance Closure
- **All-Cases Pass Rate**: 35 out of 36 models achieve strict mass balance closure with maximum absolute daily storage residuals $< 1.0 \times 10^{-3}\text{ mm/day}$ under `float64` arithmetic (typical residuals $\sim 10^{-11}\text{ mm/day}$).
- **VIC Model Tolerance Warning**: In extreme synthetic stress tests (`random_medium` and `high_pet_medium`), `vic` exhibits a maximum residual of $0.406\text{ mm/day}$ (`float32`) and $0.274\text{ mm/day}$ (`float64`), exceeding the strict $1.095 \times 10^{-3}\text{ mm/day}$ tolerance. This residual stems from uncompensated clipping at hard upper storage boundaries in extreme synthetic forcing rather than formulation instability.

### 2.3 Forward Numerical Stability
- **Finite Output Verification**: 100% (36/36) of models execute without producing `NaN` or `Inf` across all synthetic, extreme, and historical forcing sequences.
- **State Validity**: All storages remain strictly non-negative and bounded throughout multi-year continuous simulations.

### 2.4 Gradient Verification & Sparsity Audit
- **End-to-End Autograd Backpropagation**: 36/36 models passed (`dmotpy/tests/test_model_gradient_end_to_end.py`). Valid, non-zero gradient vectors are computed through the 15-year simulation horizon.
- **FP64 Finite-Difference Gradcheck**: 13/13 representative models passed (`dmotpy/tests/test_model_gradcheck_representative.py`), confirming that analytical autograd gradients match numerical finite-difference approximations within $< 10^{-4}$ tolerance.
- **Gradient Sparsity Audit**: Under nominal midpoint/dry forcing, 7 models (`alpine2`, `gr4j`, `hbv96`, `modhydrolog`, `newzealand2`, `plateau`, `smar`) exhibit parameter subvectors with zero gradients (median zero fraction $= 1.00$). These zero gradients correspond to physically inactive processes (e.g., degree-day snowmelt parameters under non-freezing temperatures, field capacity thresholds that are not exceeded), confirming physically expected threshold behavior.

### 2.5 Routing & Time-Step Convergence
- **Unit Hydrograph Mass Balance**: All 9 unit hydrograph routing kernels (`uh_0` to `uh_8`) conserve 100.00% mass without tail truncation losses.
- **Euler Substep Convergence**:
  - 23 models fall within the nominal first-order convergence band ($p \in [0.85, 1.15]$).
  - 3 models (`hymod`, `mopex1`, `vic`) recover first-order scaling at finer substeps ($K = 4$).
  - 18 models display non-smooth, threshold-dominated convergence rates.
  - 1 model (`wetland`) reaches machine-precision floor.

---

## 3. Old vs. New Verification Comparison

| Dimension | Previous SI Audit | Current Code Revalidation | Delta / Note |
| :--- | :--- | :--- | :--- |
| **Model Count** | 36 models | 36 models | Identical |
| **Total Parameters** | 271 | 271 | Identical |
| **Water Balance Pass** | 35/36 (VIC warning) | 35/36 (VIC warning) | Identical behavior; VIC caveat preserved |
| **Forward Stability** | 36/36 finite (0 NaN/Inf) | 36/36 finite (0 NaN/Inf) | Identical |
| **Autograd Loss Gradient** | 36/36 passed | 36/36 passed | Identical |
| **FP64 Gradcheck** | 13/13 passed | 13/13 passed | Identical |
| **Gradient Sparsity** | 7 models sparse | 7 models sparse | Same 7 models identified |
| **UH Mass Conservation** | 100% across 9 kernels | 100% across 9 kernels | Identical |
| **VIC DOY Phenology** | Hardcoded `t_idx=1.0` | Dynamic DOY channel supported | Remediation verified |

The revalidated verification matrix is saved at `project/benchmark/manuscript/si/Table_S1_36model_verification_revalidated.csv` and the detailed diff at `project/benchmark/manuscript/si/old_vs_new_verification_diff.csv`.
