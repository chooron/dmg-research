# Fresh Validation Protocol Specification

**Target**: Complete 36-Model Differentiable Hydrological Suite (`dmotpy`)  
**Execution Environment**: Python 3.10 virtual environment (`/home/jingxin/code/dmg-research/.venv/bin/python`), PyTorch on Linux (WSL)  
**Protocol Version**: 2026.09-FRESH  
**Working Directory**: `/home/jingxin/code/dmg-research`

---

## 1. Scope & Execution Commands

The fresh validation suite comprises seven execution commands covering all aspects of model correctness, numerical stability, gradient flow, and numerical convergence:

| Command ID | Category | Target Test Script / Module | Execution Command | Output Artifact |
| :---: | :--- | :--- | :--- | :--- |
| `CMD-01` | Registry & Contract | Model metadata & parameter bounds | `./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_registry_contract.py` | `fresh_36model_registry_results.csv` |
| `CMD-02` | Water Balance | 12 forcing regimes $\times$ 3 precision modes | `./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_water_balance.py` | `fresh_water_balance_results.csv` |
| `CMD-03` | Forward Stability | Synthetic & historical smoke tests | `./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_forward_stability.py` | `fresh_forward_stability_results.csv` |
| `CMD-04` | End-to-End Autograd | 36-model MSE loss backpropagation | `./.venv/bin/python -m pytest dmotpy/tests/test_model_gradient_end_to_end.py -v` | `fresh_gradient_end_to_end_results.csv` |
| `CMD-05` | FP64 Gradcheck | 13 representative models PyTorch gradcheck | `./.venv/bin/python -m pytest dmotpy/tests/test_model_gradcheck_representative.py -v` | `fresh_gradcheck_results.csv` |
| `CMD-06` | Routing & UH Mass | 9 UH convolution kernels | `./.venv/bin/python -m pytest dmotpy/tests/test_uh_tail_mass_balance.py dmotpy/tests/test_unithydro_consistency.py -v` | `fresh_routing_results.csv` |
| `CMD-07` | Euler Convergence | Discretization substeps $K \in \{1, 2, 4, 8\}$ vs 256 | `./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_euler_convergence.py` | `fresh_euler_convergence_results.csv` |

---

## 2. Technical Formulations, Tolerances & Fixtures

### 2.1 Model Registry and Parameter Bounds
- **Model Portfolio (36)**: `alpine1`, `alpine2`, `australia`, `collie1`, `collie2`, `collie3`, `flexb`, `flexi`, `flexis`, `gr4j`, `gsfb`, `hbv96`, `hillslope`, `hymod`, `ihacres`, `modhydrolog`, `mopex1`, `mopex2`, `mopex3`, `mopex4`, `mopex5`, `newzealand1`, `newzealand2`, `penman`, `plateau`, `simhyd`, `smar`, `susannah1`, `susannah2`, `tank`, `tcm`, `topmodel`, `us1`, `vic`, `wetland`, `xinanjiang`.
- **Criteria**: Exactly 36 unique models; total calibrated parameters sum strictly to 271; 110 state store allocations (96 unique dynamic state variables).

### 2.2 Water Balance Closure
- **Forcing Regimes (12 Standard + 3 Snow)**:
  1. `zero_zero_pet_short`: $T=7, P=0, \text{PET}=0$
  2. `zero_pos_pet_short`: $T=7, P=0, \text{PET}>0$
  3. `impulse_short`: $T=10$, single rain pulse
  4. `shifted_impulse_short`: $T=10$, shifted pulse
  5. `constant_medium`: $T=365$, steady rain & PET
  6. `alternating_medium`: $T=365$, alternating wet/dry
  7. `random_medium`: $T=365$, uniform random $P \in [0, 50]$, $\text{PET} \in [0, 8]$
  8. `very_dry_medium`: $T=365$, low precipitation
  9. `very_wet_medium`: $T=365$, intense precipitation
  10. `high_pet_medium`: $T=365$, elevated evaporative demand
  11. `low_pet_long`: $T=1000$
  12. `random_long`: $T=1000$
  13. Snow fixtures for snow-enabled models: `snow_cold_warm`, `snow_transition_batch`, `snow_mixed_short`.
- **Tolerances**:
  - `NEARZERO = 1.0e-6`
  - Step residual tolerance (FP64): $\max(1.0 \times 10^{-8}, N_{\text{states}} \times \text{NEARZERO} \times 1.1)$
  - Full-period absolute tolerance (FP64): $\max(1.0 \times 10^{-7}, N_{\text{states}} \times T \times \text{NEARZERO})$
  - Full-period relative tolerance: $1.0 \times 10^{-8}$
  - Target nominal threshold for PASS: Maximum daily residual $< 1.0 \times 10^{-3}\text{ mm/day}$.

### 2.3 Forward Numerical Stability
- **Criteria**: Simulated streamflow $Q_t \ge 0$, internal states $S_t \ge 0$, finite loss values, zero `NaN` occurrences, zero `Inf` occurrences.

### 2.4 Gradient Autograd & Gradcheck
- **Autograd Criterion**: Non-zero, finite gradient vector $\nabla_{\boldsymbol{\theta}} \mathcal{L} \in \mathbb{R}^{P_m}$ evaluated via reverse-mode automatic differentiation on $T=5, N_{\text{basins}}=2$.
- **Gradcheck Subset (13 Models)**: `flexb`, `flexi`, `flexis`, `tcm`, `gsfb`, `topmodel`, `hbv96`, `vic`, `hymod`, `alpine1`, `australia`, `collie1`, `modhydrolog`.
- **Gradcheck Parameters**: `torch.float64`, $\epsilon = 1.0 \times 10^{-6}$, $\text{atol} = 1.0 \times 10^{-4}$, $\text{rtol} = 1.0 \times 10^{-3}$.

### 2.5 Unit Hydrograph Routing Mass Conservation
- **Kernels (9)**: `uh_identity_0`, `uh_half_1`, `uh_full_2`, `uh_tri_3`, `uh_tri_4`, `uh_exp_5`, `uh_gamma_6`, `uh_uniform_7`, `uh_delay_8`.
- **Criteria**: Non-negativity ($w_i \ge 0$), sum normalization ($\sum w_i = 1.0 \pm 1.0 \times 10^{-7}$), mass conservation under 1D causal convolution.

### 2.6 Euler Time-Step Convergence
- **Substeps**: $K \in \{1, 2, 4, 8\}$ substeps per day versus high-resolution benchmark $K_{\text{ref}} = 256$.
- **Convergence Rate Estimator**: $p = \log_2(e_{k-1}/e_k) / \log_2(h_{k-1}/h_k)$.
- **Classification Categories**:
  - `FIRST_ORDER_NOMINAL`: $p \in [0.85, 1.15]$ at standard $K=1$.
  - `FIRST_ORDER_RECOVERED`: $p \in [0.85, 1.15]$ achieved at finer substeps ($K=4$).
  - `THRESHOLD_NON_SMOOTH`: Non-smooth convergence due to discrete bucket thresholding ($p < 0.85$).
  - `ANALYTICAL_STORE`: Model incorporates exact analytical production solution (e.g., `gr4j`).
  - `STEP_SATURATION`: Model uses step-saturation discontinuity (e.g., `hillslope`).
  - `PRECISION_FLOOR`: Error reaches machine precision limit (e.g., `wetland`).
