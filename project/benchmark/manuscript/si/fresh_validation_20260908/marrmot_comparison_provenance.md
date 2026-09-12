# MARRMoT Performance Comparison: Provenance and Revalidated Statistics

**Audit Target**: Empirical comparison between DMOT Individual Calibration (IC Full300 CMA-ES) and archived MARRMoT reference calibrations  
**Execution Timestamp**: 2026-09-08T14:35:00Z  
**Primary Archive Source**: `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/`

---

## 1. Study Configuration Alignment

- **Models**: 36 conceptual models (identical across both suites).
- **Basins**: 531 CAMELS-US catchments, with 17,380 overlapping model–basin pairs ($480\text{--}483$ basins per model).
- **Temporal Alignment**:
  - Parameter estimation: `1980-10-01` to `1995-09-30` (15 years)
  - Evaluation: `1995-10-01` to `2010-09-30` (15 years)
- **Forcing Inputs**: Daymet ($P, T, \text{PET}_{\text{Hargreaves}}$).

---

## 2. Revalidated Comparative Statistics

### 2.1 Model-Level Ranking & Efficiencies
- **Evaluation Period (1995–2010)**:
  - Spearman rank correlation across 36 models: $\mathbf{\rho = 0.7079}$ ($p = 1.38 \times 10^{-6}$).
  - Model-level median KGE difference ($\text{KGE}_{\text{DMOT}} - \text{KGE}_{\text{MARRMoT}}$): Median $= \mathbf{+0.0324}$, $\text{IQR} = \mathbf{0.0575}$.
- **Calibration Period (1980–1995)**:
  - Spearman rank correlation across 36 models: $\mathbf{\rho = 0.7385}$ ($p = 2.73 \times 10^{-7}$).
  - Model-level median KGE difference: Median $= \mathbf{+0.0329}$, $\text{IQR} = \mathbf{0.0650}$.

### 2.2 Catchment-Level Paired Distribution
- **Evaluation Period ($N = 17,380$ pairs)**:
  - Median paired difference: $\Delta\text{KGE} = \mathbf{+0.0274}$ ($\text{IQR} = \mathbf{0.1104}$, $\text{Q25} = -0.0137, \text{Q75} = +0.0967$).
  - DMOT IC achieves higher/equal evaluation KGE in **69.57%** of catchment instances.
- **Calibration Period ($N = 17,388$ pairs)**:
  - Median paired difference: $\Delta\text{KGE} = \mathbf{+0.0269}$ ($\text{IQR} = \mathbf{0.0869}$, $\text{Q25} = -0.0041, \text{Q75} = +0.0827$).
  - DMOT IC achieves higher/equal calibration KGE in **77.74%** of catchment instances.

---

## 3. Scientific Permissibility & Framing Rules

- **Allowed**: Cite as macro-level performance context demonstrating that the differentiable models occupy a consistent performance range and structure ranking relative to reference MARRMoT calibrations.
- **Prohibited**: Do NOT claim exact parameter-level numerical identity or ODE-solver replication.
