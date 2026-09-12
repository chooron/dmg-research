# CURRENT_PROJECT_DPL_AUDIT_SYNTHESIS_20260831

**Audit Target**: `dmg-research` Benchmark & dmotpy Core Tracks  
**Paths Audited**:  
- `project/benchmark` (`/home/jingxin/code/dmg-research/project/benchmark`)
- `dmotpy` (`/home/jingxin/code/dmg-research/dmotpy`)

**Audit Date**: 2026-08-31  
**Synthesis Agent**: Main Agent (Synthesizing Agent A & Agent B Audits)

---

## 1. Executive Conclusion

### 1.1 What Has Been Verified So Far?
1. **Hydrological Model Physics and Numerical Core are Closed & Sound (Layers 1 & 3)**:
   - All 36 MARRMoT-derived hydrological step functions in `dmotpy/models/core/` are mass-conservative, numerically stable under daily explicit Euler stepping, and possess unbroken autograd computational graphs verified via FP64 `gradcheck` and end-to-end gradient unit tests.
   - Differentiable unit hydrographs (UH 0..8) preserve mass balance and provide valid convolutional and stepwise gradients.
   - The differentiable KGE loss in `dmotpy/losses.py` correctly calculates sample-variance-based KGE with numerical safeguards ($\epsilon = 10^{-5}$) and enforces strict error raising (`FloatingPointError`) on non-finite predictions.
2. **Canonical Individual Calibration (IC) Baseline is Frozen and Verified (Layer 2)**:
   - The canonical IC pipeline (`ic_dpl_aligned_full300_20260819_final`) executed 10-start, 300-generation active CMA-ES in FP64 across all 531 CAMELS basins on training period `1980-10-01..1995-09-30` (15 hydrological years) with 5-year repeated cycle warmup (1,825 days).
   - 35 of 36 models completed strict Full300; `simhyd` completed Generation 280 (accepted as frozen baseline).
   - Test evaluation on `1995-10-01..2010-09-30` was strictly out-of-sample and evaluated post-hoc.
3. **Root Cause of Apparent Model Failures Disproved (Flex Family Diagnostic)**:
   - The apparent severe IC–dPL gap in `flexb`, `flexi`, and `flexis` was conclusively proven on 2026-08-31 to be an artifact of **training/evaluation objective mismatch** (730-day random short-window sampling vs 14-year continuous simulation) rather than model code bugs. Under identical windowed protocols, dPL matches or outperforms IC (e.g. `flexb` dPL 0.4027 vs IC 0.3375).
   - Parameter boundary concentration was verified to be a structural MARRMoT bounds misspecification rather than an autodiff artifact (IC CMA-ES hits bounds as frequently as dPL).

### 1.2 What is Truly Blocking the 36-Model IC–dPL Scientific Comparison?
The primary blockers are **not in the hydrological equations or dmotpy library**, but reside entirely in the **benchmark dPL training and model-selection protocol (Layer 4)**:
1. **Validation Leakage in Model Selection & Stopping**:
   - `k_full_retrain.py` stopped training via `PLATEAU_STOP` based on the 531-basin median KGE evaluated on the test/validation period (`1995-10-01..2010-09-30`).
   - The reported headline score was selected post-hoc as `best = max(epochs, key=lambda row: float(row["validation_median_kge"]))` from the same period used as the evaluation benchmark.
2. **Checkpoint Provenance Disconnect (`best.pt` Missing)**:
   - Checkpoints were saved only every 10 epochs. For 33 of 36 models, the reported `best_epoch` has no matching checkpoint file on disk.
   - Downstream parameter atlas and reproducibility analyses reloaded late-epoch terminal checkpoints (e.g. `epoch_060.pt`), which suffered post-peak degradation.
3. **Training vs Evaluation Horizon Mismatch**:
   - dPL trained on 730-day random windows (365-day warmup) but evaluated on 15-year continuous simulations, penalizing models with slow multi-year baseflow/groundwater dynamics.
4. **Spatial Holdout (OOB/PUR) Has Not Been Executed**:
   - All 36 models in the canonical run were trained jointly on all 531 basins (`VALID_OOB_DPL = 0`).

---

## 2. Evidence Hierarchy

| Topic | Status | Evidence | Confidence |
|---|---|---|---|
| **Hydrological Forward** | `VERIFIED` | `dmotpy/models/core/*.py`, `dmotpy/tests/test_core_registry_metadata.py` (219/219 passed), `test_core_water_balance.py` | High |
| **Numerical Stability** | `VERIFIED` | `dmotpy/tests/test_euler_substep_convergence_all_core.py`, explicit Euler convergence across 36 models | High |
| **Autodiff & Gradients** | `VERIFIED` | `dmotpy/tests/test_model_gradcheck_representative.py` (FP64 gradcheck on 9 representative models), `test_model_gradient_end_to_end.py` | High |
| **IC Convergence** | `VERIFIED` | `results/ic_dpl_aligned_full300_20260819_final/`, 35 models at 300 gen + `simhyd` at 280 gen, 10 starts | High |
| **dPL KGE Implementation** | `VERIFIED` | `dmotpy/losses.py:65`, `_columnwise_kge_loss` with $\epsilon=10^{-5}$, `FloatingPointError` on non-finites | High |
| **dPL Checkpoint Selection** | `UNVERIFIED_OR_CONFLICTED` | `k_full_retrain.py:112` (`best = max(epochs, key=lambda row: float(row["validation_median_kge"]))`), checkpoints saved only mod 10, no `best.pt` | High (Confirmed Protocol Defect) |
| **Parameter Mapping (`auto` vs `linear`)** | `VERIFIED` | `dmotpy/models/hydrology_model.py:160`, 46 log-mapped parameters identified across 36 models | High |
| **Warmup Protocol** | `VERIFIED` | `dmotpy/models/hydrology_model.py:363` (`no_grad` + `detach`), IC (1825d continuous) vs dPL (365d window) divergence | High |
| **Gradient Clipping** | `PARTIALLY_VERIFIED` | `k_full_retrain.py:192` (`clip_grad_norm_(max_norm=1.0)` executed every batch), but activation frequency not logged | Medium |
| **Boundary Saturation** | `PARTIALLY_VERIFIED` | Flex diagnosis proved bounds hit by IC too; Jacobian suppression under log mapping verified mathematically | Medium |
| **Learning Rate & Schedule** | `UNVERIFIED_OR_CONFLICTED` | Fixed AdamW lr ($10^{-3}$) without scheduler; 36-model LR sensitivity never executed | Low (Suspicion Only) |
| **Multi-Seed dPL** | `PLANNED_ONLY` | Benchmark 36-model `auto100` ran only SEED=42; 3-seed runs exist only in separate `hydrodiag` project | High (Confirmed Benchmark Status) |
| **Out-of-Basin (OOB)** | `PLANNED_ONLY` | All 531 basins trained jointly in `k_full_retrain.py`; `VALID_OOB_DPL = 0` | High |
| **Ungauged Prediction (PUR)** | `PLANNED_ONLY` | Planned in roadmap; zero 36-model PUR experiments executed | High |

---

## 3. Canonical Experiment Provenance

### 3.1 IC Canonical Pipeline (`IC_CANONICAL_STATUS`)
- **Artifact Path**: `project/benchmark/results/ic_dpl_aligned_full300_20260819_final/`
- **Runner**: `project/benchmark/scripts/run_36model_benchmark.py` (via `scripts/run_full_benchmark.sh`)
- **Optimizer**: Batched Active CMA-ES (10 independent starts per basin, best-of-10 selected on training fitness).
- **Generations**: 300 generations (simhyd stopped at 280 gen by user decision).
- **Precision**: Float64 (`torch.float64`) throughout.
- **Periods**: Train `1980-10-01..1995-09-30` (15 yr), Test `1995-10-01..2010-09-30` (15 yr), Warmup 1,825 days (5-year repeated cycle).
- **Status**: **FROZEN & VERIFIED BASELINE**.

### 3.2 dPL Canonical `auto100` Pipeline
- **Artifact Path**: `project/benchmark/results/dpl_full_retrain_20260813/auto100/` + `results/dpl_flexb_retrain_20260830/auto100/`
- **Runner**: `project/benchmark/scripts/diagnostics/k_full_retrain.py`
- **Architecture**: `CatchmentParameterizer` MLP [256, 256], LayerNorm, GELU, Dropout 0.05, 35 Caravan attributes (z-score normalized), zero-initialized output layer.
- **Optimizer**: AdamW, lr=$10^{-3}$, weight_decay=$10^{-4}$, gradient clipping `max_norm=1.0`.
- **Sampling**: Random 730-day windows (365-day warmup, 365-day scored), batch size 100, 169 steps/epoch.
- **Basins**: All 531 basins trained jointly (no spatial holdout).
- **Stopping & Selection**: `PLATEAU_STOP` on validation median KGE; headline score selected via `max(epochs, validation_median_kge)`.
- **Status**: **VALID AS EXPLORATORY DIAGNOSTIC; COMPROMISED AS BENCHMARK EVALUATION BASELINE**.

### 3.3 Historical dPL Runner (`run_dpl_benchmark_dmg_native.py`)
- **Stopping Criterion**: Early stopping on **Training Loss** (`min_epochs=50`, `patience=10`, `min_delta=1e-4`).
- **Checkpointing**: Successfully saved `best.pt` based on training loss improvement.
- **Status**: **SUPERSEDED OLD RUNNER**. Did NOT produce the current 36-model canonical benchmark artifacts.

### 3.4 Specialized Diagnostic Pipelines
- **Flex Protocol Diagnosis (`results/flex_protocol_diagnosis_20260831/`)**: Evaluated `flexb`, `flexi`, `flexis` under aligned windowed evaluation; proved objective mismatch was responsible for apparent gap.
- **Parameter Attribute Atlas (`results/parameter_attribute_atlas_20260829/`)**: Offline extraction of parameter-attribute correlations across 531 basins using `latest_saved_epoch_file` (`epoch_060.pt`).

---

## 4. Confirmed Issues

1. **Validation-Based Epoch Selection (`CONFIRMED_PROTOCOL_ISSUE`)**:
   - `k_full_retrain.py` evaluates validation median KGE on `1995-10-01..2010-09-30` at every epoch, triggers `PLATEAU_STOP` when this metric stalls, and selects `best_epoch = max(validation_median_kge)` from the CSV log. This metric is then reported as the model's benchmark performance.
2. **Missing Exact Best Checkpoint Files**:
   - Checkpoints are saved only every 10 epochs. For 33 of 36 models, no checkpoint corresponding to `best_epoch` exists. Downstream diagnostic scripts reloaded `epoch_060.pt` or `epoch_070.pt`.
3. **Training vs Evaluation Warmup/Horizon Inconsistency**:
   - dPL is trained with 365-day warmup on 730-day random slices, while IC and evaluation run 15-year continuous simulations with 1,825-day / 365-day continuous warmups.
4. **VIC Phenology Day-of-Year Hardcoding**:
   - `dmotpy/models/core/vic.py:77` hardcodes `t_idx = torch.ones_like(P)` (Day 1). `data_contract.py` excludes VIC from `CALENDAR_MODELS`. VIC operated under static phenology in all runs.
5. **No Spatial Holdout in 36-Model Benchmark**:
   - 100% of the 36-model benchmark was trained on all 531 basins (`VALID_OOB_DPL = 0`).

---

## 5. Suspected but Unverified Issues

1. **Learning Rate Inappropriateness / Need for Scheduler**:
   - Fixed lr $10^{-3}$ without decay is suspected of causing late-epoch parameter drift and boundary migration, but a multi-model LR / scheduler sweep (e.g. Cosine Annealing, $10^{-4}$ vs $10^{-3}$) has not been systematically conducted across the 36 models.
2. **Gradient Clipping Frequency**:
   - `clip_grad_norm_(network.parameters(), max_norm=1.0)` is applied every batch, but the exact pre-clip norm and clip activation percentage were not recorded.
3. **Neural Network Capacity Limitation**:
   - Whether a 2-layer [256, 256] MLP with 35 Caravan attributes has sufficient expressive capacity for complex 15-parameter models (e.g. `hbv96`, `modhydrolog`, `xinanjiang`) across 531 diverse basins remains unverified against wider/deeper architectures.
4. **Log-Mapping Jacobian Boundary Suppression**:
   - While mathematically proved that $\frac{\partial \theta}{\partial u} \to L \ln(U/L)$ is small near the lower bound, whether this is the primary driver of calibration lag for specific parameters has not been isolated from physical non-identifiability.

---

## 6. Historical Validation Already Completed (DO NOT REPEAT)

| Completed Validation | Scope / Evidence | What It Proves | What It Does NOT Prove |
|---|---|---|---|
| **36 Model Forward & Water Balance** | `dmotpy/tests/test_core_water_balance.py`, `test_core_registry_metadata.py` (219/219 passed) | All 36 models execute stably and conserve mass under daily time stepping. | Does not evaluate calibration efficiency or dPL trainability. |
| **PyTorch Autograd Correctness** | `dmotpy/tests/test_model_gradcheck_representative.py` (FP64 gradcheck) | PyTorch automatic differentiation produces exact analytical gradients across representative models. | Does not guarantee absence of flat loss plateaus or boundary saturation during training. |
| **Canonical IC CMA-ES Baseline** | `results/ic_dpl_aligned_full300_20260819_final/` (35 Full300 + 1 Gen280) | Established a highly converged, reliable individual basin calibration benchmark. | Does not establish regionalization or attribute-transfer capability. |
| **Flex Family Diagnostic** | `results/flex_protocol_diagnosis_20260831/` (`flexb`, `flexi`, `flexis`) | Apparent gap was caused by windowed vs continuous evaluation protocol, not hydrological bugs; boundary concentration is a property of MARRMoT parameter bounds. | Does not solve how to train dPL on long horizons efficiently. |
| **Unit Hydrograph Routing Verification** | `dmotpy/tests/test_unithydro_consistency.py`, `test_uh_tail_mass_balance.py` | All UH components (UH 0..8) preserve mass and provide exact differentiable transformations. | Does not optimize channel lag parameters. |

---

## 7. Historical Experiments Planned but Not Closed

1. **`linear100` 36-Model Arm**:
   - Planned as a control arm in `k_full_retrain.py`; executed for only 6 control models (`collie1`, `gr4j`, `mopex1`, `ihacres`, `mopex4`, `hillslope`). The remaining 30 models were **never run** (`PLANNED_ONLY`).
2. **Multi-Seed 36-Model dPL Benchmark**:
   - `k_full_retrain.py` ran only single seed (`SEED=42`). Multi-seed runs (3 seeds) were conducted in `project/hydrodiag` for a different study, but never executed for the 36-model benchmark (`PLANNED_ONLY`).
3. **Systematic Learning Rate & Scheduler Sweep**:
   - LR sensitivity ($10^{-4}, 5\times 10^{-4}, 10^{-3}, 2\times 10^{-3}$, StepLR, CosineAnnealing) planned in diagnostic prompts but never executed across 36 models (`PLANNED_ONLY`).
4. **Continuous Simulation dPL Training**:
   - Training dPL directly on full 15-year continuous sequences (matching IC evaluation) was piloted on Flex models but never scaled to the 36-model benchmark (`PLANNED_ONLY`).
5. **Spatial Generalization (OOB / PUB / PUR)**:
   - 5-fold cross-validation or spatial holdout was planned in project roadmaps but zero runs were completed in the benchmark repository (`PLANNED_ONLY`).

---

## 8. Current Scientific Findings and Contamination Risk

| Scientific Finding / Analysis | Current Status | Contamination Risk & Usability |
|---|---|---|
| **IC Parameter Atlas & Restart Sensitivity** | `ROBUST_TO_CURRENT_PROTOCOL` | **Fully usable**. Derived strictly from the frozen IC CMA-ES runs on `1980..1995` and independent test on `1995..2010`. Free from dPL protocol defects. |
| **Hydrological Model Mass Balance & Structural Checks** | `ROBUST_TO_CURRENT_PROTOCOL` | **Fully usable**. Model equations, Euler integration, and unit hydrograph tests are independent of training protocols. |
| **Seen-Basin IC–dPL Performance Comparisons (`D_seen`)** | `EXPLORATORY_ONLY` | **Cannot be frozen as final paper numbers**. dPL numbers are contaminated by validation-based stopping (`PLATEAU_STOP`), post-hoc `max(val_kge)` selection, and window/continuous horizon mismatch. |
| **Attribute–Parameter Matrices & Atlas (`parameter_attribute_atlas_20260829`)** | `EXPLORATORY_ONLY` | **Directionally informative but numerically unanchored**. Reloaded `epoch_060.pt` terminal weights (which suffered post-peak degradation) rather than exact `best_epoch` weights. |
| **Out-of-Basin (OOB) / PUR Generalization Claims** | `NEEDS_RECOMPUTATION` | **Zero evidence exists**. All 36 models trained on 100% of basins. Any claim about regionalization requires new OOB experiments. |

---

## 9. Provenance Conflicts

1. **VIC Calendar Forcing**:
   - *Historical Docs*: Listed VIC as requiring Day-of-Year forcing for seasonal phenology.
   - *Current Code*: `data_contract.py` defines `CALENDAR_MODELS = frozenset({"mopex4", "mopex5"})`. `vic.py:77` hardcodes `t_idx = 1.0`. VIC was trained as a 3-channel non-calendar model.
2. **Best Checkpoint vs Saved Checkpoint**:
   - *CSV Log*: Records `best_epoch = 62` with high validation KGE.
   - *Disk Artifacts*: Checkpoint directory contains only `epoch_060.pt` and `epoch_070.pt`. Downstream analysis reloaded `epoch_060.pt`.
3. **Runner Attribution**:
   - *Historical Reports*: Early project summaries referenced `run_dpl_benchmark_dmg_native.py` (which stopped on training loss and saved `best.pt`).
   - *Canonical Reality*: The 36-model `auto100` artifact was produced by `k_full_retrain.py` (which stopped on validation KGE and did not save `best.pt`).

---

## 10. Current Project Status

| Project Layer | Layer Description | Current Status | Summary |
|---|---|---|---|
| **Layer 1** | Hydrological Model Implementations | **CLOSED** | 36 models verified; mass balance and Euler convergence validated; tests passing. |
| **Layer 2** | IC CMA-ES Baseline | **CLOSED** | Full300 active CMA-ES baseline frozen and verified across all 531 CAMELS basins. |
| **Layer 3** | dmotpy Core Implementation | **CLOSED** | Autodiff graph, differentiable KGE loss, and parameter mapping mechanisms verified. |
| **Layer 4** | dPL Optimization & Training Protocol | **REOPENED** | Validation leakage in early stopping, missing `best.pt`, fixed LR, and window/continuous mismatch require protocol adjustment. |
| **Layer 5** | IC–dPL Scientific Comparison | **REOPENED** | Final numerical comparisons paused until Layer 4 training protocol is finalized. |
| **Layer 6** | Out-of-Basin (OOB) / PUR Regionalization | **NOT_STARTED** | Spatial holdout training and ungauged regionalization benchmarks not yet executed. |
