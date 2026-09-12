# BENCHMARK_CURRENT_STATE_AUDIT_20260831

**Audit Target**: `project/benchmark` (`/home/jingxin/code/dmg-research/project/benchmark`)  
**Audit Scope**: Read-only, evidence-based total audit and inventory of benchmark models, runners, training protocols, checkpoints, and scientific analyses.  
**Date**: 2026-08-31  
**Audit Agent**: Agent A (Benchmark Track)

---

## 1. Executive Summary

A comprehensive, evidence-based audit of `project/benchmark` was conducted across code, configurations, runtime logs, and result artifacts. The core conclusions are:

1. **IC (Individual Calibration) Canonical Baseline is Frozen and Verified**:
   - The canonical IC pipeline (`ic_dpl_aligned_full300_20260819_final`) executed 10-start, 300-generation active CMA-ES with FP64 precision over all 531 CAMELS basins using training period `1980-10-01..1995-09-30` (15 hydrological years) with 5-year repeated warm-up (1825 days).
   - 35 of 36 models completed strict Full300 (generation 300). `simhyd` stopped at generation 280 and was accepted as final by user decision.
   - Validation performance on `1995-10-01..2010-09-30` was evaluated strictly post-hoc on frozen parameters; test scores never entered parameter selection.

2. **dPL Canonical 36-Model `auto100` Training Protocol is Fully Decoded**:
   - The canonical 36-model dPL training artifact (`results/dpl_full_retrain_20260813/auto100` + `results/dpl_flexb_retrain_20260830/auto100`) was generated exclusively by `project/benchmark/scripts/diagnostics/k_full_retrain.py`.
   - **Critical Protocol Issue (Validation Leakage / Stopping Rule)**: Training stopped via `PLATEAU_STOP` based on the **validation median KGE** over all 531 basins (min 50 epochs, patience 10, min delta 0.001). Furthermore, the headline performance metric in `health.csv` and accuracy summary tables is selected as `best = max(epochs, key=lambda row: float(row["validation_median_kge"]))` from the epoch log, while the same validation period (`1995-2010`) is treated as the reported evaluation benchmark.
   - **All 531 basins were trained jointly**: `VALID_OOB_DPL = 0`. No spatial holdout or out-of-basin (OOB) training was executed for the 36 models.

3. **Checkpoint Provenance Disconnect Identified**:
   - Checkpoints in `k_full_retrain.py` were saved **only every 10 epochs** (`epoch % 10 == 0`) and at `epoch == 100`.
   - No `best.pt` file was preserved. For models where `best_epoch` was not a multiple of 10 (e.g., `gr4j` best epoch 20 [saved], but `penman` ep 14, `plateau` ep 16, `smar` ep 21, `susannah1` ep 21, `newzealand1` ep 24, `topmodel` ep 28, `ihacres` ep 32, `hillslope` ep 39, `hymod` ep 39, `alpine1` ep 43, `xinanjiang` ep 46, `mopex2` ep 47, `mopex4` ep 51, `hbv96` ep 53, `flexb` ep 62, `flexi` ep 62), **no exact checkpoint matching `best_epoch` exists on disk**.
   - Downstream offline diagnostic studies (e.g. `parameter_attribute_atlas_20260829`, `r1_r2_nontraining_complete_20260829`) loaded the `latest_saved_epoch_file` (e.g., `epoch_060.pt`), which corresponds to the post-plateau terminal state rather than the validation-best state.

4. **Scientific vs Protocol Gap Clarification (Flex Family Diagnostic)**:
   - The diagnosis completed on 2026-08-31 (`results/flex_protocol_diagnosis_20260831/`) proved that the apparent IC–dPL performance gap on Flex models (`flexb`, `flexi`, `flexis`) is predominantly driven by **training/evaluation objective mismatch** (continuous 14-year single-warmup vs 730-day random short-window sampling), rather than hydrological equation defects or neural failure. Under the window protocol, dPL actually outperforms IC on `flexb` (0.4027 vs 0.3375).
   - Parameter boundary concentration was proven to be structural (MARRMoT bounds misspecification): IC independent CMA-ES hits bounds as much or more than dPL.

---

## 2. Canonical IC State (`IC_CANONICAL_STATUS`)

### 2.1 Configuration and Execution Parameters

| Parameter | Canonical Value | Code / Config Source |
|---|---|---|
| **Runner Script** | `scripts/run_36model_benchmark.py` / `scripts/run_full_benchmark.sh` | `project/benchmark/scripts/run_36model_benchmark.py` |
| **Evaluation Script** | `scripts/evaluate_ic_aligned_gen300.py` | `project/benchmark/scripts/evaluate_ic_aligned_gen300.py` |
| **Config File** | `configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` | `project/benchmark/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` |
| **Optimizer** | Batched Active CMA-ES, full covariance | `src/batched_cmaes.py` |
| **Starts per Basin** | 10 independent starts (best-of-10 selection on training fitness) | `optimization.starts: 10` |
| **Generations** | 300 generations | `optimization.generations: 300` |
| **Population Size** | Tiered by parameter dimension: dim 1: 8; dim 4–6: 12; dim 7–10: 16; dim 12–15: 20 | `population_by_dimension` |
| **Initial Std Dev** | `stdev_init: 0.10` | `optimization.stdev_init: 0.10` |
| **Precision** | Float64 (`torch.float64`) throughout | `results/ic_dpl_aligned_full300_20260819_final/README.md:6` |
| **Training Period** | `1980-10-01` to `1995-09-30` (5,478 days = 15 hydrological years) | `data.train` |
| **Warm-up Period** | `1980-10-01` to `1981-09-30` repeated 5 times = 1,825 days | `warmup.repetitions: 5`, total 1825 d |
| **Objective Function** | `streaming_kge` (FP64 moment accumulation, `eps=0.1`), train-only | `src/objective.py:streaming_kge` |
| **Parameter Mapping** | Latent space $\to$ `sigmoid` $\to$ linear physical bounds | `src/parameter_transform.py` |
| **Basin Set** | All 531 CAMELS basins | `data/531sub_id.txt` |
| **Test/Validation Period** | `1995-10-01` to `2010-09-30` (evaluated strictly post-hoc) | `data.test` |

### 2.2 Model Completion Status (Full300 vs Simhyd)

- **35 Models**: Reached full `generation: 300` (`chunk_0_gen_300.pt`), verified with `src/checkpoint_guard.py:validate_canonical_checkpoint`.
- **`simhyd`**: Reached `generation: 280` (`chunk_0_gen_280.pt`). Note from `status_summary.json`: *"accepted latest generation 280 checkpoint as final by user request"*. It is classified as `RELAXED_SEEN_35` rather than `STRICT_FULL300_34`.

### 2.3 Historical IC Artifacts & Deprecation Ledger

| Historical Artifact Path | Status | Reason Deprecated / Invalid | Canonical Replacement |
|---|---|---|---|
| `remote_runs/20260729_120525` | **DEPRECATED / INVALID** | Pilot run with only 30 generations and 5 starts; erroneously loaded in early diagnosis | `results/ic_dpl_aligned_full300_20260819_final` |
| `results/all36_dpl_gap_diagnosis_20260812` (IC column) | **DEPRECATED / INVALID** | Used gen-30 pilot checkpoints, producing negative IC scores for MOPEX1–3 (-0.12, -0.04, -0.16) | `results/all36_ic_gen300_aligned_20260812` |
| `configs/full_run_10starts_300gen_warm1980_1981x5.yaml` / `results/full300_final_36models_evaluation` | **HISTORICAL (Non-Aligned)** | Trained on `1989-01-01..1998-12-31`, tested `1999..2009`; valid Full300 run but temporal window not aligned with dPL | `configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml` |
| `results/ic_dpl_aligned_full300_20260819_final` | **CANONICAL BASELINE** | Aligned 1980–1995 training period, 1825 d warmup, 531 basins, gen-300 (simhyd gen-280) | Authoritative IC artifact |

---

## 3. Canonical 36-Model dPL `auto100` Protocol

### 3.1 Code and Script Provenance
The sole runner responsible for the current 36-model dPL benchmark is:
```text
project/benchmark/scripts/diagnostics/k_full_retrain.py
```
Orchestrated by: `results/dpl_full_retrain_20260813/run_20260815/scheduler.sh` (35 models) and `results/dpl_flexb_retrain_20260830/` (flexb rerun).

### 3.2 Protocol Inventory

```text
[Data & Basins]
  - Basins: 531 CAMELS basins (data/531sub_id.txt), ALL jointly trained
  - Basin Holdout: None (VALID_OOB_DPL = 0)
  - Training Period: 1980-10-01 to 1995-09-30 (5,478 days = 15 hydrological years)
  - Validation Period: 1994-10-01 to 2010-09-30 (16 years: 365 d warmup + 15 years scored)
  - Scored Validation Window: 1995-10-01 to 2010-09-30 (5,479 days)

[Per-Epoch Sampling Dynamics]
  - Batch Size: 100 basins per step (torch.randperm(531)[:100])
  - Steps per Epoch: 169 steps
  - Window Length: 730 days (2 years)
  - Warmup Days: 365 days (first 365 days excluded from loss)
  - Scored Days: 365 days (days 366 to 730 evaluated)
  - Window Catalog: Filtered via build_informative_kge_catalog (std(Q_obs) >= 0.01 mm/day)
  - Random Seed: Seed = 42 (single seed)

[Neural Parameter Network]
  - Input Features: 35 CAMELS Catchment Physical Attributes
  - Preprocessing: Log1p on skewed cols (11 area, 23 conductivity, 34 permeability), then z-score across all 531 basins
  - Architecture: CatchmentParameterizer (MLP)
      Input (35) -> Linear(256) -> LayerNorm(256) -> GELU -> Dropout(0.05)
                 -> Linear(256) -> LayerNorm(256) -> GELU -> Dropout(0.05)
                 -> Linear(N_params)
  - Precision: torch.float64 (FP64)
  - Output Activation: Sigmoid (output in [0, 1])
  - Midpoint Initialization: Linear weight = 0, bias = 0 -> raw logit = 0 -> sigmoid(0) = 0.5 (exact normalized midpoint)
  - Parameter Mapping (Hydrology Layer): "auto"
      if lower > 0 and upper / lower >= 100: log interpolation
      else: linear interpolation

[Optimizer & Loss]
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - Learning Rate Scheduler: None (constant lr=1e-3)
  - Gradient Clipping: nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
  - Regularizer: None active (weight = 0.0)
  - Loss Function: Differentiable KGE Loss = 1.0 - mean(valid_basin_kge)
  - KGE Epsilon: eps = 0.1, moments computed in FP64 via streaming_kge

[Model Selection & Early Stopping - THE CORE PROTOCOL FINDING]
  - Stopping Metric: Validation median KGE over all 531 basins evaluated every epoch
  - Minimum Epochs: MIN_EPOCHS = 50
  - Patience: PATIENCE = 10 epochs
  - Plateau Delta: PLATEAU_EPS = 0.001
  - Maximum Epochs: 100 epochs
  - Code Logic:
      if STOP_ON_PLATEAU and epoch >= MIN_EPOCHS:
          if median > best_median + PLATEAU_EPS:
              best_median = median
              stall = 0
          else:
              stall += 1
          if stall >= PATIENCE:
              status = "PLATEAU_STOP"
              break
  - Best Epoch Selection (Post-hoc):
      best = max(epochs, key=lambda row: float(row["validation_median_kge"]))
```

---

## 4. Runner Provenance (`RUNNER_PROVENANCE_TABLE`)

| Runner Name | File Path | Stopping Criterion | Best Model Selection | Checkpoints Saved | Scope of Artifacts in Benchmark | Canonical Status |
|---|---|---|---|---|---|---|
| **`k_full_retrain.py`** | `project/benchmark/scripts/diagnostics/k_full_retrain.py` | Plateau on **Validation Median KGE** (`min_epochs=50`, `patience=10`, `eps=0.001`) | `max(epochs, validation_median_kge)` from CSV log | `epoch_010.pt`, `epoch_020.pt`, ... every 10 epochs only; NO `best.pt` | **All 36 models in current benchmark** (`dpl_full_retrain_20260813`, `dpl_flexb_retrain_20260830`) | **CURRENT CANONICAL dPL RUNNER** |
| **`run_dpl_benchmark_dmg_native.py`** | `project/benchmark/scripts/run_dpl_benchmark_dmg_native.py` | Early stop on **Training Loss** (`min_epochs=50`, `patience=10`, `min_delta=1e-4`) | Epoch with lowest **Training Loss** | `best.pt` (saved on train loss improvement) + `epoch_05.pt` | Historical Round 12/13 runs and early native pilots | **SUPERSEDED OLD RUNNER** (Not used for 36-model headline results) |
| **`h_training_pilot.py`** | `project/benchmark/scripts/diagnostics/h_training_pilot.py` | Fixed epochs or plateau on validation KGE | Post-hoc CSV selection | `checkpoints/<model>/<branch>_epoch_*.pt` | Pilot runs B0/B1/B2a/B2b for 6 representative models | Diagnostic Pilot |
| **`run_36model_benchmark.py`** | `project/benchmark/scripts/run_36model_benchmark.py` | Fixed 300 generations (CMA-ES) | Best-of-10 training fitness per basin | `chunk_*_gen_300.pt` every 5 gens + final | All 36 models IC baseline | **CURRENT CANONICAL IC RUNNER** |

---

## 5. Checkpoint Provenance Audit

Audit of `results/dpl_full_retrain_20260813/auto100` and `results/r1_r2_nontraining_complete_20260829/checkpoint_protocol_audit.csv`:

1. **Saved Checkpoint Grid**:
   - Checkpoints are saved strictly at `epoch % 10 == 0` (e.g. 10, 20, 30, 40, 50, 60, 70, 80, 90, 100).
   - If a run halts at epoch 63 via `PLATEAU_STOP`, epoch 63 checkpoint is **not saved**.
2. **Missing `best.pt`**:
   - `k_full_retrain.py` does **not** write a `best.pt` checkpoint.
   - For **33 of 36 models**, `best_epoch` does not coincide with a saved checkpoint:
     - `penman`: best epoch 14 (stop 69) $\to$ no checkpoint for ep 14 (available: 10, 20, 30, 40, 50, 60)
     - `plateau`: best epoch 16 (stop 65) $\to$ no checkpoint for ep 16
     - `gr4j`: best epoch 20 (stop 70) $\to$ checkpoint `epoch_020.pt` exists by coincidence
     - `smar`: best epoch 21 (stop 61) $\to$ no checkpoint for ep 21
     - `susannah1`: best epoch 21 (stop 63) $\to$ no checkpoint for ep 21
     - `newzealand1`: best epoch 24 (stop 63) $\to$ no checkpoint for ep 24
     - `topmodel`: best epoch 28 (stop 65) $\to$ no checkpoint for ep 28
     - `ihacres`: best epoch 32 (stop 64) $\to$ no checkpoint for ep 32
     - `hillslope`: best epoch 39 (stop 64) $\to$ no checkpoint for ep 39
     - `hymod`: best epoch 39 (stop 64) $\to$ no checkpoint for ep 39
     - `alpine1`: best epoch 43 (stop 64) $\to$ no checkpoint for ep 43
     - `modhydrolog`: best epoch 43 (stop 64) $\to$ no checkpoint for ep 43
     - `wetland`: best epoch 44 (stop 86) $\to$ no checkpoint for ep 44
     - `xinanjiang`: best epoch 46 (stop 61) $\to$ no checkpoint for ep 46
     - `mopex2`: best epoch 47 (stop 61) $\to$ no checkpoint for ep 47
     - `mopex4`: best epoch 51 (stop 61) $\to$ no checkpoint for ep 51
     - `susannah2`: best epoch 51 (stop 61) $\to$ no checkpoint for ep 51
     - `collie1`: best epoch 53 (stop 63) $\to$ no checkpoint for ep 53
     - `hbv96`: best epoch 53 (stop 63) $\to$ no checkpoint for ep 53
     - `collie2`: best epoch 55 (stop 65) $\to$ no checkpoint for ep 55
     - `gsfb`: best epoch 55 (stop 65) $\to$ no checkpoint for ep 55
     - `mopex5`: best epoch 55 (stop 65) $\to$ no checkpoint for ep 55
     - `newzealand2`: best epoch 55 (stop 65) $\to$ no checkpoint for ep 55
     - `us1`: best epoch 55 (stop 65) $\to$ no checkpoint for ep 55
     - `mopex3`: best epoch 56 (stop 62) $\to$ no checkpoint for ep 56
     - `flexb`: best epoch 62 (stop 72) $\to$ no checkpoint for ep 62 (`epoch_060.pt` used)
     - `flexi`: best epoch 62 (stop 69) $\to$ no checkpoint for ep 62 (`epoch_060.pt` used)
     - `alpine2`: best epoch 63 (stop 73) $\to$ no checkpoint for ep 63
     - `mopex1`: best epoch 63 (stop 73) $\to$ no checkpoint for ep 63
     - `australia`: best epoch 65 (stop 75) $\to$ no checkpoint for ep 65
     - `tank`: best epoch 65 (stop 69) $\to$ no checkpoint for ep 65
     - `flexis`: best epoch 70 (stop 80) $\to$ checkpoint `epoch_070.pt` exists by coincidence
     - `collie3`: best epoch 82 (stop 92) $\to$ no checkpoint for ep 82
     - `vic`: best epoch 90 (stop 100) $\to$ checkpoint `epoch_090.pt` exists by coincidence
     - `tcm`: best epoch 92 (stop 100) $\to$ no checkpoint for ep 92
     - `simhyd`: best epoch 63 (stop 100) $\to$ no checkpoint for ep 63
3. **Artifact Disconnection**:
   - The reported accuracy tables (`accuracy_table_all36.md`) report the KGE number printed at `best_epoch` in `epochs.csv`.
   - Any script that actually loads neural weights to predict or extract parameters (e.g. `parameter_attribute_atlas_20260829`) loaded `latest_saved_epoch_file` (e.g. `epoch_060.pt`), where performance had decayed from the peak.

---

## 6. Completed Validation Inventory

The following items have been fully tested, audited, and verified; **they do not require re-running**:

| Component / Test | Target Models | Outcome | What It Proves | What It Does NOT Prove |
|---|---|---|---|---|
| **Differentiable KGE Alignment** | All 36 models | `PASS` (`test_dpl_kge_alignment.py`) | Benchmark `compute_differentiable_kge` mathematically matches canonical IC `streaming_kge(eps=0.1)` and produces finite autograd gradients | Does not prove training loss convergence or absence of plateau |
| **VIC Saturation Derivative Explosion Patch** | `vic` | `PASS` (`test_saturation_2.py`, `run_vic_resumes.py`) | Clamping storage deficit to `nearzero = 1e-6` in `saturation_2` eliminates gradient explosion ($>10^6$) and passes Float64 `gradcheck` | Does not prove VIC dPL performance matches IC |
| **MOPEX4 / MOPEX5 Calendar & Mass Balance** | `mopex4`, `mopex5` | `PASS` (`verify_mopex4_canonical_chain.py`, `run_mopex4_two_param_interception_audit.py`) | Calendar DOY channel forcing on GPU is date-aligned; analytic vs autograd gradients pass; water balance error $< 10^{-12}$ | Does not eliminate parameter compensation |
| **Flex Family Protocol Attribution** | `flexb`, `flexi`, `flexis` | `PASS` (`flex_protocol_diagnosis_20260831`) | IC–dPL gap is driven by continuous vs short-window objective mismatch; dPL window KGE $\ge$ IC window KGE | Does not provide full continuous long-sequence dPL training results |
| **Parameter Bounds Diagnosis** | All 36 models | `PASS` (`flex_protocol_diagnosis_20260831`, `parameter_attribute_atlas_20260829`) | Parameter boundary sticking is not a neural pathology; IC CMA-ES sticks to bounds just as frequently due to MARRMoT prior bounds | Does not fix parameter bounds |
| **Attribute Normalization & Z-score** | 531 basins | `PASS` (`dpl/attributes.py`) | 35 Catchment physical attributes are cleanly parsed and normalized with log1p on skewed distributions | Does not prove 35 attributes are sufficient for ungauged transfer |
| **Checkpoint Resume Integrity** | Multiple models | `PASS` (`test_checkpoint_resume.py`) | CPU and CUDA RNG states, optimizer states, and network weights restore bitwise identically | Does not overcome missing intermediate `best_epoch` saves |

---

## 7. Planned-but-Not-Completed Inventory

The following experiments were planned, designed, or scripted, but **have NOT been executed or closed**:

| Planned Experiment | Script / Design Reference | Current Status | Missing Artifact / Blocker |
|---|---|---|---|
| **36-Model Full `linear100` vs `auto100` Sensitivity** | `k_full_retrain.py --arm linear100` | `PLANNED_ONLY` (6–9 models pilot run in round 12/13; never completed for all 36 models) | No 36-model `linear100` epochs, health, or checkpoints exist |
| **36-Model Multi-Seed dPL Uncertainty** | `r1_r2_nontraining_complete_20260829/README.md` | `PLANNED_ONLY` (Current 36-model auto100 has single seed=42 only) | No seeds 43, 44, etc. exist for 36 models (only 4-basin MOPEX4 pilot had 3 seeds) |
| **Formal Out-of-Basin (OOB / PUR) Cross-Validation** | `results/r1_r2_nontraining_complete_20260829/future_oob_pur_training_protocol.md` | `PLANNED_ONLY` (`VALID_OOB_DPL = 0`) | Staged 8-model / 7-HUC regional holdout design was created but 0 OOB models have been trained |
| **Warm-up Length Sensitivity (365d vs 730d vs 1825d)** | Discussion in handoffs | `PLANNED_ONLY` | No systematic warm-up duration ablation on dPL training exists |
| **Learning Rate Schedule Sensitivity** | Fixed 1e-3 | `PLANNED_ONLY` | No cosine/step LR decay comparison against constant 1e-3 |
| **Continuous Long-Sequence dPL Training (Eliminating Window Mismatch)** | Raised in `flex_protocol_diagnosis_20260831` | `PLANNED_ONLY` | dPL has never been trained on continuous 14-year sequences |
| **Exact `best.pt` Checkpoint Preservation** | Raised in `dpl_training_rule_audit_20260831` | `PLANNED_ONLY` | Runner modification not yet applied to 36-model runs |

---

## 8. Known Conflicts and Provenance Gaps

1. **Stopping Criterion Conflict Between Runners**:
   - `run_dpl_benchmark_dmg_native.py` stops on **training loss** (`1e-4` min delta).
   - `k_full_retrain.py` stops on **validation median KGE** (`0.001` plateau eps).
   - Documents prior to 2026-08-16 occasionally conflated these two stopping rules.
2. **Reported Score vs Checkpoint State Disconnection**:
   - Reported dPL validation KGE in `accuracy_table_all36.md` is taken from `epochs.csv` at `best_epoch`.
   - However, no checkpoint file exists for `best_epoch` in 33 of 36 models.
   - When parameter matrices were extracted for Chapter 4 Atlas, `latest_saved_epoch_file` was loaded instead, introducing an unquantified parameter drift.
3. **IC vs dPL Parameter Mapping Mismatch**:
   - IC uses `linear` physical parameter mapping for all models.
   - dPL uses `auto` mapping (log mapping if `upper/lower >= 100`, e.g., `s1max` in FlexB, `crak` in Simhyd).
   - In normalized coordinate space $[0, 1]$, ranks are preserved, but physical parameter derivatives and loss geometry differ.
4. **`torch.compile` Fullgraph Documentation Conflict**:
   - `TRAINING_LOG.md` states `torch.compile(step_function, fullgraph=True)`.
   - `dmotpy/models/hydrology_model.py` calls `torch.compile(fn)` without `fullgraph=True`.

---

## 9. Usable Scientific Findings vs Contaminated Findings

### 9.1 Tier 1: Fully Usable Findings (Robust to Training Protocol)

- **IC Parameter–Attribute Atlas**: The IC baseline (`results/ic_dpl_aligned_full300_20260819_final`) is completely independent of dPL training protocols. Top controlling attributes (`high_prec_dur`, `frac_snow`, `gvf_diff`, `soil_conductivity`, `elev_mean`) reflect genuine hydrological model properties.
- **CMA-ES Restart Reproducibility**: 10-start restart spread (SD 0.0000..0.0310) accurately characterizes IC optimization landscape stability.
- **Model Physics & Numerical Differentiability Audits**: All 36 hydrological forward models, unit hydrograph convolutions, mass balance closures, and analytical/autograd gradient checks remain valid.
- **Flex Family Protocol Mismatch Finding**: The finding that continuous vs windowed objective explains $\sim 80\%$ of the IC–dPL gap is mathematically verified.

### 9.2 Tier 2: Usable as Exploratory / Proxy Evidence Only

- **`D_seen` Distribution Across Models**: The seen-basin performance difference (median $D_{\text{seen}} \approx 0.046$) is a valid descriptive index of how well a single shared MLP fits 531 basins relative to 531 individual CMA-ES fits under the existing protocol.
- **Attribute–Parameter Rank Correlation Agreement ($R_2$)**: The median Spearman profile agreement of 0.6458 (strict34) demonstrates non-trivial structural learning in the neural parameterizer, even with parameter drift.
- **Identifiability & Parameter Boundary Hit Rates**: Boundary hit rates in IC and dPL confirm prior bounds limitations in MARRMoT models.

### 9.3 Tier 3: Contaminated / Invalidated Findings (Must NOT Be Used as Paper Evidence)

- **Interpreting Validation KGE as Out-of-Basin (OOB) Generalization**: The current dPL validation score is temporal validation on *seen basins* with *validation-based early stopping*. It must **never** be cited as ungauged basin performance.
- **Direct Physical Parameter Value Equivalence between IC and dPL**: Because IC used linear mapping while dPL used auto mapping, physical parameters cannot be equated without accounting for the log transform.
- **Exact Numeric Convergence Bounds of dPL**: Because `k_full_retrain.py` stopped at epoch 50–92 via validation plateau without LR scheduling and without continuous sequence training, current dPL scores do not represent the upper bound of differentiable parameter learning.

---

## 10. Open Questions Requiring Future Experiments

1. **If dPL is trained with continuous multi-year sequences (matching IC's evaluation objective), how much of the $D_{\text{seen}}$ gap disappears?**
2. **If dPL stopping rule is switched to training loss (or held-out basin validation), and an exact `best.pt` is saved, how does the parameter–attribute atlas correlation change?**
3. **Does a unified Linear parameter mapping (matching IC) improve parameter realization agreement on wide-range parameters (`smax`, `s1max`)?**
4. **What is the true multi-seed variance of dPL over the full 36 models?**
5. **How well do the 8 selected representative models generalize under true spatial OOB / PUR cross-validation?**

---

## 11. Exact Evidence Paths

```text
[IC Canonical Baseline]
  - Config: project/benchmark/configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml
  - Checkpoints: project/benchmark/results/ic_dpl_aligned_full300_20260819_final/checkpoints/
  - Best Training Chunks: project/benchmark/results/ic_dpl_aligned_full300_20260819_final/best_training/
  - Status Summary: project/benchmark/results/ic_dpl_aligned_full300_20260819_final/status_summary.json
  - Evaluation Script: project/benchmark/scripts/evaluate_ic_aligned_gen300.py
  - Evaluation Summary: project/benchmark/results/ic_gen300_aligned_all36_20260831/ic_gen300_aligned_summary.csv

[dPL Canonical Baseline]
  - Training Runner: project/benchmark/scripts/diagnostics/k_full_retrain.py
  - Training Scheduler: project/benchmark/results/dpl_full_retrain_20260813/run_20260815/scheduler.sh
  - 35-Model Epoch Logs: project/benchmark/results/dpl_full_retrain_20260813/auto100/epochs.csv
  - 35-Model Health Summary: project/benchmark/results/dpl_full_retrain_20260813/auto100/health.csv
  - FlexB Retrain Log: project/benchmark/results/dpl_flexb_retrain_20260830/auto100/epochs.csv
  - 36-Model Accuracy Table: project/benchmark/results/ic_dpl_accuracy_all36_20260831/accuracy_table_all36.md

[Audits & Diagnostics]
  - Training Rule Audit: project/benchmark/results/dpl_training_rule_audit_20260831.md
  - Flex Protocol Diagnosis: project/benchmark/results/flex_protocol_diagnosis_20260831/SUMMARY.md
  - Checkpoint Protocol Audit: project/benchmark/results/r1_r2_nontraining_complete_20260829/checkpoint_protocol_audit.csv
  - Non-Training R1/R2 Summary: project/benchmark/results/r1_r2_nontraining_complete_20260829/README.md
  - Parameter Attribute Atlas: project/benchmark/results/parameter_attribute_atlas_20260829/README.md
  - Future OOB Protocol Design: project/benchmark/results/r1_r2_nontraining_complete_20260829/future_oob_pur_training_protocol.md
```

---

## 12. Final Status Matrix

| Component | Status | Evidence Path | Confidence | Notes |
|---|---|---|---|---|
| **IC Full300 Baseline** | `VERIFIED` | `results/ic_dpl_aligned_full300_20260819_final/` | High | 35 models gen-300; simhyd gen-280 accepted |
| **dPL Forward & Autodiff** | `VERIFIED` | `dpl/tests/test_dpl_kge_alignment.py` | High | Gradient flow finite, moments match FP64 KGE |
| **dPL auto100 Training (36 models)** | `VERIFIED` | `results/dpl_full_retrain_20260813/`, `results/dpl_flexb_retrain_20260830/` | High | 33 plateau stops, 3 completed 100 ep |
| **dPL Checkpoint Selection Protocol** | `VERIFIED` (Issue Confirmed) | `scripts/diagnostics/k_full_retrain.py:108` | High | Validation median KGE used for early stop & headline selection |
| **Exact `best.pt` Preservation** | `UNVERIFIED_OR_CONFLICTED` (Missing) | `results/r1_r2_nontraining_complete_20260829/checkpoint_protocol_audit.csv` | High | Only ep 10..pt saved; 0/35 models have exact best.pt |
| **IC–dPL Mapping Alignment** | `UNVERIFIED_OR_CONFLICTED` (Mismatch) | `src/parameter_transform.py` vs `k_full_retrain.py` | High | IC is pure linear; dPL is auto (log for span $\ge$ 100) |
| **Flex Family Diagnostic** | `VERIFIED` | `results/flex_protocol_diagnosis_20260831/` | High | Gap explained by objective mismatch (continuous vs window) |
| **VIC Saturation Fix** | `VERIFIED` | `scripts/diagnostics/test_saturation_2.py` | High | Clamped storage deficit eliminates gradient blowup |
| **Linear100 vs Auto100 (All 36)** | `PARTIALLY_VERIFIED` | `scripts/diagnostics/round13_finalize.py` | Medium | Only 6–9 control models evaluated |
| **Multi-Seed dPL (All 36)** | `UNVERIFIED_OR_CONFLICTED` | `k_full_retrain.py:34` | High | Single seed=42 only in benchmark |
| **Out-of-Basin (OOB / PUR) dPL** | `PLANNED_ONLY` | `future_oob_pur_training_protocol.md` | High | VALID_OOB_DPL = 0; 0 runs executed |
| **Attribute–Parameter Atlas ($R_2$)** | `VERIFIED` (As Seen-Basin Proxy) | `results/parameter_attribute_atlas_20260829/` | High | Median profile rho = 0.6458; labeled SEEN_BASIN_PROXY |

---
*Report compiled by Agent A (Benchmark Track) on 2026-08-31.*
