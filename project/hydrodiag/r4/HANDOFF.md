# Handoff: R4 Real-Basin Shared Soil-Water State Consistency Pipeline

**Status**: $\mathbf{STOP\ EXPANDING\ R4}$ (All evidence chains, robustness modules, and figure-ready tables are complete and frozen).  
**Location**: `project/hydrodiag/manuscript/scripts/` & `project/hydrodiag/r4/`  
**Target Paper Section**: Results 3.4 (R4: Real-Catchment Process-State Verification)  

---

## 1. Scientific Motivation & Core Question

R2 and R3 demonstrated in synthetic experiments that omitting explicit snow accumulation and ablation leaves compensated parameter errors and severely distorted internal state trajectories despite achieving competitive streamflow predictive skill ($Q$).

**R4 asks**:
> Does this structural omission translate into divergent downstream shared soil-water state trajectories ($W_{\text{total}} = WU + WL + WD$) in real catchments, and does explicit snow representation (CN) provide higher consistency against external soil moisture references along an environmental snow-burden gradient?

---

## 2. Technical & Data Setup

### 2.1 Catchment Sample
- **Catchment Set**: Canonical 531 CAMELS-US basins (`data/531sub_id.txt`).
- **Evaluation Period**: Test period 1995-10-01 .. 2010-09-30 (5,479 contiguous daily steps).
- **Continuous Forward**: Full 12,418-day simulation (1980-10-01 .. 2014-09-30) from zero initial states, sliced to test period (no artificial state reset).

### 2.2 External Reference
- **Source**: Caravan v1.1 CAMELS-US NetCDF (`G:\Dataset\Caravan\timeseries\netcdf\camels\camels_<id>.nc`), cached locally at `results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz` (32.57 MB).
- **Primary Reference**: $SM_{100} = 0.07 L_1 + 0.21 L_2 + 0.72 L_3$ [m³/m³] (0–100 cm depth-weighted composite).
- **Sensitivity Reference**: $SM_{289} = (0.07 L_1 + 0.21 L_2 + 0.72 L_3 + 1.89 L_4) / 2.89$ [m³/m³] (0–289 cm full-profile composite).
- **Semantics**: External process-state reference (ERA5-Land daily basin-average), **NOT ground truth**.

### 2.3 Model Shared State
- **Primary State**: $W_{\text{total}} = WU + WL + WD$ [mm] (total XAJ catchment tension water store).
- **Comparability Rule**: Strict standardized / anomaly / timing dynamics only; **no absolute storage conversion (mm vs m³/m³)** and **no 1-to-1 depth horizon mapping**.

### 2.4 Environmental Snow-Burden Axis
- **Primary Axis**: Snow-17 ensemble-median annual maximum SWE [mm] over the test period (WY 1996..2010) from `results/r4_swe_reference_v1/`.
- **Secondary Axis**: Snow-17 median SWE-positive days.
- **Baseline Axis**: CAMELS static catchment attribute `frac_snow` (index 3).

---

## 3. Results Summary & Numerical Evidence

### 3.1 State Consistency Gain by Snow-Burden Quantile (Main Table 4)

Evaluated across 531 catchments partitioned by Snow-17 SWE burden ($Q_0$: 0–2 mm, $Q_1$: 2–35 mm, $Q_2$: 35–212 mm, $Q_3$: 212–1866 mm):

| Regime | Snow Regime (Quantile) | Catchments ($N$) | Median SWE (mm) | Base Anom Corr | CN Anom Corr | $\Delta$ Anomaly Corr (CN $-$ Base) | Base 7d Corr | CN 7d Corr | $\Delta$ 7d Corr (CN $-$ Base) |
|---|---|---|---|---|---|---|---|---|---|
| **dPL (Seed 42)** | $Q_0$ (0–2 mm) | 133 | 0.0 | 0.817 | 0.813 | $-0.001$ | 0.880 | 0.882 | $-0.000$ |
| | $Q_1$ (2–35 mm) | 133 | 23.7 | 0.798 | 0.801 | $-0.000$ | 0.872 | 0.876 | $+0.000$ |
| | $Q_2$ (35–212 mm) | 132 | 68.7 | 0.781 | 0.769 | $+0.000$ | 0.870 | 0.869 | $-0.001$ |
| | **$Q_3$ (212–1866 mm)** | **133** | **359.4** | **0.567** | **0.659** | **$+0.069$** [IQR: 0.168] | **0.723** | **0.821** | **$+0.063$** |
| **dPL (Seed 123)** | $Q_0$ (0–2 mm) | 133 | 0.0 | 0.812 | 0.818 | $+0.001$ | 0.877 | 0.880 | $+0.001$ |
| | $Q_1$ (2–35 mm) | 133 | 23.7 | 0.800 | 0.800 | $+0.002$ | 0.872 | 0.876 | $+0.001$ |
| | $Q_2$ (35–212 mm) | 132 | 68.7 | 0.774 | 0.771 | $+0.001$ | 0.865 | 0.869 | $-0.001$ |
| | **$Q_3$ (212–1866 mm)** | **133** | **359.4** | **0.576** | **0.656** | **$+0.050$** [IQR: 0.171] | **0.722** | **0.813** | **$+0.062$** |
| **IC Fused (5x200, Sens.)** | $Q_0$ (0–2 mm) | 133 | 0.0 | 0.784 | 0.793 | $-0.000$ | 0.851 | 0.852 | $-0.001$ |
| | $Q_1$ (2–35 mm) | 133 | 23.7 | 0.777 | 0.775 | $+0.000$ | 0.861 | 0.859 | $+0.001$ |
| | $Q_2$ (35–212 mm) | 132 | 68.7 | 0.737 | 0.749 | $+0.002$ | 0.839 | 0.852 | $+0.003$ |
| | **$Q_3$ (212–1866 mm)** | **133** | **359.4** | **0.559** | **0.651** | **$+0.065$** [IQR: 0.174] | **0.726** | **0.794** | **$+0.044$** |

### 3.2 Key Robustness Findings

1. **Response Shape (Deciles D01 $\to$ D10)**:
   - D01–D08 (SWE 0–134 mm): $\Delta\text{Anomaly} \approx 0.000$ (95% CI spans 0).
   - D09 (SWE median 307 mm): $\Delta\text{Anomaly} = +0.037$.
   - D10 (SWE median 751 mm): $\Delta\text{Anomaly} = \mathbf{+0.131}$ [$95\%$ CI: $+0.101, +0.153$], $\Delta\text{7d} = \mathbf{+0.276}$.
   - **Pattern**: Primarily high-snow emergence (SWE $\ge 200$ mm) rather than a linear continuous gradient.
2. **Process-Phase Concentration (Table S4)**:
   - **Phase 2 (Active Melt / Spring Recharge, 155,819 basin-days)**:
     - Base Anomaly Corr crashes to $0.226 \sim 0.273$.
     - CN Anomaly Corr remains high at $0.573 \sim 0.591$.
     - $\Delta\text{Anomaly} = \mathbf{+0.195 \sim +0.251}$, $\Delta\text{Daily} = \mathbf{+0.209}$.
   - **Phase 3 & 4 (Post-Melt & Summer Dry-Down, 984,821 basin-days)**:
     - Base and CN converge completely ($\Delta\text{Anomaly} \in [-0.008, -0.001]$).
   - **Proof**: The state discrepancy is specifically triggered during spring snowmelt recharge.
3. **Timing Error Reduction**:
   - **Spring Wet-Up Timing**: CN reduces absolute recharge onset error by **$14 \sim 18$ days** (Base error $\sim 39\text{ d} \to$ CN $\sim 23\text{ d}$).
   - **Soil Moisture Peak Timing**: Base peaks **$-25$ to $-26.5$ days too early** (false winter saturation in Dec–Feb), while CN corrects this toward spring snowmelt.
4. **Leave-One-Region-Out & Trimming**:
   - Zero sign flips across all 18 HUC regions ($\rho \in [0.11, 0.34]$).
   - Trimming top 1% and top 5% extreme-SWE catchments preserves positive association and high-snow gain ($+0.06 \sim +0.10$).
5. **Multiple Regression Controlling for $\Delta\text{KGE}$**:
   - Standardized $\beta_1(\text{SWE})$ remains strictly positive across all regimes ($+0.010 \sim +0.034$).

---

## 4. Artifact Inventory

All formal artifacts reside under `results/r4_phase1_soil_official/`:

| File | Description | Size / Rows |
|---|---|---|
| `basin_state_consistency.csv` | Basin-level state consistency (531 basins × 2 models × 3 regimes) | 3,187 rows |
| `paired_structural_effects.csv` | Paired $\Delta C(\text{CN} - \text{Base})$ per basin across all metrics | 1,594 rows |
| `timing_metrics_basin_year.csv` | Water-year level peak & wet-up timing metrics | 47,791 rows |
| `timing_metrics_basin_summary.csv` | Basin-level median timing errors and IQRs | 3,187 rows |
| `snow_burden_quartile_summary.csv` | $Q_0 \sim Q_3$ quantile summary table (Main Table 4 source) | 13 rows |
| `robustness_performance_subsets.csv` | Similar-KGE subsets ($|\Delta\text{KGE}| \le 0.02, \le 0.05$) | 10 rows |
| `robustness_controlled_regressions.csv` | Controlled OLS regression with 2,000-replicate bootstrap CIs | 13 rows |
| `robustness_leave_one_region_out.csv` | Leave-one-region-out cross-validation across 18 HUC regions | 58 rows |
| `robustness_extreme_swe_trimming.csv` | Extreme-SWE trimming (top 1%, top 5%) | 10 rows |
| `robustness_swe_decile_shape.csv` | Decile response shape (D01..D10) | 31 rows |
| `robustness_process_phase_consistency.csv` | 4-phase process-conditioned consistency (Table S4 source) | 4,213 rows |
| `robustness_timing_sensitivity.csv` | Timing definition sensitivity across 7d/14d/21d & annual/spring windows | 13 rows |
| `r4_phase1_soil_official_report.json` | Master JSON report (state consistency) | 42 KB |
| `r4_robustness_report.json` | Master JSON report (robustness checks) | 4.8 KB |

Manuscript tables & figures:
- `manuscript/tables/Table4_soil_state_consistency.md` & `.tex`
- `manuscript/tables/TableS4_process_phase_and_robustness.md` & `.tex`
- `manuscript/figures/figure4_r4_soil_consistency.png` (600 dpi) & `.pdf`

---

## 5. Paper Writing Guidance (Claim Boundaries)

### Allowed Claims
1. In strongly snow-affected catchments ($Q_3$, SWE $\ge 200$ mm), explicit snow representation (CN) significantly improves shared downstream soil moisture consistency against Caravan ERA5-Land $SM_{100}$ compared to Base ($\Delta\text{Anomaly} \approx +0.05 \sim +0.07$, $\Delta\text{7d} \approx +0.06$).
2. This state consistency advantage is process-specific and overwhelmingly concentrated in **Phase 2 (Active Melt / Spring Recharge)** ($\Delta\text{Anomaly} \approx +0.20$), while converging to zero in summer dry-down.
3. Explicit snow physics substantially reduces spring recharge onset timing error by $14 \sim 18$ days and corrects Base's systematic false-winter-saturation bias ($-25$ days).
4. Results are consistent across independent dPL training seeds and the fused-IC sensitivity regime, scale-invariant across daily, 7-day, and monthly aggregations, and robust to regional omission and extreme SWE trimming.

### Forbidden Claims
1. **DO NOT claim a uniform linear continuous gradient**: characterize the response as *high-snow emergence / strongly snow-affected catchments*.
2. **DO NOT call ERA5-Land soil moisture "truth"**: use *external process-state consistency reference*.
3. **DO NOT claim 1-to-1 depth correspondence** between XAJ layers and ERA5-Land layers.
4. **DO NOT claim CN uniquely outperforms TGD2 in soil moisture**: TGD2 observation-trained checkpoints are pending (`TGD2_PENDING`), so only Base vs CN contrast is formally proven.
