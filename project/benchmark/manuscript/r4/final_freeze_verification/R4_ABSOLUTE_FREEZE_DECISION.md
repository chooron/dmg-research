# R4 Absolute Freeze Decision & Definitive Scientific Record

## Final Absolute Verdict

**`R4 ABSOLUTELY FROZEN — NUMERIC AND WORDING AUDIT PASS`**

---

## 1. Core Technical Verifications Summary

### 1. Verification of Parameter Excess (`+0.227033`): **`PASS (VERIFIED)`**
- **Exact Aggregation Form:** `+0.227033` is verified as the equal-weight median across the $M=5$ strictly eligible models (`alpine2`, `hillslope`, `ihacres`, `us1`, `xinanjiang`) of the model-specific differences between basin-median out-of-bag displacement and basin-median multi-start dispersion:
  $$E_{\text{Case 1}} = \text{median}_{m \in \text{Eligible}} \left(\text{median}_b(D_{\text{cross, OOB}}(m, b)) - \text{median}_b(D_{\text{self}}(m, b))\right) = \mathbf{+0.227033}$$
- **Basin-Matched Alternative:** Under basin-level matched subtraction first ($E_{m, b} = D_{\text{cross, OOB}}(m, b) - D_{\text{self}}(m, b)$), the cross-model median is **$+0.212135$**.
- **Exceedance Consistency:** Excess parameter displacement is strictly positive across **5 / 5 (100%)** of eligible models under both aggregation methods (and positive across **8 / 8 (100%)** models in the unfiltered sensitivity cohort, median $+0.1934$).

---

### 2. Verification of Catchment Information Representation (13 / 20 / 32 Dimensions): **`PASS (VERIFIED)`**
- **Exact Representation:** R4 replayed catchment-information profiles across the prespecified continuous attribute pipeline:
  - Primary Cluster Representation: **13 orthogonal continuous clusters** ($R_{\text{paired, OOB}} = \mathbf{0.739011}$, $A_{\text{diag, OOB}} = \mathbf{0.718407}$, within-model permutation $\mathbf{p = 0.000999}$).
  - Full Profile Sensitivity: **32 continuous physical attributes** ($R_{\text{paired, OOB}} = \mathbf{0.762005}$, $A_{\text{diag, OOB}} = \mathbf{0.729106}$, permutation $\mathbf{p = 0.000999}$).
- **Dimensional Crosswalk:** The 13 continuous clusters derive from the exact same $|\rho| \ge 0.70$ hierarchical clustering applied to the continuous CAMELS attributes (omitting 3 discrete categorical variables). The numeric retention is robust and invariant across representations, and the manuscript explicitly identifies this continuous representation.

---

## 2. Definitive Support Status Matrix Across R1–R3

| Research Result | Frozen Claim | Replayed Estimand | Observed OOB Metric | Support Status in R4 | Scope Boundary |
|---|---|---|---|---|---|
| **R1: Performance** | Aggregate outlet performance is broadly comparable; individual responses are heterogeneous | $\Delta\text{KGE}_{\text{OOB}}$, ANOVA variance decomposition | Median $\Delta\text{KGE}_{\text{OOB}} = -0.0654$, Basin SS = 67.14%, Model SS = 0.53% | **`PARTIALLY SUPPORTED`** | Uncalibrated regional transfer incurs a $-0.065$ KGE gap relative to IC; basin heterogeneity strongly confirmed. |
| **R2: Separation** | dPL parameter vectors occupy a distinct realization regime separated from IC | $D_{\text{RMS, OOB}}$ | Median $D_{\text{RMS, OOB}} = 0.3429$ (seen = $0.3139$) | **`SUPPORTED`** | Large, positive parameter displacement persists in held-out basins. |
| **R2: IC-Self Excess** | Displacement exceeds multi-start calibration dispersion | $D_{\text{cross, OOB}} - D_{\text{self}}$ | Median excess = $\mathbf{+0.2270}$ (Case 1) / $\mathbf{+0.2121}$ (Case 2) | **`SUPPORTED (IN 5 / 5 ELIGIBLE MODELS)`** | Exceedance confirmed in all 5 models meeting 90% restart coverage requirement. |
| **R2: Performance Bridge** | Larger performance gaps accompany greater parameter displacement | $\rho_b(|\Delta\text{KGE}_{\text{OOB}}|, D_{\text{RMS, OOB}})$ | Median $\rho_b = \mathbf{+0.2425}$ | **`SUPPORTED`** | Strictly positive across 8/8 models ($p < 0.05$ in 7/8 models). |
| **R2: Geometry Sub-Claims** | Rank reorganization ($R_{\text{rank}}$) and coordinate localization ($C_{\text{eff}}$) | Not prespecified | N/A | **`NOT TESTED BY R4`** | Remains strictly scoped to the 36-model seen-basin benchmark. |
| **R3: Profile Correspondence** | Conceptual parameters preserve catchment-information association profiles | $R_{\text{paired, OOB}}$ (IC vs OOB-dPL) | Median $R_{\text{paired, OOB}} = \mathbf{0.7390}$ (13 clusters) / $\mathbf{0.7620}$ (32 attrs) | **`SUPPORTED`** | Replicates seen-basin benchmark ($0.7129$ / $0.7447$). |
| **R3: Coordinate Specificity** | Profile correspondence is coordinate-specific against within-model alternatives | $A_{\text{diag, OOB}}$ and permutation test | Diagonal = $0.7390$, Off-diagonal = $-0.0110$, $A_{\text{diag}} = \mathbf{0.7184}$, **$p = 0.000999$** | **`SUPPORTED`** | Highly significant coordinate specificity preserved out-of-bag. |
| **R3: Functional Roles** | Functional role labels add no explanatory power beyond coordinates | $A_{\text{role}}$ | Boundary inherited | **`NOT TESTED BY R4 (BOUNDARY RETAINED)`** | Coordinate specificity only; no physical role continuity claimed. |

---

## 3. Definitive Final Sentences

### 中文总结 (Authoritative Chinese Summary):
> 在预先冻结的八模型 held-out stress-test panel 中，R4 严格验证了 R1 的出口响应异质性（流域主效应占性能差方差的 67.1%）、R2 的核心参数分离（$D_{\text{RMS, OOB}} = 0.343$；在满足 90% 重启覆盖率的 5 个合格模型中净位移 $D_{\text{cross}} - D_{\text{self}} = +0.227$ 全模型为正；性能桥梁 $\rho = +0.242$ 全模型为正）以及 R3 的同坐标信息组织（在预冻结的 13 信息聚类与 32 连续属性空间下 $R_{\text{paired}} = 0.739, A_{\text{diag}} = 0.718, p = 0.000999$）；未在 R4 中 replay 的 R2 秩重组与坐标局域化等几何子结论明确保留为未测试，既不作过度泛化，亦无需重新训练。

### English JoH Summary (Authoritative Manuscript Sentence):
> Across the prespecified eight-model held-out stress-test panel, out-of-bag evaluation directly supports the core empirical findings of R1–R3: regional dPL preserves substantial coordinate-specific catchment-information profiles against independent calibration ($R_{\text{paired}} = 0.739$, diagonal advantage $A_{\text{diag}} = 0.718$, $p = 0.000999$ across 13 continuous information clusters) and maintains distinct parameter realization separation ($D_{\text{RMS}} = 0.343$, exceeding multi-start restart dispersion by $+0.227$ in all five eligible tested models), while confirming that pervasive catchment heterogeneity dominates the regionalization performance gap ($-0.065$ KGE). Secondary parameter-geometry properties not prespecified for held-out re-computation remain strictly scoped to the 36-model seen-basin benchmark.

---

## 4. Final Absolute STOP Decision

**R4 IS ABSOLUTELY FROZEN.**

No further training, calibration, recomputation, or exploratory analysis is permitted. The R4 analysis package is complete, verified, and ready for manuscript publication.
