# R4 Final Freeze Decision & Definition Alignment Synthesis

## Final Overall Verdict

**`B. FREEZE R4 — PARTIAL HELD-OUT SUPPORT; R3 REPRESENTATION QUALIFIED`**

---

## 1. Formal Results & Sub-Claim Freeze Ledger

### 1.1 R1: Outlet Performance Response
- **Overall R1 Verdict:** **`PARTIALLY SUPPORTED`**
- **Exact Numeric Baseline:**
  - IC Median KGE (8 models): **0.649873**
  - Seen-dPL Median KGE (8 models): **0.648533**
  - OOB-dPL Median KGE (8 models): **0.581336**
  - Median $\Delta\text{KGE}_{\text{OOB}}$: **-0.065384** (regionalization penalty of $-0.057526$ relative to seen dPL)
- **Heterogeneity & Variance Shares:**
  - OOB-dPL outperforms locally calibrated IC in **26.13%** of individual catchment instances.
  - Variance decomposition: **Model Additive = 0.53%**, **Basin Additive = 67.14%**, **Remainder = 32.33%**.
- **Scope Qualification:** Regional transfer without local calibration introduces an aggregate performance gap of $-0.065$ KGE, but strongly preserves R1's primary thesis of pervasive place-specific heterogeneity.

---

### 1.2 R2: Parameter Realization Response
- **Overall R2 Verdict:** **`PARTIALLY SUPPORTED`**
- **Sub-Claim Status Matrix:**
  1. `R2 raw parameter separation (D_RMS)`: **`SUPPORTED`**
     - Median $D_{\text{RMS, OOB}} = \mathbf{0.342886}$ (vs. seen $0.313853$).
     - Median shift between seen-dPL and OOB-dPL parameters is only **$0.100356$**.
  2. `R2 separation beyond IC-self`: **`SUPPORTED (IN 5 / 5 ELIGIBLE TESTED MODELS)`**
     - In the 5 models meeting the strict 90% multi-start restart coverage threshold (`alpine2`, `hillslope`, `ihacres`, `us1`, `xinanjiang`), median excess displacement is **$D_{\text{cross, OOB}} - D_{\text{self}} = \mathbf{+0.227033}$** (positive in 5/5 models).
     - The remaining 3 models (`hbv96`, `mopex4`, `newzealand2`) are retained in the frozen `INSUFFICIENT_REFERENCE` category ($D_{\text{cross, OOB}} - D_{\text{self}} = +0.1934$ across all 8 models).
  3. `R2 performance-parameter bridge`: **`SUPPORTED`**
     - Spearman correlation $\rho_b(|\Delta\text{KGE}_{\text{OOB}}|, D_{\text{RMS, OOB}})$ is strictly positive in **8 / 8 models** (model-equal median **$+0.242456$**, $p < 0.05$ in 7/8 models).
  4. `R2 cross-catchment rank reorganization (R_rank)`: **`NOT TESTED BY R4`** (scoped to seen benchmark).
  5. `R2 coordinate localization (C_eff)`: **`NOT TESTED BY R4`** (scoped to seen benchmark).
  6. `R2 contraction boundary`: **`NOT TESTED BY R4`** (canonical IC dependent; boundary retained).

---

### 1.3 R3: Catchment Information Organization
- **Overall R3 Verdict:** **`STRONGLY SUPPORTED (UNDER PRESPECIFIED ALTERNATIVE REPRESENTATIONS)`**
- **Representation Specification:** **`ALTERNATIVE REPRESENTATION SUPPORT`**
  - Evaluated on the prespecified 13 continuous information clusters and the full 32 continuous attribute space (derived from CAMELS continuous attributes at $|\rho| \ge 0.70$), which correspond to the continuous-variable subset of the frozen 20-dimensional information space.
- **Sub-Claim Status Matrix:**
  1. `R3 same-parameter profile correspondence (R_paired)`: **`SUPPORTED`**
     - Model-equal median $R_{\text{paired, OOB}} = \mathbf{0.739011}$ (13 clusters) and **$0.762005$** (32 continuous attributes), matching seen-basin values ($0.712912$ / $0.744685$; Full36 seen = $0.715789$).
  2. `R3 same-coordinate specificity (A_diag)`: **`SUPPORTED`**
     - Diagonal profile alignment (median $0.7390$) markedly exceeds within-model off-diagonal alternatives (median $-0.0110$), yielding an out-of-bag diagonal advantage of **$A_{\text{diag, OOB}} = \mathbf{0.718407}$** (13 clusters) and **$0.729106$** (32 attributes).
     - Within-model parameter-label permutation test (1,000 permutations) yields **$p = 0.000999$** ($p < 0.001$).
  3. `R3 rank-residual specificity`: **`NOT TESTED BY R4 (SEEN EVIDENCE INHERITED)`**.
  4. `R3 functional-role boundary`: **`NOT TESTED BY R4 (BOUNDARY RETAINED)`** (information organization is coordinate-specific only; functional role extensions remain null).
- **Supporting Continuity Finding:**
  - OOB-dPL parameter–attribute relationships correlate with seen-dPL at **$0.9616$** (median $0.9574$), with **99.93%** non-trivial sign retention and **96.44%** information cluster retention. (Classified strictly as supporting continuity of the neural parameterizer).

---

## 2. Definitive Summary Statements

### 中文总结 (Authoritative Chinese Summary):
> 在预先冻结的八模型 held-out stress-test panel 中，R4 严格验证了 R1 的出口响应异质性（流域主效应占性能差方差的 67.1%）、R2 的核心参数分离（$D_{\text{RMS, OOB}} = 0.343$；在满足 90% 重启覆盖率的 5 个合格模型中净位移 $D_{\text{cross}} - D_{\text{self}} = +0.227$ 全模型为正；性能桥梁 $\rho = +0.242$ 全模型为正）以及 R3 的同坐标信息组织（在预冻结的 13 信息聚类与 32 连续属性空间下 $R_{\text{paired}} = 0.739, A_{\text{diag}} = 0.718, p = 0.000999$）；未在 R4 中 replay 的 R2 秩重组与坐标局域化等几何子结论明确保留为未测试，既不作过度泛化，亦无需重新训练。

### English JoH Summary (Authoritative Manuscript Sentence):
> Across the prespecified eight-model held-out stress-test panel, out-of-bag evaluation directly supports the core empirical findings of R1–R3: regional dPL preserves substantial coordinate-specific catchment-information profiles against independent calibration ($R_{\text{paired}} = 0.739$, diagonal advantage $A_{\text{diag}} = 0.718$, $p = 0.000999$ across 13 continuous information clusters) and maintains distinct parameter realization separation ($D_{\text{RMS}} = 0.343$, exceeding multi-start restart dispersion by $+0.227$ in all five eligible tested models), while confirming that pervasive catchment heterogeneity dominates the regionalization performance gap ($-0.065$ KGE). Secondary parameter-geometry properties not prespecified for held-out re-computation remain strictly scoped to the 36-model seen-basin benchmark.

---

## 3. Final STOP Rule

**R4 FINAL AUDIT COMPLETE — FREEZE AND STOP.**

All technical definitions, dimensional crosswalks, IC-self references, and manuscript statements are mathematically verified and frozen. No further training, calibration, model additions, or exploratory analyses are permitted.
