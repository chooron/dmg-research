# Figure 7 (F7) Production Notes & Design Specifications

## 1. Scientific Objectives and Evidence Architecture
Figure 7 serves as the closing interpretive synthesis figure of Manuscript R3, answering:
$$\boxed{\text{\textbf{How far can same-coordinate specificity be interpreted?}}}$$

The figure establishes a two-panel definitive boundary narrative:
$$\boxed{\text{Beyond raw rank similarity (a)} \longrightarrow \text{Hydrological functional-role boundary (b)}}$$

---

## 2. Panel-by-Panel Architecture & Data Audit

### Panel (a): Beyond Rank Similarity (HERO Scatter)
- **Scientific Role**: Hostile check verifying whether same-coordinate information specificity is merely a reflection of raw parameter-value rank continuity ($R_{\mathrm{rank}}$).
- **Data Source**: `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/tables/matching_coordinate_results.csv` (`spec == 'primary_k3_no_caliper'`).
- **Data Space**: $n = 270$ matched parameter coordinates across 35 multi-parameter models (`collie1` structurally unmatched as $P=1$).
- **Matching Specification**:
  - For each parameter $(p,p)$, matched against the $k=3$ nearest off-diagonal alternatives by absolute rank distance ($|Q_{\mathrm{rank}}(p,p) - Q_{\mathrm{rank}}(p,q)|$), no caliper.
  - $x$-axis: Mean information correspondence of matched off-diagonal alternatives.
  - $y$-axis: Same-coordinate information correspondence.
- **Key Empirical Statistics**:
  - Model-equal residual advantage: $\Delta = A_{\mathrm{info}|\mathrm{rank}} = 0.295238 \approx \mathbf{0.295}$.
  - Model-clustered 95% bootstrap CI: $[\mathbf{0.254386}, \mathbf{0.429073}] \approx [0.254, 0.429]$.
  - Sign-flip test: $p = 0.000400 < \mathbf{0.001}$.
  - Model consistency: $100\%$ ($35/35$ finite model summaries $> 0$).

### Panel (b): Functional-Role Boundary (Paired-Dot/Line Plot)
- **Scientific Role**: Tests whether same-coordinate specificity extends to broader hydrological functional roles (e.g., across distinct soil storage, runoff partitioning, or routing parameters).
- **Data Source**: `project/benchmark/results/joh_functional_role_diagnostic_20260905/tables/13_ROLE_OFFDIAGONAL_ADVANTAGE_BY_MODEL.csv` (`confidence_filter == 'high'`, `valid_role_parameter_count > 0`).
- **Data Space**: $n = 29$ conceptual models possessing valid same-role and different-role alternative parameter contrasts.
- **Key Empirical Statistics**:
  - Model-equal same-role alternative median: $\mathbf{-0.027820} \approx -0.028$.
  - Model-equal different-role alternative median: $\mathbf{+0.005263} \approx +0.005$.
  - Role off-diagonal advantage: $A_{\mathrm{role}} = \mathbf{-0.118797} \approx -0.119$.
  - Role-count-preserving permutation test: empirical $p = \mathbf{0.805719} \approx 0.806$.
  - Positive models: $12 / 29$ ($41.4\%$).
- **Pre-specified HESS-Prior Robustness (in Caption)**:
  - $D_{\rho} = -0.00033$, 95% CI: $[-0.00669, 0.02653]$, $p = 0.850$.

---

## 3. Visual & Aesthetic Execution
- **Layout**: Asymmetric 2-column layout (Panel a HERO $\approx 64\%$ width, Panel b $\approx 36\%$ width).
- **Palette**: Deep Navy (`#1e3a5f`) for observed coordinates, Purple (`#7b1fa2`) for role medians, Neutral Slate (`#64748b`, `#cbd5e1`) for paired lines and reference boundaries.
- **Minimal Text Principles**: Zero explanation boxes, zero flowchart arrows, zero redundant annotations; only 1 unboxed statistical summary per panel.
- **Export Specifications**:
  - Resolution: 600 DPI PNG ($6900 \times 3120$ px), white background.
  - Format: PNG only (no PDF).
  - File Paths:
    - Primary: `project/benchmark/manuscript/r3/figures/F7.png`
    - Synchronized: `project/benchmark/manuscript/r3/F7_main.png`
    - Script: `project/benchmark/manuscript/r3/scripts/plot_F7.py`
    - Caption: `project/benchmark/manuscript/r3/F7_caption.md`
