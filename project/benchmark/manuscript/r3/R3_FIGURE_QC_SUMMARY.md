# R3 Figure Quality Control & Cross-Figure Synthesis Summary

**Document Purpose**: Comprehensive synthesis and formal quality control (QC) audit for **Figures 5, 6, and 7** of the Journal of Hydrology (JoH) manuscript (R3).  
**Target Directory**: `project/benchmark/manuscript/r3/` (`\\wsl.localhost\Debian\home\jingxin\code\dmg-research\project\benchmark\manuscript\r3`)  
**Status**: **PASSED — ALL 3 FIGURES FORMALLY GENERATED AT 600 DPI (PNG ONLY, NO PDF/INTERMEDIATES)**

---

## 1. Figure Architecture and Scientific Question Division

The three figures implement the progressive scientific narrative of R3 without semantic overlap or statistical inflation:

$$\boxed{ \text{F5: What is retained?} \longrightarrow \text{F6: Is it coordinate-specific?} \longrightarrow \text{F7: Reference scale \& interpretation boundary} }$$

| Figure | Scientific Question | Hero Panel | Evidence & Headline | Hydrological Subject Matter |
|---|---|---|---|---|
| **Figure 5** | *How much catchment–parameter association structure is retained across calibration paradigms, and is that overlap symmetric?* | **(a) Nested Association Ledger** ($5{,}420 \to 902 \to 712 \to 692$) | - $78.9\%$ of IC-stable cells remain strong in dPL<br>- Only $20/712$ ($2.8\%$) sign reversals<br>- Bidirectional asymmetry: $P(\text{dPL}\|\text{IC})=78.9\%$ vs $P(\text{IC}\|\text{dPL})=32.9\%$ | Catchment parameter–information associations across 20 physical attribute dimensions |
| **Figure 6** | *Is the retained correspondence preferentially aligned with the same model parameter coordinate rather than alternative coordinates?* | **(a) Representative-Model Correspondence Atlas** (Real $C_{m,p,q}$ matrices for GR4J, SIMHYD, HBV96) | - Headline $A_{\mathrm{diag}} = 0.6150 \gg \text{null } 0.0015$ ($p = 0.000999$)<br>- Ensemble breadth: $35/35$ models $A_m > 0$<br>- Top-1 share = $57.56\%$ (vs $13.28\%$ random) | Real conceptual hydrological model parameters and mathematical state variables |
| **Figure 7** | *How strong is cross-paradigm coordinate specificity relative to within-IC repeatability, and does that specificity extend to broad hydrological functional roles?* | **(a) Model-Level Paired Coordinate Reference** ($A_{\mathrm{diag}}^{\mathrm{self}} = 0.981$ vs $A_{\mathrm{diag}}^{\mathrm{cross}} = 0.604$, $n=34$) | - Repeatability gap: $\Delta_A = +0.339$ ($31/34 > 0$)<br>- Profile correlation gap: $\Delta_R = +0.217$ ($35/35 > 0$)<br>- Functional-role boundary: $A_{\mathrm{role}} = -0.119$ ($p = 0.806$) | Within-IC calibration repeatability reference vs. broad hydrological functional roles |

---

## 2. Core Scientific Integrity & Frozen Numbers Check

Every number across all three figures, captions, and design notes matches the frozen empirical audit tables with zero tolerance deviations:

### 2.1 Figure 5 Verification
- **All candidate cells**: $N = 5{,}420$
- **IC strong/stable cells**: $n = 902$ ($|\rho_{\mathrm{IC}}| \ge 0.20$ and bootstrap sign stability $\ge 0.95$, identically matching $|\rho_{\mathrm{IC}}| \ge 0.20$ with Jaccard $1.000$).
- **Mutually strong cells**: $n = 712$ ($78.936\%$ of IC-stable cells).
- **Mutually strong & same sign**: $n = 692$ ($97.191\%$ of mutually strong cells, $76.718\%$ of IC-stable cells).
- **Mutually strong & sign-flipped**: $n = 20$ ($2.809\%$ of mutually strong cells).
- **Conditioning asymmetry**:
  - $P(\text{dPL strong} \mid \text{IC strong}) = 712 / 902 = 78.936\%$
  - $P(\text{IC strong} \mid \text{dPL strong}) = 712 / 2{,}163 = 32.917\%$
- **Threshold progression ladder ($n = 902$)**:
  - Same sign: $849/902 = 94.124\%$
  - Same sign $+ |\rho_{\mathrm{dPL}}| \ge 0.10$: $791/902 = 87.694\%$
  - Same sign $+ |\rho_{\mathrm{dPL}}| \ge 0.20$: $692/902 = 76.718\%$
  - Same sign $+ |\rho_{\mathrm{dPL}}| \ge 0.30$: $517/902 = 57.317\%$

### 2.2 Figure 6 Verification
- **Headline Coordinate Specificity**: $A_{\mathrm{diag}} = 0.615038$ across $n = 35$ multi-parameter models (`collie1` excluded as 1-parameter model).
- **Permutation Null**: Null mean $= 0.001536$, empirical $p = 0.000999$ ($N = 1{,}000$ label permutations).
- **Ensemble Breadth**: $35/35$ models ($100\%$) have $A_m > 0$; median $= 0.6150$, $\mathrm{IQR} = [0.4508, 0.7643]$.
- **Rank Metrics ($N = 271$ parameters)**:
  - Top-1 share: $156/271 = 57.565\%$ (random expectation $= 13.284\%$)
  - Top-2 share: $193/271 = 71.218\%$ (random expectation $= 26.199\%$)
  - Top-3 share: $228/271 = 84.133\%$ (random expectation $= 39.114\%$)
- **Dimensionality Associations**: $\rho(\mathrm{Top1}, P_m) = -0.564$ ($p = 0.00042$), $\rho(A_m, P_m) = -0.283$ ($p = 0.100$).

### 2.3 Figure 7 Verification
- **$A_{\mathrm{diag}}$ Paired Reference ($n = 34$ models)**:
  - IC-self median $= 0.981203$, matched cross median $= 0.603759$.
  - Paired difference $\Delta = +0.339474$, $95\%$ bootstrap CI: $[0.202256, 0.428571]$.
  - Positive models: $31/34$ ($91.2\%$); 3 negative models transparently reported (`collie2` $-0.720$, `gr4j` $-0.123$, `tcm` $-0.393$).
- **$R_{\mathrm{paired}}$ Paired Reference ($n = 35$ models)**:
  - IC-self median $= 0.967669$, matched cross median $= 0.731579$.
  - Paired difference $\Delta = +0.217293$, $95\%$ bootstrap CI: $[0.156391, 0.284211]$.
  - Positive models: $35/35$ ($100\%$).
- **Strict $R^2$ Subset Sensitivity**:
  - $R_{\mathrm{paired}}$ ($n=23$): IC-self $= 0.983459$ vs matched cross $= 0.742105$ ($\Delta = +0.216541$).
  - $A_{\mathrm{diag}}$ ($n=22$): IC-self $= 1.026316$ vs matched cross $= 0.701504$ ($\Delta = +0.285338$).
- **Functional-Role Interpretation Boundary ($n = 36$ models)**:
  - $A_{\mathrm{role}} = -0.118797$, permutation empirical $p = 0.805719$, positive models $12/36$ ($33.3\%$).
  - Same-role off-diagonal median $= -0.027820$, cross-role median $= +0.005263$.

---

## 3. Demotion of Non-Hero Statistical Detail to SI / Notes

To preserve hydrological readability in the main manuscript, secondary and purely statistical analyses were deliberately routed to design notes and SI tables:
1. **271-Row Abstract Pseudo-Matrix**: Replaced in F6 hero by 3 concrete, physically interpretable conceptual models (GR4J, SIMHYD, HBV96); full 271-row rank atlas retained for Supplementary Information.
2. **Raw Attribute Space Redundancy**: Main text figures exclusively focus on the 20 orthogonal information dimensions; raw 35-attribute space summaries are detailed in tabular SI.
3. **Leave-One-Out (LOO) Model Jackknife**: Jackknife sensitivity tests are documented in `F6_design_notes.md` rather than occupying visual canvas.
4. **HESS Prior-Weighting Sub-tests**: Secondary parameter-weighting role tests are documented in `F7_design_notes.md`.

---

## 4. Reviewer-Risk Mitigation & Boundary Discipline

| Risk Point | Potential Reviewer Misinterpretation | Explicit Safeguard in Main Figures & Captions |
|---|---|---|
| **Reference Status** | Interpreting within-IC reference as an "upper bound" or "ceiling" | Explicitly labeled as a **calibration-scale repeatability comparator**, reflecting independent gradient-descent restarts under a $\Delta\mathrm{KGE} \le 0.01$ rule. |
| **Physical Identity** | Interpreting coordinate specificity as proof of physical parameter identity preservation | Caption explicitly clarifies that correspondence represents **preferential coordinate alignment** across empirical parameterizations, not invariant physical truth. |
| **Causal Attribution** | Interpreting self–cross gap as an isolated causal paradigm effect | Caption explicitly notes that an empirical multi-start dPL-self reference was not part of the canonical experiment, preventing isolated causal claims. |
| **dPL Validity** | Interpreting $P(\text{IC}\|\text{dPL}) = 32.9\%$ as proof that dPL associations are invalid | Explicitly clarified in F5 panel (b) that asymmetry reflects the **denser association field** naturally generated by attribute-constrained mapping. |
| **Role Boundary** | Interpreting negative $A_{\mathrm{role}}$ as cross-role parameters being "more similar" | Clarified as an **interpretation boundary**: broad functional role sharing confers zero additional correspondence advantage once exact coordinates are excluded. |

---

## 5. Visual, Typographic, and Export Compliance

- **File Format & Resolution**: All figures exported as single, standalone `.png` files at **600 DPI** with no auxiliary PDF or temporary files.
- **Typography**: Unified Serif family (`Times New Roman` / `STIXGeneral`), matching Journal of Hydrology publication specifications.
- **Color Semantics**:
  - Independent Calibration (IC / Within-IC Self): Deep Navy / Slate Blue (`#2b5c8f` / `#1f4e79`)
  - Cross-Paradigm (IC–dPL): Warm Terracotta / Amber (`#d95f02` / `#c2593f`)
  - Retained / Agreement: Deep Forest Jade (`#2a7b4c`)
  - Sign Flip / Disagreement: Coral Red (`#d32f2f`)
  - Baselines / Null: Neutral Slate Grey (`#718096` / `#8c9ba5`)
- **Consistent Model Ordering**: Models in F6 panel (c) and F7 panel (a) follow an identical hydrological progression sorted by model parameter dimensionality ($P_m$, ascending).

---

## 6. Deliverable Artifact Registry

All required deliverable files are in place in `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/`:

- `F5_main.png` (6176 × 4115 px, 600 DPI, 1.47 MB)
- `F5_caption.md`
- `F5_design_notes.md`
- `F6_main.png` (7031 × 5058 px, 600 DPI, 1.67 MB)
- `F6_caption.md`
- `F6_design_notes.md`
- `F7_main.png`
- `F7_caption.md`
- `F7_design_notes.md`
- `R3_FIGURE_QC_SUMMARY.md`
- Python plotting scripts under `scripts/plot_r3_figure5.py`, `scripts/plot_r3_figure6.py`, and `scripts/plot_r3_figure7.py`.

**Verdict**: Figures 5, 6, and 7 meet all scientific, statistical, and Journal of Hydrology publication standards.
