# R4 Eight-Model Subset Coverage Audit Using Pre-OOB Descriptors

## 1. Executive Summary

This audit quantifies how comprehensively the eight models selected for the R4 out-of-bag (OOB) held-out-basin experiment (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`) span the design space and empirical response regimes of the 36-model benchmark.

**Strict Anti-Leakage Protocol:** All metrics, structural classifications, and empirical response statistics evaluated here derive strictly from pre-OOB data sources frozen on or before 2026-09-01 (`SEENBASIN_MASTER_MODEL_SUMMARY.csv`, `09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv`, `OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv`). Zero OOB held-out outcomes are used.

### Primary Coverage Verdict:
- **Structural Diversity:** **STRONG SPANNING COVERAGE**. The eight models span parameter counts from 5 to 15 (71.4% range span), state counts from 1 to 5 (100% range span), proportional snow representation (37.5% vs 33.3% in full benchmark), balanced routing architectures (50% base vs 50% endpoint), and contrasting runoff-generation concepts (bucket storage, saturation excess, non-linear cascades).
- **Seen-Basin Response Diversity:** **STRONG 4-QUADRANT SPANNING COVERAGE**. The subset provides exact orthogonal 2-model coverage across all four combinations of flexibility gap ($G_{\text{seen}}$) and parameter reproducibility ($R_{\text{seen}}$), spans 99.3% of the IC restart uncertainty range ($U$), 67.0% of the parameter displacement range ($D_{\theta}$), and maintains baseline performance viability ($K_{\text{joint}} \ge 0.561$).
- **Representativeness Limitation:** While the subset spans contrasting structural and empirical archetypes effectively, it does not constitute an identically distributed or random probability sample of the 36 models. It is intentionally stratified to cover contrasting extremes and performance-viable formulations.

---

## 2. Quantitative Comparison: 8 Selected Models vs. Full 36-Model Benchmark

### Summary Statistics Table

| Descriptor / Metric | Category | 36-Model Range | 36-Model Median (IQR) | 8-Model Range | 8-Model Median (IQR) | Range Span (%) | Quadrant / Domain Coverage |
|---|---|---|---|---|---|---|---|
| **Free Parameters ($P$)** | Structural | $[1, 15]$ | $7.00$ ($[5.75, 9.25]$) | $[5, 15]$ | $7.50$ ($[6.00, 10.50]$) | $71.4\%$ | Low ($5, 6$), Mid ($7, 8, 10$), High ($12, 15$) |
| **State Count ($S$)** | Structural | $[1, 5]$ | $4.00$ ($[2.00, 5.00]$) | $[1, 5]$ | $2.00$ ($[2.00, 4.25]$) | $100.0\%$ | Full discrete span ($1, 2, 4, 5$) |
| **Snow Process ($S_{\text{snow}}$)** | Structural | $12/36$ ($33.3\%$) | N/A | $3/8$ ($37.5\%$) | N/A | Balanced | HBV, Alpine2, MOPEX4 vs 5 rain-only |
| **Routing Form** | Structural | Mix (Base/End) | N/A | $4$ Base / $4$ End | N/A | $50.0\% / 50.0\%$ | Balanced convolution vs direct |
| **Flexibility Gap ($G_{\text{seen}}$)** | Empirical | $[-0.0100, 0.0734]$ | $+0.0104$ ($[0.0062, 0.0163]$) | $[-0.0100, 0.0263]$ | $+0.0101$ ($[0.0024, 0.0146]$) | $43.6\%$* | 4 Low $G$ ($\le 0.0104$) / 4 High $G$ ($> 0.0104$) |
| **Reproducibility ($R$)** | Empirical | $[0.3342, 0.9448]$ | $0.7326$ ($[0.5984, 0.8460]$) | $[0.5384, 0.9143]$ | $0.7474$ ($[0.5892, 0.8417]$) | $61.6\%$ | 4 Low $R$ ($< 0.7326$) / 4 High $R$ ($\ge 0.7326$) |
| **Displacement ($D_{\theta}$)** | Empirical | $[0.1506, 0.5927]$ | $0.3844$ ($[0.3114, 0.4203]$) | $[0.1742, 0.4704]$ | $0.3802$ ($[0.3146, 0.3923]$) | $67.0\%$ | Spans $D_{\text{low}}$ ($0.17$) to $D_{\text{high}}$ ($0.47$) |
| **Restart Uncertainty ($U$)** | Empirical | $[0.0000, 0.3164]$ | $0.1656$ ($[0.0980, 0.2267]$) | $[0.0022, 0.3164]$ | $0.1680$ ($[0.1088, 0.2372]$) | $99.3\%$ | Spans deterministic ($0.002$) to non-unique ($0.316$) |
| **IC KGE ($K_{\text{IC}}$)** | Empirical | $[0.3897, 0.7331]$ | $0.6215$ ($[0.5646, 0.6530]$) | $[0.5617, 0.7331]$ | $0.6499$ ($[0.6315, 0.6845]$) | $49.9\%$ | Performance gate $K_{\text{joint}} \ge 0.561$ satisfied |
| **dPL KGE ($K_{\text{dPL}}$)** | Empirical | $[0.3982, 0.7572]$ | $0.6093$ ($[0.5633, 0.6393]$) | $[0.6004, 0.7572]$ | $0.6485$ ($[0.6059, 0.6794]$) | $43.7\%$ | Performance gate $K_{\text{joint}} \ge 0.561$ satisfied |
| **Signal Count ($A$)** | Empirical | $[0, 24]$ | $16.00$ ($[10.75, 20.00]$) | $[7, 21]$ | $16.50$ ($[8.00, 18.50]$) | $58.3\%$ | Low signal ($7, 8$) to high signal ($20, 21$) |

*\*Note on $G_{\text{seen}}$ range:* The unselected extreme $G > 0.0263$ region in the 36-model benchmark consists exclusively of degraded models (`vic`, `flexb`, `collie3`) that failed the baseline performance gate ($K_{\text{joint}} < 0.561$). Within the performance-viable model population, the 8-model subset covers $100\%$ of the viable $G$ spectrum.

---

## 3. Individual Profile of the Eight Tested Models

| Model | Quadrant & Role | $P$ | States | Snow | Routing | Runoff Architecture | $G_{\text{seen}}$ (pct) | $R_{\text{seen}}$ (pct) | $D_{\theta}$ (pct) | $U$ (pct) | $K_{\text{joint}}$ (pct) | $A$ (pct) | Conceptual Archetype |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **alpine2** | Q1: REP | 6 | 2 | Yes | Base | Simple alpine bucket + degree-day melt | $+0.0094$ (43%) | $0.8751$ (83%) | $0.2658$ (6%) | $0.1555$ (37%) | $0.6651$ (86%) | 8 (16%) | Parsimonious snow-dominated bucket |
| **hbv96** | Q1: CONTRAST | 15 | 5 | Yes | Endpoint | Multi-zone snow, soil moisture, dual non-linear reservoir | $-0.0099$ (0%) | $0.8252$ (69%) | $0.3884$ (51%) | $0.2258$ (74%) | $0.7331$ (100%) | 7 (11%) | High-dimensional classical benchmark |
| **newzealand2** | Q2: REP | 8 | 2 | No | Endpoint | Two-layer infiltration & drainage cascade | $-0.0095$ (3%) | $0.5384$ (14%) | $0.3720$ (43%) | $0.2714$ (83%) | $0.5617$ (26%) | 17 (60%) | Flexible conceptual layer model |
| **xinanjiang** | Q2: CONTRAST | 12 | 4 | No | Base | Tension water storage + parabolic saturation excess | $+0.0063$ (26%) | $0.6696$ (37%) | $0.3308$ (34%) | $0.1805$ (54%) | $0.6557$ (83%) | 21 (86%) | Variable contributing area saturation excess |
| **ihacres** | Q3: REP | 6 | 1 | No | Endpoint | Non-linear loss module + linear routing cascade | $+0.0108$ (51%) | $0.9143$ (89%) | $0.3886$ (54%) | $0.0764$ (23%) | $0.6320$ (71%) | 20 (76%) | Unit-hydrograph transfer function |
| **us1** | Q3: CONTRAST | 5 | 2 | No | Base | Parsimonious excess soil water store | $+0.0263$ (86%) | $0.8305$ (71%) | $0.1742$ (3%) | $0.0022$ (6%) | $0.6004$ (37%) | 16 (51%) | Low-complexity identifiable store |
| **hillslope** | Q4: REP | 7 | 2 | No | Endpoint | Topographic gradient-based hillslope routing | $+0.0217$ (83%) | $0.5739$ (20%) | $0.4032$ (57%) | $0.1196$ (31%) | $0.6039$ (43%) | 18 (66%) | Non-linear hillslope discharge |
| **mopex4** | Q4: CONTRAST | 10 | 5 | Yes | Base | SAC-SMA derived multi-storage formulation | $+0.0122$ (54%) | $0.5943$ (23%) | $0.4704$ (89%) | $0.3164$ (100%) | $0.7093$ (91%) | 8 (16%) | Intermediate complex conceptual store |

---

## 4. Evaluation of Coverage Properties

### 4.1. Structural Formulations Spanned
1. **Low to High Parameter Complexity:** Free parameters range from $P=5$ (`us1`) to $P=15$ (`hbv96`).
2. **State Dimensionality:** Internal states range from $S=1$ (`ihacres`) to $S=5$ (`hbv96`, `mopex4`).
3. **Snow Dynamics:** Includes snow-explicit formulations (`alpine2`, `hbv96`, `mopex4`) alongside snow-free/rainfall-dominant models (`xinanjiang`, `ihacres`, `us1`, `hillslope`, `newzealand2`).
4. **Hydrological Mechanisms:** Spans unit-hydrograph transfer functions (IHACRES), saturation-excess parabolic tension stores (Xin'anjiang), classical multi-reservoir cascades (HBV96, MOPEX4), and hillslope kinematic routing (Hillslope).

### 4.2. Response Regimes Spanned
1. **Fidelity under Shared Parameter Learning ($G_{\text{seen}}$):** Spans models where dPL matches or exceeds IC ($G \le 0$; `hbv96`, `newzealand2`) to models where IC maintains a distinct optimization edge ($G > 0.02$; `hillslope`, `us1`).
2. **Parameter Reproducibility ($R_{\text{seen}}$):** Spans models where parameter–attribute correlations are robustly retained across calibrations ($R > 0.85$; `ihacres`, `alpine2`) to models where parameter realizations are highly variable ($R < 0.60$; `newzealand2`, `hillslope`, `mopex4`).
3. **Parameter Equifinality / Non-Uniqueness ($U$):** Spans almost perfectly unique calibrations ($U=0.002$; `us1`) to highly non-unique multi-start landscapes ($U=0.316$; `mopex4`).

---

## 5. Explicit Limitations of the Subset

While the eight models span contrasting conceptual formulations and response regimes, several boundaries of the 36-model benchmark are intentionally or structurally unrepresented:
1. **Degraded / Sub-Viable Models Excluded:** Models with $K_{\text{joint}} < 0.561$ (e.g. `collie3`, `flexb`, `vic`) were excluded by design. Thus, R4 cannot evaluate OOB behavior in numerically failing or severely mis-specified models.
2. **Single-Parameter Toy Models Excluded:** Ultra-parsimonious models ($P < 5$, e.g. `australia` $P=1$, `gsfb` $P=2$) were not selected.
3. **Lineage Redundancy Pruned:** Only one member of the MOPEX family (`mopex4`) was retained; other members (`mopex1`, `mopex2`, `mopex3`, `mopex5`) were excluded to prevent family clustering.
4. **Non-Random Sampling:** Because the subset was deterministically selected to cover quadrant extremes and maximize 5D contrast, sample statistics computed across the eight models cannot be interpreted as unbiased population estimates of the 36-model benchmark.

---

## 6. Coverage Audit Verdict

- **Structural Coverage:** **ADEQUATE TO STRONG** (spans major process and complexity axes).
- **Seen-Basin Response Coverage:** **STRONG** (spans $2 \times 2$ orthogonal $G \times R$ grid and 5D contrast).
- **Statistical Representativeness:** **NOT SUPPORTED FOR UNBIASED INFERENCE**; the subset is an intentionally stratified, contrast-maximizing panel.
