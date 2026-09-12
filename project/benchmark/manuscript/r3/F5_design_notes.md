# Figure 5 (F5) Production Notes & Design Specifications

## 1. Scientific Objectives and Evidence Architecture
Figure 5 serves as the opening empirical figure of Manuscript R3, answering:
$$\boxed{\text{\textbf{How is catchment–parameter association structure reorganized from IC to dPL?}}}$$

The figure establishes a continuous, standard scientific narrative:
$$\boxed{\text{Continuous joint geometry (a)} \longrightarrow \text{Hydrological-information redistribution (b)} \longrightarrow \text{Cross-model overlap beyond independence (c)}}$$

---

## 2. Panel-by-Panel Architecture & Data Audit

### Panel (a): Continuous Cross-Paradigm Association Geometry
- **Aspect Ratio**: Exact square `set_aspect('equal')` over identical axis ranges $[-0.82, 0.82]$.
- **Reference Line Hierarchy**:
  - $x=0, y=0$: Thin light grey coordinate lines (`#cbd5e1`, lw=0.55).
  - $y=x$: Dark dashed identity line (`#334155`, lw=0.90).
  - $x = \pm 0.20, y = \pm 0.20$: Thin dotted intensity threshold grid lines (`#94a3b8`, lw=0.50).
- **Geometrically Precise 5-Class Annotations**:
  - Center ($|x|<0.20, |y|<0.20$): `Neither strong\n3,067`
  - Top-middle ($|x|<0.20, y\ge 0.20$): `dPL-only\n1,451`
  - Right-middle ($x\ge 0.20, |y|<0.20$): `IC-only\n190`
  - Top-right ($x\ge 0.20, y\ge 0.20$): `Both strong\n(same sign)\n692`
  - Top-left ($x\le -0.20, y\ge 0.20$): `Both strong\n(sign-flip)\n20 (2.8%)`
- **Marginal KDEs**:
  - Top marginal ($\rho_{\mathrm{IC}}$): Green solid curve + light green fill, overlaid with purple dashed curve ($\rho_{\mathrm{dPL}}$).
  - Right marginal ($\rho_{\mathrm{dPL}}$): Purple solid curve + light purple fill, overlaid with green dashed curve ($\rho_{\mathrm{IC}}$).
  - Shared vertical/horizontal scale limit ($d_{\max} = 3.3554$).
- **Colorbar**: Horizontal mini-scale bar placed outside underneath Panel (a).

### Panel (b): Redistribution of Association Strength across Catchment-Information Dimensions (HERO)
- **Status**: Strictly preserved code, primitives, colors, labels, and domain grouping.
- **Graphic Type**: 20-row ridgeline density plot of $\Delta|\rho| = |\rho_{\mathrm{dPL}}| - |\rho_{\mathrm{IC}}|$.
- **Right-Side Alignment & Alternating Bands**:
  - 20 dimension labels aligned along the right side of the curves.
  - Alternating subtle light-grey (`#f4f6f8`) and white (`#ffffff`) background bands for the 5 physical domains.
  - Group labels positioned on the right margin of each band: `Climate & Snow (5)`, `Soil & Terrain (5)`, `Vegetation (4)`, `Geology (5)`, `Topography (1)`.
- **Continuous Progressive Diverging Colormap**:
  - Continuous gradient: Green ($\Delta|\rho| < 0$) $\to$ Light neutral grey near zero $\to$ Purple ($\Delta|\rho| > 0$).
  - Deep grey outline curve (`#1e293b`).
  - Median indicator: Solid dark circle for each dimension.

### Panel (c): Strong-Association Overlap Across Conceptual Models
- **Graphic Type**: 36-model Empirical Cumulative Distribution Function (ECDF) with individual model rug ticks.
- **X-axis Extension & Baseline**:
  - Extended range: $[0.80, 3.05]$.
  - Sub-independence region $[0.80, 1.0)$ shaded in light neutral grey (`#f1f5f9`), highlighting 0 models with $E_m < 1.0$.
  - Independence baseline: Solid vertical line at $x=1.0$ (`#334155`).
  - Median reference: Dotted line and marker at $E_m = 2.00\times$.
- **Model Annotations**:
  - Lowest model: `flexb (1.24×)`.
  - Degenerate single-parameter model: `collie1† (2.86×)`.

---

## 3. Visual Balance & Space Allocation
- **Master Column Gap**: Minimized horizontal spacing between Left Column (a/c) and Right Column (b) (`wspace = 0.11`).
- **Typography**: Serif (`Times New Roman` / `STIXGeneral`), publication grade.
- **Export Formats**:
  - PNG: 600 DPI ($7440 \times 4680$ px), white background.
  - PDF: Vector PDF output.
- **File Locations**:
  - Script: `project/benchmark/manuscript/r3/scripts/plot_F5.py`
  - Figures: `project/benchmark/manuscript/r3/figures/F5.png` and `F5.pdf`
  - Main sync: `project/benchmark/manuscript/r3/F5_main.png` and `F5_main.pdf`
  - Caption: `project/benchmark/manuscript/r3/F5_caption.md`
  - Design Notes: `project/benchmark/manuscript/r3/F5_design_notes.md`
