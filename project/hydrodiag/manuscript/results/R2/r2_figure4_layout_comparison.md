# Figure 4 Candidate Layout Exploration & Comparison Report

## Executive Summary

This report presents the design, visual evaluation, and comparative analysis of candidate layouts for **Figure 4 (F4)**. Figure 4 serves as the orthogonal parameter-layer space occupation and organization figure, answering:

> **How do Base and CN occupy the shared 15D parameter space, and what distinct parameter-space organization signatures do IC and dPL leave behind?**

All candidate layouts strictly reuse the frozen R2 datasets across 531 basins, follow HESS/Copernicus publication standards, inherit visual conventions from Figures 1–3, and comply with the **PNG-only output constraint** (no PDF outputs generated).

---

## Candidate Layouts Overview

| Candidate | Layout Grid | Panel (a) | Panel (b) | Panel (c) | Dimensions | Visual Style / Feel |
|---|---|---|---|---|---|---|
| **Candidate A** *(Actual Parameter Occupation Triptych)* | 1×3 Aligned Triptych (1.25 : 1 : 1) | Global Base−CN Paired Shifts ($\Delta z = z_{\text{Base}} - z_{\text{CN}}$) + 95% CI | IC Base vs CN Actual Occupation ($z \in [0,1]$, Median + IQR + 5–95%) | dPL Base vs CN Actual Occupation ($z \in [0,1]$, Median + IQR + 5–95%) | 11.0" × 7.5" | **Hydrological Result Feel**: Directly displays the parameter distributions occupied by Base and CN. |
| **Candidate B** *(Derived Diagnostics Triptych)* | 1×3 Aligned Triptych (1.25 : 1 : 1) | Global Base−CN Paired Shifts ($\Delta z = z_{\text{Base}} - z_{\text{CN}}$) + 95% CI | Boundary Concentration Rate Change ($\Delta \text{Boundary}$) at $\epsilon = 0.01$ | Dispersion / IQR Change ($\Delta \text{IQR}$) | 11.0" × 7.5" | **Diagnostic Dashboard Feel**: Displays derived statistical summary metrics ($\Delta \text{Boundary}$, $\Delta \text{IQR}$). |

---

## Detailed Visual & Scientific Evaluation

### 1. Candidate A — Actual Parameter Occupation Triptych (`Figure4_layout_A_occupation_triptych.png`)
- **Information Value Add**: Outstanding. Panels `(b)` and `(c)` show the *actual parameter ranges* ($z \in [0,1]$) occupied across 531 basins.
- **Clarifying IC Multi-modality**: Solves a critical potential misinterpretation: under IC, several parameters (e.g. $b, c, k$) have a signed median shift near 0 ($\Delta z pprox 0$). Panel `(b)` shows that Base and CN parameters under IC are heavily concentrated at physical boundaries ($z=0$ or $z=1$), making signed median shift near 0 a symptom of boundary saturation rather than lack of parameter movement.
- **Clarifying dPL Re-centering**: Panel `(c)` shows dPL's continuous parameter distributions in the interior of the physical space ($z \in (0,1)$), demonstrating clear global re-centering between Base and CN.
- **Hydrological Result vs Dashboard**: Has a strong hydrological research feel by presenting actual parameter space occupation rather than abstract error metrics.

### 2. Candidate B — Derived Organization Diagnostics Triptych (`Figure4_layout_B_diagnostics_triptych.png`)
- **Information Value Add**: Useful as a summary diagnostic. Panel `(b)` shows boundary concentration rate changes ($\Delta 	ext{Boundary}$) and Panel `(c)` shows dispersion changes ($\Delta 	ext{IQR}$).
- **Limitations**: Panels `(b)` and `(c)` display abstract derived delta metrics rather than actual parameter values, giving the figure a statistical "diagnostic dashboard" feel rather than a physical hydrological result.

---

## Preliminary Recommendation for Human Review

**Preliminary Recommendation: Candidate A**

### Key Reasons:
1. **Physical Intuition**: Candidate A directly shows the actual parameter distributions ($z \in [0,1]$), making it intuitive for hydrological readers to see where parameters land.
2. **Explaining IC vs dPL**: Clearly reveals the structural difference between IC's boundary-concentrated occupation and dPL's interior continuous occupation.
3. **Non-Redundant with Figure 3**: Figure 3 covers the snow gradient ($f_{	ext{snow}}$), excess separation ($excess$), and core parameter shifts ($u_m, k_i, c_i$). Candidate A covers the overall 15D parameter space occupation across all 531 basins without repeating snow gradient plots.

*Final selection is left for human review via the generated review package.*
