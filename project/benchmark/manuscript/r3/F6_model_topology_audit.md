# F6 Representative Model Topology, Provenance, and Publication-Quality Audit

**Manuscript:** Journal of Hydrology R3 Revision  
**Figure:** Figure 6 (F6) — Parameter-level Specificity in Conceptual Hydrological Models  
**Execution Environment:** Python 3.10 (.venv) + Matplotlib 3.10.0 (Unified Pixel Canvas Architecture $1448 \times 1086$ px, Scale=1.0, 600 DPI)  
**Output Figures:**
- `project/benchmark/manuscript/r3/figures/F6_coordinate_specificity_main.png` (600 DPI, PNG, Publication Final)
- `project/benchmark/manuscript/r3/figures/F6.png` (600 DPI, PNG)
**Plot Script:** `project/benchmark/manuscript/r3/scripts/plot_F6.py`

---

## 1. Scientific Objective and Visual Structure

### 1.1 Scientific Question
> **In cross-calibration paradigm comparisons, is parameter–information organization preferentially retained on the same specific model parameter coordinate rather than randomly dispersed across alternative parameter coordinates?**

### 1.2 Core Metric
The parameter-level specificity advantage is quantified by:
$$
\mathit{adv}_{m,p} = C_{m,p,p} - \operatorname{median}_{q \neq p} C_{m,p,q}
$$
where $C_{m,p,q}$ is the cross-method Spearman profile correspondence between the Information Cluster (IC) profile of parameter $p$ and the differentiable Parameter Learning (dPL) profile of parameter $q$.

### 1.3 Figure Layout (3-Panel Architecture)
- **Left Column: (b) Ensemble Same-Coordinate Specificity (Hero Panel, $n = 35$ models)**
  - Base canvas coordinates: `(3, 3, 776, 1083)`
  - Grouping columns on left: `Store complexity` ($1\mathrm{S} \to 6\mathrm{S}$ with brackets and $(n=\dots)$ counts) and `Model` names.
  - Model names right-aligned to axis line with clean padding, avoiding any text overlap.
  - Full lollipop data axis `px_axes(223, 95, 752, 990)` displaying true $A_m$ values.
  - Ensemble median line ($\mathrm{Median}\ A_m = 0.615$) and IQR shading ($[0.451, 0.764]$).
  - Permutation null test inset moved up to $y \in [680, 850]$ px (level of `hbv96` and $5\mathrm{S}$ stratum) and enlarged to $195 \times 170$ px for clear readability ($A_{\mathrm{diag}} = 0.615, p < 0.001$).
- **Right Top: (a) Parameter-Level Specificity in Representative Model Structures**
  - Base canvas coordinates: `(785, 3, 1445, 723)`
  - 5 representative models stacked vertically as 5 horizontal water-flow strips (`IHACRES`, `TOPMODEL`, `VIC`, `Xinanjiang`, `HBV96`).
  - Straight/orthogonal arrows only; white process boxes with dark borders; 5-level teal specificity markers.
  - Bottom legend: `Higher marker fill = larger parameter-level specificity advantage`.
- **Right Bottom: (c) Top-$k$ Rank Consequence**
  - Base canvas coordinates: `(785, 733, 1445, 1083)`
  - 3 rows (Top 1, Top 2, Top 3) evaluated on **270 eligible parameters**:
    - Top 1: Random $13.0\%$, Observed **$57.4\%$ (155/270)** ($4.42\times$ enrichment)
    - Top 2: Random $25.9\%$, Observed **$71.1\%$ (192/270)** ($2.75\times$ enrichment)
    - Top 3: Random $38.9\%$, Observed **$84.1\%$ (227/270)** ($2.16\times$ enrichment)

---

## 2. Canonical Model Data Binding & Stratum Counts

| Stratum | Model Count | Models (in ascending $A_m$ order within stratum) |
|:---:|:---:|---|
| **1S** | 4 | `ihacres` (0.742), `newzealand1` (0.959), `wetland` (1.246), `collie2` (1.411) |
| **2S** | 12 | `susannah1` (0.389), `gr4j` (0.422), `collie3` (0.438), `newzealand2` (0.447), `plateau` (0.456), `topmodel` (0.467), `hillslope` (0.531), `susannah2` (0.615), `simhyd` (0.620), `us1` (0.744), `alpine1` (0.802), `alpine2` (0.914) |
| **3S** | 5 | `australia` (0.255), `gsfb` (0.277), `flexb` (0.418), `vic` (0.538), `penman` (0.644) |
| **4S** | 5 | `tank` (0.349), `mopex1` (0.497), `xinanjiang` (0.632), `flexi` (0.763), `tcm` (0.909) |
| **5S** | 8 | `mopex3` (0.454), `mopex4` (0.538), `mopex5` (0.606), `hbv96` (0.747), `modhydrolog` (0.759), `flexis` (0.765), `hymod` (0.848), `mopex2` (0.960) |
| **6S** | 1 | `smar` (0.222) |

---

## 3. Readability & Publication Quality Audit Checklist

- [x] **Panel (b) Hero status affirmed:** Occupies ~53% width, full height of figure.
- [x] **Panel (b) y-axis label alignment:** Right-aligned at $x=212$ with uniform 11 px padding to axis line ($x=223$), zero text-line collision.
- [x] **Permutation null test inset repositioned & enlarged:** Moved up to $y \in [680, 850]$ px alongside `hbv96` and enlarged by $>2.5\times$ in area.
- [x] **Panel (a) 5 models stacked vertically in 1 column:** IHACRES, TOPMODEL, VIC, Xinanjiang, HBV96.
- [x] **Panel (c) evaluated on 270 eligible parameters:** Top-1 (57.4%), Top-2 (71.1%), Top-3 (84.1%).
- [x] **All data 100% verified against canonical R3 tables.**
- [x] **High-resolution 600 DPI PNG exported.**

```text
F6 DRAFT COMPLETE
```
