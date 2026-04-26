You are an expert scientific visualization engineer helping prepare all main figures for a hydrology manuscript targeted at HESS / Journal of Hydrology / WRR style, with HESS-like visual restraint, high information density, and publication-level polish.

Your job is to batch-generate all main figures for the manuscript from the available project data, using a unified, elegant, information-rich visual system. Do not produce simplistic line charts, bar charts, or generic boxplots as the dominant figure type. The figures must look like a coherent paper-level figure suite, not a collection of unrelated plots.

The paper is NOT an accuracy benchmark paper. The manuscript’s central storyline is:

1. The main goal is NOT to show one model has universally better predictive accuracy.
2. The core question is which parameter-learning formulation better recovers basin attribute–parameter relationships.
3. Distributional formulation is strongest in relationship reliability.
4. The three formulations share a clear dominant-control core rather than telling three unrelated stories.
5. Distributional does not depart from the shared hydrologic rule set; instead it recovers more stable and clearer gradients in parameter means and parameter uncertainty.
6. Therefore, distributional is more suitable for interpretable parameter learning, regionalization reasoning, and ungauged hydrology discussion.

You must preserve this narrative. Never turn the figure suite into “distributional wins everything” or “best accuracy model” storytelling.

==================================================
A. NON-NEGOTIABLE MANUSCRIPT LOGIC
==================================================

Respect the following writing/figure logic:

- Results only answer: what is the evidence?
- Discussion answers: what does it mean?
- Section 3.2 proves reliability.
- Section 3.3 summarizes shared structure.
- Section 3.4 reveals gradients and hydrologic meaning.
- Do not let Figures for 3.3 or 3.4 drift back into merely repeating “distributional is more stable.”
- Post-hoc explainability is only supporting evidence, not the main evidence.
- Parameter uncertainty gradients may be described as structured/organized, but not all patterns should be overstated as pure identifiability signals.
- Snow/routing-sensitive parameters should be visually treated as more model-sensitive and potentially compensation-driven.

==================================================
B. OUTPUT DELIVERABLES
==================================================

Create a reproducible figure-generation pipeline with the following outputs:

1. A project-level figure configuration file:
   - config/figure_style.yaml
   - config/figure_palette.yaml

2. A reusable visualization utility module:
   - src/figure_utils.py

3. One script per main figure:
   - scripts/make_fig01_performance_merged.py
   - scripts/make_fig02_cross_seed_param_stability.py
   - scripts/make_fig03_cross_seed_corr_stability.py
   - scripts/make_fig04_cross_loss_corr_stability.py
   - scripts/make_fig05_shared_dominant_core.py
   - scripts/make_fig06_matrix_similarity.py
   - scripts/make_fig07_explainability_support.py
   - scripts/make_fig08_parameter_mean_gradients.py
   - scripts/make_fig09_representative_mean_gradients.py
   - scripts/make_fig10_parameter_uncertainty_gradients.py
   - scripts/make_fig11_conceptual_synthesis.py

4. A batch runner:
   - scripts/make_all_figures.py

5. Output files:
   - figures/main/Fig01_performance_merged.pdf
   - figures/main/Fig01_performance_merged.png
   ...
   - figures/main/Fig11_conceptual_synthesis.pdf
   - figures/main/Fig11_conceptual_synthesis.png

6. A QC report:
   - figures/main/qc_report.md
   This report must include:
   - figure dimensions
   - file sizes
   - min/max font size detected
   - whether any labels overlap
   - whether any colorbars are clipped
   - whether the figure passes grayscale legibility checks
   - whether all fonts are embedded in PDF outputs if possible
   - whether each figure stayed within the target file size budget

If any required data for a figure are missing, do NOT fabricate. Instead:
- fail gracefully for that figure,
- write a structured TODO entry in qc_report.md,
- list exactly which data objects / columns / files are missing,
- continue generating the other figures.

==================================================
C. DATA DISCOVERY RULES
==================================================

Before plotting, inspect the project to discover existing data assets. Search for likely inputs such as:

- basin-level performance metrics by model
- paired basin-wise performance differences
- seed-wise inferred parameter values
- seed-wise attribute–parameter correlation matrices
- loss-wise attribute–parameter correlation matrices
- basin attributes
- parameter means
- parameter standard deviations / uncertainty terms
- explainability outputs (e.g., SHAP / permutation importance / integrated gradients summaries)
- basin geometry / shapefile / lat-lon tables
- model metadata
- parameter group metadata
- attribute group metadata

Expected model names:
- deterministic
- mc_dropout
- distributional

Do not rename these arbitrarily in the source data. Internally, you may display them as:
- Deterministic
- MC dropout
- Distributional

Expected conceptual variable families:
- storage/recharge-related parameters
- routing parameters
- snow/cold-region parameters
- runoff nonlinearity / threshold / partition parameters

If parameter family metadata are absent, infer them conservatively from parameter names only when obvious; otherwise mark them as “unclassified”.

==================================================
D. GLOBAL FIGURE STYLE SYSTEM
==================================================

Important:
There is no official public HESS color palette that you need to reproduce literally. Instead, use a HESS-oriented hydrology palette:
- restrained
- publication-safe
- colorblind-friendly
- mature and non-flashy
- consistent across all figures

Use the following visual grammar consistently.

------------------------------------------
D1. Typography
------------------------------------------

Use a clean sans-serif font consistently across all figures:
Preferred:
- Arial
Fallback:
- Helvetica
- DejaVu Sans

Do NOT mix font families across figures.

Target final font sizes at publication scale:
- panel letters: 10.5 pt bold
- figure title inside figure: avoid unless absolutely needed
- axis labels: 8.5–9.0 pt
- tick labels: 7.5–8.0 pt
- legend labels: 7.5–8.0 pt
- subplot annotations: 7.0–8.0 pt
- colorbar title: 8.0 pt
- colorbar ticks: 7.0–7.5 pt
- network node labels: minimum 7.0 pt after final export
- conceptual diagram labels: 7.5–8.5 pt

Never allow any text smaller than 6.8 pt at final exported size.
Never let titles/labels become oversized and visually clumsy.
Use sentence case, not title case screaming.

------------------------------------------
D2. Figure sizes and export
------------------------------------------

Create figures in final-use dimensions, not oversized then shrunk carelessly.

Preferred widths:
- single-column candidate: 85–90 mm
- 1.5-column / intermediate composite: 120–140 mm
- double-column main figure: 175–180 mm

Default for most main composite figures:
- width = 178 mm
- height adjusted by content, generally 120–220 mm

Export both:
- vector PDF for manuscript use
- 600 dpi PNG for quick review / slide preview

Keep file size efficient:
- prefer vector elements whenever possible
- rasterize only very dense layers (e.g., huge scatter/hexbin backgrounds) if needed
- avoid bloated PDFs
- target each figure to remain visually sharp and ideally below 5 MB if practical

------------------------------------------
D3. Core HESS-oriented palette
------------------------------------------

Use this palette consistently.

Model identity colors:
- Deterministic: #355C7D   (deep muted blue)
- MC dropout:    #2A9D8F   (teal-green)
- Distributional:#C06C3E   (muted burnt orange)

Neutral/support colors:
- Dark text:     #1F2430
- Mid gray:      #6E7783
- Light gray:    #D9DEE5
- Very light bg: #F6F8FA
- Panel divider: #C9D1D9

Sequential palette for magnitude/stability:
- light to dark blue-gray:
  ["#F7FBFF", "#DCEAF4", "#AFCBE0", "#6E9FC2", "#355C7D", "#20364F"]

Sequential palette for organized uncertainty:
- light to dark teal:
  ["#F2FBF9", "#CDEEE8", "#8FD3C7", "#4FB5A5", "#2A9D8F", "#1E6F67"]

Diverging palette for signed correlations / signed differences:
- negative = muted blue-purple
- neutral = near-white
- positive = muted earth-red
Use:
["#4C6A92", "#8FA8C9", "#EDEFF2", "#D9A58F", "#B5654A"]

Important:
- no rainbow colormap
- no jet
- no saturated neon colors
- no red-green opposition as the sole encoding
- no pure black heavy fills except text

------------------------------------------
D4. Line widths, grids, frames
------------------------------------------

Use restrained line widths:
- axis spine: 0.6–0.8 pt
- major tick line: 0.6 pt
- minor tick line: 0.4 pt
- network edges: 0.6–2.2 pt depending on effect size
- map boundary lines: 0.4–0.8 pt
- annotation connector lines: 0.5–0.8 pt

Gridlines:
- use only when they improve reading
- very light gray
- thin
- never dominate the data

Spines:
- remove top/right spines for standard charts unless panel logic requires framing
- for heatmaps/matrices, use subtle framing

------------------------------------------
D5. Legends and colorbars
------------------------------------------

Legends must be compact, aligned, and ordered logically:
- deterministic
- MC dropout
- distributional

Avoid giant floating legends.
Prefer direct labeling when clean.

Colorbars:
- consistent width
- consistent tick density
- clear titles
- no crowding
- align with main matrix/map panels

------------------------------------------
D6. Layout discipline
------------------------------------------

All figures must look like one paper.

Rules:
- consistent outer margins
- balanced whitespace
- aligned panel edges
- aligned colorbars
- consistent panel letter positions (top-left inside or just outside)
- consistent spacing between panels
- do not create a patchwork of tiny unreadable subplots
- use 3–5 strong panels rather than 8–12 weak micro-panels unless absolutely necessary

Preferred panel letter style:
- bold uppercase A, B, C...
- top-left
- same offset in all figures

------------------------------------------
D7. Hydrology-specific visual tone
------------------------------------------

The figures should feel:
- quantitative
- hydrologically grounded
- mature
- non-gimmicky
- readable under peer review
- attractive without being decorative

Avoid:
- infographic-style icons everywhere
- excessive gradients for decoration only
- pseudo-3D effects
- glossy effects
- crowded annotation arrows

==================================================
E. GLOBAL DO-NOT-DO LIST
==================================================

Do NOT make the figures primarily as:
- simple line plots
- basic bar charts
- generic boxplots as the main visual
- repeated small multiples that say the same thing
- huge scatter clouds without density summarization
- visually loud benchmark-comparison charts

Instead prefer:
- clustered heatmaps
- signed matrix views
- uncertainty/stability masks
- spatial maps
- network / bipartite graphs
- density fields / hexbin / smoothed response surfaces
- compact representative archetype panels
- similarity heatmaps + embedding views + block decomposition

==================================================
F. REQUIRED FIGURE SUITE
==================================================

There are 11 main figures after merging the original performance overview and basin-wise performance difference figure into one composite figure.

--------------------------------------------------
FIGURE 01. PERFORMANCE MERGED
Purpose:
- Provide a brief sanity check only.
- Show that all three models are viable.
- Show that deterministic is somewhat steadier on structural metrics.
- Show that MC dropout may have some pointwise error advantages.
- Show that distributional is NOT being sold on accuracy.
- Do not let this figure dominate the paper.

Design:
Create one high-information composite figure with 4 panels.

Panel A: model × metric performance matrix
- Rows: evaluation metrics
- Columns: model
- In each cell encode:
  - median basin performance
  - IQR or spread
  - relative rank
- Use a compact annotated heatmap, not bars

Panel B: basin-wise paired performance difference maps
- Show spatial pattern of:
  - distributional - deterministic
  - distributional - MC dropout
- Prefer two aligned CONUS basin maps or basin centroid maps
- Use the same diverging color scale across both maps
- Include subtle state or region outlines only if they improve orientation
- Do not use visually busy basemaps

Panel C: environmental ordering carpet / difference strip matrix
- Order basins by one or two key environmental gradients (e.g., aridity, snow fraction, elevation)
- Show paired differences across metrics as a compact heatmap/carpet
- This panel should reveal structured heterogeneity instead of only average wins/losses

Panel D: ternary/simplex-like dominance summary or compact dominance matrix
- Summarize where each basin’s best relative model lies across multiple metrics
- Avoid gimmicky ternary styling if unreadable; fallback to a highly polished categorical summary tile matrix

Visual emphasis:
- Keep this figure controlled and concise
- It must read as sanity check, not central evidence

--------------------------------------------------
FIGURE 02. CROSS-SEED STABILITY OF INFERRED PARAMETER VALUES
Purpose:
- Show reproducibility of inferred parameter values across random seeds.
- This is evidence for parameter-field stability, not yet the final relationship reliability claim.

Design:
4 panels maximum.

Panel A:
- parameter × model stability heatmap
- metric could be normalized seed variability, ICC-like score, or robust CV
- cluster parameters by family
- annotate parameter family side bars

Panel B:
- basin-ordered parameter stability carpet
- select a manageable subset of key parameters or use family summaries
- basins ordered by environmentally meaningful axis
- show whether instability concentrates in specific hydrologic settings

Panel C:
- compact density/ridge summary for 4–6 representative parameters
- compare seed distributions across models
- keep this compact and elegant

Panel D:
- stability class summary:
  - stable core
  - intermediate
  - model-sensitive / unstable
- use compact tiles or dot matrix, not bars

Important:
- do not overclaim that stable parameter values are automatically more physically correct

--------------------------------------------------
FIGURE 03. CROSS-SEED STABILITY OF ATTRIBUTE–PARAMETER CORRELATIONS
Purpose:
- This is one of the core figures of the paper.
- Show cross-seed reliability of inferred attribute–parameter relationships.
- Establish that distributional provides more coherent and reliable relationship recovery.

Design:
This should be a strong matrix-based composite figure.

Panel A:
- mean signed correlation matrix for Deterministic

Panel B:
- mean signed correlation matrix for MC dropout

Panel C:
- mean signed correlation matrix for Distributional

All three panels must:
- share identical row/column ordering
- use identical diverging scale
- group rows by basin attribute family if possible
- group columns by parameter family if possible

Panel D:
- cross-seed variability / instability matrix
- preferably one panel comparing all three models side-by-side or stacked compactly
- use sequential palette distinct from signed-correlation palette

Panel E:
- reliability mask / consensus mask
- show cells with:
  - stable sign
  - sufficient magnitude
  - acceptable cross-seed variability
- visually emphasize robust cells

Panel F:
- compact summary of top robust relationships
- show sign, effect strength, and reliability class
- this can be a tile summary rather than a ranked bar list

Critical message:
- distributional should appear more coherent at matrix level
- but never phrase or visualize it as universally superior in every sense

--------------------------------------------------
FIGURE 04. CROSS-LOSS STABILITY OF ATTRIBUTE–PARAMETER CORRELATIONS
Purpose:
- Extend reliability from seed robustness to loss robustness.
- Show which relationships survive changes in loss formulation.

Design:
4 panels.

Panel A:
- compact stacked matrix summaries by loss for each model, or vice versa
- preserve consistent ordering

Panel B:
- cross-loss variability heatmap
- identify sensitive vs invariant relationship regions

Panel C:
- robust relationship mask under both seed and loss perturbation
- emphasize relationships that are stable under both sources of variation

Panel D:
- 2D classification matrix:
  - stable across seeds only
  - stable across losses only
  - stable across both
  - sensitive to both
- present as elegant tile classes, not pie charts

Important:
- this figure should make the “reliability” claim stronger and more reviewer-proof
- it should not look redundant with Figure 03; differentiate the visual encoding clearly

--------------------------------------------------
FIGURE 05. SHARED DOMINANT-CONTROL CORE ACROSS MODELS
Purpose:
- Show that the three models share a dominant hydrologic control core.
- Distinguish shared relationships from model-sensitive relationships.
- Support the claim that distributional strengthens shared structure rather than departing from it.

Design:
Main panel must be a highly polished bipartite or two-column consensus network.

Panel A:
- bipartite graph
- left: basin attributes
- right: parameters
- edge width = relationship magnitude
- edge color = sign
- edge style / opacity = consensus class
- group nodes by hydrologic family

Consensus classes:
- shared-stable across all three models
- shared but weaker
- model-sensitive
- formulation-enhanced but still aligned with shared core

Panel B:
- compact class summary tile matrix or upset-style summary
- show counts or dominant examples of each relationship class

Panel C:
- optional mini callouts for 3–5 exemplary relations with hydrologic labels

Styling:
- must be elegant, sparse, readable
- avoid spaghetti-network appearance
- threshold edges intelligently
- use node alignment and grouping to reduce clutter

--------------------------------------------------
FIGURE 06. MATRIX-LEVEL SIMILARITY OF RELATIONSHIP STRUCTURES
Purpose:
- Show that distributional does not deviate from the shared hydrologic rule set.
- Show similarity structure across models, seeds, and losses.

Design:
4 panels.

Panel A:
- similarity heatmap among all inferred correlation matrices
- entries may represent matrix correlation, Mantel-like similarity, Frobenius-based transformed similarity, etc.

Panel B:
- 2D embedding (MDS / UMAP / PCoA / t-SNE only if justified)
- each point = one matrix instance
- color = model
- shape = seed or loss grouping
- confidence ellipse or hull optional if clean

Panel C:
- block summary:
  - within-model similarity
  - cross-model similarity
  - within-loss similarity
  - cross-loss similarity
- use compact annotated tiles, not bars if possible

Panel D:
- subsystem-level decomposition
- compare similarity separately for:
  - storage/recharge subset
  - snow/routing subset
  - runoff nonlinearity subset
- use aligned mini heatmaps or compact tiles

Important:
- this figure should visually support “shared rule set + strengthened coherence”
- not “distributional invents a new physics”

--------------------------------------------------
FIGURE 07. POST-HOC EXPLAINABILITY AS SUPPORTING EVIDENCE
Purpose:
- Only supporting evidence.
- Show that post-hoc explainability broadly aligns with the main relationship evidence.
- This figure must not overshadow the correlation-based evidence.

Design:
3 or 4 panels maximum.

Panel A:
- agreement matrix between explainability ranking and correlation-derived relationship ranking/signature

Panel B:
- overlap structure of top relationships
- use compact upset-style or overlap tile summary
- avoid Venn diagrams unless exceptionally clean

Panel C:
- representative parameter fingerprints
- small, elegant fingerprints for 3–5 parameters
- could be compact tiles or radial alternatives only if publication-clean
- no giant SHAP beeswarm as main panel

Panel D:
- agreement by relation class
- e.g. shared-stable vs model-sensitive
- use tile/dot encoding, not bars if possible

Important:
- this figure is subordinate
- visually lighter than Figures 03–06

--------------------------------------------------
FIGURE 08. LARGE-SCALE GRADIENTS IN PARAMETER MEANS
Purpose:
- Start the transition from reliability to hydrologic gradients.
- Show what distributional learns about basin-scale organization in parameter means.

Design:
This should be one of the major visual anchors of the paper.

Panel A:
- attribute-bin × parameter heatmap of parameter means
- order attributes and parameters meaningfully
- use family clustering and side annotations
- use a sequential palette distinct from uncertainty

Panel B:
- parameter-family structure strip / annotation band
- clearly identify storage/recharge, routing, snow/cold-region, runoff nonlinearity families

Panel C:
- 3–4 mini spatial maps of key parameter means across basins
- select only the most interpretable parameters
- maps must align stylistically with Figure 01 maps

Panel D:
- gradient sign summary or compact directional matrix
- indicate whether parameter means increase/decrease along key environmental axes

Important:
- emphasize large-scale organized gradients, not noisy local scatter

--------------------------------------------------
FIGURE 09. REPRESENTATIVE MEAN GRADIENTS AND BASIN ARCHETYPES
Purpose:
- Make the gradients interpretable and hydrologically concrete.
- Connect mean gradients to representative basin groups/case examples.

Design:
4 panels.

Panel A:
- 2D density or response field for aridity vs selected parameter mean

Panel B:
- 2D density or response field for snow fraction / coldness vs selected parameter mean

Panel C:
- 2D density or response field for topography / relief / landscape control vs selected parameter mean

Panel D:
- basin archetype panel
- show 4–6 representative basin groups with:
  - small map locator
  - attribute profile strip
  - parameter fingerprint strip
- keep archetype cards compact and scientifically clean

Avoid:
- ordinary scatterplots with thousands of opaque points
Use:
- hexbin
- density contours
- smoothed conditional surfaces
- quantile ribbons only if truly needed

--------------------------------------------------
FIGURE 10. LARGE-SCALE GRADIENTS IN PARAMETER UNCERTAINTY
Purpose:
- Show that parameter uncertainty is also environmentally structured.
- Keep an explicit caution that not all patterns should be overinterpreted as pure identifiability signals.

Design:
4 panels.

Panel A:
- attribute-bin × parameter heatmap of parameter uncertainty / std / spread

Panel B:
- caution mask overlay or annotation band
- identify regions/parameters potentially affected by:
  - mean–std coupling
  - boundary effects
  - weaker identifiability
  - model-sensitive process compensation

Panel C:
- joint organization view of mean vs uncertainty
- could be a paired tile matrix or compact bivariate structural summary

Panel D:
- mini spatial maps of key uncertainty patterns

Important:
- use language-neutral visual design:
  do not visually imply that all high uncertainty is bad
- show structure, not alarm

--------------------------------------------------
FIGURE 11. CONCEPTUAL SYNTHESIS
Purpose:
- Discussion-level synthesis figure.
- Integrate the full paper logic:
  reliability -> shared dominant-control core -> structured mean gradients -> structured uncertainty gradients -> interpretable regionalization / ungauged reasoning

Design:
This is a conceptual figure, but it must still look quantitative and manuscript-ready, not cartoonish.

Suggested structure:
Layer 1:
- three parameter-learning formulations
- predictive performance as sanity check only

Layer 2:
- relationship reliability
- shared hydrologic rule set / dominant-control core
- clearer parameter mean gradients
- structured parameter uncertainty

Layer 3:
- interpretable parameter learning
- regionalization reasoning
- ungauged hydrology implications

Embed small quantitative motifs:
- matrix thumbnail
- network thumbnail
- gradient heatmap thumbnail
- uncertainty thumbnail

Styling:
- use clean arrows / flow structure
- avoid clip-art
- avoid decorative icons unless minimalist and internally consistent

==================================================
G. AXIS, LABEL, AND PANEL PROFESSIONAL STANDARDS
==================================================

Apply the following rules to every figure.

1. Axis labels
- concise
- scientific
- no verbose sentences
- units included where appropriate
- same capitalization style across all figures

2. Tick labels
- never too dense
- never rotated unless necessary
- if rotated, use consistent angle and alignment
- preserve readability at final manuscript size

3. Heatmaps
- preserve square-ish cells when useful
- do not force impossibly tiny labels
- group separators should be subtle
- use side annotations for families rather than text overload inside cells

4. Maps
- minimalist map style
- no flashy basemap
- soft boundaries
- consistent extent across comparable panels
- consistent projection
- use same color scale across comparable map panels

5. Networks
- no edge pile-up
- threshold and route edges to avoid clutter
- prioritize readability over exhaustive display
- annotate only the most important nodes and relations

6. Density panels
- use enough bins / smoothing for stable perception
- avoid over-smoothing away structure
- include contour or quantile cues only if they add interpretation

7. White space
- leave enough room between panels
- no cramped colorbars
- no legend collisions
- no text touching panel borders

==================================================
H. QUALITY CONTROL BEFORE SAVING
==================================================

Before finalizing each figure, automatically check:

- minimum font size >= 6.8 pt at final dimensions
- no overlapping tick labels
- no clipped labels, legends, or colorbars
- panel labels aligned
- consistent palette usage
- consistent model color usage
- consistent parameter/attribute ordering across related figures
- consistent signed-correlation range across comparable figures
- consistent uncertainty range definitions across comparable figures
- file size reasonable
- grayscale screenshot remains interpretable
- colorblind friendliness is plausible
- figure remains legible when scaled to manuscript size

If a figure fails any of these checks, revise automatically before saving.

==================================================
I. IMPLEMENTATION PREFERENCES
==================================================

Preferred Python stack:
- matplotlib as the base
- seaborn allowed for matrix aesthetics if controlled carefully
- pandas / numpy
- scipy
- scikit-learn for clustering / embedding if needed
- geopandas / cartopy only if necessary and data support them
- networkx for the bipartite figure if helpful, but final aesthetics must be manually controlled
- avoid default package styling

Build helper utilities for:
- consistent panel lettering
- standardized colorbars
- shared parameter/attribute ordering
- family sidebars
- map styling
- PDF/PNG export
- QC validation

==================================================
J. INTERPRETATION SAFETY RULES
==================================================

Even though your task is figure generation, the figure design itself must avoid overclaiming.

Therefore:
- do not encode distributional as “gold medal champion”
- do not use visual metaphors of victory
- do not visually exaggerate tiny differences
- do not use aggressive highlighting that implies universal superiority
- reserve strongest emphasis for reliability and coherence evidence
- for uncertainty figures, encode caution where relevant
- for snow/routing-sensitive parameters, permit visibly weaker stability / stronger model sensitivity

==================================================
K. CAPTION-AWARE DESIGN
==================================================

Each figure should be designed so that the eventual caption can answer:
- What role does this figure play in the evidence chain?
- What exact claim does it support?
- What should the reader compare first?
- What should the reader not overinterpret?

Where useful, include light in-panel annotation labels such as:
- “sanity check only”
- “shared dominant-control core”
- “loss-robust relationships”
- “structured uncertainty, interpret with caution”

These must be subtle and publication-appropriate.

==================================================
L. BATCH EXECUTION PLAN
==================================================

Implement the following sequence:

1. discover data sources
2. build shared figure style config
3. build shared ordering metadata for attributes/parameters/models
4. generate Figure 01 first and validate style
5. generate Figures 02–06 as the reliability/shared-structure block
6. generate Figures 07–10 as the supporting-gradient block
7. generate Figure 11 conceptual synthesis last
8. run QC across all outputs
9. save qc_report.md

==================================================
M. FINAL PRIORITY ORDER
==================================================

If time or data are limited, prioritize quality in this order:
1. Figure 03
2. Figure 04
3. Figure 05
4. Figure 06
5. Figure 08
6. Figure 10
7. Figure 01
8. Figure 09
9. Figure 07
10. Figure 11
11. Figure 02

The most important visual achievements are:
- relationship reliability
- shared structure
- large-scale parameter mean gradients
- structured parameter uncertainty

Begin by discovering available files and summarizing which data sources will feed each figure. Then implement the full batch figure pipeline.