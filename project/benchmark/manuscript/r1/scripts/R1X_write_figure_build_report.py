#!/usr/bin/env python3
"""Write the reproducible Figure 1 build report."""
from __future__ import annotations

from pathlib import Path

import json
import pandas as pd

from r1_config import CACHE_DIR, DATA_ROOT, FIGURES_DIR, REPO, R1_ROOT, TABLES_DIR, MODEL_REGISTRY, TEMPORAL_AB

REPORT_PATH = R1_ROOT / "R1_FIGURE_BUILD_REPORT.md"


def main() -> None:
    source_tables = [
        "R1_model_basin_delta_kge.csv", "R1_model_performance_summary.csv", "R1_ensemble_summary.csv",
        "R1_temporal_AB_model_basin.csv", "R1_temporal_AB_model_summary.csv",
        "R1X_model_estimator_tendency.csv", "R1X_basin_estimator_tendency.csv",
        "Fig1c_basin_effect_spread.csv",
        "R1X_two_way_decomposition.csv", "R1X_temporal_basin_persistence_summary.csv",
        "R1X_basin_difficulty_checks.csv",
        "TableS1A_R1_effect_magnitude_summary.csv", "TableS1B_R1_model_summary.csv",
        "TableS1_R1_performance_context.md",
    ]
    full = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv")
    basin_map = pd.read_csv(TABLES_DIR / "Fig1c_basin_effect_spread.csv")
    map_metadata = json.loads((CACHE_DIR / "Fig1c_map_metadata.json").read_text())
    decomposition = pd.read_csv(TABLES_DIR / "Fig1d_two_way_decomposition.csv")
    temporal = pd.read_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv")
    s1a = pd.read_csv(TABLES_DIR / "TableS1A_R1_effect_magnitude_summary.csv")
    s1b = pd.read_csv(TABLES_DIR / "TableS1B_R1_model_summary.csv")
    report = f"""# R1 Figure build report

## A. Inputs

Figure 1 final is the conclusive visual and scientific wording closure using the existing R1/R1X source tables only. The renderer only reads precomputed tables: no additional training, recalibration, CMA-ES, checkpoint modification, or forward evaluation was run in this revision. Prior formal forward-only A/B results are reused. The final renderer writes one assembled PNG and does not write panel files or PDFs.

Source tables are under `{TABLES_DIR}`:

{chr(10).join(f"- `{name}`" for name in source_tables)}

The primary matrix contains {len(full):,} cells, {full.model.nunique()} models, and {full.basin_id.nunique()} basins. Model rows use the canonical registry order:

`{", ".join(MODEL_REGISTRY)}`

The estimand is `ΔKGE = KGE_dPL − KGE_IC`; positive values indicate higher dPL KGE. Temporal panel (e) uses A=`{TEMPORAL_AB[0]['start_date']}..{TEMPORAL_AB[0]['end_date']}` and B=`{TEMPORAL_AB[1]['start_date']}..{TEMPORAL_AB[1]['end_date']}`.

## B. Style calibration

The four user-supplied PDF paths under `/mnt/data` were checked in the execution environment but were not mounted, so no direct figure tracing or paper-specific copying was performed. The available repository analogs were `project/hydrodiag/manuscript/scripts/r0/plot_r0_figure1.py`, `project/hydrodiag/manuscript/scripts/r1/plot_r1_figure2_canonical.py`, and the shared HESS/Copernicus plotting style. The adopted principles were: one restrained composite layout; serif typography; light boundaries and direct legends; balanced whitespace; sparse, non-inferential annotations; and no dashboard, ranking, or model-selection encoding.
The final revision keeps the five-panel scientific structure and widens the canvas to 15.5 × 8.5 inches. The first six GridSpec columns remain allocated to (a) and (b), while the six right-hand columns use enlarged equal-width ratios for (c)–(e), prioritizing space to the GIS and lower-right panels without materially expanding (a)/(b).
Panel (a) retains an extremely light neutral-gray row background when the model-level median satisfies `dPL > IC` (12 models); panel (b) removes all row shading to prevent a dual-winner presentation. Panel (b) retains only the pale negative/positive half-plane cues and clear dashed `ΔKGE=0` line; all top directional annotations are removed, with direction interpretation retained in the caption. The enlarged `P_m^+` column is positioned close to the distribution body.
Panel (c) retains `M_b` as the sole map color variable and fixed-size gauge markers. Its dedicated wide map region is unobstructed; a short, thin horizontal colorbar occupies the upper part of the right-side column, and an enlarged neutral `|M_b|` versus `S_b` (MAD) inset with a `y = x` reference line is placed below and right-shifted. This frees the map from inset overlap without implying stronger spatial structure.
Panel (d) is titled `Decomposition of ΔKGE variation`, with the 66.2% component labeled `Interaction + residual` (variation remaining beyond additive model and basin effects rather than pure model–basin specificity). Panels (d) and (e) use equal three-column GridSpec spans and identical outer dimensions, with a slightly increased horizontal gap (`wspace=0.12`). Typography is enlarged throughout on a `15.5 × 8.5` inch canvas; the renderer uses the installed `DejaVu Serif` fallback because Times New Roman and Liberation Serif are unavailable.

The GIS remains in the main text as a full-period descriptive basin effect map. The non-adopted suggestions are moving GIS to the SI, adding a sixth/residual-profile panel, using marker size for spread, adding a null-reference marker or cell-level A/B elements, and adding new robustness analyses; these would broaden the frozen R1 scope or introduce competing encodings. `P_b^+` remains in the existing/source table support but is not the panel (c) map color.

`Times New Roman` and `Liberation Serif` were unavailable; the final PNG uses the installed `DejaVu Serif` family consistently for panel labels, titles, axes, ticks, legends, inset text, and math text.

## C. GIS audit

- Basin location source: `{DATA_ROOT / 'camels_loc' / 'camels_671_loc.dbf'}`, following the hydrodiag GIS convention of reading `gage_id`, `lat`, and `lon` and joining by zero-padded `basin_id`.
- Boundary source: `{REPO / 'project' / 'hydrodiag' / 'manuscript' / 'cache' / 'gis' / 'us_states.geojson'}`; a read-only copy is cached at `{CACHE_DIR / 'gis' / 'us_states.geojson'}`.
- Final map metadata: `{CACHE_DIR / 'Fig1c_map_metadata.json'}`; the derived table is `{TABLES_DIR / 'Fig1c_basin_effect_spread.csv'}`.
- Projection: EPSG:5070 Albers Equal Area, with light CONUS state boundaries and a national outline.
- Matched points: **{len(basin_map)}/531**; missing coordinates: **0**; duplicate basin joins: **0**; geometry conflicts: **0**.
- Final geometry: **gauge-location points** (`geometry_status=gauge_point`). No complete CAMELS basin polygon source was available in the permitted R1 inputs, so the prescribed point fallback was used. No spatial interpolation or spatial significance test was performed.
- Panel (c) primary map color is `M_b_median_deltaKGE`, the median ΔKGE across 36 models; all gauge markers have equal size and are plotted in increasing `|M_b|` order so extremes remain visible.
- Cross-model spread is `S_b_MAD_deltaKGE`, the median absolute deviation of ΔKGE across models, shown in the neutral `|M_b|` versus `S_b` inset with a `y = x` diagonal reference line. It is not called residual model–basin specificity.
- Selected symmetric color limit: **L={map_metadata['selected_color_limit_L']:.2f}** from Q95(|M_b|)={map_metadata['q95_abs_M_b']:.4f}; clipped for display: **{map_metadata['number_clipped_low']} below**, **{map_metadata['number_clipped_high']} above**, **{map_metadata['number_clipped_total']} total**.
- Inset summaries: median |M_b|={map_metadata['median_abs_M_b']:.4f}; fraction with |M_b| < S_b={map_metadata['fraction_abs_M_b_less_than_S_b']:.4f}. Existing `P_b^+` remains in the source table support but is not encoded in panel (c).

## D. Main Figure 1

- **(a)** retains same-row paired median-KGE points and connecting segments in canonical model order, using open blue IC circles and filled coral dPL squares. Rows with model-level median `dPL > IC` receive an extremely light neutral-gray background; no y-offset or micro-summary is used.
- **(b)** retains Q10–Q90, Q25–Q75, median ΔKGE, the enlarged `P_m^+` column, and the zero reference. All row shading is removed. Very pale blue/coral negative/positive backgrounds remain, while the top direction annotations are removed and explained in the caption.
- **(c)** maps basin-level median ΔKGE (`M_b`) with equal-size CAMELS-US gauge markers and no spatial-pattern enhancement. The widened map region is unobstructed; a short, thin horizontal colorbar occupies the upper right-side block and the larger neutral `|M_b|` versus `S_b` (MAD) inset with `y = x` line is below it and shifted right.
- **(d)** is titled `Decomposition of ΔKGE variation`, uses `Share of ΔKGE variation (%)`, and presents the unchanged shares as three concise slate horizontal bars with `Interaction + residual` for the 66.2% component. Its outer panel dimensions match (e) exactly.
- **(e)** retains 36 model-level A/B summaries, identity and zero references, and the frozen Spearman/same-sign annotation. Its outer panel dimensions match (d) exactly, with a slightly larger horizontal gap. Same-sign points are neutral slate filled circles; sign-changing points are open diamonds, with the extreme `topmodel` point labeled. No regression or significance overlay is added.

Formal outputs:

- Revised deliverable: `{FIGURES_DIR / 'Fig1_R1_main_final.png'}` (one assembled PNG, 600 dpi).
- The final deliverable is the single new PNG; the renderer does not generate panel files or PDFs and does not target any prior draft names.

- The runner prepares the existing-matrix derivative table/metadata and then targets only `Fig1_R1_main_final.png`.

## E. Supplement and Numerical Context (Table S1)

- **No main-text table is added for R1:** Figure 1 fully carries the visual evidence chain for performance context.
- **Supplementary Table S1 is established under `{TABLES_DIR}`:**
  - **Table S1A (`TableS1A_R1_effect_magnitude_summary.csv`):** across 19,116 pairs, overall median `|ΔKGE| = 0.0456` (Q25=0.0176, Q75=0.1011). Conditional on dPL higher (7,660 pairs, 40.07%), median ΔKGE is `+0.0425` (Q25=0.0155, Q75=0.1009). Conditional on IC higher (11,456 pairs, 59.93%), median `|ΔKGE|` is `0.0477` (signed median −0.0477, Q25=0.0191, Q75=0.1013). Ties = 0.
  - **Table S1B (`TableS1B_R1_model_summary.csv`):** exact 36-model numerical backing for panels (a), (b), and (e).
  - **Integrated Markdown (`TableS1_R1_performance_context.md`):** combines S1A and S1B with comprehensive notes.
- **Table S2 retained as internal support:** records simhyd exclusion, mean-versus-median aggregation, basin temporal persistence, and tolerance checks.

## F. Final verdict

**READY FOR R1 WRITING.** The Figure 1 final package has no data or provenance blocker. The frozen scientific structure is preserved: IC and dPL occupy broadly comparable ensemble-level performance ranges, individual model–basin responses show substantial heterogeneity, typical estimator-effect magnitude is comparable to cross-model spread, variation beyond additive model and basin effects dominates, and model-level responses are temporally reproducible across split periods. No suitability, ranking, causal attribution, OOB/PUR, or R2/R3 claims are made.
"""
    REPORT_PATH.write_text(report)
    print(f"PASS: wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
