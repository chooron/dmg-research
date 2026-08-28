# Final Supplement asset package

This directory is the frozen submission-asset layer for the HESS Supplement. Original results, legacy files, main manuscript files, and Word documents are not moved, deleted, or overwritten.

## Frozen SI architecture

| Asset | Final role | Final path |
|---|---|---|
| Table S1 | Calibrated parameter definitions, bounds, units, and structural membership | `tables/Table_S1/` |
| Table S2 | R1 KGE × CT and R3 denominator-threshold sensitivity | `tables/Table_S2/` |
| Table S3 | Controlled-recovery distributions and generating-field robustness | `tables/Table_S3/` |
| Figure S1 | Leave-one-HUC2-out regional omission robustness | `figures/Figure_S1/Figure_S1.png` |
| Figure S2 | TGD temperature-response and fixed-shape sensitivity | `figures/Figure_S2/Figure_S2.png` |
| Figure S3 | Component-wise truth-relative parameter/state/flux errors | `figures/Figure_S3/Figure_S3.png` |
| Figure S4 | High-snow seasonal liquid-water delivery and storage trajectories | `figures/Figure_S4/Figure_S4.png` |
| Figure S5 | Outcome-independent external-state examples and population corroboration | `figures/Figure_S5/Figure_S5.png` |

## Asset map

| Asset | Scientific role | Main R section | Final file | Source | README | Status |
|---|---|---|---|---|---|---|
| Table S1 | Production parameter contract | Methods / R1–R5 | `tables/Table_S1/Table_S1.csv` | `models/parameter_specs.py` | `tables/Table_S1/README.md` | CREATE_FROM_EXISTING_DATA |
| Table S2 | Threshold-screen sensitivity | R1 / R3 | `tables/Table_S2/Table_S2_panelA.csv`; `Table_S2_panelB.csv` | R1 CT and R3 denominator CSVs | `tables/Table_S2/README.md` | REGENERATE_FROM_EXISTING_DATA |
| Table S3 | Recovery tails and field construction | R3 / reviewer robustness | `tables/Table_S3/Table_S3_panelA.csv`; `Table_S3_panelB.csv` | reviewer-2 frozen summaries | `tables/Table_S3/README.md` | CREATE_FROM_EXISTING_DATA |
| Figure S1 | HUC-2 regional omission robustness | R1 / R3 / R5 | `figures/Figure_S1/Figure_S1.png` | regional LORO CSVs | `figures/Figure_S1/README.md` | RENUMBER_ONLY |
| Figure S2 | TGD response and response-shape sensitivity | R3 | `figures/Figure_S2/Figure_S2.png` | TGD response and shape CSVs | `figures/Figure_S2/README.md` | RENUMBER_ONLY |
| Figure S3 | Component error decomposition | R3 | `figures/Figure_S3/Figure_S3.png` | R3 component CSVs | `figures/Figure_S3/README.md` | CREATE_FROM_EXISTING_DATA |
| Figure S4 | Seasonal pathway and storage deviation | R3 | `figures/Figure_S4/Figure_S4.png` | Figure 6 seasonal summary/arrays | `figures/Figure_S4/README.md` | RENUMBER_ONLY |
| Figure S5 | External-state examples/population context | R4 | `figures/Figure_S5/Figure_S5.png` | R4 selection/population audit and frozen PNG | `figures/Figure_S5/README.md` | RENUMBER_ONLY |

Machine-readable registry: `_audit/supplement_asset_registry.csv`.

## Symbol conventions

| Symbol | Meaning / code mapping |
|---|---|
| `P_t` | Daily precipitation forcing / code `precip_t` |
| `T_t` | Daily mean temperature / code `temp` |
| `E_{p,t}` | Dataset-supplied potential evapotranspiration; legacy code/data aliases may be `PET` or `pet` |
| `P_t^*` | Effective liquid-water input to XAJ; code `effective_precipitation`/`effective` |
| `S_t^g` | TGD generic storage; code `storage` |
| `A_t` | XAJ tension-water capacity curve |
| `\tau_t` | TGD temperature-conditioned residence time; code `tgd2_tau`/`tau_*` |
| `r_t` | TGD daily retention; code `tgd2_retention`/`retention_*` |
| `G_t` | CN snow-water-equivalent storage |
| `M_t` | CN melt contribution |
| `W_{U,t}`, `W_{L,t}`, `W_{D,t}` | XAJ upper, lower, deep tension storages; code `wu`, `wl`, `wd` |
| `W_t` | Total tension storage, `W_{U,t}+W_{L,t}+W_{D,t}`; code `wt` |
| `Q_{i,t}`, `Q_{g,t}` | Interflow and groundwater routing fluxes; code `qi`, `qg` |
| `Q_t` | Total outlet discharge; code `qsim`/`Q` |

Full mapping and unresolved local intermediate names: `_audit/symbol_registry.md`.

## Caption assembly instructions

For each final asset, a caption-writing GPT should:

1. Read the final Figure/Table itself first.
2. Read that asset's `README.md`.
3. Read `caption_facts.md` for figures, or the table README for tables.
4. Describe only supported composition, source, sample, metric, uncertainty, and visual encoding facts.
5. Avoid repeating Results/Discussion arguments in the caption.
6. Do not add information absent from the asset, README, caption facts, or cited source data.
7. Respect every interpretation boundary, especially the HUC-2 regional-omission, generating-field, and external-state qualifiers.

## Retired assets

Legacy matched-control parameter-gradient and detailed Base–TGD parameter-distribution figures are excluded from final SI and documented in `retired_assets/README.md`. Their original files remain in `manuscript/archive/tables_si_legacy/`.

The existing alternative-generating-field PNG is also not a final figure; its machine-readable comparison is represented in Table S3 Panel B. Existing old-numbered Supplement copies remain untouched.

## Audit and execution records

- `_audit/preflight.md` — project/environment/status snapshot.
- `_audit/supplement_asset_registry.csv` — status registry.
- `_audit/symbol_registry.md` — Figure 2/Methods/code notation audit.
- `_audit/numerical_consistency_report.md` — numerical locks and conflicts.
- `supplement_manifest.csv` — machine-readable final package manifest.
- `SUPPLEMENT_ASSET_EXECUTION_REPORT.md` — final execution report.

No main manuscript text, Results/Discussion text, main Figure 2, Supplement Word document, or canonical result file is modified by this package.
