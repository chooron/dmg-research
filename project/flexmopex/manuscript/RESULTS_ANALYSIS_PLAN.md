# Results analysis organization and execution plan

## Scope and evidence boundary

This plan is for the completed, model-modified CAMELS-671 formal results in
`results/formal_671_unified_nmul1_tail3/`. The final inventory is 103 runs:

- 88 main runs: core/reference, LOPO, and LORO families;
- 15 sensitivity runs: 3 seeds each for `nmul=1`, `4`, `8`, `16`, and `32`.

The plan separates analyses that can be computed from the downloaded JSON
metrics/checkpoints from analyses that require large prediction/state arrays.
No training is part of this plan. DFlex remains outside the formal evidence
because its formal provenance/configuration is not tracked.

## Results sections

### Results 3.1 — Predictive performance and model complexity

**Question.** Does the learned process-structure model retain hydrological
predictive skill relative to the basic/full references while introducing a
controlled structural representation?

**Primary inputs.**

- `results/formal_671_core/`
- `results/formal_671_reference/`
- per-run `metrics.json`, `metrics_agg.json`, `early_stopping.json`

**Candidate code.**

- `scripts/build_paper_tables.py`
- `scripts/analyze_kge_conditioning.py`
- `scripts/evaluate_optimal_vs_final_epochs.py`

**Planned outputs.** Performance/complexity table, NSE/KGE summaries, seed
summary, and an explicit best-vs-final checkpoint audit.

**Gate.** First build a compact metrics inventory and verify that model labels,
alpha values, seeds, train/test splits, and epoch identifiers are consistent.
Do not aggregate before this gate passes.

### Results 3.2 — Alpha path and performance–complexity trade-off

**Question.** How does the structural regularization strength change predictive
performance, process weights, active-process counts, and structural complexity?

**Primary inputs.**

- `results/formal_671_core/lambda*/`
- `results/formal_671_reference/`
- `early_stopping.json` and metrics JSON files

**Candidate code.**

- `scripts/build_paper_tables.py`
- `scripts/synthesize_r19_results.py`
- `scripts/analyze_kge_conditioning.py`

**Planned outputs.** Alpha-path table, performance–complexity curve, primary
alpha justification, and seed-aware uncertainty summary.

**Gate.** Confirm which lambda/alpha configurations are present in the final
671 inventory. Historical scripts refer to `results/block1_alpha_path/` and
must not silently consume that legacy path.

### Results 3.3 — Process weights and hydroclimatic organization

**Question.** Are learned process contributions and structural coordinates
organized by basin attributes in a stable and interpretable way?

**Primary inputs.**

- final/best checkpoint outputs for the Flex runs;
- basin attributes and `data/gage_id.npy`;
- compact per-basin metrics and weight/coordinate exports when available.

**Candidate code.**

- `scripts/analyze_basin_attributes.py`
- `scripts/analyze_process_gradients.py`
- `scripts/plot_structural_consistency_figures.py`
- `scripts/build_paper_tables.py`
- `scripts/audit_structural_calibration.py`

**Planned outputs.** Attribute–weight/coordinate associations, spatial or
attribute summaries, process-level tables, and seed-stability checks.

**Gate.** Inspect checkpoint/output schemas before loading tensors. Prefer
streaming one run at a time and writing compact CSV/JSON intermediates. Do not
read all basin-by-time arrays into memory.

### Results 3.4 — Parameter-space readout and process mechanism

**Question.** Do structural coordinates and process weights correspond to
parameter-space organization or counterfactual process effects, rather than
being only predictive labels?

**Primary inputs.**

- best/final checkpoints from the formal Flex runs;
- parameter/weight exports and compact basin-level tables;
- only the minimum forward trajectories needed for a defined diagnostic.

**Candidate code.**

- `scripts/analyze_process_counterfactuals.py`
- `scripts/analyze_gate_gradient_separability.py`
- `scripts/oracle_representability_data.py`
- `scripts/oracle_representability_baselines.py`
- `scripts/oracle_representability_probe.py`
- `scripts/oracle_representability_null.py`
- `scripts/analyze_basin_interception_benefit.py`
- `scripts/compute_interception_oracle.py`

**Planned outputs.** Parameter-readout association tables, process-level
counterfactual summaries, representability/null audits, and mechanism figures.

**Gate.** Run a small one-basin smoke calculation first. Any GPU forward pass
must use bounded basin batches and explicitly release tensors. No full
counterfactual sweep is allowed until its memory estimate is documented.

### Results 3.5 — Leave-one-region-out transfer

**Question.** Do learned structural coordinates and process decisions transfer
to held-out hydroclimatic regions?

**Primary inputs.**

- `results/formal_671_loro/`
- region and seed metadata;
- LORO metrics and structural outputs.

**Candidate code.**

- `scripts/build_paper_tables.py`
- `scripts/build_loro_groups.py`
- `scripts/build_531_loro_groups.py`
- `scripts/collect_full_lopo_ablation.py` (only where its input contract matches)
- `scripts/plot_structural_consistency_figures.py`

**Planned outputs.** Region-wise predictive transfer, continuous-coordinate
transfer, categorical process-decision transfer, and region-level summary tables.

**Gate.** Validate the region mapping and held-out-region identity before
computing any pooled result. Never infer region identity from directory
modification times.

### Sensitivity and validation analyses

These analyses support the main Results claims and should be kept separate from
the primary tables:

- **LOPO validation:** `results/formal_671_lopo/`,
  `scripts/collect_full_lopo_ablation.py`;
- **nmul sensitivity:** `results/formal_671_nmul/`,
  `scripts/collect_nmul_ablation.py`;
- **best-vs-final training choice:**
  `scripts/evaluate_optimal_vs_final_epochs.py`;
- **KGE and conditioning:** `scripts/analyze_kge_conditioning.py`;
- **structural calibration and identifiability:**
  `scripts/audit_structural_calibration.py`,
  `scripts/audit_pure_x35_bce_and_identifiability.py`;
- **gate/gradient and interception diagnostics:** the corresponding
  `analyze_*`, `compute_*`, `screen_*`, and `validate_*` scripts.

The exact mapping of LOPO and nmul text to main versus supplementary Results
must be fixed after the manuscript section headings are recovered; until then,
these are validation analyses and must not be presented as a new primary claim.

## Imported script status

`manuscript/scripts/` currently contains 62 Python analysis scripts. They were
copied without execution. The source repository's historical scripts include
legacy paths (`results/block1_*`, `results/block3_loro`, and older R15–R19
experiments); these are retained for provenance but are not automatically
trusted for the final 671 run. Each script needs an input-path/schema check
before use.

Recommended status labels for the next pass:

- `READY_AFTER_INVENTORY`: JSON/metadata-only scripts after path verification;
- `NEEDS_ADAPTER`: scripts expecting legacy result roots or old field names;
- `GPU_SMOKE_REQUIRED`: scripts that perform model forward passes;
- `LEGACY_NOT_PRIMARY`: R15–R19/reconciliation/replay diagnostics not tied to
  the final 671 formal inventory.

## Safe execution order

1. Inventory all 103 run directories and metric schemas.
2. Build a compact run manifest and metric table; CPU threads = 1.
3. Run one small Results 3.1 smoke aggregation.
4. Review counts, model labels, alpha, seeds, and missing fields.
5. Generate Results 3.1 tables before moving to later sections.
6. Add one section at a time; use one process by default and bounded batches for
   GPU inference.
7. Store every output under `manuscript/analysis/` or
   `manuscript/figures/`, never overwrite raw `results/`.
8. Write a per-stage log and validation manifest before proceeding.
