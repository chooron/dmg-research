# Final R2 Analysis Pipeline

## Scientific question

R2 characterizes how bound-normalized parameter realizations change when moving from basin-wise independent calibration (IC) to shared attribute-constrained dPL, and which parts of that reorganization remain beyond archived IC multi-start variability. All outputs are descriptive associations. R2 does not establish a preferred method, physical correctness, causal functional roles, compensation, or identifiability improvement.

## Reproduction

From the repository root, the frozen environment is expected at `.venv/bin/python`:

```bash
cd project/benchmark/manuscript/r2
bash scripts/run_all.sh
```

`run_all.sh` clears only derived cache groups, preserves `cache/inputs/`, runs Modules 1–6 and the validation gate, and writes `cache/checksums.sha256`. Source products under `project/benchmark/results/` are read-only inputs.

## Final modules

1. `01_parameter_separation_icself.py`: bound-normalized `D_RMS` relative to the archived one-sided IC-self reference.
2. `02_rank_reorganization.py`: all-36 descriptive rank preservation and the strict 23-model IC-self subset.
3. `03_coordinate_localization_icself.py`: raw coordinate concentration and coordinate-specific IC-self excess adjustment.
4. `04_distribution_reference_sensitivity.py`: matched 23-model canonical/consensus/self-reference CR geometry.
5. `05_outlet_parameter_bridge.py`: short within-model outlet-performance bridge.
6. `06_r2_r3_rank_linkage_audit.py`: supporting rank-to-information hostile audit.

`99_validate_frozen_results.py` checks all frozen values, model denominators, verdicts, seed policy, and aggregation caveats. `r2_common.py` records source hashes and the strict IC restart contract.

## Frozen claims and boundaries

- IC–dPL(seed 42) parameter separation is substantial relative to archived IC multi-start dispersion (`D_RMS` model-equal median ≈0.384375; one-sided cross-minus-self difference ≈0.218775).
- Cross-catchment ordering is substantially reorganized; the strict IC-self benchmark is limited to 23/36 eligible models and remains `INCONCLUSIVE`.
- Displacement shows some coordinate concentration, which persists after coordinate-specific archived IC-self adjustment in the strict subset.
- Apparent canonical contraction is `CANONICAL-IC DEPENDENT`, not universal.
- The outlet-performance bridge is moderate and supporting only.
- dPL has only canonical seed 42. `D_cross`, `E_paradigm`, and `R_paradigm` remain unavailable; the IC-self comparison is one-sided and is not IC truth. No E/excess result is a causal decomposition.
- `M_self≈0.000165` in Module 3 is a coordinate/basin median archived IC-self reference, not the earlier parameter-vector RMS `D_self` aggregation.

## Cache layout

`cache/inputs/` stores the frozen contract, exact 23/13 model lists, parameter bounds/order, input manifest, and source checksums. `cache/separation/`, `rank/`, `localization/`, `distribution/`, `performance_bridge/`, and `r2_r3_linkage/` store derived module summaries and manifests. `cache/final/` stores the frozen summary, claim–evidence map, validation report, run log, master manifests, status, audit, and checksums. Large raw model products are not copied.

## Cleanup and superseded branches

The cleanup audit records are `CLEANUP_PLAN.md`, `DELETION_MANIFEST.tsv`, and `CLEANUP_REPORT.md`. The old 00–09 manuscript scripts and caches were exploratory or superseded, including clustering/taxonomy, basin-regime, landing/role, spatial/influence, old contraction interpretations, and the incorrect 36-model Agent B primary aggregation. Historical audit/source products remain under `project/benchmark/results/` and were not deleted. Future dPL multi-seed work, if ever added, is an independent stochastic robustness check and does not redefine this R2 pipeline.
