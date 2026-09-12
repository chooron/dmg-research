# R1 Data Audit

## Scope

Current R1 uses IC-CMA-ES XAJ-Base, XAJ-TGD, XAJ-CN and dPL-MLP XAJ-Base, XAJ-TGD, XAJ-CN, and HBV. The period contract is warmup 1980-10-01..1981-09-30; train 1981-10-01..1995-09-30; test 1995-10-01..2010-09-30. The CAMELS-531 list is `/home/jingxin/code/dmg-research/data/531sub_id.txt`.

## Inference provenance

Existing daily simulation exports and partition summaries are reused in this extension. No training, resume, recalibration, preprocessing, or inference job was launched. The current provisional dPL XAJ-TGD artifact is selected from the common valid periodic checkpoint epoch recorded in `r1_inference_audit.md`; the current exported result is epoch 100. Existing merged Parquet files are verified for row-count integrity and are not rewritten.

Forcing and observations come from `/home/jingxin/code/dmg-research/data/camels_dataset` with forcing order `P,T,PET`. Observed discharge is raw ft3/s and is converted to mm/day using the repository's `area_gages2` index-11 conversion. Nonfinite and negative flows are masked; zero is valid. dPL uses the repository robust median/IQR normalization and sigmoid physical mapping, including inverse-log mapping for TGD2 residence times. IC uses the repository Lite `ModelAdapter` with selected restarts based only on stored train-period KGE.

## Metric definition

In `full` mode, statistics are computed immediately from the same aligned target and predictions used to write the daily Parquet exports; `statistics` mode recomputes them from those daily exports. The authoritative repository evaluator is standard KGE(Q): `1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`, with `alpha=std_sim/std_obs`, `beta=mean_sim/mean_obs`, finite nonnegative paired mask, minimum 30 paired days, and invalid zero-variance observations excluded. It is not CV-ratio KGE-prime.

NSE, PBIAS, and RMSE use the same paired valid-day mask. PBIAS is `100*sum(sim-obs)/sum(obs)`; positive values indicate simulated excess.

## Signatures

Water years begin October 1. CT is the first day cumulative flow reaches 50% of annual flow. AMJJ is April-July flow divided by water-year flow. Complete water years are calculated first and basin summaries retain `valid_years`; primary CT and AMJJ effects require at least five complete water years and a three-year minimum sensitivity is retained. SPO is excluded from active R1 under `excluded_from_R1_incomplete_prespecified_definition`. Daily simulation data are available; the limitation is definition and reproducibility because cumulative start date, search window, reference discharge, no-pulse handling, incomplete-year handling, tied-minimum handling, and date encoding are not prespecified.

## Extension estimands

Generalization exposure is `E_enhanced-base = (KGE_enhanced,test - KGE_base,test) - (KGE_enhanced,train - KGE_base,train)` for CN-Base, TGD-Base, and CN-TGD, calculated basin by basin within each paradigm. IC-dPL transfer is `KGE_IC - median_seed(KGE_dPL)` for each XAJ structure and period. All paired summaries use matched basin IDs.

The snow-specific CN-TGD relationship uses `KGE_CN,test - KGE_TGD,test`. The combined interaction model is `effect_value ~ frac_snow + paradigm_dPL + frac_snow:paradigm_dPL`, with IC-CMA-ES as the reference category and cluster-robust standard errors clustered by `basin_id`.

CT and AMJJ error reductions are `|E_Base| - |E_CN|`, `|E_Base| - |E_TGD|`, and `|E_TGD| - |E_CN|`; positive values indicate lower error for the second model in the stored first-minus-second convention. Effects are calculated from basin-level medians over complete water years. The primary set requires five valid years and the sensitivity set requires three.

Ordinary paired basin bootstrap uses 10,000 resamples and seed `20260730`. The spatial sensitivity uses the authoritative `data/basin_groups/group_11.npy` through `group_17.npy` seven-region LORO grouping documented in `project/flexmopex/run_model.py`; region blocks are sampled with replacement. Restart signature robustness is unavailable because non-selected restart daily CT/AMJJ series do not exist. dPL seed robustness calculates basin effects within each seed (`42`, `123`, `2026`) before reporting seed estimates and the across-seed median.

See `r1_inference_audit.md`, `r1_daily_simulation_inventory.csv`, `r1_exclusion_log.csv`, and `r1_result_manifest.json`.

## Full-sample aggregation pass

The aggregation pass reads the existing basin-level performance, structural-effect, generalization-effect, signature, attribute, and snow-relationship CSV files. It does not read or rewrite daily Parquet files and launches no inference. Primary absolute metrics use `all_531_basins`. dPL primary metrics use within-seed basin metrics followed by the median across seeds; IC primary metrics use the selected restart chosen by train-period KGE. Primary signature summaries require five complete water years; the three-year result remains sensitivity only. Bootstrap intervals target the basin-level median with 10,000 resamples and seed `20260730`. Regional sensitivity uses the existing seven LORO region files. Snow-stratified rows use the existing fixed S1-S5 frac_snow boundaries and are descriptive stratified summaries within the same R1 tables. Remaining checks add basin-wise IC-dPL transfer A/B/D, transfer-gradient interaction with TGD as the reference structure and basin-clustered standard errors, continuous snow-gradient exposure, and primary five-water-year CT/AMJJ absolute and paired reduction summaries. Regional slope intervals are seven-region block-bootstrap OLS slope sensitivities; robust slopes retain ordinary Theil-Sen confidence intervals.

## Reproducibility audit

The pre-fix comparison classified all CSV and Markdown outputs as byte-identical and all parsed scientific JSON values as identical. The only pre-fix JSON difference was the invocation-specific temporary `--output-root` embedded in `r1_result_manifest.json`; `r1_execution.log` differed only in timestamps, runtime paths, and git-status snapshots. The fix stores the canonical reproduction command in the manifest, uses explicit UTF-8/LF/`%.17g` summary serialization, and records deterministic artifact hashes without hashing the manifest itself. Bootstrap master seed is `20260730`; keyed estimands use the existing SHA-256-derived seed helper. The three post-fix processes with `PYTHONHASHSEED=1`, `777`, and `20260731` produced identical inventories, schemas, exact parsed scientific values, and deterministic artifact bytes. The remaining volatile artifact is `r1_execution.log`.
