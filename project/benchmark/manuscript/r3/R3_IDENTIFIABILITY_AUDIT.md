# R3-F Identifiability and Reproducibility Audit

## Question
Is poorer IC parameter identifiability associated with weaker cross-estimator relationship persistence?

## Data and provenance
Existing archived restart uncertainty `/home/jingxin/code/dmg-research/project/benchmark/results/seenbasin_remaining_analysis_20260901/agent_C/C05_IC_RESTART_PARAMETER_UNCERTAINTY.csv` and availability gate `/home/jingxin/code/dmg-research/project/benchmark/results/seenbasin_remaining_analysis_20260901/agent_C/C05_RESTART_DATA_AVAILABILITY_GATE.csv`; primary R3 parameter profiles and R2 boundary/tie/IC bootstrap tables. No restart, optimizer, training, or checkpoint modification was performed.

## Estimand
Outcome is each model×parameter information-cluster profile reproducibility. Predictors are median restart_u SD/IQR/range and fitness separation on normalized parameter coordinates; boundary fraction, tie fraction, IC relationship stability, model skill, and parameter count are covariates/descriptors.

## Denominator
Parameter denominator is 271 common model-parameter cells; basin-level restart aggregation requires 531 basins and archived 10 starts per cell. Pooled, within-model, model-level, and leave-one-model-out association levels are distinct.

## Method
Spearman associations and 1000-row bootstrap CIs are reported without filtering on outcome. LOO ranges assess influential models. The result is not a mixed-effects causal decomposition.

## Result
Parameter-level pooled association between median restart-u SD and profile reproducibility was rho=-0.374376 (p=1.92026e-10, n=271); bootstrap CI [-0.473632,-0.270936]. Model-level and within-model rank associations plus LOO ranges are saved.

## Sensitivity
Boundary saturation and low restart spread can coexist with poor cross-estimator persistence, and restart spread may reflect equifinality rather than statistical noise. Sensitivity does not identify a unique mechanism.

## Adversarial interpretation
A reviewer may say identifiability explains all disagreement. This audit can support only an association; residual model/parameter heterogeneity must be assessed from the profile, dominant, and construction-artifact tables. Use language 'associated with weaker persistence' only; never say a percentage of disagreement is caused by identifiability.

## Verdict
R3_F_READY

## Execution metadata
- runtime_seconds: `2.426`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
