# R3 IC-Stable to dPL Magnitude-Retention Audit

## Question
Among relationships selected only by IC stability, how often does dPL retain the same sign and an appreciable relationship magnitude?

## Data and provenance
Frozen relationship matrix `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`; primary frozen information-cluster space at threshold 0.70 and raw-attribute sensitivity. IC restart uncertainty comes from `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IDENTIFIABILITY_REPRODUCIBILITY.csv` and is not rerun.

## Estimand
Eligibility is IC-only: |rho_IC|>=0.20 and IC bootstrap sign probability>=0.95. For each eligible paired cell report rho_IC, rho_dPL, same sign, absolute magnitudes, delta, ratio, strength categories, and transitions. P(dPL retained | IC stable) is reported separately from P(IC support | dPL strong).

## Denominator
Primary denominators are 902 all36 information-cluster cells and 862 exclude-SIMHYD cells; raw and model-level denominators are explicit in the tables. All36 has 36 models and exclude_simhyd has 35; deterministic LOO omits each model once. Bootstrap CIs use 1000 paired cell resamples with fixed seeds.

## Method
Compute pooled and model-equal retention at dPL |rho| thresholds 0.10, 0.20, and 0.30, median absolute changes and ratios with bootstrap CIs, strength transition tables, per-model summaries, LOO ranges, and a descriptive restart-identifiability split/association.

## Result
1. Among IC-stable information-cluster relationships, same-sign retention is 0.941242 (849/902).

2. Same sign plus dPL |rho|>=0.10/0.20/0.30 is 0.876940/0.767184/0.573171; model-equal medians are 0.873397/0.769841/0.548589.

3. Median absolute IC/dPL strength is 0.268044/0.343520; median delta |rho|=0.054752 with bootstrap CI [0.041448, 0.067693], and median ratio=1.185404 with CI [1.141273, 1.231778]. The typical magnitude is preserved to strengthened, not attenuated, but the delta IQR is 0.209873.

4. Transition counts are saved in `R3_IC_TO_DPL_STRENGTH_TRANSITIONS.csv`: same-sign strong=517, same-sign moderate=175, same-sign near-zero=58, same-sign attenuated-small=99, opposite-sign=53 in the all36 primary denominator.

5. Retention is heterogeneous but not model-fragile: the all36 LOO p20 rate ranges from 0.758701 to 0.771889, and p30 from 0.563805 to 0.582949.

6. Exclude-SIMHYD gives denominator 862 and p10/p20/p30=0.872390/0.758701/0.563805; it does not change the conclusion.

7. Higher-identifiability/lower-restart-SD cells have p20=0.787625, p30=0.596990; lower-identifiability/higher-SD cells have p20=0.726974, p30=0.526316. The linear restart-SD association with dPL magnitude is rho=-0.074642, so any identifiability link is modest and descriptive.

8. This is P(dPL retained | IC stable), not the previous reverse conditional P(IC-supported | dPL strong)≈0.32. Closure verdict: R3_MAGNITUDE_RETENTION_STRONG.

## Sensitivity
Raw attributes are secondary. Exclude-SIMHYD and LOO are reported without changing the IC denominator rule. IC-stable sign agreement is checked against the prior audit denominator; a dPL magnitude threshold is never used to redefine eligibility.

## Adversarial interpretation
Sign retention can coexist with attenuation. The previous construction-artifact result P(IC-supported | dPL-strong)≈32% has the reverse conditional direction and a different denominator; neither percentage is interchangeable with P(dPL retained | IC stable). Retention remains cross-estimator association, not independent causal or hydrological validation.

## Verdict
R3_MAGNITUDE_RETENTION_STRONG

## Execution metadata
- runtime_seconds: `8.537`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
