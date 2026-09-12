# R3 Parameter-Identity Correspondence Audit

## Question
Is same-parameter IC--dPL correspondence higher than cross-parameter correspondence when every dPL profile and attribute gradient is preserved?

## Data and provenance
Frozen relationship matrix `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r3/tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv` in primary 0.70 information-cluster and secondary raw-attribute spaces. No estimator or parameter result is used to change the frozen feature space.

## Estimand
For every model and within-model IC parameter p and dPL parameter q, C[p,q] is Spearman across the frozen information-cluster relationship vectors. Primary A_m is the median over p of C[p,p] minus the median over q != p of C[p,q]; A_overall is the model-equal median over models with a defined off-diagonal contrast. Top-1/top-3 use the rank of the diagonal among all dPL parameters for the same IC row.

## Denominator
The full cross matrix has 4790 rows. Diagonal/top-rank summaries use 36 models/all36 and 35 models/exclude_simhyd; off-diagonal and advantage contrasts use 35 of 36 and 34 of 35 valid model contributors because one-parameter collie1 has no off-diagonal contrast. Permutation nulls use 1000 fixed-seed dPL parameter-label permutations and preserve every profile; LOO rows report both retained and valid model counts.

## Method
Compute exact average-tie Spearman via row ranks; repeat after dPL parameter-label permutation without changing any dPL profile values. Raw-attribute space is sensitivity. No role-constrained null is imposed because role metadata has heterogeneous confidence and would condition on a separate closure result.

## Result
1. Same-parameter correspondence is higher than cross-parameter correspondence: all36 information-cluster diagonal median=0.715789, model-equal off-diagonal median=-0.021805, and diagonal advantage=0.615038. The model-equal median fraction of parameter rows with diagonal > off-diagonal is 1.000000. The advantage/off-diagonal contrast has 35 valid model contributors out of 36 because collie1 has one parameter and no off-diagonal comparison.

2. The primary 1000-permutation parameter-label null has mean=0.001536, 95% interval=[-0.123308, 0.112801], empirical p=0.000999; its observed/valid contrast denominator is 35/36 (the one-parameter collie1 advantage is undefined). It preserves complete IC/dPL profiles and all dPL attribute gradients while destroying only same-parameter identity.

3. Model-equal diagonal top-1/top-3 fractions are 0.625000/0.875000. Exclude-SIMHYD advantage=0.610526, null mean=-0.000157, p=0.000999; its advantage/null contrast has 34/35 valid model contributors. Exclude-SIMHYD top-1/top-3 are available in the summary table.

4. Deterministic information-cluster LOO advantage ranges are 0.610526–0.617293 for all36 with 34–35 valid contributors per omission, and 0.606015–0.615038 for exclude-SIMHYD with 33–34 valid contributors. Raw-attribute identity sensitivity is also positive and above its label null.

5. Generic shared dPL attribute gradients do not explain most of the original profile correspondence: the same-parameter diagonal retains the original profile correspondence scale while cross-parameter medians are near zero/slightly negative. The identity-specific effect is therefore left after preserving generic dPL structure.

6. The evidence supports the phrase same-parameter-coordinate profile persists across estimators at the frozen association-profile level. It does not establish causal parameter identity or a universal physical law.

7. Closure verdict: R3_PARAMETER_IDENTITY_SUPPORTED.

## Sensitivity
All36/exclude_simhyd, raw sensitivity, and deterministic LOO are reported. The null is identity-specific and is not the earlier basin-row null. Small models retain their full parameter matrices; one-parameter models contribute diagonal/top-rank information but not an undefined off-diagonal contrast.

## Adversarial interpretation
A significant A would defend same-coordinate persistence beyond generic cross-parameter dPL gradients, but not causality or physical law. A null result would downgrade the original profile correspondence to generic profile similarity; the report therefore retains both diagonal and off-diagonal values and does not switch statistics after seeing the null.

## Verdict
R3_PARAMETER_IDENTITY_SUPPORTED

## Execution metadata
- runtime_seconds: `138.774`
- git_commit: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
