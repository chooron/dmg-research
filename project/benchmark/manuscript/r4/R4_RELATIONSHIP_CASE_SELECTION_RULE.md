# R4 Relationship Case Selection Rule

## Scope and provenance
No prior R4 relationship-case preregistration or frozen manifest was found in the repository. The deterministic rule below therefore freezes four cases from the final seen-basin IC–dPL atlas before any OOB relationship coefficient is read.

Seen evidence source:
- `project/benchmark/results/ic_dpl_seenbasin_formal_20260901/09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv`
- Relationship implementation: `project/benchmark/scripts/diagnostics/parameter_attribute_atlas.py`
- Attribute-cluster sensitivity source: `project/benchmark/results/parameter_attribute_atlas_followup_20260829/attribute_redundancy_groups.csv`, threshold `|rho| >= 0.70`

The selected population is the PRIMARY 8 models, all 531 basins, continuous attributes, and `strict_full300=True`. Parameters are excluded if either seen IC or seen dPL normalized coordinates are near-constant (SD ≤ 0.01) or boundary-concentrated (≥50% within 0.01 of either bound).

## Relationship classes inherited from R2/R3 seen evidence
Using basin-wise Spearman correlation with average ranks and the existing descriptive rules, applied in this precedence order:

1. `sign-changing`: both `|rho| >= 0.10`, opposite signs;
2. `attenuated`: IC `|rho| >= 0.20`, dPL `|rho| < 0.10`;
3. `dPL-emergent`: dPL `|rho| >= 0.20`, IC `|rho| < 0.10`;
4. `persistent/reproduced`: same sign, both `|rho| >= 0.20`, and both within-parameter absolute-rho ranks ≤ 10;
5. otherwise `weak/unresolved`.

## Deterministic case selection
One case is selected from each class in this order: persistent/reproduced, attenuated, dPL-emergent, sign-changing.

- Persistent: descending persistent-attribute model recurrence, descending `|rho_IC|`, descending `|rho_dPL|`, then ascending model/parameter/attribute.
- Attenuated: descending `|rho_IC|`, ascending `delta_abs_rho = |rho_dPL|-|rho_IC|`, then ascending model/parameter/attribute.
- dPL-emergent: descending `|rho_dPL|`, descending `delta_abs_rho`, then ascending model/parameter/attribute.
- Sign-changing: descending `|rho_IC|`, descending `|rho_dPL|`, then ascending model/parameter/attribute.

## Freeze declaration
**OOB relationship coefficients had not been inspected or used for case selection before `R4_RELATIONSHIP_CASES_FROZEN.csv` was written and hashed.** The manifest is now frozen; later OOB analysis may evaluate these cases but may not alter them.

The manifest includes the seen rho values only as pre-OOB selection evidence. It is not a truth label, causal claim, or model-suitability selection.
