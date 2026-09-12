# R3 Audit Log

## Scope

Final read-only synthesis of frozen R3 source products and A–E audits. Existing `project/benchmark/manuscript/r3/scripts/` and `cache/` were inventoried and left unchanged. No training, recalibration, simulation, new dPL seed, bootstrap, permutation, source overwrite, or R4 was performed.

## Provenance

- Baseline repository HEAD at inventory: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`.
- Frozen universe: 36 models, 531 common basins/model, 35 descriptors, 20 information dimensions, 271 coordinates, 5,420 information-cluster cells.
- Authoritative paths and hashes are listed in `R3_SOURCE_MANIFEST.json` and `R3_SOURCE_CHECKSUMS.sha256`.
- Source-checksum verification from repository root: `22/22 PASS`.

## A–E audit dispositions

- Agent A primary correspondence: **PASS WITH LIMITATION**.
- Agent B same-coordinate specificity: **PASS WITH QUALIFIER**.
- Agent C functional-role negative controls: **PASS WITH LIMITATION**.
- Agent D R2→R3 linkage and conditioning: **PASS WITH LIMITATION**.
- Agent E hostile review: **PASS WITH LIMITATION**; no concrete blocker and no rerun recommended.

## Prompt/source reconciliation

The prompt targets agree with the authoritative products at the reported precision. The exact authoritative values used are:

- `R_paired=0.7157894737`;
- stable-cell retention `849/902=0.9412416851`;
- `A_diag=0.6150375940`, SIMHYD-excluded `0.6105263158`;
- `A_role=-0.1187969925`;
- HESS-prior `D_rho=-0.0003301577`;
- `A_info|rank=0.295238`, CI `[0.254386,0.429073]`;
- conditioned divergence `+0.028033`, CI `[0.017163,0.043610]`;
- conditioned paired correspondence `-0.141353`, CI `[-0.215038,-0.078947]`.

No substantive prompt/source mismatch was found. Rounded prompt values were not used to modify any source.

## Evidence boundaries retained

The 94.12% sign-retention rate is conditional on an IC-only selected 902-cell subset, not all 5,420 cells. `R_paired` is profile correspondence, not identity. `A_diag` is same-coordinate label alignment, not physical semantics. Role-level continuity and HESS-prior flexibility are negative controls. Rank continuity is acknowledged; rank-matched residual specificity is the protection against reducing R3 to raw rank continuity. Performance conditioning is observational and composition-sensitive; secondary `D_theta` is mathematically coupled.

## Superseded/out-of-scope interpretations

Broad functional-role continuity, parameter physical-meaning/identity claims, “dPL strips noise,” `94.1%` all-cell retention, `A_diag` as a conservative physical estimate, causal performance conditioning, IC-as-truth, dPL superiority, compensation, and any new attributes/clusters/seeds/training/R4 are not for manuscript use. Historical source products remain preserved.
