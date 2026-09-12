# S2 Manuscript Readiness

| Section | Ready facts | Missing facts | Required action | Manuscript risk |
|---|---|---|---|---|
| S2.1 | Active XAJ/GR4J/SIMHYD/HBV equations and states | Exact canonical SIMHYD variant | Cite implementation and qualify canonical comparison | Medium |
| S2.2 | Basic two-parameter CN, partition, G/eTG and melt | Literature wording only | Write implemented CemaNeige variant first | Low |
| S2.3 | TGD storage, frozen temperature stats and tau equation | Published dPL checkpoint identity | Confirm run manifest if exact experiment must be named | Medium |
| S2.4 | Parameter counts and code names | Formal author-selected matching convention | Use code-name table; do not claim equal process DOF | Low |
| S2.5 | Same-day wrapper order and unchanged PET | None for code path | Write coupling matrix into S2 | Low |
| S2.6 | Clamp/where/tanh/expm1/UH inventory | Derivative behavior at every branch | Retain implementation caveats | Medium |
| S2.7 | TGD/CN preprocessing diagnostics and partial host probes | Full XAJ/GR4J/HBV whole-system balance | Expose daily traces or leave claim unresolved | High |
| S2.8 | Bounds and normalized mapping | Production checkpoint-specific values | Cite active parameter spec | Low |
| S2.9 | HBV implementation and reference role | Exact author-selected HBV literature edition | Cite source recorded in reference CSV | Low |

The main attribution risk is treating TGD as snow or claiming CN/TGD have the same number of states. The code shows a conservative generic delay versus an explicit snow accounting variant.
