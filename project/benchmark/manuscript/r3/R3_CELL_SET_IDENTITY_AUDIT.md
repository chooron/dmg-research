# R3 Cell-Set Identity Audit

## Scope

This is read-only post-processing of the frozen R3 information-cluster relationship table. The unique key is `(model, parameter_index, feature_index)`, with the report labels `model`, `parameter`, and `information_dimension`. No threshold or bootstrap rule was changed.

## Definitions

- `S_stable`: `abs(rho_IC) >= 0.20` and `IC_bootstrap_sign_probability >= 0.95`.
- `S_absrho20`: `abs(rho_IC) >= 0.20`, without the bootstrap-sign condition.
- dPL is not used to define either set.

## Set comparison

| Quantity | Value |
|---|---:|
| Total information-cluster cells | 5420 |
| `|S_stable|` | 902 |
| `|S_absrho20|` | 902 |
| Intersection | 902 |
| `S_stable - S_absrho20` | 0 |
| `S_absrho20 - S_stable` | 0 |
| Symmetric difference | 0 |
| Jaccard | 1.000000000000 |

The two 902-cell definitions are **exactly identical** in this frozen dataset. The exported union audit is `tables/R3_CELL_SET_IDENTITY_DIFF.csv`; rows marked `same` are members of both sets, while non-`same` rows would be the identity differences.

The correct wording is: the bootstrap-sign criterion is empirically redundant for this frozen dataset at this threshold. This does not establish redundancy in general.

## F5 nested ledger decision

Because the cell identities are identical, F5 may use the single nested ledger `5420 -> 902 -> 712 -> 692`. The nested data are in `tables/R3_F5_NESTED_LEDGER.csv` when identity is exact; otherwise the separate-estimand output would be `tables/R3_F5_SEPARATE_ESTIMANDS.csv`.

At `abs(rho_IC), abs(rho_dPL) >= 0.20`, the strong-both count is 712; among those, 692 have the same sign and 20 are sign-flipped.

## Bidirectional conditioning

`tables/R3_F5_BIDIRECTIONAL_CONDITIONING.csv` contains both conditional directions and the complete four-way counts. The exact conditional probabilities are:

- `P(dPL strong | IC strong) = 712/902 = 0.789356984479`.
- `P(IC strong | dPL strong) = 712/2163 = 0.329172445677`.

The reverse conditional is not a validity or effectiveness claim. dPL parameter values are generated through attribute-to-parameter mapping, so denser attribute–parameter associations can have a constructive component. The two directions describe association-system asymmetry only.

## Provenance

- Relationship source: `tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`.
- IC selection columns: `rho`, `bootstrap_sign_probability`; R3 frozen thresholds 0.20 and 0.95.
- dPL is used only for classification after IC set construction.
- Source table rows: 5420 information-cluster cells.
