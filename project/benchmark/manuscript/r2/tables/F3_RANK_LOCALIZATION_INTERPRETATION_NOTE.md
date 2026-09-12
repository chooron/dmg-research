# F3 rank/localization interpretation note

DECOMPOSITION LEVEL:

Top-k localization is decomposed at the model × basin level.  For each basin,
normalized squared coordinate contributions are ranked within that basin.  The
resulting C_eff and cumulative top-k shares are first summarized across basins
within each model and only then across the 36 models.  There is no frozen
model-level total vector whose coordinate identity is ranked once across all
basins.

WHAT CAN BE SAID:

High basin-wise top-1/top-2 shares can coexist with only moderate coordinate-wise
cross-catchment rank correspondence because the identity of the largest
contributors can vary among basins.  In that situation, displacement is
concentrated within individual basin decompositions while the rank ordering of
any fixed coordinate across catchments is only partly retained.  This is a
coherent descriptive reading of the two estimands, not a contradiction.

The rank atlas measures tie-corrected Spearman correspondence for each native
model-coordinate across the 531 basins.  The localization summaries measure
concentration of squared normalized displacement within each basin.  They use
different reductions and should be read side by side without treating either as
an explanation of the other.

WHAT CANNOT BE SAID:

Figure 3 does not test whether coordinates with larger displacement are the
same coordinates with lower rank correspondence.

Neither top-k share nor C_eff establishes parameter importance, sensitivity,
physical dominance, mechanism, or a model-family/functional-role claim.
The strict23 rows remain reference-qualified descriptive checks and cannot be
upgraded to a full36 inference.  No causal relation between rank loss and
localization is tested here.
