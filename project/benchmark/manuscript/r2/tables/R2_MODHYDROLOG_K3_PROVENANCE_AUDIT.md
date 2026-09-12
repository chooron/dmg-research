# modhydrolog:k3 provenance audit

## Verdict

```text
MODHYDROLOG_K3_VERDICT = CURRENT CANONICAL
```

## Frozen source and formulas

The frozen normalized source array is:

```text
results/joh_direct_parameter_change_diagnostic_20260905/r2/cache/modhydrolog_normalized_parameter_matrices.npz
```

It contains `IC`, `dPL`, `DeltaTheta`, and `basin_ids`. The `IC` and `dPL` arrays are `(531, 15)`; coordinate `14` is `k3` and the basin count is 531. `DeltaTheta[:,14]` equals `dPL[:,14] - IC[:,14]` exactly.

For this coordinate the canonical diagnostics use:

```text
M = median_b |theta_dPL_norm - theta_IC_norm|
R = Spearman(theta_IC_norm, theta_dPL_norm)
```

The F4 audit builder computes `M` as the median absolute frozen coordinate difference. The canonical rank input is the frozen coordinate-wise Spearman result in `cache/fig2c_rank_correspondence.parquet`; the independent recomputation below applies the same tie-corrected Spearman definition directly to the frozen arrays.

## Independent one-coordinate recomputation

Using only `IC[:,14]`, `dPL[:,14]`, and the 531 aligned basin rows from the frozen NPZ source:

```text
M = 0.3880192475023052
R = 0.016289203982133137
max |DeltaTheta - (dPL - IC)| = 0.0
```

The coordinate ranges are finite and bound-normalized:

```text
IC range  = [4.390533681936669e-15, 1.0]
dPL range = [1.913992946356302e-06, 0.972598433494568]
```

## Cross-artifact comparison

| Artifact | M | R | Result |
|---|---:|---:|---|
| `tables/F4_ALL_COORDINATE_DIAGNOSTICS.csv` | 0.388019247502 | 0.0162892039821 | agrees |
| `tables/F4_CANDIDATE_POOL.csv` (`C14`) | 0.388019247502 | 0.0162892039821 | agrees |
| `tables/F4_CANDIDATE_RANKINGS.csv` | 0.388019247502 | 0.0162892039821 | agrees |
| `tables/F4_EXEMPLAR_SELECTION_AUDIT.md` | 0.388 (rounded) | 0.016 (rounded) | agrees |
| `scripts/plot_r2_figure4_exemplars.py` | reads the F4 diagnostics row for the badge | reads the F4 diagnostics row | agrees |
| older expected value cited for blocker resolution | 0.3879 | 0.0164 | not present in the current R2 repository |

A repository-wide search of the R2 materials found no earlier artifact containing the exact pair `0.3879 / 0.0164`. The current canonical table, candidate pool, ranking table, frozen arrays, and plotting input are internally consistent. The older pair is therefore not authoritative and is not an alternative aggregation supported by the current source tree.

## Authoritative values and synchronization

The authoritative manuscript values are:

```text
M = 0.3880192475023052  (display: 0.3880 or 0.388)
R = 0.016289203982133137 (display: 0.0163 or 0.016)
```

Synchronized artifacts:

- `tables/F4_ALL_COORDINATE_DIAGNOSTICS.csv` — already canonical;
- `tables/F4_CANDIDATE_POOL.csv` — already canonical;
- `tables/F4_CANDIDATE_RANKINGS.csv` — already canonical;
- `tables/F4_EXEMPLAR_SELECTION_AUDIT.md` — already consistent at displayed precision;
- `scripts/plot_r2_figure4_exemplars.py` — already reads the canonical row and displays three decimals.

No numerical table repair or new analysis was required.
