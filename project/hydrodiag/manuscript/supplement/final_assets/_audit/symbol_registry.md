# Supplement symbol registry

This audit aligns the final Supplement asset package with the production-truth equations in `manuscript/methods_supplement_production_audit.md` and the current Figure 2 source `manuscript/scripts/r1/plot_r1_figure2.py`. Figure 2 is not modified in this pass.

## Forcing and upstream interface

| Formal display symbol | Figure 2 / Methods source | Production code or data name | Meaning | Status |
|---|---|---|---|---|
| `P_t` | Methods P03, Section 3.2 | forcing precipitation / `precip_t` | Daily precipitation forcing, mm d⁻¹ | CONFIRMED |
| `T_t` | Methods P03, TGD/CN equations | forcing temperature / `temp` | Daily mean temperature, °C | CONFIRMED |
| `E_{p,t}` | Methods P03 and Stage 1 | forcing PET, sometimes legacy `PET`/`pet_t` in audit columns | Dataset-supplied potential evapotranspiration, before host scaling | CONFIRMED; display as `E_{p,t}` only |
| `E_{p,t}^{adj}` | Methods Stage 1 | `pet_adj` | Host-scaled PET, `max(k E_{p,t}, 0)` | CONFIRMED |
| `P_t^*` | Methods upstream equations | `effective_precipitation`, `effective`, or equivalent model input | Effective liquid-water input supplied to the XAJ host | CONFIRMED |

Base uses `P_t^{*,Base} = P_t`. TGD uses `P_t^{*,TGD}` after generic temperature-conditioned memory. CN uses `P_t^{*,CN} = P_t^r + M_t` after explicit snow accumulation and melt.

## TGD and CN upstream states

| Formal display symbol | Production code / source key | Meaning | Status |
|---|---|---|---|
| `S_t^g` | TGD2 code `storage`, output key `final_states.storage` | TGD generic precipitation-memory storage | CONFIRMED |
| `A_t` | Methods XAJ Stage 2 | Tension-water capacity curve evaluated before runoff generation | CONFIRMED in equations; local XAJ code spelling is implementation-specific |
| `\tau_t` | TGD2 output `tgd2_tau`; response CSV `tau_*` | Temperature-conditioned residence time | CONFIRMED; use `\tau_t` in Supplement displays, not `\tau(T)` |
| `r_t` | TGD2 output `tgd2_retention`; response CSV `retention_*` | Daily retention factor, `exp(-1/\tau_t)` in code | CONFIRMED |
| `T_{ref}` | `TGD2_T_REF_C` | Fixed TGD gate reference temperature, 0 °C | CONFIRMED |
| `s_T` | `TGD2_T_SCALE_C` | Fixed TGD gate temperature scale, 2 °C | CONFIRMED |
| `f_{solid,t}` | CN/CemaNeige solid fraction | Fraction of precipitation treated as solid | CONFIRMED |
| `P_t^s`, `P_t^r` | CemaNeige solid/rain components | Solid and rain precipitation components | CONFIRMED |
| `G_t` | CemaNeige snow water equivalent state | Snowpack water equivalent | CONFIRMED |
| `eTG_t` | CemaNeige thermal state | Snowpack thermal state | CONFIRMED |
| `M_t` | CemaNeige melt output | Melt contribution to effective input | CONFIRMED |
| `C_{TG}` | `cn_ctg` | Snowpack thermal weighting coefficient | CONFIRMED |
| `K_f` | `cn_kf` | Degree-day melt factor | CONFIRMED |

## XAJ host states and fluxes

| Formal display symbol | Production code / CSV key | Meaning | Status |
|---|---|---|---|
| `W_{U,t}` | `wu`, `wu_new` | Upper-layer tension-water storage | CONFIRMED |
| `W_{L,t}` | `wl`, `wl_new` | Lower-layer tension-water storage | CONFIRMED |
| `W_{D,t}` | `wd`, `wd_new` | Deep-layer tension-water storage | CONFIRMED |
| `W_t` or `W_t^{tot}` | `wt` or sum `wu + wl + wd` | Total tension-water storage | CONFIRMED; use `W_t` in final Figure S3/S4 displays |
| `Q_{i,t}` | `qi`, `qi_t`, `qi_store` | Interflow routing flux/state | CONFIRMED |
| `Q_{g,t}` | `qg`, `qg_t`, `qg_store` | Groundwater routing flux/state | CONFIRMED |
| `Q_t` | `qsim`, `Q` | Total simulated outlet discharge | CONFIRMED |
| `R_t` | local runoff generation variable | Excess rainfall/runoff before source separation | CONFIRMED in Methods; not used as a final displayed state unless source data provides it |
| `FR_t`, `SS_t`, `AU_t`, `RS_t`, `RI_t`, `RG_t` | implementation locals such as `fr`, `s_next`, `rs`, `qi`, `qg` | XAJ free-water and source-separation intermediates | PARTIAL; do not invent a display mapping where a source column is absent |

The production equation is `W_t = W_{U,t} + W_{L,t} + W_{D,t}`. Final Figure S3 therefore uses only the canonical keys `wu`, `wl`, `wd`, `wt`, `qi`, and `qg`; the legacy component renderer's `s` row is not promoted as a substitute for `W_t`.

## Figure 2 conventions and conflicts

- Figure 2 strata are `S1`–`S5` by `frac_snow`, with counts 165, 156, 121, 34, and 55.
- Formal Methods equations use `P_t`, `T_t`, `E_{p,t}`, and `P_t^*`; any source column named `PET` or `pet` is an implementation alias, not the manuscript display symbol.
- Figure 2 uses Base orange (`#D55E00`), TGD green (`#009E73`), and CN blue (`#0072B2`). Its panel-(a) IC/dPL diagnostic colors are a separate visual layer. Final Supplement figures retain the structure palette and encode IC/dPL with marker fill/shape or line style rather than introducing extra hues.
- The Figure 2 implementation and the formal equation audit use different local spellings for several XAJ intermediates (`pet_adj`, `s_next`, `rs`, `qi`, `qg`). This is an implementation-name difference, not evidence of a changed scientific definition.
- The formal audit describes `\tau_t` and `r_t`, whereas some existing plot titles and CSV labels use `\tau(T)` and `retention`. Final Figure S2 displays the formal time-indexed symbols where applicable.
- No complete one-to-one display mapping was found for every local XAJ intermediate (`S_mm`, `S_0`, `O_i`, `O_g`, `R_i`, `R_g`) across Figure 2, equations, and production code. These are marked `UNRESOLVED_SYMBOL_MAPPING` and are not used as invented labels in final assets.

## Source references

- `manuscript/scripts/r1/plot_r1_figure2.py`
- `manuscript/methods_supplement_production_audit.md`, Sections 3.1–3.7
- `models/parameter_specs.py`
- `models/tgd2.py`
- `models/xaj.py`
- `results/r3_misspec_analysis_v1/state_excess.csv`
- `manuscript/scripts/r3/protocol_misspec_v1.json`
