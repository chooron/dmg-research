# Table S2.8. Definitions, implemented bounds, units, and transformations of active parameters

All bounds below are from the active project code; they are not literature recommendation ranges.

## Panel A. XAJ

| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |
|---|---|---|---:|---:|---|---|
| k | `xaj_k` | ratio of potential ET to reference crop evaporation | 0.5 | 2.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| B | `xaj_b` | tension-water capacity-curve exponent | 0.1 | 2.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| IM | `xaj_im` | impervious-area fraction | 0.0 | 0.3 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| UM | `xaj_um` | upper tension-water capacity | 5.0 | 50.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| LM | `xaj_lm` | lower tension-water capacity | 20.0 | 200.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| DM | `xaj_dm` | deep tension-water capacity | 20.0 | 200.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| C | `xaj_c` | deep-layer evaporation coefficient | 0.05 | 0.3 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| SM | `xaj_sm` | areal mean free-water capacity | 5.0 | 100.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| EX | `xaj_ex` | free-water capacity-curve exponent | 0.1 | 2.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| KI | `xaj_ki` | daily interflow outflow coefficient multiplying free-water state | 0.0 | 0.7 | 1/day in the daily implementation | linear normalized-to-physical; runtime: no individual clamp; joint rescaling when KI+KG>=1 |
| KG | `xaj_kg` | daily groundwater outflow coefficient multiplying free-water state | 0.0 | 0.7 | 1/day in the daily implementation | linear normalized-to-physical; runtime: no individual clamp; joint rescaling when KI+KG>=1 |
| CI | `xaj_ci` | interflow output-state memory/recession coefficient | 0.1 | 1.0 | dimensionless daily coefficient | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| CG | `xaj_cg` | groundwater output-state memory/recession coefficient | 0.9 | 1.0 | dimensionless daily coefficient | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| a_UH | `xaj_a` | Gamma UH shape parameter before hydrodl2 offset | 0.0 | 2.9 | dimensionless | linear normalized-to-physical; runtime: hydrodl2 uh_gamma: relu(a)+0.1 |
| theta_UH | `xaj_theta` | Gamma UH scale parameter | 0.0 | 6.5 | day | linear normalized-to-physical; runtime: hydrodl2 uh_gamma: relu(theta)+0.5 |
## Panel B. GR4J

| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |
|---|---|---|---:|---:|---|---|
| X1 | `x1` | GR4J production-store capacity | 10.0 | 1200.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| X2 | `x2` | GR4J groundwater exchange coefficient | -5.0 | 3.0 | mm/day in the project daily equation | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| X3 | `x3` | GR4J routing-store capacity | 20.0 | 5000.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| X4 | `x4` | GR4J unit-hydrograph time base | 1.1 | 10.0 | day | linear normalized-to-physical; runtime: compute_gr4j_uh_ordinates: max(X4,1e-3) |
## Panel C. SIMHYD

| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |
|---|---|---|---:|---:|---|---|
| INSC | `simhyd_insc` | interception capacity | 1e-06 | 50.0 | mm | linear normalized-to-physical; runtime: max(INSC,1e-6) |
| COEFF | `simhyd_coeff` | maximum infiltration capacity coefficient | 1e-06 | 400.0 | mm/day in the project equation | linear normalized-to-physical; runtime: max(COEFF,1e-6) |
| SQ | `simhyd_sq` | infiltration capacity exponent | 0.0 | 10.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| SMSC | `simhyd_smsc` | soil-moisture storage capacity | 1.0 | 1000.0 | mm | linear normalized-to-physical; runtime: max(SMSC,1e-6) |
| SUB | `simhyd_sub` | interflow proportionality coefficient | 0.0 | 1.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| CRAK | `simhyd_crak` | groundwater recharge proportionality coefficient | 0.0 | 1.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| K | `simhyd_k` | groundwater recession coefficient | 0.0 | 1.0 | 1/day daily fraction | linear normalized-to-physical; runtime: clamp(K,0,1) |
| ETMUL | `simhyd_etmul` | PET multiplier | 0.1 | 3.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| a_UH | `simhyd_a` | Gamma UH shape parameter before hydrodl2 offset | 0.0 | 2.9 | dimensionless | linear normalized-to-physical; runtime: hydrodl2 uh_gamma: relu(a)+0.1 |
| theta_UH | `simhyd_theta` | Gamma UH scale parameter | 0.0 | 6.5 | day | linear normalized-to-physical; runtime: hydrodl2 uh_gamma: relu(theta)+0.5 |
## Panel D. HBV

| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |
|---|---|---|---:|---:|---|---|
| BETA | `parBETA` | soil-moisture control exponent for recharge | 1.0 | 6.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| FC | `parFC` | field capacity of soil store | 50.0 | 1000.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| K0 | `parK0` | upper-zone quick-flow recession coefficient | 0.05 | 0.9 | 1/day | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| K1 | `parK1` | upper-zone interflow recession coefficient | 0.01 | 0.5 | 1/day | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| K2 | `parK2` | lower-zone baseflow recession coefficient | 0.001 | 0.2 | 1/day | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| LP | `parLP` | fraction of FC controlling PET reduction | 0.2 | 1.0 | dimensionless fraction of FC | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| PERC | `parPERC` | maximum percolation rate from upper to lower zone | 0.0 | 10.0 | mm/day | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| UZL | `parUZL` | upper-zone threshold for quick flow | 0.0 | 100.0 | mm | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| TT | `parTT` | rain-snow threshold temperature | -2.5 | 2.5 | degC | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| CFMAX | `parCFMAX` | degree-day melt factor | 0.5 | 10.0 | mm/(degC*day) | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| CFR | `parCFR` | refreezing coefficient | 0.0 | 0.1 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| CWH | `parCWH` | snowpack liquid-water holding coefficient | 0.0 | 0.2 | dimensionless fraction | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
## Panel E. CN

| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |
|---|---|---|---:|---:|---|---|
| CTG | `cn_ctg` | snowpack thermal-state memory coefficient | 0.0 | 1.0 | dimensionless | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
| Kf | `cn_kf` | CemaNeige degree-day melt factor | 0.0 | 10.0 | mm/(degC*day) | linear normalized-to-physical; runtime: none on parameter; derived expressions have state/flux min/max |
## Panel F. TGD

| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |
|---|---|---|---:|---:|---|---|
| alpha | `tgd_alpha` | fraction of precipitation entering generic delay storage | 0.0 | 1.0 | dimensionless fraction | linear normalized-to-physical; runtime: clamp(alpha,0,1) |
| tau | `tgd_tau` | baseline generic-delay release timescale | 0.001 | 90.0 | day | log interpolation; runtime: clamp(tau,1e-6,3650); dynamic tau clamped again |
| beta | `tgd_beta` | temperature sensitivity of release timescale | -4.0 | 4.0 | dimensionless | linear normalized-to-physical; runtime: none on beta itself; enters exp(-beta*tanh(...)) |

Table note: dPL first generates a sigmoid-normalized value and maps it to the physical range. IC-XNES uses the same physical bounds. `tgd_tau` uses logarithmic interpolation; other active parameters use linear interpolation. Runtime clamps are not calibration bounds. Gamma UH shape and scale have hydrodl2 positive offsets. KI/KG have a joint sum constraint. `parCWH` retains the code metadata/equation conflict recorded in the audit.
