# S2 Equations Candidate

This candidate reports the implemented equations. `P_t`, `T_t`, and `PET_t` are the daily tensors accepted by the shared model interface. All `max`, `min`, `clamp`, `where`, and epsilon terms below are implementation operations, not editorial simplifications.

### S2.1 XAJ

#### Inputs and states

States are `WU, WL, WD, S, FR, QI, QG` and a 14-sample surface-runoff UH buffer. Initial values are half of the corresponding capacities for WU/WL/WD/S and zero for the remaining states (models/xaj.py:467-493).

#### Daily flux equations

`prcp=max(P_t,0)`, `PET_a=max(k PET_t,0)`. The code computes `EU`, `EL`, and `ED` in the nested XAJ branches, with `ED=min(ED,WD)`. Let `W_0=min(WU+WL+WD,WM-eps)`, `WM=UM+LM+DM`, `PE=max(prcp-EU-EL-ED,0)`, `WMM=WM(1+B)`, and `A=WMM[1-(1-W_0/(WM+eps))^(1/(1+B))]`. The code's piecewise expression then computes `R`, followed by `R_I=KI S FR` and `R_G=KG S FR`.

#### State updates and routing

The code updates WU/WL/WD with its branch-specific formulas, updates FR and S with storage limits, then updates `QI=CI QI_old+(1-CI)R_I(1-IM)` and `QG=CG QG_old+(1-CG)R_G(1-IM)`. Surface input is `RS_adj=RS(1-IM)+IM PE`. It is routed through the finite 15-ordinate gamma UH; final output is `Q=RS_routed+QI+QG` (models/xaj.py:118-171, 308-342).

#### Parameters

The 14 active XAJ parameters and bounds are in `results/s2_parameter_manifest.csv`.

### S2.2 GR4J

#### Inputs and states

States are production store `S_prod`, routing store `S_route`, UH1 buffer length 15, and UH2 buffer length 30. Defaults are `0.5 X1`, `0.5 X3`, and zero buffers (models/gr4j.py:144-190).

#### Daily flux equations

The implementation branches on `P_t>=PET_t`, computes `P_N`, `E_N`, `P_S`, `E_S` with tanh equations, updates `S_prod`, and computes percolation `Perc=S_prod[1-(1+(4S_prod/(9X1))^4)^(-1/4)]`. `P_R=clamp(Perc+P_N-P_S,0,inf)` is split as `0.9P_R` and `0.1P_R` into UH1 and UH2 (models/gr4j.py:17-77).

#### State updates and routing

UH ordinates are S-curve differences, normalized with `eps=1e-8`, using x4 and finite lengths 15/30. The exchange term is `F=X2(S_route/(X3+eps))^3.5`; the routing store receives UH1 and F, produces `Q_R`, while `Q_D=max(UH2+F,0)` and `Q=Q_R+Q_D` (models/gr4j.py:78-108; models/unit_hydro.py:18-71).

### S2.3 SIMHYD

`I=min(INSC_safe,PET_safe,P)`, `I_f=min(COEFF_safe exp(-SQ soil_ratio),P-I)`, `R_D=P-I-I_f`, `R_I=SUB soil_ratio I_f`, `R_G=CRAK soil_ratio(I_f-R_I)`. Soil ET, overflow, groundwater and runoff updates follow models/simhyd.py:52-118. Instant runoff is routed by a finite normalized gamma UH with a 14-sample continuation buffer (models/simhyd.py:165-200).

### S2.4 HBV reference

HBV uses states SNOWPACK, MELTWATER, SM, SUZ, SLZ. Rain/snow is a hard threshold at `parTT`; melt is `min(max(CFMAX(T-TT),0),SNOWPACK)`; refreezing and snow liquid retention follow the exact order in models/hbv.py:30-71. The soil and upper/lower-zone equations are also implemented there. It is a standalone registry model, not one of the CN/TGD wrappers.

### S2.5 Module coupling

For CN, `effective_precip=rain+melt` from `G/eTG`; for TGD, `effective_precip=(1-alpha)P+release` from delay storage `S`; PET is passed unchanged. Each wrapper performs module update then host update on the same day (models/composed.py:46-53; models/composed_temperature_delay.py:32-110).

### S2.6 Differentiability and smoothing

TGD uses `tanh(clamp(z,-5,5))` and `-expm1(-1/tau_t)`. Other modules retain hard `where`/`min`/`max` branches but add eps denominators, capacity clamps and UH normalization. The exact inventory is `results/s2_threshold_and_smoothing_inventory.csv`.

### S2.7 Mass balance and reference verification

TGD preprocessing has a per-step residual of `S_old+P-effective-S_new`; CN has the analogous `P-effective-G_new` diagnostic over a zero-initial snow store. Host-wide balances are only claimed where current auxiliaries expose all terms; see `results/s2_mass_balance_results.csv`.

### S2.8 Parameter bounds

See `results/s2_parameter_manifest.csv`; `tgd_tau` uses log interpolation in `ablation/ic_core/parameter_adapter.py:56-67`, all other active physical parameters use linear interpolation.

### S2.9 HBV snow-process reference

The HBV reference is the implemented explicit snow routine above. It should be described as a model reference, not as evidence that CN and HBV are mathematically identical.
