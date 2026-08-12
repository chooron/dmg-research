# Model Source Notes

## Model Sources

### HBV
- **Source URL**: https://github.com/mhpi/hydrodl2/blob/master/src/hydrodl2/models/hbv/hbv.py
- **Reference paper**: Feng, D. et al. (2022). WRR, 58, e2022WR032404.
- **Formulas**: All formulas from the reference implementation.
- **Note**: HBV includes its own snow module (degree-day melt, refreezing, water holding
  capacity). No external CemaNeige needed.

### GR4J
- **Source URL**: https://github.com/kratzert/RRMPG/blob/master/rrmpg/models/gr4j_model.py
- **Reference paper**: Perrin, C. et al. (2003). J. Hydrol., 279(1), 275-289.
- **Formulas**: Production store (eq. 3, 4), percolation (eq. 5), unit hydrographs
  (eq. 16, 17), groundwater exchange (eq. 18).
- **Unit hydrograph implementation**: Differentiable S-curve difference method, inspired
  by dmotpy DplUHBase pattern (reference only, not imported). Ordinates computed as
  S(t)-S(t-1) with tensor operations; x4 is fully differentiable. Sequential UH buffer
  updates occur within the compiled step kernel. A standalone `torch.nn.functional.conv1d`
  -based routing function is also available in `models/unit_hydro.py`.
- **Reference consulted**: `/home/jingxin/code/dmg-research/dmotpy/models/unithydro/`
  (pattern only; dmotpy is NOT imported or depended upon).

### XAJ (XinAnJiang)
- **Source URL**: https://github.com/OuyangWenyu/hydromodel/blob/master/hydromodel/models/xaj.py
- **Reference book**: 《水文预报》(Hydrologic Forecasting), 5th edition.
- **Formulas retained**: Three-layer evaporation, tension water capacity curve, free water
  reservoir separation (surface / interflow / groundwater), linear reservoir routing
  (interflow + groundwater), channel lag routing.
- **Routing**: Does not include Muskingum routing. The hydromodel reference treats
  routing (CSL/MZ) as a configurable external module, not as a defining part of the
  core XAJ runoff-generation structure. Channel lag routing is used instead.
- **Excluded**: `sources5mm` division method, CSL/MZ routing methods.
- **Adjustments**: NumPy/scipy operations converted to PyTorch. Lag routing uses
  fixed-size history buffer (max 15 days). HF variant only (not EH).

### CemaNeige
- **Source URL**: https://github.com/kratzert/RRMPG/blob/master/rrmpg/models/cemaneige_model.py
- **Reference**: Valery, A. (2010). PhD thesis. Cemagref.
- **Parameters**: CTG and Kf only. The snow-cover threshold is fixed as
  `0.9 * 365.25 * mean(solid_precipitation)`.
- **Formulas**: Solid/liquid fraction, thermal state update, degree-day melt,
  instantaneous SCA ratio, and actual melt factor.
- **Adjustments**: Single elevation zone and on-the-fly solid-fraction calculation;
  the external `frac_solid_prec` input is replaced by the project's tensor forcing
  interface.

### CemaNeigeHyst
- **Source URL**: https://github.com/kratzert/RRMPG/blob/master/rrmpg/models/cemaneigehyst_model.py
- **Reference**: Valery, A. (2010). PhD thesis. Cemagref. Riboust, P. et al. (2019).
  J. Hydrol. Hydromech., 67, 70-81.
- **Parameters**: CTG, Kf, Thacc, and Rsp.
- **Formulas**: Solid/liquid fraction, thermal state update, degree-day melt, SCA
  hysteresis (accumulation/ablation phases), and actual melt factor.
- **Adjustments**: Single elevation zone, simplified solid fraction calculation on the
  fly, and Psolannual estimated from forcing statistics.

### PrecipitationDelay placebo control
- **Reference**: This is an internal control module, not a snow model or a direct
  implementation from RRMPG. It is designed to match the two free parameters of
  the basic CemaNeige composition.
- **Parameters**: `pd_alpha` is the fraction of precipitation entering temporary
  storage (`[0, 1]`); `pd_tau` is the release time scale in days (`[1e-3, 90]`).
- **Update**: `S_pre = S[t-1] + alpha * P[t]`,
  `R[t] = (1 - exp(-1/tau)) * S_pre`,
  `S[t] = S_pre - R[t]`, and
  `P_star[t] = (1-alpha) * P[t] + R[t]`.
- **Purpose**: `P_star` replaces raw precipitation before GR4J, XAJ, or SIMHYD.
  The module uses no temperature or snow variables, conserves water exactly, and
  reaches the original-model limit at `alpha = 0`.

## Parameter Ranges

### Ranges from Source Implementations
- **GR4J**: x1[10,1200], x2[-5,3], x3[20,5000], x4[1.1,10] — from RRMPG
- **CemaNeige**: CTG[0,1], Kf[0,10] — from RRMPG
- **CemaNeigeHyst**: CTG[0,1], Kf[0,10], Thacc[0,1000], Rsp[0,1] — from RRMPG

### Provisional Ranges (暂定)
- **HBV**: All 12 parameters from hydrodl2 reference. Validated against HBV-light manual.
- **XAJ**: All 15 parameters from hydromodel reference (removed `xaj_cs` since Muskingum
  routing is not used). The bounds for `xaj_lag` are provisional — the source allows lag
  up to 10 days, but in practice it may be larger for large basins.

## Alignment Note

This project does NOT align with MARRMoT, dmotpy, or pymarrmot. All implementations
are independently derived from the original publications and the specified GitHub
source implementations. dmotpy's unit hydrograph pattern was consulted as an
implementation reference only; no dmotpy code is imported or depended upon.

## Unit Hydrograph Implementation Note

The GR4J unit hydrograph ordinate computation in `models/unit_hydro.py` follows the
dmotpy DplUHBase S-curve difference pattern:

1. Compute S-curve values at integer time points t=1,2,...,max_len
2. UH ordinates = S(t) - S(t-1) (central difference)
3. Normalize ordinates to sum to 1
4. UH routing via `F.conv1d` with `groups=batch_size`

This pattern has been verified as equivalent to the original GR4J S-curve formulation
by the dmotpy project. Our implementation is an independent re-implementation.
