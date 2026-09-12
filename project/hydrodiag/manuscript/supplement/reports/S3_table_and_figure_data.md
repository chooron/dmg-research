# S3 table and figure data

## Table S3.1

| Host | Structure | Dimension | Population | Initial width | Learning rate | Generations | Evaluations | Restarts | Initial center | Stopping rule | Seed protocol |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| XAJ | XAJ | 15 | 48 | 0.25 | center=1.0; covariance=0.6*(3+log(D))/(D*sqrt(D)) | 400 | 19200 | 3 | deterministic LHS per basin/start | fixed generations; no early stop | 0 |
| XAJ | XAJ_TGD | 18 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| XAJ | XAJ_CN | 17 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| GR4J | GR4J | 4 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| GR4J | GR4J_TGD | 7 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| GR4J | GR4J_CN | 6 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| SIMHYD | SIMHYD | 10 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| SIMHYD | SIMHYD_TGD | 13 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| SIMHYD | SIMHYD_CN | 12 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| HBV | HBV | 13 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| XAJ | XAJ_PD | 17 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| GR4J | GR4J_PD | 6 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| SIMHYD | SIMHYD_PD | 12 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |

## Figure S3.1

No active foundation-531 optimizer scan was found. Existing large-scale scans are retained in the CSV with `LEGACY_NOT_ACTIVE` status.

**Caption draft.** Optimizer calibration evidence available in the repository. Foundation-531 calibration is unresolved; legacy screening scans use a different 1988-1998 date protocol and are shown only as provenance.

## Figure S3.2

No formal foundation-531 restart or budget trace was found. Screening traces are not promoted to 531 production evidence.

**Caption draft.** Estimation adequacy diagnostics. Foundation-531 restart dispersion, budget saturation, and convergence coverage are unavailable from the current result inventory; screening traces are retained for audit traceability only.

## Objective

`KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`; dPL minimizes the mean of `1-KGE` over valid sampled basin windows. The IC implementation has no epsilon in the GPU KGE formula; dPL uses epsilon-stabilized differentiable KGE.

## Protocol

Foundation 531 dates are warmup 1980-10-01 to 1981-09-30, calibration 1981-10-01 to 1995-09-30, evaluation 1995-10-01 to 2010-09-30. Both routes are configured around 365-day warm-up/prediction windows, but exact shared evaluation equivalence remains unresolved.
