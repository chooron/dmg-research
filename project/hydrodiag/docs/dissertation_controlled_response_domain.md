# Dissertation controlled finite response domain

This record supersedes the previous pre-Phase-0 attempt to approximate legacy
native `C=1` endpoints with a finite `tau_0`. It does not modify legacy XAJ or
historical calibration results.

## Legacy/public XAJ

`XAJ_PARAM_SPECS` and the public `XAJ` class retain the historical contract:

```text
CI in [0.1, 1.0]
CG in [0.9, 1.0]
```

The previous endpoint audit remains at:

```text
results/pre_phase0_native_scale/endpoint_audit_summary.json
```

With the native default `QI0=QG0=0.1`, `C=1` is a singular
Q-to-latent-storage inversion/infinite-timescale initialization limit. It is
not part of the new controlled dissertation comparison domain.

## Controlled native N domain

The new controlled native reference uses exactly the native XAJ equations but a
variant-specific parameter specification:

```text
CI in [0.1, 0.9]
CG in [0.9, 0.998]
```

The classes are:

```text
XAJControlledN
XAJControlledNLite
```

They are exported for direct controlled forward/tests only and are not added
to training registries or experiment configurations. `XAJDE` and `XAJGE` use
the same controlled CI/CG specification because their response subsystem is
native. Legacy `XAJ`, `XAJLite`, XAJ+CemaNeige and XAJ+TGD2 continue to use
their original specs.

## Controlled tau0 range

For the common finite native response envelope:

```text
tau_min = -1 / log(0.1)   = 0.43429448190325187 day
tau_max = -1 / log(0.998) = 499.49983316645478 day
```

`tau_0` retains log mapping. `beta=[0.5,2.0]` and the fixed numerical
reference scale remain unchanged:

```text
NATIVE_XAJ_LATENT_Z0 = 3.1553493591016335
```

The Z0 value is a robust numerical reference scale derived from finite native
response realizations. It conditions the nonlinear storage-discharge
parameterization; it is not a physical estimate of basin groundwater storage.
The earlier 432-basin derivation remains the source of this numerical scale;
it is not claimed to represent all 531 basins.

Historical calibration results are not retroactively interpreted as having used
the controlled bounds.
