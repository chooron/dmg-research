# Pre-Phase-0 subsurface-response numerical freeze

This record freezes only numerical semantics and scale. It does not start a
calibration, Phase 0 experiment, dPL run, or 531-basin structure comparison.

## Native source and Z0

The source is the completed native `XAJ` IC result:

```text
results/xaj_base_cmaes_531_batched_paired_v2/raw/xaj/
```

There are 5,310 completed records (531 basins x 10 starts). For each basin the
existing stored `train_objective` was used to select one already-calibrated
parameter vector; no optimizer was called. Native XAJ was then run by its
existing compact step kernel over the existing CAMELS forcing.

For finite `CI, CG < 1`, the latent native response storage is:

```text
Z_I = CI / (1-CI) * QI
Z_G = CG / (1-CG) * QG
Z_N = Z_I + Z_G
```

The two-level equal-basin center is:

```text
m_i    = median_t(log(Z_N,i,t)), Z_N,i,t > 0
logZ0  = median_i(m_i)
Z0     = exp(logZ0)
```

The audit found 432 eligible finite basins and 99 basins with `CI==1` or
`CG==1`. The latter are a singular no-recession endpoint: their latent
storage is infinite for nonzero Q and they are not silently epsilon-clamped.
Every eligible basin had 12,418 positive storage samples.

Frozen fixed scalar:

```text
Z0 = 3.1553493591016335
```

This is a pre-Phase-0 native-XAJ latent-storage scaling constant, not a fitted
parameter and not a basin-specific value. The complete distribution is in:

```text
results/pre_phase0_native_scale/native_latent_scale_summary.json
```

## Native recursion equivalence

Native XAJ uses:

```text
Q_t = C Q_(t-1) + (1-C) R_t
```

For `C < 1`, define:

```text
Z_(t-1) = C/(1-C) Q_(t-1)
Z_a     = Z_(t-1) + R_t
Z_t     = C Z_a
Q_t     = (1-C) Z_a
```

This is the explicit latent-storage realization used by the D_R/G_R linear
semantics. The equivalent time scale is:

```text
tau = -dt / log(C), dt = 1 day
```

The exact `C=1` endpoint remains singular and is not represented by a finite
tau.

## tau0 range

Native parameter bounds are:

```text
CI in [0.1, 1.0] -> tau in [0.43429448190325187, infinity)
CG in [0.9, 1.0] -> tau in [9.491221581029905, infinity)
```

Among the 432 finite selected native basins, the largest observed finite tau
was `14005.203764512573` days (CG). The frozen finite operational envelope
uses the smallest native lower bound and a 0.1 log-scale safety margin above
that observed maximum:

```text
tau0 in [0.43429448190325187, 15478.143902262878] days
```

This range is not a claim that the singular native `C=1` point has a finite
equivalent tau. `beta` remains `[0.5, 2.0]` with log mapping.

## Diagnostics

G_R full XAJ diagnostics now expose:

- `z_available`
- `z`
- `extinction_mask`
- `log_z_ratio = log(z_available / Z0)` on positive available storage
- extinction count and positive-available count
- `f_extinct`
- mean, standard deviation, median, IQR, p05 and p95 of `log_z_ratio`

Invalid log-ratio entries are represented as zero together with the positive
availability mask; they are not included in the summaries. No basin filtering
is performed from these diagnostics.
