# S3 claims and limitations

## Fully supported

- Foundation 531 data contract: 531 basins, 35 attributes, explicit periods, units, and manifest fingerprints.
- XNES is the active optimizer implementation in the active stage-1 design; coordinates are normalized and clipped before physical mapping.
- The dPL architecture, mapping, optimizer, schedule, sampler, and launcher seed protocol (42, 123, 2026) are code/config verified.

## Supported with qualification

- “Independent estimation used multiple restarts” applies to the XAJ screening design, not a verified full 531 production inventory.
- “The routes target KGE(Q)” is supported; “exactly the same objective” is not, because implementations differ.
- “The routes use the foundation date protocol” is configured, while complete state/mask/evaluation equivalence remains untested.
- “dPL uses one parameter vector per basin per forward” is code-supported for the runner, but the full production checkpoint inventory is absent.

## Prohibited

- XNES reached the global optimum.
- Independent estimation recovered the true parameter set.
- Independent estimation is the compensation upper bound.
- Both routes used exactly the same objective.
- Both routes used exactly the same warm-up/evaluation path.
- All basins converged.
- Three seeds were equivalent.
- Runtime is fully reproducible.

Minimum supplement action: add the formal foundation-531 IC manifest/checkpoints/traces, full dPL production checkpoints and three-seed logs, and a deterministic same-input IC/dPL equivalence test before upgrading any qualified statement.
