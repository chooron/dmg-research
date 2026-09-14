# FUSE-78 specification inputs

- `structures_78.json` is extracted verbatim in value from the paper
  repository's `list_decision_78.txt` (78 rows, semicolon-delimited source).
- `parameter_catalog.json` is extracted from the paper template's
  `fuse_zConstraints_snow.txt` (37 union coordinates with fit/default/bounds).
- `catalog_78.json` is a generated, fully materialized snapshot (masks/topology/solver included) for downstream tooling; regenerate it from `dfuse.spec`, rather than editing it by hand.
- `dfuse.spec` is the only conversion authority: `get_structure(id)` derives
  decision codes, active state/parameter masks, selected fluxes, topology, and
  solver configuration from those inputs.  No second model-ID/mask table is
  maintained.

The source paths and pinned commit are recorded in `vendor/provenance.json`.
