# Manuscript scripts

All manuscript-facing code lives below this directory and is grouped by result family.

```text
manuscript/scripts/shared/  shared style and path helpers
manuscript/scripts/r1/      R1 statistics, Figure 1/2, Tables 1/S1–S3
manuscript/scripts/r2/      R2 statistics, Figure 3/4, Tables S4/S5
manuscript/scripts/r3/      R3 figures, tables, and process-data exports
manuscript/scripts/r4/      R4 figures, tables, and provenance-guarded generators
```

R3/R4 computational and manuscript-facing modules are colocated in
`manuscript/scripts/r3/` and `manuscript/scripts/r4/` so imports and path
resolution use one canonical package location.

Generated assets must use these locations:

- figures, including supplementary figure panels: `manuscript/figures/`
- tables: `manuscript/tables/`
- intermediate data, logs, audits, and temporary exports: `manuscript/cache/`

Run commands from the repository worktree root or from `project/hydrodiag` with
an explicit `--project-root`/`--results-root` when a script supports those flags.
No script may use an absolute parent-checkout path as a default output location.
