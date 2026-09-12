# R4 Handoff

- Analysis status: Steps 5–12 complete; no training was launched.
- Frozen manifest SHA256: `44dc1426818bf16aac7acb70c94fe022e7cd2671ab658d89a75570f8d7277a14`.
- Master QC: PASS for 8/8 models.
- Bootstrap device: `NVIDIA GeForce RTX 3060`; peak allocated VRAM `379304960` bytes; peak RSS `1207.5` MiB.
- Current three-tier judgment: **A. broadly retained**.
- Run `python project/benchmark/manuscript/r4/scripts/r4_make_figure.py` to render the single planned figure from frozen figure-data CSVs.
- `KGE_IC`/`KGE_dPL_seen` comparison columns remain blank by design because IC is not rerun in R4.
