# Upstream reference area

The repositories under `vendor/upstream/` are local, ignored Git clones used as
read-only provenance/reference oracles. Their exact remote, ref, commit, clone
date, and dirty state are recorded in [`provenance.json`](provenance.json).

No upstream source is copied into `dfuse/`; the first-stage kernel is an
independent tensor implementation driven by the extracted specifications.
