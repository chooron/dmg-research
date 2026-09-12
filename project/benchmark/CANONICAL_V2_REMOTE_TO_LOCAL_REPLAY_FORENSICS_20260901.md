# Canonical dPL v2 Remote-to-Local Replay Forensics (2026-09-01)

## Final verdict

- **Verdict: `B. REPRODUCED_AFTER_DATA_CONTRACT_MATCH`**.
- **`CANONICAL_V2_VALID = YES`**, conditional on the recovered remote data contract and post-training evaluator contract documented below.
- The stored Canonical v2 TEST results were reproduced for all 36 models with the original local `best.pt` files, using local GPU, `model.eval()`, `torch.no_grad()`, and no optimizer/backward/checkpoint writes.
- This identifies the primary mismatch as **data/input contract drift**, not checkpoint reconstruction or a stored TEST-generation bug.

## 1. Remote repository and source provenance

- Remote repository: `/root/dmg-research`.
- Remote HEAD: `7d1132bf5c9ee0114a1a12dd10720101f6ca3b74`.
- Remote HEAD is a commit on `master` and contains the seven S4D ablation configuration changes shown in `SHA_7d1132bf_PROVENANCE.txt`.
- The active Canonical v2 runner, `dmotpy/`, `project/benchmark/dpl/`, `project/benchmark/src/`, and `project/benchmark/scripts/` were **untracked working-tree source** on the remote host; they are absent from the tracked `7d1132bf` archive.
- `3caca37a4243ae0a95ebe9cc4f22998672ddf464` is **not present in the remote object database** (`NOT_FOUND`). No remote source archive was created for it.
- The local repository does contain `3caca37a`, but it is an unrelated HESS supplement commit and is not the remote Canonical source.
- The best available remote active source was recovered as `remote_current_active_source_snapshot.tar.gz`; its extracted 398-file manifest matches the captured remote file SHA-256 list 398/398. Because the source was untracked, this remains a working-tree snapshot rather than a commit-addressed source, with the tracked `source_7d1132bf.tar.gz` retained separately.

## 2. Checkpoint metadata semantics

- For `alpine1`, `gr4j`, `penman`, and `vic`, both `best.pt["git_sha"]` and `best_metadata.json["git_sha"]` equal the remote HEAD `7d1132bf...`.
- Each run's `source_commit.txt` also records `7d1132bf...`.
- `source_commit` and `source_dirty` are not separate populated fields in the checkpoint payload or metadata; the SHA field is the repository HEAD fingerprint written by `get_git_sha()`.
- The earlier comparison against local `SOURCE_COMMIT.txt` (`3caca37a...`) was therefore a **declared-source provenance mismatch**, not a state/checkpoint mismatch.
- All 36 local `best.pt` SHA-256 values match the remote SHA-256 values: **36/36**.

## 3. Remote environment

- Kernel: Linux 5.4.0-162-generic, x86_64.
- Python: `/root/miniconda3/bin/python`, Python 3.12.3.
- PyTorch: `2.8.0+cu128`, CUDA runtime 12.8.
- Remote forensic-time `torch.cuda.is_available()`: `False`; the remote host had no usable GPU at capture time.
- Historical Canonical run artifacts record `DEVICE: cuda:0` and `TORCH_VERSION: 2.8.0+cu128` in all 36 `environment.txt` files; remote training GPU usage is therefore **recorded as YES**, while remote GPU availability during this forensic session was **NO**.
- Full package capture is in `REMOTE_PYTHON_ENV.txt`; per-run environment captures are in `REMOTE_RUN_ENVIRONMENTS.txt`.

## 4. Data/input provenance

The exact remote canonical native loader resolves data under `/root/dmg-research/data` and reads:

- `531sub_id.txt`
- `camels_dataset`
- `gage_id.npy`
- `caravan_671_attributes.npy` when present; otherwise attributes from `camels_dataset`
- calendar features are generated from source date ranges; no separate calendar file is read by the canonical runner.

Remote/local comparison:

- `531sub_id.txt`, `camels_dataset`, `gage_id.npy`, `camels_forcing_v2.pkl`, and `camels_dates.npy`: checksums match.
- Remote `caravan_671_attributes.npy`: present, 188,008 bytes, checksum `686366653e5cbcac00ac24ecb20b710b4940ccf73fe1957f27f7da1969dfd825`.
- Main local data root: `caravan_671_attributes.npy` was absent.
- The remote Caravan attribute matrix was downloaded only into the isolated replay directory, not into the main repository.
- After constructing the isolated remote-contract data root, the replay manifest matched remote input checksums **7/7**.

The decisive finding is that the remote runner preferred the remote Caravan attribute matrix, while the previous local replay fell back to the attributes embedded in `camels_dataset`.

## 5. Exact evaluator recovered

Best available recovered source (untracked remote working tree; file-hash validated):

`candidate_remote_snapshot/project/benchmark/scripts/canonical_v2/run_canonical_v2_model.py`

The recovered post-hoc evaluator uses:

- `model.eval()` and `torch.no_grad()`;
- TEST forcing/targets converted to `float32`;
- float64 parameterizer/network;
- CUDA `backend="compile"` when available;
- TEST forcing `1994-10-01..2010-09-30`, scored TEST `1995-10-01..2010-09-30`;
- evaluation warmup 365 days;
- KGE epsilon 0.1;
- dynamic calendar forcing for calendar-input models;
- best-checkpoint reload before the `Phase.EVAL` TEST call.

No training, optimizer construction/step, backward pass, early stopping, or checkpoint save was used by the replay script.

## 6. Four-model differential source/data matrix

The four-model matrix contains all candidate outcomes in `FOUR_MODEL_SOURCE_MATRIX.csv`.

| Source/data candidate | alpine1 | gr4j | penman | vic |
|---|---:|---:|---:|---:|
| local source + local data | FAIL (-0.020728) | FAIL (-0.028067) | ERROR (obsolete `truncate:90`) | FAIL (-0.007947) |
| remote active source + local data | FAIL (-0.020728) | FAIL (-0.028067) | FAIL (-0.033436) | FAIL (-0.007947) |
| local source + remote data | PASS | PASS | ERROR (obsolete `truncate:90`) | PASS |
| local source + remote data + `detach` semantic override | — | — | PASS |
| remote active source + remote data | **PASS** | **PASS** | **PASS** | **PASS** |
| tracked `7d1132bf` source archive | NOT RUN: active canonical source was untracked | NOT RUN | NOT RUN | NOT RUN |
| remote `3caca37a` object | NOT RUN: SHA absent | NOT RUN | NOT RUN | NOT RUN |

The local source reproduces the normal three models once the remote data contract is supplied. For Penman, the local remediation rejects the historical `truncate:90` label; forcing its documented semantic equivalent, `detach`, reproduces the stored result. The remote active source accepts the historical label as the old no-op behavior did.

## 7. Local GPU 36-model replay

The official full replay was run only after the recovered remote source/data candidate passed the 4/4 gate; the timestamps and explicit `FOUR_MODEL_GATE_PASS=TRUE` are recorded in `REPLAY_ORDER.txt`.

- Output: `project/benchmark/results/canonical_v2_remote_forensics_20260901/LOCAL_GPU_36MODEL_REPLAY.csv`.
- Before loading any data, the replay script validated `source_root/data/*` against the supplied remote manifest and wrote `LOCAL_GPU_36MODEL_REPLAY_LOADED_DATA_MANIFEST.csv`; all loaded paths/checksums matched.
- GPU: **YES**, NVIDIA GeForce RTX 3060, 12 GB.
- Replay GPU: **YES**, NVIDIA GeForce RTX 3060, 12 GB; remote forensic-time GPU: **NO**, while historical remote training artifacts record `DEVICE: cuda:0`.
- Backend: `compile` for all 36 models.
- Models: **36/36**.
- Valid basins: **531/531 for every model**.
- Model-level median gate (`abs(delta) <= 1e-10`): **36/36 passed**.
- Maximum absolute model-level median delta: **4.440892098500626e-16**.
- Maximum absolute model-level mean delta: **6.163634047595679e-10**.
- Maximum absolute model-level Q25 delta: **4.218370819319972e-09**, consistent with GPU reduction/percentile numerical variation; no median gate failed.
- Penman: reproduced.
- VIC: reproduced.
- No stored TEST output was overwritten.

## 8. Cause classification

- **A. Source snapshot drift:** not the primary cause. The exact remote active source is needed for provenance, but local evaluator code reproduces the normal models after data correction.
- **B. Evaluator contract drift:** secondary Penman label drift. Local code now rejects historical `truncate:90`; remote code treats it as the historical no-op/detach semantic.
- **C. Data/input drift:** **identified primary cause**. The remote evaluation used `caravan_671_attributes.npy`, which was absent from the previous local data root.
- **D. Checkpoint reconstruction drift:** not supported; all 36 local and remote checkpoint hashes match, and all payloads reload correctly.
- **E. Stored TEST generation bug:** not supported; the recovered source plus recovered input contract reproduces all 36 stored medians.

## 9. Scope controls and deviations

- Training started: **NO**.
- Optimizer/backward/early stopping: **NO**.
- Canonical checkpoint modified: **NO**.
- Remote working tree modified: **NO**; only `/root/canonical_v2_forensics_20260901/` was created.
- H1 was not retrained and no H1 comparison was fabricated.
- VIC IC was not rerun.
- A local-current-source 36-model no-grad diagnostic was also produced before the final differential matrix was complete; it was retained as `LOCAL_GPU_LOCAL_CURRENT_36MODEL_DIAGNOSTIC.csv`. It did not modify any artifact. The official 36-model replay is the recovered remote-source/remote-data run above.

## 10. Evidence paths

Local forensic directory:

`project/benchmark/results/canonical_v2_remote_forensics_20260901/`

Required evidence includes:

- `REMOTE_HEAD.txt`
- `REMOTE_STATUS.txt`
- `REMOTE_LOG.txt`
- `REMOTE_REFLOG.txt`
- `REMOTE_WORKTREE.diff`
- `REMOTE_INDEX.diff`
- `REMOTE_PYTHON_ENV.txt`
- `REMOTE_RUN_ENVIRONMENTS.txt`
- `REMOTE_DATA_PATHS.txt`
- `SHA_7d1132bf_PROVENANCE.txt`
- `SHA_3caca37a_PROVENANCE.txt`
- `SHA_COMPARISON.diff`
- `REMOTE_CHECKPOINT_METADATA.csv`
- `CHECKPOINT_METADATA_SEMANTICS.md`
- `REMOTE_BEST_PT_SHA256.csv`
- `LOCAL_BEST_PT_SHA256.csv`
- `REMOTE_VS_LOCAL_BEST_PT.csv`
- `REMOTE_DATA_INPUT_MANIFEST.csv`
- `LOCAL_DATA_INPUT_MANIFEST.csv`
- `REMOTE_VS_LOCAL_DATA_MANIFEST.csv`
- `LOCAL_REPLAY_DATA_INPUT_MANIFEST.csv`
- `REMOTE_VS_LOCAL_REPLAY_DATA_MANIFEST.csv`
- `FOUR_MODEL_SOURCE_MATRIX.csv`
- `LOCAL_GPU_36MODEL_REPLAY.csv`
- `LOCAL_GPU_36MODEL_REPLAY_LOADED_DATA_MANIFEST.csv`
- `FOUR_MODEL_REMOTE_CURRENT.csv` and `FOUR_MODEL_REMOTE_CURRENT.log`
- `REPLAY_ORDER.txt`
- `source_7d1132bf.tar.gz`
- `remote_current_active_source_snapshot.tar.gz`
- `REMOTE_SOURCE_SNAPSHOT_STATUS.txt`
- `REMOTE_ACTIVE_SOURCE_FILE_SHA256.txt`
- `REMOTE_ACTIVE_SOURCE_SNAPSHOT_SHA256.txt`
- `LOCAL_ACTIVE_SOURCE_SNAPSHOT_SHA256.txt`
- `REMOTE_VS_EXTRACTED_ACTIVE_SOURCE_MANIFEST.csv`
- `remote_repo_refs.bundle`

Isolated source roots:

`/home/jingxin/code/dmg-research_replay_forensics/`
`/home/jingxin/code/dmg-research_replay_forensics/run_gpu_candidate_replay.py`

## 11. Final repository state

- Main repository `/home/jingxin/code/dmg-research` was not reset, cleaned, checked out, or overwritten.
- Existing user modifications and untracked files remain preserved.
- New replay code and all remote evidence are outside the main source tree except for this report and the forensic results directory.
- `git status --short` was captured before and after execution; no existing user-authored change was discarded.
