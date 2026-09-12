# Canonical dPL v2 Post-Training Consistency Audit (2026-09-01)

## 1. Executive conclusion

**Verdict: `NOT_READY`**

The frozen Canonical v2 run has complete model artifacts and internally coherent checkpoint payloads, but it does not pass the final consistency gate: the required no-grad replay using the current local evaluator disagrees with the stored post-hoc test results for all 36 models, all 36 checkpoint metadata Git SHAs disagree with the declared source commit, and the Phase H H1 `best.pt` files required for the paired comparison are not present in the local artifact set. No training, optimizer step, backward pass, early stopping, checkpoint update, protocol change, or TEST-informed selection was performed in this audit.

The stored dPL outputs remain frozen and are not replaced. Formal IC–dPL reanalysis must wait for recovery of the exact training/evaluation environment and H1 checkpoints, followed by a resolved replay gate.

## 2. 36-model recomputed summary

- Model count: **36/36**; DONE: **36**; FAILED: **0**.
- All models have **531/531 valid basin rows**.
- Across the 36 model-level Test KGE medians: median **0.6092708961** (0.6093), mean **0.6016415052**, Q25 **0.5632773694**, Q75 **0.6393107095**.
- The existing report value `0.6093` is the correctly rounded **36-model model-level median**. The existing `0.4284` is the median of the 36 per-model basin-level Q25 values, retained only as a descriptive statistic; it is not Q25 across model-level medians (which is **0.5633**).

| Model | Test median | Test mean | Test Q25 | Valid | Best ep | Best train loss | Warmup/scored | Seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpine1` | 0.589518 | 0.563134 | 0.479862 | 531 | 39 | 0.503334 | 730/365d | 42 |
| `alpine2` | 0.665110 | 0.619434 | 0.555112 | 531 | 39 | 0.469745 | 730/365d | 42 |
| `australia` | 0.633575 | 0.526857 | 0.376621 | 531 | 73 | 0.483033 | 730/365d | 42 |
| `collie1` | 0.398226 | 0.312569 | 0.132074 | 531 | 36 | 0.813098 | 730/365d | 42 |
| `collie2` | 0.607391 | 0.519863 | 0.427842 | 531 | 36 | 0.555893 | 730/365d | 42 |
| `collie3` | 0.625736 | 0.550381 | 0.454004 | 531 | 91 | 0.525261 | 730/365d | 42 |
| `flexb` | 0.449418 | 0.400268 | 0.259935 | 531 | 82 | 0.613125 | 730/365d | 42 |
| `flexi` | 0.483037 | 0.435535 | 0.305617 | 531 | 45 | 0.594292 | 730/365d | 42 |
| `flexis` | 0.558976 | 0.505431 | 0.404662 | 531 | 67 | 0.511349 | 730/365d | 42 |
| `gr4j` | 0.650165 | 0.568051 | 0.465975 | 531 | 62 | 0.476753 | 730/365d | 42 |
| `gsfb` | 0.621494 | 0.553166 | 0.458144 | 531 | 40 | 0.513395 | 730/365d | 42 |
| `hbv96` | 0.757150 | 0.715515 | 0.674550 | 531 | 63 | 0.312610 | 730/365d | 42 |
| `hillslope` | 0.603950 | 0.535864 | 0.432683 | 531 | 49 | 0.517600 | 730/365d | 42 |
| `hymod` | 0.607820 | 0.544010 | 0.445809 | 531 | 45 | 0.491936 | 730/365d | 42 |
| `ihacres` | 0.631955 | 0.542695 | 0.434820 | 531 | 82 | 0.502471 | 730/365d | 42 |
| `modhydrolog` | 0.638090 | 0.567082 | 0.486483 | 531 | 62 | 0.497784 | 730/365d | 42 |
| `mopex1` | 0.564979 | 0.502142 | 0.392738 | 531 | 42 | 0.583836 | 730/365d | 42 |
| `mopex2` | 0.706827 | 0.664292 | 0.603291 | 531 | 60 | 0.414139 | 730/365d | 42 |
| `mopex3` | 0.710972 | 0.665375 | 0.603554 | 531 | 60 | 0.413267 | 730/365d | 42 |
| `mopex4` | 0.709333 | 0.664203 | 0.601368 | 531 | 60 | 0.401793 | 730/365d | 42 |
| `mopex5` | 0.735264 | 0.686012 | 0.635653 | 531 | 51 | 0.376321 | 730/365d | 42 |
| `newzealand1` | 0.610721 | 0.531174 | 0.429044 | 531 | 42 | 0.551851 | 730/365d | 42 |
| `newzealand2` | 0.606581 | 0.518251 | 0.383222 | 531 | 51 | 0.558273 | 730/365d | 42 |
| `penman` | 0.550432 | 0.463919 | 0.316625 | 531 | 42 | 0.593976 | 365/365d | 42 |
| `plateau` | 0.642972 | 0.571531 | 0.499421 | 531 | 36 | 0.471873 | 730/365d | 42 |
| `simhyd` | 0.629759 | 0.551326 | 0.455544 | 531 | 42 | 0.509786 | 730/365d | 42 |
| `smar` | 0.601292 | 0.535152 | 0.427251 | 531 | 24 | 0.514862 | 730/365d | 42 |
| `susannah1` | 0.626848 | 0.551078 | 0.415763 | 531 | 40 | 0.536417 | 730/365d | 42 |
| `susannah2` | 0.492156 | 0.417187 | 0.275780 | 531 | 39 | 0.687469 | 730/365d | 42 |
| `tank` | 0.612604 | 0.531936 | 0.420144 | 531 | 82 | 0.533983 | 730/365d | 42 |
| `tcm` | 0.465255 | 0.443324 | 0.315002 | 531 | 39 | 0.598827 | 730/365d | 42 |
| `topmodel` | 0.504562 | 0.427236 | 0.256031 | 531 | 42 | 0.613527 | 730/365d | 42 |
| `us1` | 0.600448 | 0.515891 | 0.391957 | 531 | 36 | 0.535502 | 730/365d | 42 |
| `vic` | 0.532346 | 0.462993 | 0.328816 | 531 | 42 | 0.600172 | 730/365d | 42 |
| `wetland` | 0.564711 | 0.473757 | 0.341000 | 531 | 62 | 0.604418 | 730/365d | 42 |
| `xinanjiang` | 0.669418 | 0.587719 | 0.485207 | 531 | 62 | 0.441989 | 730/365d | 42 |

Independent raw-basin recomputation agrees with the stored `test_evaluation.json` statistics to <=1.2e-16 for the reported median, mean, and Q25 fields.

## 3. Provenance of “16 models / 0.6093”

Classification: **A — 0.6093 is the 36-model model-level median**, not a 16-model subset. The current report generator uses every `model` in `canonical_manifest.yaml` (`n_models: 36`) and computes the median over all completed rows. Neither the generated `CANONICAL_V2_REPORT.md` nor the generator contains “16 models”. “16 models” came from the earlier assistant summary text and was a prose typo. The generator was corrected to state the 36-model denominator and to distinguish model-level Q25/Q75 from the descriptive median of per-model basin Q25 values; the training artifacts were not changed.

## 4. Checkpoint provenance

- External source provenance: **0/36** `PROVENANCE_VALID`; **36/36** `PROVENANCE_MISMATCH` because metadata Git SHA does not match the declared source commit.
- Internal checkpoint consistency: **36/36**.
- Every `best.pt`, `best_metadata.json`, and SHA-256 sidecar exists; all sidecar hashes match the local `best.pt` bytes.
- For all 36 models, metadata epoch = payload epoch = runtime epoch = the unique `epoch_metrics.csv` row; metadata/runtime train loss matches that row at absolute tolerance 1e-12.
- Every best epoch is strictly before the terminal logged epoch; no terminal/latest checkpoint was mistaken for `best.pt`. The runner source reloads `best.pt` before setting `Phase.EVAL` and calling the test evaluator.
- All 36 metadata and payload files record Git SHA `7d1132bf5c9ee0114a1a12dd10720101f6ca3b74`, while `SOURCE_COMMIT.txt` records `3caca37a4243ae0a95ebe9cc4f22998672ddf464`; source-SHA match for metadata+payload: **0/36**.

| Model | Best ep | Log ep | Best loss | Log loss | SHA | Source SHA | Best reloaded | Not terminal | Status |
|---|---:|---:|---:|---:|---|---|---|---|---|
| `alpine1` | 39 | 39 | 0.503333979 | 0.503333979 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `alpine2` | 39 | 39 | 0.469744855 | 0.469744855 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `australia` | 73 | 73 | 0.483033410 | 0.483033410 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `collie1` | 36 | 36 | 0.813097844 | 0.813097844 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `collie2` | 36 | 36 | 0.555893015 | 0.555893015 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `collie3` | 91 | 91 | 0.525261123 | 0.525261123 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `flexb` | 82 | 82 | 0.613124817 | 0.613124817 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `flexi` | 45 | 45 | 0.594291788 | 0.594291788 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `flexis` | 67 | 67 | 0.511348642 | 0.511348642 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `gr4j` | 62 | 62 | 0.476753175 | 0.476753175 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `gsfb` | 40 | 40 | 0.513394726 | 0.513394726 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `hbv96` | 63 | 63 | 0.312609559 | 0.312609559 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `hillslope` | 49 | 49 | 0.517600133 | 0.517600133 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `hymod` | 45 | 45 | 0.491936327 | 0.491936327 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `ihacres` | 82 | 82 | 0.502471478 | 0.502471478 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `modhydrolog` | 62 | 62 | 0.497783942 | 0.497783942 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `mopex1` | 42 | 42 | 0.583836269 | 0.583836269 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `mopex2` | 60 | 60 | 0.414138698 | 0.414138698 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `mopex3` | 60 | 60 | 0.413266781 | 0.413266781 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `mopex4` | 60 | 60 | 0.401792616 | 0.401792616 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `mopex5` | 51 | 51 | 0.376321492 | 0.376321492 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `newzealand1` | 42 | 42 | 0.551850646 | 0.551850646 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `newzealand2` | 51 | 51 | 0.558272965 | 0.558272965 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `penman` | 42 | 42 | 0.593976417 | 0.593976417 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `plateau` | 36 | 36 | 0.471873490 | 0.471873490 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `simhyd` | 42 | 42 | 0.509785726 | 0.509785726 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `smar` | 24 | 24 | 0.514862069 | 0.514862069 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `susannah1` | 40 | 40 | 0.536417178 | 0.536417178 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `susannah2` | 39 | 39 | 0.687468756 | 0.687468756 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `tank` | 82 | 82 | 0.533982580 | 0.533982580 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `tcm` | 39 | 39 | 0.598827493 | 0.598827493 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `topmodel` | 42 | 42 | 0.613527082 | 0.613527082 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `us1` | 36 | 36 | 0.535501978 | 0.535501978 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `vic` | 42 | 42 | 0.600171551 | 0.600171551 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `wetland` | 62 | 62 | 0.604418486 | 0.604418486 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |
| `xinanjiang` | 62 | 62 | 0.441988610 | 0.441988610 | OK | MISMATCH | YES | YES | `PROVENANCE_MISMATCH` |

## 5. TEST isolation

- Source-guarded result: **36/36**.
- All 36 rows report `train_phase_test_access=NO`, `selection_used_test=NO`, `early_stop_used_test=NO`, and `posthoc_test_only=YES`.
- `epoch_metrics.csv` has no TEST metric columns; `best_metadata.json` has no TEST fields. Training selection and plateau stopping use `train_loss`; the source guard rejects test evaluation while `Phase.TRAIN`, and the TEST call occurs only after best-checkpoint reload and `Phase.EVAL` transition.
- Local per-run `stdout.log`/`stderr.log` files were not archived in the downloaded artifact set (**0/36 complete**), so this is a source-plus-artifact audit rather than a complete log-forensic audit. The root ledger is present and records post-completion test values.

## 6. Stored versus unified no-grad TEST replay

Replay contract: local CPU, `model.eval()`, `torch.no_grad()`, FP64 parameterizer, 531 basins, TEST `1995-10-01..2010-09-30`, evaluation warmup 365d, KGE epsilon 0.1, current local forcing/data contract, and no truncate semantics.
- Evaluator provenance note: the canonical runner source casts stored TEST forcing/target tensors to float32, whereas this audit intentionally used FP64 as required. This is a strict FP64 replay, not yet a byte-equivalent reproduction of the stored evaluator.
- A source-faithful FP32-input diagnostic was also run: its replay gate was 0/36, with median absolute delta **0.0290453607** and maximum **0.0473197912**. Therefore the mismatch is not explained by the FP64-versus-FP32 choice alone.
- Replay execution errors: **0/36**; replay gate `abs(delta median) <= 1e-10`: **0/36**. All non-error executions are explicitly marked `EXECUTION_OK_GATE_FAILED` when the gate fails.
- Median absolute model-median delta: **0.0290453604**; maximum absolute delta: **0.0473197931**. All 36 deltas are negative and materially larger than floating-point tolerance; this is an unexplained environment/source/data provenance discrepancy, not a scientific correction.

| Model | Stored median | Replayed median | Delta | Stored Q25 | Replayed Q25 | Max basin | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `alpine1` | 0.589518 | 0.568791 | -0.020728 | 0.479862 | 0.452490 | 0.969422 | `FAIL` |
| `alpine2` | 0.665110 | 0.633873 | -0.031237 | 0.555112 | 0.507911 | 1.296051 | `FAIL` |
| `australia` | 0.633575 | 0.586255 | -0.047320 | 0.376621 | 0.292433 | 1.822365 | `FAIL` |
| `collie1` | 0.398226 | 0.381870 | -0.016356 | 0.132074 | 0.132593 | 0.515273 | `FAIL` |
| `collie2` | 0.607391 | 0.569044 | -0.038347 | 0.427842 | 0.370211 | 0.492925 | `FAIL` |
| `collie3` | 0.625736 | 0.582402 | -0.043334 | 0.454004 | 0.343723 | 0.817424 | `FAIL` |
| `flexb` | 0.449418 | 0.423371 | -0.026047 | 0.259935 | 0.217218 | 0.667873 | `FAIL` |
| `flexi` | 0.483037 | 0.479495 | -0.003542 | 0.305617 | 0.268589 | 0.516100 | `FAIL` |
| `flexis` | 0.558976 | 0.542174 | -0.016802 | 0.404662 | 0.357168 | 1.007239 | `FAIL` |
| `gr4j` | 0.650165 | 0.622099 | -0.028067 | 0.465975 | 0.405731 | 0.718217 | `FAIL` |
| `gsfb` | 0.621494 | 0.607894 | -0.013600 | 0.458144 | 0.440325 | 4.268196 | `FAIL` |
| `hbv96` | 0.757150 | 0.731052 | -0.026098 | 0.674550 | 0.599100 | 1.071588 | `FAIL` |
| `hillslope` | 0.603950 | 0.568550 | -0.035399 | 0.432683 | 0.395227 | 0.475402 | `FAIL` |
| `hymod` | 0.607820 | 0.586293 | -0.021527 | 0.445809 | 0.409990 | 0.491569 | `FAIL` |
| `ihacres` | 0.631955 | 0.610918 | -0.021037 | 0.434820 | 0.405807 | 0.457972 | `FAIL` |
| `modhydrolog` | 0.638090 | 0.603893 | -0.034197 | 0.486483 | 0.427806 | 0.694590 | `FAIL` |
| `mopex1` | 0.564979 | 0.534703 | -0.030277 | 0.392738 | 0.341812 | 0.748668 | `FAIL` |
| `mopex2` | 0.706827 | 0.673485 | -0.033342 | 0.603291 | 0.557057 | 1.208205 | `FAIL` |
| `mopex3` | 0.710972 | 0.672159 | -0.038813 | 0.603554 | 0.555258 | 0.977790 | `FAIL` |
| `mopex4` | 0.709333 | 0.662901 | -0.046432 | 0.601368 | 0.534053 | 1.106065 | `FAIL` |
| `mopex5` | 0.735264 | 0.700326 | -0.034938 | 0.635653 | 0.558535 | 1.014065 | `FAIL` |
| `newzealand1` | 0.610721 | 0.573944 | -0.036777 | 0.429044 | 0.376620 | 0.614031 | `FAIL` |
| `newzealand2` | 0.606581 | 0.559938 | -0.046642 | 0.383222 | 0.328686 | 0.453009 | `FAIL` |
| `penman` | 0.550432 | 0.516996 | -0.033436 | 0.316625 | 0.258915 | 0.571361 | `FAIL` |
| `plateau` | 0.642972 | 0.614064 | -0.028908 | 0.499421 | 0.442412 | 0.686962 | `FAIL` |
| `simhyd` | 0.629759 | 0.600576 | -0.029183 | 0.455544 | 0.422868 | 0.547485 | `FAIL` |
| `smar` | 0.601292 | 0.584016 | -0.017276 | 0.427251 | 0.394381 | 0.800461 | `FAIL` |
| `susannah1` | 0.626848 | 0.593258 | -0.033590 | 0.415763 | 0.374334 | 1.019027 | `FAIL` |
| `susannah2` | 0.492156 | 0.475501 | -0.016654 | 0.275780 | 0.230570 | 0.462576 | `FAIL` |
| `tank` | 0.612604 | 0.575909 | -0.036694 | 0.420144 | 0.377497 | 0.722064 | `FAIL` |
| `tcm` | 0.465255 | 0.447391 | -0.017864 | 0.315002 | 0.283767 | 0.471560 | `FAIL` |
| `topmodel` | 0.504562 | 0.481378 | -0.023184 | 0.256031 | 0.259379 | 0.568853 | `FAIL` |
| `us1` | 0.600448 | 0.576075 | -0.024373 | 0.391957 | 0.348907 | 0.972130 | `FAIL` |
| `vic` | 0.532346 | 0.524399 | -0.007947 | 0.328816 | 0.311375 | 0.787737 | `FAIL` |
| `wetland` | 0.564711 | 0.544012 | -0.020699 | 0.341000 | 0.330603 | 0.620431 | `FAIL` |
| `xinanjiang` | 0.669418 | 0.628609 | -0.040810 | 0.485207 | 0.443794 | 0.938523 | `FAIL` |

Largest median discrepancies are retained in the replay CSV; no stored test result was overwritten and no TEST result was used to modify a protocol or select a checkpoint.

## 7. H1 versus v2 common-TEST paired comparison

The intended comparison is same TEST / same evaluator only. It is **blocked**: all eight Phase H H1 run directories contain `best_metadata.json` and inner-validation artifacts but no `best.pt` in the local repository. The eight expected files are `H01_gr4j_730`, `H02_hbv96_1825`, `H03_flexb_730`, `H04_mopex4_1825`, `H05_topmodel_730`, `H06_xinanjiang_1825`, `H07_collie1_730`, and `H08_penman_1825`. The prior remote endpoint was unreachable during this audit (TCP accepted then closed before an SSH banner), so no substitute checkpoint was invented.

- Paired rows emitted: **8/8**, all `BLOCKED_MISSING_H1_BEST_PT`.
- Consequently, no defensible same-TEST `delta_median`, `delta_q25`, fraction improved, or median basin delta can be reported. No cross-period H1 INNER-VAL versus v2 TEST subtraction is reported.

| Model | H1 checkpoint | H1 median | V2 replay median | Same-TEST delta | Status |
|---|---|---:|---:|---:|---|
| `collie1` | `H07_collie1_730/best.pt` missing | NA | 0.381870 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `flexb` | `H03_flexb_730/best.pt` missing | NA | 0.423371 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `gr4j` | `H01_gr4j_730/best.pt` missing | NA | 0.622099 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `hbv96` | `H02_hbv96_1825/best.pt` missing | NA | 0.731052 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `mopex4` | `H04_mopex4_1825/best.pt` missing | NA | 0.662901 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `penman` | `H08_penman_1825/best.pt` missing | NA | 0.516996 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `topmodel` | `H05_topmodel_730/best.pt` missing | NA | 0.481378 | NA | `BLOCKED_MISSING_H1_BEST_PT` |
| `xinanjiang` | `H06_xinanjiang_1825/best.pt` missing | NA | 0.628609 | NA | `BLOCKED_MISSING_H1_BEST_PT` |

## 8. Model-specific anomalies

- The all-model replay mismatch is the primary unresolved anomaly; it includes both ordinary models and VIC, so it cannot be attributed to one hydrology formula from this audit alone.
- Penman replay used 365d evaluation warmup and full forward with no truncate behavior; the training semantics are 365-day no-grad warm-up with state detach followed by full backpropagation only over the scored period. Stored/replayed median delta was **-0.033436**, within the all-model mismatch pattern. The frozen config retains the historical `truncate:90` label; it is not interpreted as a TBPTT effect.
- VIC current evaluator source contains dynamic-DOY handling, but frozen-run provenance is unresolved because its run metadata SHA differs from the declared protocol SHA; exact replay delta was **-0.007947**. The old VIC IC baseline remains invalid for formal paired IC–dPL analysis.

## 9. VIC status

`VIC_IC_RERUN_REQUIRED_BEFORE_FINAL_PAIRED_ANALYSIS`. Do not rerun VIC IC in this audit. The VIC dPL checkpoint/result is not deleted or altered.

## 10. Readiness for scientific reanalysis

**`NOT_READY`** — internal checkpoint coherence and source-level TEST isolation pass, but external checkpoint source provenance fails for 36/36, stored-versus-replayed TEST fails for 36/36, and the H1 common-TEST comparison is unavailable. Resolve all three provenance blockers before IC–dPL reanalysis.

## Audit artifacts

- `project/benchmark/results/dpl_canonical_v2_20260831/POST_AUDIT_CANONICAL_V2_RECOMPUTED_SUMMARY.csv`
- `project/benchmark/results/dpl_canonical_v2_20260831/POST_AUDIT_CHECKPOINT_PROVENANCE.csv`
- `project/benchmark/results/dpl_canonical_v2_20260831/POST_AUDIT_TEST_ISOLATION.csv`
- `project/benchmark/results/dpl_canonical_v2_20260831/POST_AUDIT_CANONICAL_V2_TEST_REPLAY.csv` (FP64 replay)
- `project/benchmark/results/dpl_canonical_v2_20260831/POST_AUDIT_CANONICAL_V2_TEST_REPLAY_SOURCE_FP32.csv` (source-dtype diagnostic)
- `project/benchmark/results/dpl_canonical_v2_20260831/POST_AUDIT_H1_VS_V2_COMMON_TEST.csv`
- Corrected derived report: `project/benchmark/results/dpl_canonical_v2_20260831/CANONICAL_V2_REPORT.md`

## Scope controls

- GPU used: **NO**; local CPU replay only. Remote GPU was not used.
- Training started: **NO**.
- Optimizer/backward/early stopping/checkpoint update: **NO**.
- Canonical protocol modified: **NO**. Only the derived report generator/report wording was corrected.
- VIC IC, 3-seed, OOB, PUR, and parameter-atlas reanalysis: **NOT STARTED**.

## Git status at report generation

```text
M dmotpy/data_contract.py
 M dmotpy/models/core/vic.py
 M dmotpy/models/hydrology_model.py
 M project/benchmark/scripts/diagnostics/e2_boundary_kink.py
 M project/benchmark/scripts/diagnostics/full_model_fd_warmup_modes.py
 M project/benchmark/scripts/diagnostics/k_full_retrain.py
 M project/benchmark/scripts/diagnostics/round12_edge_probe.py
 M project/benchmark/scripts/diagnostics/warmup_gradient_contract.py
 M project/benchmark/src/model_registry.py
 M project/hydrodiag/manuscript/scripts/supplement/plot_huc2_loro_robustness.py
 M project/hydrodiag/manuscript/supplement/final_assets/figures/Figure_S1/Figure_S1.png
 M project/hydrodiag/manuscript/supplement/final_assets/figures/Figure_S1/caption_facts.md
 M project/hydrodiag/manuscript/supplement/final_assets/figures/Figure_S1/plot_Figure_S1.py
 M project/hydrodiag/manuscript/supplement/final_assets/tables/Table_S3/Table_S3.md
 M project/hydrodiag/manuscript/supplement/final_assets/tables/Table_S3/Table_S3_panelA.csv
 M project/hydrodiag/manuscript/supplement/final_assets/tables/Table_S3/Table_S3_panelB.csv
?? dmotpy/DMOTPY_CURRENT_STATE_AUDIT_20260831.md
?? dmotpy/tests/test_vic_doy_remediation.py
?? project/benchmark/BENCHMARK_CURRENT_STATE_AUDIT_20260831.md
?? project/benchmark/CANONICAL_V2_POST_TRAINING_CONSISTENCY_AUDIT_20260901.md
?? project/benchmark/CURRENT_PROJECT_DPL_AUDIT_SYNTHESIS_20260831.md
?? project/benchmark/DPL_NOGPU_PHASE_FINAL_REPORT_20260831.md
?? project/benchmark/PENMAN_TRUNCATE90_CLEANUP_AND_SYNC_REPORT_20260831.md
?? project/benchmark/PENMAN_TRUNCATE90_CORRECTION_NOTE_20260831.md
?? project/benchmark/PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md
?? project/benchmark/scripts/ablation/
?? project/benchmark/scripts/canonical_v2/
?? project/benchmark/scripts/diagnostics/parameter_attribute_atlas.py
?? project/benchmark/scripts/diagnostics/parameter_attribute_atlas_followup.py
?? project/benchmark/scripts/diagnostics/run_agent2_replay.py
?? project/benchmark/scripts/diagnostics/run_agent3_audit.py
?? project/benchmark/scripts/diagnostics/run_agent4_decomposition.py
?? project/benchmark/scripts/diagnostics/run_task_s0_evaluation.py
?? project/benchmark/tests/
?? project/hydrodiag/CH3_3_3_PARAMETER_COMPENSATION_AUDIT.md
?? project/hydrodiag/CH3_3_4_CONTROLLED_RECOVERY_AUDIT.md
?? project/hydrodiag/manuscript/supplement/HESS_Supplement.docx:Zone.Identifier
?? project/hydrodiag/manuscript/supplement/final_assets.zip
```
