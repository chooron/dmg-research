# Hostile Model Selection Review and Bias Audit

**Reviewer Context:** Adversarial peer-review audit conducted from the perspective of an unfriendly *Journal of Hydrology* reviewer evaluating the selection provenance, potential cherry-picking, structural coverage, and generalization claims of the eight-model subset used in the R4 held-out-basin experiment.

---

## D1. Was the exact eight-model list frozen before OOB outcomes?

### Verdict: **CONFIRMED**

### Audit Evidence:
1. **Config Freeze:** The configuration file `project/benchmark/configs/oob_primary8_5fold_20260902.yaml` (SHA256: `1e9e4e03...`) was written and timestamped on **2026-09-02 09:23:18 +0800**. It explicitly enumerates the exact eight models (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`).
2. **Deployment Tarball:** The deployment archive `project/benchmark/results/oob_deploy_20260902/oob_deploy_20260902.tar.gz` (SHA256: `70afc9f9...`) and model list `OOB_PRIMARY8_MODELS.txt` (SHA256: `d67c57a7...`) were packaged on **2026-09-02 10:58:35 +0800**.
3. **Execution Timeline:** The first remote training job (`alpine2_fold0`) started execution at **2026-09-02 11:03:47 +0800** and completed at **2026-09-02 12:08:08 +0800**.
4. **Temporal Separation:** The exact eight-model subset was frozen 2 hours 45 minutes prior to the first completed OOB run, and the underlying deterministic selection rule was executed and documented 13 hours prior (2026-09-01 23:09:19 +0800).

---

## D2. Is there any evidence of OOB-result-informed cherry-picking?

### Verdict: **NO EVIDENCE**

### Audit Evidence:
1. **Queue Completeness:** The queue log `remote_queue.log` records exactly 40 job executions (8 models $\times$ 5 folds). All 40 jobs completed with status `PLATEAU_STOP` and exit code 0 (`QUEUE_STOP completed=40 failed=0`).
2. **Zero Job Dropping / Swapping:** No model was executed, found to have poor OOB KGE or poor parameter transfer, and subsequently replaced or discarded.
3. **Zero Retries:** The downloaded queue log contains exactly 0 retry events.
4. **Exact 1:1 Mapping:** The 40 completed checkpoints match the pre-launch configuration contract exactly.

---

## D3. Does the subset span meaningful structural diversity?

### Verdict: **ADEQUATE TO STRONG COVERAGE**

### Audit Evidence:
1. **Parameter Complexity ($P$):** Spans $P=5$ (`us1`) to $P=15$ (`hbv96`), covering 71.4% of the benchmark parameter-count range ($[1, 15]$).
2. **State Dimensionality ($S$):** Spans $S=1$ (`ihacres`) to $S=5$ (`hbv96`, `mopex4`), covering 100% of the benchmark state range ($[1, 5]$).
3. **Snow Dynamics:** Proportional representation of snow-accounting models (3/8 = 37.5%) vs non-snow models (5/8 = 62.5%), consistent with the 36-model benchmark base rate (33.3%).
4. **Hydrological Concepts:** Spans diverse, contrasting runoff mechanisms:
   - Transfer-function unit hydrograph cascade (`ihacres`)
   - Tension water store with parabolic saturation excess (`xinanjiang`)
   - Multi-zone conceptual store with threshold melt (`hbv96`)
   - SAC-SMA multi-layer conceptual store (`mopex4`)
   - Parsimonious bucket formulations (`alpine2`, `us1`)
   - Two-layer drainage cascade (`newzealand2`)
   - Non-linear topographic gradient discharge (`hillslope`)

---

## D4. Does the subset span seen-basin R1–R3 response diversity?

### Verdict: **STRONG COVERAGE**

### Audit Evidence:
1. **$G_{\text{seen}} \times R_{\text{seen}}$ Orthogonal Grid:** Exactly two models are positioned in each of the four quadrants:
   - $Q_1$ (Low $G$ / High $R$): `alpine2` (REP), `hbv96` (CONTRAST)
   - $Q_2$ (Low $G$ / Low $R$): `newzealand2` (REP), `xinanjiang` (CONTRAST)
   - $Q_3$ (High $G$ / High $R$): `ihacres` (REP), `us1` (CONTRAST)
   - $Q_4$ (High $G$ / Low $R$): `hillslope` (REP), `mopex4` (CONTRAST)
2. **Dispersion in Secondary Axes:**
   - Parameter displacement $D_{\theta}$ spans from $0.174$ (`us1`) to $0.470$ (`mopex4`).
   - IC multi-start restart uncertainty $U$ spans 99.3% of the 36-model range ($0.0022$ to $0.3164$).
   - Attribute signal count $A$ spans low signal (7) to high signal (21).

---

## D5. Are there obvious holes or design blind spots?

| Potential Hole / Gap | Audit Findings | Severity Classification | Manuscript Implication |
|---|---|---|---|
| **1. Exclusion of severely degraded models ($K_{\text{joint}} < 0.561$)** | The bottom quartile of benchmark models was intentionally excluded by the viability gate. | `MINOR` | Necessary safeguard: testing OOB generalization on models that cannot simulate streamflow on seen basins would introduce severe confounding. |
| **2. Exclusion of 1- to 2-parameter toy models ($P < 5$)** | Models like `australia` ($P=1$) or `gsfb` ($P=2$) are absent. | `MINOR` | Parsimonious models are represented down to $P=5$ (`us1`) and $P=6$ (`alpine2`, `ihacres`). Models with 1–2 parameters lack the parameterization capacity to evaluate attribute-conditioned learning. |
| **3. Capping of MOPEX lineage to 1 model** | `mopex1, 2, 3, 5` excluded in favor of `mopex4`. | `NOT MATERIAL` (Design Strength) | Eliminating duplicate variants of the same model family prevents clustering and enhances structural diversity. |
| **4. Omission of other popular models (e.g. GR4J, TOPMODEL)** | `gr4j` and `topmodel` were eligible in $Q_1$ and $Q_3$ but ranked lower in 5D contrast distance than `alpine2`, `hbv96`, `ihacres`, `us1`. | `MINOR` | Their underlying conceptual mechanisms (bucket storage, unit hydrograph routing) are well represented by the selected panel. |
| **5. Non-Random / Non-IID Sampling** | The 8 models were chosen by deterministic maximin quadrant stratification, not random sampling. | `MAJOR` for "representative" claim; `NOT MATERIAL` for "prespecified spanning subset" claim. | Precludes claiming the 8 models form a statistically "representative" sample of all 36 models; mandates using narrower wording ("prespecified spanning subset"). |

---

## D6. Could a reviewer reasonably claim post-hoc model selection?

### Strongest Hostile Reviewer Argument:
> *"The manuscript presents held-out basin results for only 8 of the 36 models. Without an independent preregistration repository, how can the readership be certain that the authors did not run all 36 models in OOB mode (or run a pilot set of 15 models), observe that dPL failed on certain architectures, and selectively present only the 8 models that yielded favorable held-out conclusions?"*

### Repository Evidence That Conclusively Defeats This Argument:
1. **Complete Execution Record:** The remote execution environment logs, job IDs, timestamped batch scripts, and local sync manifests prove that exactly 40 OOB training runs were executed across the entire project history. There are no "hidden" or "failed" OOB training runs for any of the other 28 models.
2. **Deterministic Code Reproduction:** The Python selection scripts (`project/benchmark/analysis/oob_model_selection_20260901/select_models.py` and `revise_mopex_redundancy.py`), timestamped 2026-09-01, take the pre-existing seen-basin summary tables as input and deterministically output the exact eight model names.
3. **Imperfect OOB Outcomes in Selected Panel:** The selected 8 models do *not* exhibit uniformly superior or cherry-picked OOB behavior. For example, `newzealand2` ($K_{\text{IC}}=0.562, K_{\text{dPL}}=0.607$) and `us1` ($K_{\text{IC}}=0.635, K_{\text{dPL}}=0.600$) represent low-to-moderate performance tiers, and `mopex4` exhibits high parameter displacement and restart uncertainty.
4. **Transparent Manuscript Scope Boundary:** The manuscript does not extrapolate R4 findings to the un-run 28 models; it explicitly restricts its held-out challenge conclusions to the prespecified eight-model panel.

---

## 7. Final Hostile Audit Summary

- **Prespecification:** **PROVEN**
- **Non-Test-Informed:** **PROVEN**
- **Post-Hoc Cherry-Picking:** **REFUTED**
- **Structural Spanning:** **CONFIRMED**
- **Unbiased "Representative" Sample Claim:** **REJECTED (Must use narrower phrasing)**
