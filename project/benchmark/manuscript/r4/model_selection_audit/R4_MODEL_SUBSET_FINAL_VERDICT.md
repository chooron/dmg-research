# Final verdict

**B. PRESPECIFIED SPANNING SUBSET SUPPORTED - NO RERUN**

---

# 1. Was the subset prespecified?

**YES (CONFIRMED).**

The eight models (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`) were deterministically selected on **2026-09-01 23:09:19 +0800**, formally frozen into configuration file `project/benchmark/configs/oob_primary8_5fold_20260902.yaml` (SHA256: `1e9e4e03...`) on **2026-09-02 09:23:18 +0800**, and deployed to the remote compute cluster in `oob_deploy_20260902.tar.gz` on **2026-09-02 10:58:35 +0800**.

The first OOB training job (`alpine2_fold0`) completed at **2026-09-02 12:08:08 +0800**. Thus, the exact eight-model subset was frozen and deployed **2 hours 45 minutes to 13 hours prior to the availability of any out-of-bag result**.

---

# 2. Was selection non-test-informed?

**YES (CONFIRMED).**

Model selection was executed using automated scripts (`select_models.py`, `revise_mopex_redundancy.py`) that operated solely on pre-existing seen-basin calibration metrics from R1–R3 ($G_{\text{seen}}, R_{\text{seen}}, D_{\theta}, U, K_{\text{joint}}, P, A$). Zero OOB held-out metrics were calculated, accessed, or known at selection time.

Forensic examination of execution logs (`remote_queue.log`) confirms that all 40 scheduled jobs (8 models $\times$ 5 folds) completed with zero job failures, zero retries, and zero post-hoc model substitutions.

---

# 3. How well does the subset span the 36-model benchmark?

**STRONG 4-QUADRANT SPANNING COVERAGE.**

- **Structural Dimension:** Spans parameter counts from $P=5$ to $P=15$ (71.4% range span), state counts from $S=1$ to $S=5$ (100% range span), proportional snow representation (37.5% vs 33.3% in full benchmark), balanced routing forms (4 base vs 4 endpoint), and contrasting runoff mechanisms (unit hydrographs, parabolic tension water saturation excess, multi-zone conceptual stores, and hillslope kinematic routing).
- **Seen-Basin Response Dimension:** Provides exact orthogonal 2-model coverage across all four quadrants of the seen-basin flexibility gap ($G_{\text{seen}}$) and parameter reproducibility ($R_{\text{seen}}$), covers 99.3% of the IC restart uncertainty range ($U$), and 67.0% of the parameter displacement range ($D_{\theta}$), while enforcing baseline performance viability ($K_{\text{joint}} \ge 0.561$).

---

# 4. Can "representative" be used?

**NO (UNSUPPORTED FOR POPULATION GENERALIZATION).**

While the eight models effectively span contrasting structural archetypes and empirical response regimes, the subset was chosen via deterministic quadrant stratification and contrast maximization rather than independent random sampling. Calling the subset "representative" would falsely imply that sample statistics computed over the eight models serve as an unbiased estimator of the 36-model benchmark population.

---

# 5. Is any rerun/reselection required?

**NO RERUN REQUIRED.**

The pre-OOB selection provenance is complete, untainted by leakage, and cryptographically verified. The 40 completed OOB runs represent a scientifically sound, computationally tractable stress test of held-out-basin parameter learning. No protocol violations or post-hoc cherry-picking occurred.

---

# 6. Exact manuscript wording to use

**Recommended Canonical Phrase:**
> **"a prespecified eight-model subset spanning contrasting conceptual model formulations"**

**Permissible Contextual Variations:**
- *"the prespecified eight-model panel"*
- *"an eight-model subset structured to span contrasting flexibility gaps and parameter reproducibility regimes"*
- *"the tested eight-model subset"*

---

# 7. Remaining limitation

The R4 held-out-basin findings apply strictly to the tested eight-model panel and provide a targeted architectural stress test of shared parameter learning. They cannot be generalized uncritically to the remaining 28 benchmark models, nor do they evaluate severely degraded models ($K_{\text{joint}} < 0.561$) or ultra-parsimonious toy formulations ($P < 5$).

---

# 8. STOP decision

**AUDIT COMPLETE. STOP.**

No further training, calibration, reselection, or simulation should be launched. The historical selection provenance is frozen and ready for manuscript handoff.
