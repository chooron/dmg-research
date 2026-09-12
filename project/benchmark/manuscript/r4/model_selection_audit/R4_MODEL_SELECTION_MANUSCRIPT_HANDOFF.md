# R4 Model Selection Manuscript Handoff

This document provides exact, audited manuscript text for Section 2 (Methods), Section 3 (Results), Section 4 (Discussion), and formal reviewer responses. All statements are verified against the pre-OOB provenance audit.

---

## 1. Methods Sentence

> "To evaluate out-of-bag regionalization under strict held-out-basin conditions without prohibitive computational cost, we prespecified an eight-model subset (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, and `hillslope`) prior to out-of-bag execution using a deterministic 4-quadrant rule across seen-basin performance flexibility ($G_{\text{seen}}$) and parameter reproducibility ($R_{\text{seen}}$), subject to a baseline performance viability gate ($K_{\text{joint}} \ge 0.561$) and a structural anti-redundancy constraint limiting the MOPEX lineage to a single member."

---

## 2. Results / R4 Boundary Sentence

> "Because the 5-fold held-out-basin evaluation was conducted on this prespecified eight-model panel spanning contrasting conceptual formulations, our out-of-bag findings directly stress-test the retention of shared parameter learning across distinct model architectures, but should not be interpreted as an exhaustive 36-model population census."

---

## 3. Discussion Limitation Sentence

> "Although the tested eight-model subset deliberately covers key structural archetypes ($P \in [5, 15]$, $S \in [1, 5]$, snow vs. rain-dominant, and diverse runoff mechanisms) and all four seen-basin flexibility–reproducibility regimes, it is an intentionally stratified stress-test panel rather than an unbiased random sample of the 36-model benchmark."

---

## 4. Reviewer-Response Paragraph

**Question:** *Why were these eight models selected, and could the selection have been informed by out-of-bag (OOB) outcomes?*

> "The eight models were selected strictly prior to out-of-bag execution via a deterministic, rule-based protocol frozen on 2026-09-01/2026-09-02, before any OOB training was launched. Candidate selection operated on the full 36-model pool and relied exclusively on pre-existing seen-basin calibration metrics to span orthogonal quadrants of the IC–dPL performance flexibility gap ($G_{\text{seen}}$) and parameter–attribute reproducibility ($R_{\text{seen}}$). Within each quadrant, models were chosen to maximize 5D feature contrast (in parameter count $P \in [5, 15]$, internal states $S \in [1, 5]$, restart uncertainty $U$, and parameter displacement $D_{\theta}$) under a baseline performance viability gate ($K_{\text{joint}} \ge 0.561$) and an explicit anti-redundancy constraint that capped MOPEX family representation at one model. Full cryptographic timestamps and execution logs confirm that the exact 8-model configuration was packaged and deployed prior to the first completed OOB run, with zero job failures, retries, or post-hoc model substitutions across the 40/40 completed 5-fold cross-validation runs. Consequently, model selection was completely non-test-informed and could not have been influenced by OOB outcomes."
