# Current Execution Logic Audit (Revised)

Last updated: after GR4J x4 differentiable fix and XAJ naming correction.

## 1. One-sentence verdict

**PASS — the current eight-model stack (including SIMHYD and gamma routing)
passes the frozen model/test gate; the historical four-model wording below is
retained as provenance for the earlier audit.**

---

## 2. What is already solid

1. **Step-kernel compile pattern**: All four step kernels properly compiled with `torch.compile(fn, fullgraph=True)`. No silent fallback. No compile of forward/loop.

2. **HBV model**: Byte-level match with hydrodl2 reference. Snow module complete (degree-day melt, refreezing, water holding capacity). 12 parameters match reference ranges.

3. **GR4J model**: Production store (eq. 3, 4), percolation (eq. 5), routing store, groundwater exchange (eq. 18), UH routing all match RRMPG reference. x4 is fully differentiable via S-curve difference method.

4. **XAJ model**: All core XAJ processes faithfully implemented from hydromodel reference — three-layer evaporation, tension water capacity curve, free water reservoir separation, three-source division, linear reservoir routing, channel lag routing. Muskingum routing is not included; the hydromodel reference treats routing as a configurable external module. 15 parameters match the current specification.

5. **CemaNeige snow modules**: The two-parameter basic routine and the four-parameter hysteresis routine are exposed separately.

6. **Composed models**: GR4J+CemaNeige, XAJ+CemaNeige, and SIMHYD+CemaNeige use the two-parameter CemaNeige preprocessing module.

7. **Interface**: Uniform forcings/params/states/aux, physical-scale param dicts, batch×time shapes.

8. **No legacy imports**: Zero runtime imports from dmotpy, marrmot, pymarrmot.

---

## 3. Critical risks (resolved from previous audits)

### RESOLVED: XAJ routing

An earlier audit flagged the Muskingum routing as a bug. This has been corrected.
The hydromodel reference itself treats routing (CSL/MZ) as a configurable module,
not a defining part of the core XAJ runoff-generation structure. Channel lag
routing + linear reservoir routing are used, which is a legitimate routing choice.
The model is correctly named XAJ.

### RESOLVED: GR4J x4 not differentiable

Replaced `.item()`-based UH ordinate computation with differentiable S-curve
difference method (`models/unit_hydro.py`). Verified by gradient tests.

**Previous finding**: `_compute_uh_ordinates` used `.item()` calls, detaching x4 from autograd.

**Resolution**: Replaced with `compute_gr4j_uh_ordinates()` in `models/unit_hydro.py` — a fully differentiable S-curve difference implementation. No `.item()`. No per-basin Python loops. Pure tensor operations. Verified by new gradient tests.

**Implementation details**:
- UH1: S1(t) = clamp(t/x4, max=1)^2.5, ordinates = S1(t) - S1(t-1)
- UH2: S2(t) = 0.5*(t/x4)^2.5 for t≤x4; 1-0.5*(2-t/x4)^2.5 for t<x4<2x4
- Ordinates normalized per basin to sum to 1
- Sequential UH buffer updates remain in the compiled step kernel
- `x4` no longer passed to step kernel (only pre-computed ordinates)

---

## 4. Remaining non-blocking caveats

1. **Warmup**: No built-in warmup in any model. Must be handled externally by training pipeline.

2. **XAJ parameter count**: 15 parameters × batch can make dPL optimization challenging. This is inherent to the experimental design.

3. **CemaNeige simplifications**: Single elevation zone and simplified solid fraction (mean temp only); Psolannual is estimated from forcing statistics.

4. **GR4J UH max length**: Fixed at UH1_MAX=15, UH2_MAX=30. Truncation acceptable since x4 range is [1.1, 10.0].

5. **No water balance / snow sanity tests**: Recommended to add before Phase B training.

---

## 5. Model-by-model summary

### HBV
- **Verdict**: PASS — ready for Phase B
- **Status**: No changes from previous audit. Correctly matches hydrodl2 reference.

### GR4J
- **Verdict**: PASS — x4 now differentiable, ready for Phase B
- **Changes**: `_compute_uh_ordinates` replaced with differentiable `compute_gr4j_uh_ordinates`. Dead code removed from step kernel. x4 removed from step kernel args (only ordinates needed).

### XAJ (XinAnJiang)
- **Verdict**: PASS — core XAJ processes faithfully implemented; Muskingum routing is optional per reference
- **Changes**: Removed `xaj_cs` parameter (Muskingum not used due to optional routing per reference).
- **Note**: The hydromodel reference treats routing (CSL/MZ) as a configurable module. Channel lag routing is used here.

### CemaNeige
- **Verdict**: PASS — two-parameter basic routine with documented simplifications

### CemaNeigeHyst
- **Verdict**: PASS — four-parameter hysteresis routine retained as a standalone interface

### GR4J+CemaNeige
- **Verdict**: PASS — inherits GR4J fix

### XAJ+CemaNeige
- **Verdict**: CONDITIONAL PASS — XAJ core is correct; inherits CemaNeige simplifications

---

## 6. torch.compile audit (unchanged)

All step kernels follow compiled step + eager Python loop pattern. No regression.

---

## 7. IC/dPL readiness

The GR4J x4 fix eliminates the previous artificial asymmetry between IC and dPL. All four model parameters (x1, x2, x3, x4) are now fully differentiable.

Remaining concern: no warmup mechanism in any model. This affects both IC and dPL equally, so it doesn't bias Δ. But it must be added before final experiments.

---

## 8. Test status

- **31 tests pass** (26 original + 5 new GR4J x4 tests)
- New tests verify:
  - x4 has nonzero gradient through full forward
  - UH ordinates are differentiable
  - conv1d routing is differentiable
  - Batch independence of UH routing
  - x4 affects timing (different x4 → different qsim)

---

## 9. Recommended next action

**CONDITIONAL PASS — ready for Phase B** after confirming:
1. Warmup strategy (external or built-in)
2. Whether to add water balance/snow sanity tests before or during Phase B

---

## 10. Superseding baseline

The four-model audit below predates SIMHYD and the unified registry. The
authoritative current model list and test result are in
[`model_test_baseline.md`](model_test_baseline.md), and the active IC/dPL
entry points are in [`training_baseline.md`](training_baseline.md).

## 11. Files inspected/modified this round

Modified:
- `models/unit_hydro.py` — NEW: differentiable UH ordinates + conv1d routing
- `models/gr4j.py` — uses new differentiable UH, removed `.item()`, removed dead code
- `models/xaj.py` — removed cs/pseudo-Muskingum, clarified XAJ naming
- `models/parameter_specs.py` — removed `xaj_cs` from XAJ specs
- `tests/test_step_compile.py` — updated GR4J and XAJ input generators
- `tests/test_gr4j_x4_gradient.py` — NEW: 5 gradient/batch/timing tests
- `docs/current_execution_logic_audit.md` — updated (this file)
