#!/usr/bin/env python3
"""Internal harness: execute all S2 audit components in one reproducible run."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv, write_json

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out)
    scripts=["audit_s2_active_model_paths.py","audit_s2_equations.py","audit_s2_parameters.py","audit_s2_coupling.py","audit_s2_mass_balance.py","audit_s2_gradients.py"]
    log=[]; started=time.time()
    for name in scripts:
        cmd=[sys.executable,str(out/"scripts"/name),"--project-root",str(root),"--output-dir",str(out)]
        t=time.time(); proc=subprocess.run(cmd,cwd=root,text=True,capture_output=True); log.append({"command":" ".join(cmd),"exit_code":proc.returncode,"seconds":time.time()-t,"stdout":proc.stdout[-1000:],"stderr":proc.stderr[-2000:]})
        if proc.returncode:
            (out/"results"/"s2_formula_audit_log.txt").write_text(json.dumps(log,indent=2)+"\n"); return proc.returncode
    # literature comparison is deliberately separated from code evidence.
    refs=[
      {"model":"GR4J","canonical_reference":"Perrin, Michel & Andréassian (2003), Improvement of a parsimonious model for streamflow simulation, Journal of Hydrology 279, 275-289","url":"https://doi.org/10.1016/S0022-1694(03)00240-6","accessed":"2026-07-27","implemented_difference":"Implemented same production/routing architecture but differentiable finite S-curve UHs, explicit eps/clamps, and a finite buffer; exact equivalence of every discretization branch is not claimed.","classification":"differentiability modification; discretization difference","status":"SUPPORTED_LITERATURE"},
      {"model":"HBV","canonical_reference":"Bergström, S. (1992), The HBV model: its structure and applications, SMHI RH No. 4","url":"https://www.smhi.se/en/research/research-departments/hydrology/hbv-model-1.16632","accessed":"2026-07-27","implemented_difference":"The current code exposes the stated HBV snow, soil, upper-zone and lower-zone steps; no separate routing convolution is present.","classification":"reference implementation comparison","status":"SUPPORTED_LITERATURE"},
      {"model":"CemaNeige","canonical_reference":"Valéry, Andréassian & Perrin (2014), As simple as possible but not simpler: what is useful in a snow accounting routine?, Journal of Hydrology 517, 1118-1129","url":"https://doi.org/10.1016/j.jhydrol.2013.11.042","accessed":"2026-07-27","implemented_difference":"The active wrapper uses the two-parameter basic variant, a 0-3 degC piecewise partition, fixed g_thresh=0.9 annual solid precipitation and instantaneous SCA; the hysteresis class is not selected by the active CN wrappers.","classification":"parameterization difference; process omitted","status":"SUPPORTED_LITERATURE"},
      {"model":"XAJ","canonical_reference":"Zhao, R.J. (1992), The Xinanjiang model applied in China, Journal of Hydrology 135, 371-381","url":"https://doi.org/10.1016/0022-1694(92)90024-W","accessed":"2026-07-27","implemented_difference":"Current implementation adds explicit differentiability floors, finite gamma UH routing and caps deep evaporation at WD; exact canonical equivalence is not claimed.","classification":"differentiability modification; parameterization difference","status":"SUPPORTED_LITERATURE"},
      {"model":"SIMHYD","canonical_reference":"Chiew, Peel & Western (2002), Application and testing of the simple rainfall-runoff model SIMHYD, in Mathematical Models of Small Watershed Hydrology and Applications","url":"https://www.sciencedirect.com/science/article/pii/B9780444513061500127","accessed":"2026-07-27","implemented_difference":"Current code is a differentiable batched variant with explicit interception, soil overflow transfer, groundwater recession and gamma-UH routing; the exact textbook variant is not inferred from the name.","classification":"parameterization difference; routing difference","status":"SUPPORTED_LITERATURE"},
    ]
    write_csv(out/"results"/"s2_reference_comparison.csv",refs)
    write_csv(out/"results"/"s2_unresolved_items.csv",[
      {"item":"Exact active 531 dPL checkpoint/model route for a published run","status":"UNRESOLVED","evidence_needed":"A run manifest or checkpoint selected for the manuscript; code default is full model, --lite is optional."},
      {"item":"Exact whole-system mass balance for XAJ/GR4J/HBV","status":"UNRESOLVED","evidence_needed":"Daily state/flux traces exposing all storage and ET terms, including UH tail; current aux is insufficient."},
      {"item":"Canonical SIMHYD variant identity","status":"UNRESOLVED","evidence_needed":"Author-selected authoritative SIMHYD variant; current implementation is reported directly without naming it canonical."},
      {"item":"CN hysteresis active use","status":"VERIFIED_CODE","evidence_needed":"Active wrappers import and call _cemaneige_step, not _cemaneige_hyst_step."},
      {"item":"Units of all model inputs in production runner","status":"INFERRED","evidence_needed":"The model interface is physical-scale tensors; active dataset manifest supplies forcing units separately."},
    ])
    with (out / "results" / "s2_one_step_results.csv").open(newline="") as f:
        forward_rows = list(csv.DictReader(f))
    with (out / "results" / "s2_gradient_check_results.csv").open(newline="") as f:
        gradient_rows = list(csv.DictReader(f))
    reports(out,root,log)
    gate={"status":"S2 FORMULA AUDIT PARTIALLY COMPLETE","core_code_inventory":True,"equation_inventory":True,"runtime_forward_probes":True,"forward_rows":len(forward_rows),"forward_failures":sum(row.get("status")=="FAIL" for row in forward_rows),"gradient_rows":len(gradient_rows),"gradient_failures":sum(row.get("status")=="FAIL" for row in gradient_rows),"gradient_checks":"representative fused-step checks completed; see per-row statuses","unresolved_items":5,"generated_at":time.strftime('%Y-%m-%dT%H:%M:%S%z'),"note":"No production code/config/history was modified. Partial status reflects unresolved exact whole-system mass balances, three low-signal float32 GR4J finite-difference rows, and published dPL checkpoint identity."}
    write_json(out/"results"/"s2_final_gate.json",gate)
    (out/"results"/"s2_formula_audit_log.txt").write_text(json.dumps({"elapsed_seconds":time.time()-started,"commands":log,"gate":gate},indent=2)+"\n")
    return 0

def reports(out:Path,root:Path,log:list)->None:
    r=out/"reports"
    report=f'''# S2 Formula Source-of-Truth Audit Report\n\n## Executive summary\n\nThe active 531 foundation configuration selects the model keys recorded in `results/s2_active_call_graph.json`. IC-XNES uses `ablation/ic_core/model_adapter.py` and defaults to full classes; dPL uses the same `models` classes through `training/dpl/run_dpl_model.py` unless `--lite` is explicitly supplied. The active structural controls are Base, basic two-parameter CemaNeige (CN), precipitation delay (PD), and three-parameter temperature-conditioned generic delay (TGD).\n\nThe implementation facts below are code facts, not textbook substitutions.\n\n## Active implementation map\n\n| Route | Evidence | Fact |\n|---|---|---|\n| IC model selection | `ablation/ic_core/model_adapter.py:14-28` | `MODEL_CLASSES` maps all active keys to the classes in `models`. |\n| IC execution | `ablation/ic_core/runtime.py:104-110` | forcing and physical parameters enter the shared model adapter. |\n| dPL execution | `training/dpl/run_dpl_model.py:84-114,627-650` | dPL registry uses the same full/lite model classes; full is default. |\n| 531 design | `ablation/configs/ic_foundation_531_v1.json:2-12` | current foundation configuration, not legacy 559. |\n\n## Model formulas\n\nThe machine-readable equation inventory is `results/s2_equation_inventory.csv`. XAJ has three-layer evaporation, tension-water capacity runoff, free-water separation, linear interflow/groundwater reservoirs and a finite 15-ordinate gamma UH. GR4J has the production store, 0.9/0.1 split, differentiable S-curve UH1/UH2, exchange term and routing store. SIMHYD has interception, exponential infiltration, interflow, recharge, soil overflow transfer, groundwater recession and finite gamma UH. HBV is a standalone five-state explicit snow/soil/response model.\n\n## CN and TGD\n\nCN calls `_cemaneige_step` (`models/composed.py:46-53`) before the host step. Its basic implementation uses a 0/3 degC piecewise solid fraction, G and eTG states, an instantaneous SCA ratio, and storage-limited melt. TGD calls `_temperature_conditioned_delay_step` (`models/temperature_delay.py:26-51`) before the host step. It has a single delay storage, frozen training temperature mean/standard deviation, bounded smooth temperature signal, dynamic tau, and conservative release. TGD has no rain/snow partition, SWE state, or melt equation. Neither module changes PET.\n\n## Coupling and routing\n\nThe coupling matrix is `results/s2_module_coupling_matrix.csv`. In every wrapper the order is raw `P,T,PET`, preprocessing, same-day `effective_precip`, host runoff, and host routing. Base bypasses preprocessing; PD is a separate temperature-agnostic control and must not be described as TGD. The initialization and routing matrix is `results/s2_initialization_and_routing.csv`; finite UH buffers are carried for continuation, while a finite output window does not include its future tail.\n\n## Parameters and bounds\n\n`results/s2_parameter_manifest.csv` contains every active key's code name, symbol, bounds, unit, scope and mapping. The physical parameter adapter is `ablation/ic_core/parameter_adapter.py:56-67`: all parameters use linear bounds except `tgd_tau`, which uses log interpolation. dPL outputs a sigmoid normalized value (`training/dpl/run_dpl_model.py:166-168`) and maps it to physical bounds. Parameters are basin-specific at model execution; network weights are shared but are not hydrological parameters.\n\n## Smoothing and numerical details\n\nThe threshold inventory is `results/s2_threshold_and_smoothing_inventory.csv`. Important implementation-specific details are epsilon denominators, fractional-power base floors, store clamps, `torch.where` branches, TGD `tanh` temperature clipping, `expm1` release calculation, and finite UH normalization. These alter derivatives and sometimes the exact discrete map; they must be reported as implementation details.\n\n## Runtime verification\n\n`results/s2_one_step_results.csv` records deterministic CPU forward probes and `results/s2_mass_balance_results.csv` records short-sequence diagnostics. `results/s2_gradient_check_results.csv` compares autograd with central finite differences for representative interior parameters. A complete whole-system mass balance remains unresolved for XAJ, GR4J and HBV because current active auxiliaries do not expose all daily storage and ET terms and UH tails; this is reported rather than silently treated as a pass.\n\n## Literature comparison\n\n`results/s2_reference_comparison.csv` separates canonical references from implemented equations. The manuscript must lead with implemented formulas. The current CemaNeige wrapper is the basic variant, not the hysteresis class, and the SIMHYD name alone does not establish a unique canonical variant.\n\n## Prohibited or unresolved manuscript claims\n\nDo not state that TGD is an explicit snow model, that external SWE is truth, that CN and TGD have identical state dimension, that all 531 design cells have completed results, or that the current implementation is exactly canonical XAJ/GR4J/SIMHYD/HBV without qualification.\n\n## Generated artifacts\n\nScripts are in `manuscript/supplement/scripts/`; results are in `manuscript/supplement/results/`; candidate equations and readiness are in this report directory.\n'''
    (r/"S2_formula_source_of_truth_report.md").write_text(report)
    (r/"S2_equations_candidate.md").write_text(candidate())
    (r/"S2_manuscript_readiness.md").write_text(readiness())
    (out/"README_S2_AUDIT.md").write_text("# S2 formula audit\n\nRun from the project root:\n\n```bash\nbash manuscript/supplement/scripts/run_s2_formula_audit.sh\n```\n\nThe audit reads active source/configuration paths and writes only `manuscript/supplement`. Partial status is explicit where runtime evidence is insufficient.\n")

def candidate()->str:
    return r'''# S2 Equations Candidate

This candidate reports the implemented equations. `P_t`, `T_t`, and `PET_t` are the daily tensors accepted by the shared model interface. All `max`, `min`, `clamp`, `where`, and epsilon terms below are implementation operations, not editorial simplifications.

### S2.1 XAJ

#### Inputs and states

States are `WU, WL, WD, S, FR, QI, QG` and a 14-sample surface-runoff UH buffer. Initial values are half of the corresponding capacities for WU/WL/WD/S and zero for the remaining states (models/xaj.py:467-493).

#### Daily flux equations

`prcp=max(P_t,0)`, `PET_a=max(k PET_t,0)`. The code computes `EU`, `EL`, and `ED` in the nested XAJ branches, with `ED=min(ED,WD)`. Let `W_0=min(WU+WL+WD,WM-eps)`, `WM=UM+LM+DM`, `PE=max(prcp-EU-EL-ED,0)`, `WMM=WM(1+B)`, and `A=WMM[1-(1-W_0/(WM+eps))^(1/(1+B))]`. The code's piecewise expression then computes `R`, followed by `R_I=KI S FR` and `R_G=KG S FR`.

#### State updates and routing

The code updates WU/WL/WD with its branch-specific formulas, updates FR and S with storage limits, then updates `QI=CI QI_old+(1-CI)R_I(1-IM)` and `QG=CG QG_old+(1-CG)R_G(1-IM)`. Surface input is `RS_adj=RS(1-IM)+IM PE`. It is routed through the finite 15-ordinate gamma UH; final output is `Q=RS_routed+QI+QG` (models/xaj.py:118-171, 308-342).

#### Parameters

The 14 active XAJ parameters and bounds are in `results/s2_parameter_manifest.csv`.

### S2.2 GR4J

#### Inputs and states

States are production store `S_prod`, routing store `S_route`, UH1 buffer length 15, and UH2 buffer length 30. Defaults are `0.5 X1`, `0.5 X3`, and zero buffers (models/gr4j.py:144-190).

#### Daily flux equations

The implementation branches on `P_t>=PET_t`, computes `P_N`, `E_N`, `P_S`, `E_S` with tanh equations, updates `S_prod`, and computes percolation `Perc=S_prod[1-(1+(4S_prod/(9X1))^4)^(-1/4)]`. `P_R=clamp(Perc+P_N-P_S,0,inf)` is split as `0.9P_R` and `0.1P_R` into UH1 and UH2 (models/gr4j.py:17-77).

#### State updates and routing

UH ordinates are S-curve differences, normalized with `eps=1e-8`, using x4 and finite lengths 15/30. The exchange term is `F=X2(S_route/(X3+eps))^3.5`; the routing store receives UH1 and F, produces `Q_R`, while `Q_D=max(UH2+F,0)` and `Q=Q_R+Q_D` (models/gr4j.py:78-108; models/unit_hydro.py:18-71).

### S2.3 SIMHYD

`I=min(INSC_safe,PET_safe,P)`, `I_f=min(COEFF_safe exp(-SQ soil_ratio),P-I)`, `R_D=P-I-I_f`, `R_I=SUB soil_ratio I_f`, `R_G=CRAK soil_ratio(I_f-R_I)`. Soil ET, overflow, groundwater and runoff updates follow models/simhyd.py:52-118. Instant runoff is routed by a finite normalized gamma UH with a 14-sample continuation buffer (models/simhyd.py:165-200).

### S2.4 HBV reference

HBV uses states SNOWPACK, MELTWATER, SM, SUZ, SLZ. Rain/snow is a hard threshold at `parTT`; melt is `min(max(CFMAX(T-TT),0),SNOWPACK)`; refreezing and snow liquid retention follow the exact order in models/hbv.py:30-71. The soil and upper/lower-zone equations are also implemented there. It is a standalone registry model, not one of the CN/TGD wrappers.

### S2.5 Module coupling

For CN, `effective_precip=rain+melt` from `G/eTG`; for TGD, `effective_precip=(1-alpha)P+release` from delay storage `S`; PET is passed unchanged. Each wrapper performs module update then host update on the same day (models/composed.py:46-53; models/composed_temperature_delay.py:32-110).

### S2.6 Differentiability and smoothing

TGD uses `tanh(clamp(z,-5,5))` and `-expm1(-1/tau_t)`. Other modules retain hard `where`/`min`/`max` branches but add eps denominators, capacity clamps and UH normalization. The exact inventory is `results/s2_threshold_and_smoothing_inventory.csv`.

### S2.7 Mass balance and reference verification

TGD preprocessing has a per-step residual of `S_old+P-effective-S_new`; CN has the analogous `P-effective-G_new` diagnostic over a zero-initial snow store. Host-wide balances are only claimed where current auxiliaries expose all terms; see `results/s2_mass_balance_results.csv`.

### S2.8 Parameter bounds

See `results/s2_parameter_manifest.csv`; `tgd_tau` uses log interpolation in `ablation/ic_core/parameter_adapter.py:56-67`, all other active physical parameters use linear interpolation.

### S2.9 HBV snow-process reference

The HBV reference is the implemented explicit snow routine above. It should be described as a model reference, not as evidence that CN and HBV are mathematically identical.
'''

def readiness()->str:
    return '''# S2 Manuscript Readiness

| Section | Ready facts | Missing facts | Required action | Manuscript risk |
|---|---|---|---|---|
| S2.1 | Active XAJ/GR4J/SIMHYD/HBV equations and states | Exact canonical SIMHYD variant | Cite implementation and qualify canonical comparison | Medium |
| S2.2 | Basic two-parameter CN, partition, G/eTG and melt | Literature wording only | Write implemented CemaNeige variant first | Low |
| S2.3 | TGD storage, frozen temperature stats and tau equation | Published dPL checkpoint identity | Confirm run manifest if exact experiment must be named | Medium |
| S2.4 | Parameter counts and code names | Formal author-selected matching convention | Use code-name table; do not claim equal process DOF | Low |
| S2.5 | Same-day wrapper order and unchanged PET | None for code path | Write coupling matrix into S2 | Low |
| S2.6 | Clamp/where/tanh/expm1/UH inventory | Derivative behavior at every branch | Retain implementation caveats | Medium |
| S2.7 | TGD/CN preprocessing diagnostics and partial host probes | Full XAJ/GR4J/HBV whole-system balance | Expose daily traces or leave claim unresolved | High |
| S2.8 | Bounds and normalized mapping | Production checkpoint-specific values | Cite active parameter spec | Low |
| S2.9 | HBV implementation and reference role | Exact author-selected HBV literature edition | Cite source recorded in reference CSV | Low |

The main attribution risk is treating TGD as snow or claiming CN/TGD have the same number of states. The code shows a conservative generic delay versus an explicit snow accounting variant.
'''

if __name__ == "__main__":
    raise SystemExit(main())
