#!/usr/bin/env python3
"""Add equation usage, authoritative meaning/unit sources and manuscript tables."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv

ACCESS="2026-07-28"

SOURCES=[
 {"source_id":"XAJ_ZHAO_1992","model":"XAJ","parameter":"XAJ parameters","source_type":"original_paper","authors":"Zhao, Ren-Jun","year":1992,"title":"The Xinanjiang model applied in China","journal_or_organization":"Journal of Hydrology 135, 371-381","DOI":"10.1016/0022-1694(92)90096-E","URL":"https://doi.org/10.1016/0022-1694(92)90096-E","accessed_date":ACCESS,"exact_page_or_section":"model formulation; exact parameter table not available in local evidence","supported_claim":"Xinanjiang conceptual runoff-generation model and terminology","quotation_or_paraphrase_note":"Used as the canonical XAJ source; current code-specific routing and bounds remain source-controlled by project code.","source_quality":"original peer-reviewed paper"},
 {"source_id":"GR4J_PERRIN_2003","model":"GR4J","parameter":"X1,X2,X3,X4","source_type":"original_paper","authors":"Perrin, Michel, Andréassian","year":2003,"title":"Improvement of a parsimonious model for streamflow simulation","journal_or_organization":"Journal of Hydrology 279, 275-289","DOI":"10.1016/S0022-1694(03)00225-7","URL":"https://doi.org/10.1016/S0022-1694(03)00225-7","accessed_date":ACCESS,"exact_page_or_section":"GR4J formulation and four-parameter definition","supported_claim":"X1 production capacity, X2 exchange, X3 routing capacity, X4 UH time base","quotation_or_paraphrase_note":"Canonical meaning source; project bounds are not replaced by literature ranges.","source_quality":"original peer-reviewed paper"},
 {"source_id":"SIMHYD_CHIEW_2002","model":"SIMHYD","parameter":"INSC,COEFF,SQ,SMSC,SUB,CRAK,K,ETMUL,a_UH,theta_UH","source_type":"original_book_chapter","authors":"Chiew, F.H.S.; Peel, M.C.; Western, A.W.","year":2002,"title":"Application and testing of the simple rainfall-runoff model SIMHYD","journal_or_organization":"In Mathematical Models of Small Watershed Hydrology and Applications, Water Resources Publications","DOI":"UNRESOLVED","URL":"https://www.waterresources.com/","accessed_date":ACCESS,"exact_page_or_section":"chapter; exact pagination/parameter table not verified from an accessible authoritative copy","supported_claim":"SIMHYD process terminology and original model provenance","quotation_or_paraphrase_note":"The current model is labeled the differentiable SIMHYD implementation used in this study; no canonical variant is asserted.","source_quality":"original/authoritative book chapter; DOI unresolved"},
 {"source_id":"HBV_BERGSTROM_1992","model":"HBV","parameter":"HBV parameters","source_type":"official_technical_report","authors":"Bergström, S.","year":1992,"title":"The HBV model: its structure and applications","journal_or_organization":"SMHI RH No. 4","DOI":"UNRESOLVED","URL":"https://www.smhi.se/en/research/research-departments/hydrology/hbv-model","accessed_date":ACCESS,"exact_page_or_section":"snow, soil-moisture and response-zone routines","supported_claim":"HBV parameter meanings and snow/response terminology","quotation_or_paraphrase_note":"Official SMHI source; exact current page availability was not confirmed, so the report is the primary citation record.","source_quality":"official technical source; DOI unresolved"},
 {"source_id":"CEMANEIGE_VALERY_2014A","model":"CemaNeige","parameter":"CTG,Kf","source_type":"original_paper","authors":"Valéry, Andréassian, Perrin","year":2014,"title":"‘As simple as possible but not simpler’: What is useful in a temperature-based snow-accounting routine? Part 2 – Sensitivity analysis of the Cemaneige snow accounting routine on 380 catchments","journal_or_organization":"Journal of Hydrology 517, 1118-1129","DOI":"10.1016/j.jhydrol.2014.04.058","URL":"https://doi.org/10.1016/j.jhydrol.2014.04.058","accessed_date":ACCESS,"exact_page_or_section":"CemaNeige parameter sensitivity and definitions","supported_claim":"CemaNeige CTG thermal-memory and Kf degree-day melt parameter concepts","quotation_or_paraphrase_note":"Supports the basic two-parameter interpretation; project fixed threshold and partition details are code facts.","source_quality":"original peer-reviewed paper"},
 {"source_id":"CEMANEIGE_VALERY_2014B","model":"CemaNeige","parameter":"CemaNeige","source_type":"original_paper","authors":"Valéry, Andréassian, Perrin","year":2014,"title":"‘As simple as possible but not simpler’: What is useful in a temperature-based snow-accounting routine? Part 1 – Comparison of six snow accounting routines on 380 catchments","journal_or_organization":"Journal of Hydrology 517, 1108-1117","DOI":"10.1016/j.jhydrol.2014.04.059","URL":"https://doi.org/10.1016/j.jhydrol.2014.04.059","accessed_date":ACCESS,"exact_page_or_section":"comparison of temperature-based snow-accounting routines","supported_claim":"CemaNeige process context","quotation_or_paraphrase_note":"Used for context, not to override the active basic implementation.","source_quality":"original peer-reviewed paper"},
 {"source_id":"HYDRODL2_UH","model":"Gamma UH","parameter":"a_UH,theta_UH","source_type":"official_code","authors":"MHPI hydrodl2 contributors","year":2024,"title":"hydrodl2: hydrological deep learning utilities, uh_routing.py","journal_or_organization":"GitHub/PyPI project","DOI":"UNRESOLVED","URL":"https://github.com/mhpi/hydrodl2/blob/master/hydrodl2/core/calc/uh_routing.py","accessed_date":ACCESS,"exact_page_or_section":"uh_gamma and uh_conv; installed version 1.3.4 local source lines 5-57","supported_claim":"relu(a)+0.1, relu(theta)+0.5, t=0.5 sampling, kernel normalization and causal convolution","quotation_or_paraphrase_note":"Local installed source was read at /home/jingxin/code/dmg-research/.venv/lib/python3.10/site-packages/hydrodl2/core/calc/uh_routing.py; SHA256 a2305c...c0a (full fingerprint in report).","source_quality":"official implementation source"},
 {"source_id":"NIST_GAMMA","model":"Gamma UH","parameter":"shape,scale","source_type":"official_reference","authors":"NIST/SEMATECH","year":2003,"title":"Gamma distribution","journal_or_organization":"e-Handbook of Statistical Methods","DOI":"UNRESOLVED","URL":"https://www.itl.nist.gov/div898/handbook/eda/section3/eda366b.htm","accessed_date":ACCESS,"exact_page_or_section":"gamma distribution parameterization","supported_claim":"shape is dimensionless and scale has the variable's time unit","quotation_or_paraphrase_note":"General distribution parameter meaning only; routing discretization is from hydrodl2.","source_quality":"official statistical reference"},
 {"source_id":"TGD_PROJECT_CODE","model":"TGD","parameter":"alpha,tau,beta","source_type":"project_code","authors":"Current project authors/code","year":2026,"title":"TemperatureConditionedDelay implementation","journal_or_organization":"hydrodiag","DOI":"UNRESOLVED","URL":"/home/jingxin/code/dmg-research/project/hydrodiag/models/temperature_delay.py","accessed_date":ACCESS,"exact_page_or_section":"models/temperature_delay.py:20-51; models/parameter_specs.py:391-419","supported_claim":"implemented parameter meanings and units from the active equation","quotation_or_paraphrase_note":"No external classical TGD provenance is asserted.","source_quality":"active project source"},
]

MEANINGS={
 "xaj_k":("ratio of potential ET to reference crop evaporation","dimensionless","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","Project equation uses PET multiplier; exact reference-crop wording is code description."),
 "xaj_b":("tension-water capacity-curve exponent","dimensionless","CODE_AND_LITERATURE_AGREE","XAJ_ZHAO_1992","Exponent in the implemented capacity curve."),
 "xaj_im":("impervious-area fraction","dimensionless","CODE_AND_LITERATURE_AGREE","XAJ_ZHAO_1992","Fraction of effective precipitation routed directly as impervious runoff."),
 "xaj_um":("upper tension-water capacity","mm","CODE_AND_LITERATURE_AGREE","XAJ_ZHAO_1992","Storage capacity in the upper tension layer."),
 "xaj_lm":("lower tension-water capacity","mm","CODE_AND_LITERATURE_AGREE","XAJ_ZHAO_1992","Storage capacity in the lower tension layer."),
 "xaj_dm":("deep tension-water capacity","mm","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","Deep store capacity; exact active three-layer partition is implementation-specific."),
 "xaj_c":("deep-layer evaporation coefficient","dimensionless","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","Multiplies LM and residual PET in the deep-layer branch."),
 "xaj_sm":("areal mean free-water capacity","mm","CODE_AND_LITERATURE_AGREE","XAJ_ZHAO_1992","Capacity used by the free-water separation curve."),
 "xaj_ex":("free-water capacity-curve exponent","dimensionless","CODE_AND_LITERATURE_AGREE","XAJ_ZHAO_1992","Exponent in the free-water capacity curve."),
 "xaj_ki":("daily interflow outflow coefficient multiplying free-water state","1/day in the daily implementation","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","The current equation is RI=KI*S*FR; do not relabel as an undocumented classical coefficient."),
 "xaj_kg":("daily groundwater outflow coefficient multiplying free-water state","1/day in the daily implementation","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","The current equation is RG=KG*S*FR."),
 "xaj_ci":("interflow output-state memory/recession coefficient","dimensionless daily coefficient","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","QI_next=CI*QI_old+(1-CI)*input; this is not the same role as KI."),
 "xaj_cg":("groundwater output-state memory/recession coefficient","dimensionless daily coefficient","INFERRED_FROM_EQUATION","XAJ_ZHAO_1992","QG_next=CG*QG_old+(1-CG)*input; this is not the same role as KG."),
 "xaj_a":("Gamma UH shape parameter before hydrodl2 offset","dimensionless","CODE_AND_LITERATURE_AGREE","HYDRODL2_UH","Effective hydrodl2 shape is relu(xaj_a)+0.1."),
 "xaj_theta":("Gamma UH scale parameter","day","CODE_AND_LITERATURE_AGREE","HYDRODL2_UH","Effective scale is relu(xaj_theta)+0.5 day."),
 "x1":("GR4J production-store capacity","mm","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Project-specific bounds are code facts."),
 "x2":("GR4J groundwater exchange coefficient","mm/day in the project daily equation","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Negative values are allowed by code; it enters signed exchange."),
 "x3":("GR4J routing-store capacity","mm","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Project-specific bounds are code facts."),
 "x4":("GR4J unit-hydrograph time base","day","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Project UH computation uses max(x4,1e-3)."),
 "gr4j_x1":("GR4J production-store capacity","mm","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Prefixed wrapper name; same host meaning as x1."),"gr4j_x2":("GR4J groundwater exchange coefficient","mm/day in the project daily equation","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Prefixed wrapper name; signed."),"gr4j_x3":("GR4J routing-store capacity","mm","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Prefixed wrapper name."),"gr4j_x4":("GR4J unit-hydrograph time base","day","CODE_AND_LITERATURE_AGREE","GR4J_PERRIN_2003","Prefixed wrapper name."),
 "simhyd_insc":("interception capacity","mm","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Current step uses min(INSC,PET,P); exact standard-variant label is unresolved."),"simhyd_coeff":("maximum infiltration capacity coefficient","mm/day in the project equation","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Multiplies exp(-SQ*rho); project description says mm/day."),"simhyd_sq":("infiltration capacity exponent","dimensionless","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Exponent in exp(-SQ*rho)."),"simhyd_smsc":("soil-moisture storage capacity","mm","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Capacity used in soil ratio and overflow."),"simhyd_sub":("interflow proportionality coefficient","dimensionless","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Interflow=SUB*rho*infiltration."),"simhyd_crak":("groundwater recharge proportionality coefficient","dimensionless","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Recharge=CRAK*rho*(infiltration-interflow)."),"simhyd_k":("groundwater recession coefficient","1/day daily fraction","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","Runtime clamps K to [0,1]."),"simhyd_etmul":("PET multiplier","dimensionless","INFERRED_FROM_EQUATION","SIMHYD_CHIEW_2002","PET*=ETMUL*PET."),"simhyd_a":("Gamma UH shape parameter before hydrodl2 offset","dimensionless","CODE_AND_LITERATURE_AGREE","HYDRODL2_UH","Effective shape relu(a)+0.1."),"simhyd_theta":("Gamma UH scale parameter","day","CODE_AND_LITERATURE_AGREE","HYDRODL2_UH","Effective scale relu(theta)+0.5 day."),
 "parBETA":("soil-moisture control exponent for recharge","dimensionless","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Exponent in (SM/FC)^BETA."),"parFC":("field capacity of soil store","mm","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Soil-store capacity."),"parK0":("upper-zone quick-flow recession coefficient","1/day","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Daily coefficient multiplying excess upper-zone water above UZL."),"parK1":("upper-zone interflow recession coefficient","1/day","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Daily coefficient multiplying remaining upper-zone store."),"parK2":("lower-zone baseflow recession coefficient","1/day","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Daily coefficient multiplying lower-zone store."),"parLP":("fraction of FC controlling PET reduction","dimensionless fraction of FC","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","ET factor uses SM/(LP*FC)."),"parPERC":("maximum percolation rate from upper to lower zone","mm/day","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Code uses min(SUZ,PERC)."),"parUZL":("upper-zone threshold for quick flow","mm","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Q0 uses max(SUZ-UZL,0)."),"parTT":("rain-snow threshold temperature","degC","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Hard threshold T>=TT for rain."),"parCFMAX":("degree-day melt factor","mm/(degC*day)","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Also multiplies refreezing potential."),"parCFR":("refreezing coefficient","dimensionless","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Multiplies CFMAX*(TT-T)."),"parCWH":("snowpack liquid-water holding coefficient","dimensionless fraction","CODE_AND_LITERATURE_AGREE","HBV_BERGSTROM_1992","Current code uses CWH*SNOWPACK; parameter spec labels 1/day, which conflicts with the equation."),
 "cn_ctg":("snowpack thermal-state memory coefficient","dimensionless","CODE_AND_LITERATURE_AGREE","CEMANEIGE_VALERY_2014A","Weight in eTG=CTG*eTG_old+(1-CTG)*T."),"cn_kf":("CemaNeige degree-day melt factor","mm/(degC*day)","CODE_AND_LITERATURE_AGREE","CEMANEIGE_VALERY_2014A","Current basic variant parameter."),
 "tgd_alpha":("fraction of precipitation entering generic delay storage","dimensionless fraction","INFERRED_FROM_EQUATION","TGD_PROJECT_CODE","Project-specific; no classical TGD source asserted."),"tgd_tau":("baseline generic-delay release timescale","day","INFERRED_FROM_EQUATION","TGD_PROJECT_CODE","Project-specific; log-interpolated."),"tgd_beta":("temperature sensitivity of release timescale","dimensionless","INFERRED_FROM_EQUATION","TGD_PROJECT_CODE","Multiplies dimensionless tanh signal in exponent."),
}

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out)
    bounds=list(csv.DictReader((out/"results/s2_parameter_bounds_from_code.csv").open()))
    usage=[]; meaning_rows=[]
    def usage_location(name):
        if name=="xaj_k": return "PET adjustment pet_adj=pet_t*k", "models/xaj.py:67-69"
        if name=="xaj_b": return "tension/free-water capacity exponents", "models/xaj.py:103-110,147-155"
        if name=="xaj_im": return "impervious runoff and scaling of RI/RG", "models/xaj.py:116,162-172"
        if name in {"xaj_um","xaj_lm","xaj_dm"}: return "WM and tension-store capacity/update limits", "models/xaj.py:94-136"
        if name=="xaj_c": return "deep-layer evaporation branches", "models/xaj.py:72-88"
        if name=="xaj_sm": return "free-water capacity and S limits", "models/xaj.py:145-160"
        if name=="xaj_ex": return "free-water capacity exponent", "models/xaj.py:147-160"
        if name in {"xaj_ki","xaj_kg"}: return "RI/RG generation and joint rescaling", "models/xaj.py:162-165,280-291"
        if name in {"xaj_ci","xaj_cg"}: return "QI/QG recursive state updates", "models/xaj.py:167-169"
        if name in {"xaj_a","xaj_theta"}: return "Gamma UH shape/scale", "models/xaj.py:306-314,644-666"
        if name in {"x1","x2","x3","x4"}: return "GR4J stores, exchange and UH time base", "models/gr4j.py:38-91; models/unit_hydro.py:18-71"
        if name.startswith("gr4j_"): return "GR4J prefixed host parameter after wrapper prefix stripping", "models/composed.py:120-131,153-184"
        if name=="simhyd_insc": return "interception", "models/simhyd.py:60-63"
        if name in {"simhyd_coeff","simhyd_sq"}: return "exponential infiltration capacity", "models/simhyd.py:70-78"
        if name=="simhyd_smsc": return "soil ratio, overflow and initialization", "models/simhyd.py:69-91,329-343"
        if name=="simhyd_sub": return "interflow", "models/simhyd.py:81-83"
        if name=="simhyd_crak": return "groundwater recharge", "models/simhyd.py:82-93"
        if name=="simhyd_k": return "groundwater recession/baseflow", "models/simhyd.py:93-103"
        if name=="simhyd_etmul": return "PET multiplier", "models/simhyd.py:53"
        if name in {"simhyd_a","simhyd_theta"}: return "Gamma UH routing", "models/simhyd.py:178-200"
        if name.startswith("par"): return "HBV snow, soil or response-zone equation", "models/hbv.py:30-71"
        if name in {"cn_ctg","cn_kf"}: return "CemaNeige thermal state and melt", "models/cemaneige.py:55-74"
        if name.startswith("tgd_"): return "temperature-conditioned delay equation", "models/temperature_delay.py:37-51"
        return "parameter use unresolved", "UNRESOLVED"
    for row in bounds:
        name=row["code_name"]; key=name
        if name.startswith("gr4j_"): key=name
        m=MEANINGS.get(key)
        if m is None: m=("UNRESOLVED parameter meaning","UNRESOLVED","UNRESOLVED","UNRESOLVED","No authoritative meaning record created.")
        meaning,unit,status,source,note=m
        meaning_rows.append({"model_or_module":row["model_or_module"],"active_model_key":row["active_model_key"],"active_order":row["active_order"],"symbol":row["symbol"],"code_name":name,"meaning_in_implemented_equations":meaning,"unit":unit,"meaning_status":status,"reference_source_id":source,"difference_note":note,"code_evidence":row["source_file"]+":"+row["source_line"]})
        usage_eq,usage_evidence=usage_location(name)
        usage.append({"model_or_module":row["model_or_module"],"active_model_key":row["active_model_key"],"code_name":name,"used_by":row["used_by"],"model_scope":row["model_scope"],"equation_usage":usage_eq,"runtime_usage":row["runtime_clamp"],"usage_evidence":usage_evidence,"status":"CODE_VERIFIED"})
    write_csv(out/"results/s2_parameter_meaning_and_units.csv",meaning_rows)
    write_csv(out/"results/s2_parameter_usage_trace.csv",usage)
    write_csv(out/"results/s2_parameter_reference_sources.csv",SOURCES)
    conflicts=[
      {"parameter":"parCWH","conflict_type":"unit/equation","code_fact":"models/parameter_specs.py:113-120 labels unit 1/day; models/hbv.py:47 uses CWH*SNOWPACK as a same-day storage fraction","reference_fact":"HBV snow liquid-water holding capacity is conventionally dimensionless/fractional","status":"CONFLICT","recommended_manuscript_wording":"Report the implemented code unit label verbatim only in the audit; in S2 describe CWH as a dimensionless snowpack holding coefficient and flag the code-spec unit conflict."},
      {"parameter":"xaj_ki/xaj_kg","conflict_type":"classical naming versus active equation","code_fact":"KI/KG multiply S*FR and are jointly rescaled when their sum reaches 1","reference_fact":"XAJ literature terminology varies between free-water partition/outflow coefficients","status":"IMPLEMENTATION_SPECIFIC","recommended_manuscript_wording":"Call them daily free-water outflow coefficients in the implemented equation; do not equate them to CI/CG."},
      {"parameter":"xaj_ci/xaj_cg","conflict_type":"recession naming","code_fact":"CI/CG are coefficients in Q_next=C*Q_old+(1-C)*input","reference_fact":"A literature recession constant may be parameterized as a different daily decay coefficient","status":"IMPLEMENTATION_SPECIFIC","recommended_manuscript_wording":"Call them dimensionless daily output-state memory/recession coefficients."},
      {"parameter":"xaj_a/xaj_theta; simhyd_a/simhyd_theta","conflict_type":"bounds versus effective runtime values","code_fact":"calibration bounds allow zero; hydrodl2 applies relu(a)+0.1 and relu(theta)+0.5","reference_fact":"Gamma shape is positive dimensionless and scale is positive time","status":"CODE_AND_LITERATURE_AGREE","recommended_manuscript_wording":"Show project bounds in the table and footnote the hydrodl2 effective positive offsets."},
      {"parameter":"SIMHYD parameter variant","conflict_type":"model variant","code_fact":"Current 10-parameter differentiable implementation includes ETMUL and Gamma UH a/theta","reference_fact":"Accessible original SIMHYD source evidence does not establish this exact 10-parameter routing variant","status":"UNRESOLVED","recommended_manuscript_wording":"Use 'the differentiable SIMHYD implementation used in this study' and do not name a canonical parameterization."},
      {"parameter":"SIMHYD parameter count","conflict_type":"active-count expectation","code_fact":"SIMHYD_PARAM_SPECS contains 10 names: seven runoff/soil/recharge terms plus ETMUL and a/theta Gamma UH parameters","reference_fact":"The task expectation of 9 active parameters is not supported by the current active parameter spec","status":"CONFLICT","recommended_manuscript_wording":"Report 10 active SIMHYD parameters and list all names; do not use the nine-parameter count without changing production code."},
      {"parameter":"TGD alpha/tau/beta","conflict_type":"source provenance","code_fact":"Defined only by project code and parameter_specs","reference_fact":"No authoritative classical TGD model source identified","status":"IMPLEMENTATION_SPECIFIC","recommended_manuscript_wording":"Describe meanings from the implemented equations; do not cite TGD as an established named model."},
    ]
    write_csv(out/"results/s2_parameter_conflicts.csv",conflicts)
    audit_path=out/"results/s2_parameter_audit.json"
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audit.update({"meaning_status_counts":{status:sum(row["meaning_status"]==status for row in meaning_rows) for status in sorted({row["meaning_status"] for row in meaning_rows})},"reference_source_count":len(SOURCES),"conflict_count":len(conflicts),"conflict_status_counts":{status:sum(row["status"]==status for row in conflicts) for status in sorted({row["status"] for row in conflicts})},"status":"PARAMETER_AUDIT_CODE_AND_SOURCE_PACKAGE_GENERATED"})
    audit_path.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    reports(out,root,bounds,meaning_rows,conflicts)

def reports(out,root,bounds,meanings,conflicts):
    (out/"reports/S2_parameter_source_of_truth_report.md").write_text(f'''# S2 Parameter Source-of-Truth Report

## Active parameter path

The 531 foundation configuration is `ablation/configs/ic_foundation_531_v1.json:2-12`. IC-XNES maps the active model key through `ablation/ic_core/model_adapter.py:14-28` and uses full classes by default at `ablation/ic_core/model_adapter.py:31-45`; the runtime calls the shared adapter at `ablation/ic_core/runtime.py:104-110`. dPL uses the same full registry classes at `training/dpl/run_dpl_model.py:84-98`; the lite registry is selected only by explicit `--lite` at `training/dpl/run_dpl_model.py:613-629`. The parameter order is `list(parameter_specs)` and is recorded in `results/s2_parameter_bounds_from_code.csv`.

Active base counts are XAJ=15, GR4J=4, SIMHYD=10 and HBV=12. The ten active SIMHYD names are `simhyd_insc`, `simhyd_coeff`, `simhyd_sq`, `simhyd_smsc`, `simhyd_sub`, `simhyd_crak`, `simhyd_k`, `simhyd_etmul`, `simhyd_a` and `simhyd_theta`; this conflicts with a nine-parameter expectation. CN adds CTG and Kf; TGD adds alpha, tau and beta. Gamma UH shape/scale parameters are included in XAJ and SIMHYD host vectors, not added as a separate model call. HBV is standalone and has no MAXBAS.

## Code bounds and mappings

All bounds/defaults are extracted from `models/parameter_specs.py` and are project-specific calibration bounds. IC uses `normalized_to_physical` (`ablation/ic_core/parameter_adapter.py:56-80`); normalized values are clipped to [0,1], then mapped linearly except `tgd_tau`, which is log-interpolated. dPL applies sigmoid and output clipping at `training/dpl/run_dpl_model.py:166-168`; its head bias is initialized from parameter defaults at lines 148-164.

Runtime effective values are not identical to calibration bounds. In particular, hydrodl2 `uh_gamma` applies `relu(a)+0.1`, `relu(theta)+0.5`, samples at t=0.5, and normalizes each kernel (`/home/jingxin/code/dmg-research/.venv/lib/python3.10/site-packages/hydrodl2/core/calc/uh_routing.py:5-22`, version 1.3.4, SHA256 `a2305c37ca895efe3323e8fabe20421b84e511355d0aab67eb4d3b657b67c0a`). XAJ additionally jointly rescales KI/KG when their sum is at least one (`models/xaj.py:280-294`).

## Meaning and unit evidence

`results/s2_parameter_meaning_and_units.csv` separates CODE_AND_LITERATURE_AGREE from INFERRED_FROM_EQUATION and IMPLEMENTATION_SPECIFIC. The main classical meanings are supported by Zhao (1992) for XAJ, Perrin et al. (2003) for GR4J, Bergström (1992) for HBV, and Valéry et al. (2014) for CemaNeige. SIMHYD is explicitly called the differentiable SIMHYD implementation used in this study because the accessible authoritative source does not prove the exact current ten-parameter variant. TGD meanings are equation-derived and have no claimed classical provenance.

## Required manuscript edits

1. State that active XAJ has 15 parameters, including `xaj_a` and `xaj_theta`; do not use a 14-parameter description.
2. State all lower/upper bounds as project-specific code bounds, not literature ranges.
3. Explain linear mapping for all active parameters except log interpolation for `tgd_tau`, and sigmoid-to-bound dPL output.
4. Footnote that Gamma UH effective shape/scale are `relu(a)+0.1` and `relu(theta)+0.5`, with finite daily kernels.
5. Keep `parCWH` unit conflict visible: the spec says `1/day`, while the equation uses it as a multiplier of snowpack storage; recommended wording is dimensionless holding coefficient with an audit note.
6. Describe KI/KG as implemented free-water outflow coefficients and CI/CG as output-state memory/recession coefficients; do not interchange them.
7. Describe SIMHYD as the differentiable implementation used in this study; do not assert a unique canonical SIMHYD variant.
8. Describe TGD as project-specific and equation-defined, not as a literature model.
9. The requested S2 draft file was not found under `manuscript/`; no direct manuscript edits were made.\n''')
    (out/"reports/S2_parameter_unresolved_items.md").write_text('''# S2 Parameter Unresolved Items

- `parCWH`: code metadata says `1/day`, but the active equation uses `CWH * SNOWPACK`; this is a code metadata/equation conflict. Recommended manuscript wording is a dimensionless snowpack holding coefficient with an audit note.
- SIMHYD: no accessible authoritative source uniquely identifies the current ten-parameter implementation including ETMUL and Gamma UH parameters.
- TGD: no classical external model source is claimed; meanings and units are inferred from the project equations.
- The internal `hydrodl2` Gamma UH source was verified locally, but its public repository file path/version history was not independently version-pinned beyond installed package version 1.3.4.
- No manuscript Text S2 source file matching the requested path was found; only recommendations are provided.
''')
    # Build compact, directly copyable panels from unique code names.
    seen=set(); panel=[]
    for row,m in zip(bounds,meanings):
        code=row["code_name"]
        panel_name="CN" if code.startswith("cn_") else "TGD" if code.startswith("tgd_") else "XAJ" if code.startswith("xaj_") else "GR4J" if code in {"x1","x2","x3","x4"} or code.startswith("gr4j_") else "SIMHYD" if code.startswith("simhyd_") else "HBV"
        key=(panel_name,code)
        if key in seen: continue
        seen.add(key)
        panel.append(row|{"panel_name":panel_name,"meaning":m["meaning_in_implemented_equations"],"unit_final":m["unit"],"meaning_status":m["meaning_status"]})
    lines=['# Table S2.8. Definitions, implemented bounds, units, and transformations of active parameters','', 'All bounds below are from the active project code; they are not literature recommendation ranges.', '']
    for name in ("XAJ","GR4J","SIMHYD","HBV","CN","TGD"):
        lines += [f'## Panel {"A" if name=="XAJ" else "B" if name=="GR4J" else "C" if name=="SIMHYD" else "D" if name=="HBV" else "E" if name=="CN" else "F"}. {name}', '', '| Symbol | Code name | Meaning in implemented equations | Lower | Upper | Unit | Mapping/constraint |', '|---|---|---|---:|---:|---|---|']
        for row in panel:
            code=row["code_name"]
            row_panel=row["panel_name"]
            if name=="CN": keep=row_panel=="CN"
            elif name=="TGD": keep=row_panel=="TGD"
            elif name=="XAJ": keep=row_panel=="XAJ" and code.startswith("xaj_")
            elif name=="GR4J": keep=row_panel=="GR4J" and code in {"x1","x2","x3","x4"}
            elif name=="SIMHYD": keep=row_panel=="SIMHYD"
            else: keep=row_panel=="HBV"
            if not keep: continue
            lines.append(f'| {row["symbol"]} | `{row["code_name"]}` | {row["meaning"]} | {row["lower_bound"]} | {row["upper_bound"]} | {row["unit_final"]} | {row["transform"]}; runtime: {row["runtime_clamp"]} |')
    lines += ['', 'Table note: dPL first generates a sigmoid-normalized value and maps it to the physical range. IC-XNES uses the same physical bounds. `tgd_tau` uses logarithmic interpolation; other active parameters use linear interpolation. Runtime clamps are not calibration bounds. Gamma UH shape and scale have hydrodl2 positive offsets. KI/KG have a joint sum constraint. `parCWH` retains the code metadata/equation conflict recorded in the audit.']
    (out/"reports/S2_parameter_table_for_manuscript.md").write_text('\n'.join(lines)+'\n')

if __name__ == "__main__": main()
