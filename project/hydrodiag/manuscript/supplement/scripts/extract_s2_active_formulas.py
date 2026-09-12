#!/usr/bin/env python3
"""Extract source-backed formula, state, routing and coupling inventories."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv, write_json

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); r=out/"results"; reports=out/"reports"
    active={"foundation_config":"ablation/configs/ic_foundation_531_v1.json:2-12","IC model map":"ablation/ic_core/model_adapter.py:14-28","IC adapter default":"ablation/ic_core/runtime.py:24-35","dPL full registry":"training/dpl/run_dpl_model.py:84-98","dPL lite registry":"training/dpl/run_dpl_model.py:100-114","dPL default selection":"training/dpl/run_dpl_model.py:627-650","forcing order":"ablation/ic_core/model_adapter.py:82-96","active_full_models":{"XAJ":"XAJ","GR4J":"GR4J","SIMHYD":"SIMHYD","HBV":"HBV","CN":{"XAJ":"XAJWithCemaNeige","GR4J":"GR4JWithCemaNeige","SIMHYD":"SIMHYDWithCemaNeige"},"TGD":{"XAJ":"XAJWithTemperatureConditionedDelay","GR4J":"GR4JWithTemperatureConditionedDelay","SIMHYD":"SIMHYDWithTemperatureConditionedDelay"}},"path_conclusion":"IC-XNES and dPL use the same full model classes by default; lite is explicit only; training and evaluation share the model forward class. HBV is standalone and is not a Base/TGD/CN wrapper.","excluded_from_this_extraction":["PD","GD","legacy 559 paths","mass-balance, gradient and training validation"]}
    write_json(r/"s2_formula_active_paths.json",active)
    formulas=[
      ("XAJ","input","P_t,T_t,PET_t enter as batch daily tensors; the full XAJ kernel uses P_t and PET_t, while T_t is validated by the shared interface and is not used in Base XAJ.","models/xaj.py:373-388; models/utils.py:30-62","VERIFIED_CODE"),
      ("XAJ","evaporation","prcp_t=max(P_t,0); PET^a_t=max(k PET_t,0); EU_t=min(WU_t+prcp_t,PET^a_t). ED_raw_t= C(PET^a_t-EU_t)-WL_t if WL_t<C LM and WL_t<C(PET^a_t-EU_t), else 0; ED_t=min(ED_raw_t,WD_t). EL_t=0 if WU_t+prcp_t>=PET^a_t; otherwise EL_t=(PET^a_t-EU_t)WL_t/(LM+eps) if WL_t>=C LM, otherwise min(C(PET^a_t-EU_t),WL_t). E_t=EU_t+EL_t+ED_t.","models/xaj.py:66-91","VERIFIED_CODE"),
      ("XAJ","tension runoff","WM=UM+LM+DM; W0=min(WU+WL+WD,WM-eps); PE=max(prcp-E,0); base=max(1-W0/(WM+eps),1e-6); A=WMM[1-base^(1/(1+B))], WMM=WM(1+B). If PE>0 and PE+A<WMM, R_cal=PE-(WM-W0)+WM[1-min((A+PE)/(WMM+eps),1)^(1+B); if PE>0 and PE+A>=WMM, R_cal=PE-(WM-W0); otherwise R_cal=0. R=max(R_cal,0); R_IM=max(PE IM,0).","models/xaj.py:93-116","VERIFIED_CODE"),
      ("XAJ","tension-state update","If PE>0, WU*=min(WU+PE-R,UM); otherwise WU*=max(WU+prcp-E,0). If PE>0 and WU_old+WL_old+PE-R>UM+LM, WD*=WU_old+WL_old+WD_old+PE-R-UM-LM; otherwise for PE>0 WD*=WD_old; if PE<=0, WD*=WD-ED. If PE>0, WL*=WU_old+WL_old+WD_old+PE-R-WU*-WD*; otherwise WL*=WL-EL. Then WU=clamp(WU*,0,UM), WL=clamp(WL*,0,LM), WD=clamp(WD*,0,DM).","models/xaj.py:118-136","VERIFIED_CODE"),
      ("XAJ","free-water separation","F_R=1[R>0]; FR*=R/(PE+eps) if F_R else FR_old; FR=clamp(FR*,0,1). SS=FR_old S/(FR*+eps) if F_R else S, then SS=min(SS,SM-eps). base_f=max(1-SS/(SM+eps),1e-6); AU=MS[1-base_f^(1/(1+EX))], MS=SM(1+EX). If PE+AU<MS, RS_fr=FR[PE-SM+SS+SM(1-min((PE+AU)/(MS+eps),1)^(1+EX)]; otherwise RS_fr=FR(PE+SS-SM). RS=clamp(F_R min(RS_fr,R),0,inf). S*=SS+[(R-RS)/(FR+eps)] if F_R else SS; S=clamp(S*,max=SM-eps). RI=KI S FR; RG=KG S FR; S_next=S(1-KI-KG).","models/xaj.py:138-165","VERIFIED_CODE"),
      ("XAJ","linear reservoirs and output","QI_next=CI QI_old+(1-CI)RI(1-IM); QG_next=CG QG_old+(1-CG)RG(1-IM). RS_adj=RS(1-IM)+R_IM. The diagnostic instantaneous output is RS_adj+QI_next+QG_next; full-model output routes RS_adj first and returns RS_routed+QI_next+QG_next.","models/xaj.py:167-178; models/xaj.py:537-575","VERIFIED_CODE"),
      ("GR4J","production store","mask=1[P_t>=PET_t]; P_N=max(P_t-PET_t,0); E_N=max(PET_t-P_t,0). r=clamp(S_prod/(X1+eps),0,1). P_S=X1(1-r^2)tanh(P_N/(X1+eps))/(1+r tanh(P_N/(X1+eps))+eps) when mask; E_S=S_prod(2-r)tanh(E_N/(X1+eps))/(1+(1-r)tanh(E_N/(X1+eps))+eps) otherwise. S_prod*=S_prod-E_S+P_S. n4=(4/9)S_prod/(X1+eps); Perc=S_prod[1-(1+n4^4)^(-1/4)]; S_prod_next=S_prod-Perc; P_R=max(Perc+P_N-P_S,0).","models/gr4j.py:33-57","VERIFIED_CODE"),
      ("GR4J","unit hydrographs","P_R1=0.9P_R and P_R2=0.1P_R. For t_i=1,...,L, d=max(X4,1e-3), ratio=t_i/d. UH1 S_1=min(ratio,1)^2.5; UH1_i=S_1(i)-S_1(i-1). UH2 S_2=0.5 ratio^2.5 when ratio<=1, otherwise 1-0.5 max(2-ratio,0)^2.5, then min(S_2,1); UH2_i=S_2(i)-S_2(i-1). Each ordinate vector is divided by its sum+1e-8. L1=15,L2=30.","models/gr4j.py:59-71; models/unit_hydro.py:18-71","VERIFIED_CODE"),
      ("GR4J","routing store and output","UH buffers shift left, append zero, and add UH ordinate times P_R1/P_R2; outputs are the first buffer entries UH1_o and UH2_o. F=X2 clamp(S_route/(X3+eps),min=0)^3.5. S_route*=max(S_route+UH1_o+F,0). q_R=S_route[1-(1+clamp(S_route/(X3+eps),min=0)^4)^(-1/4)]; S_route_next=S_route-q_R. q_D=max(UH2_o+F,0); Q_t=q_R+q_D. The buffers retain future scheduled flow for continuation; the finite window does not append an output tail.","models/gr4j.py:63-91; models/gr4j.py:152-165","VERIFIED_CODE"),
      ("SIMHYD","daily runoff generation","P=max(P_t,0); PET^*=max(ETMUL PET_t,0); I=min(min(INSC_safe,PET^*),P), INSC_safe=max(INSC,1e-6); throughfall=P-I; rho=clamp(soil/(SMSC_safe+eps),0,1), SMSC_safe=max(SMSC,1e-6); I_cap=COEFF_safe exp(-SQ rho), COEFF_safe=max(COEFF,1e-6); Infiltration=min(I_cap,throughfall); Direct=throughfall-Infiltration; Interflow=SUB rho Infiltration; Recharge=CRAK rho(Infiltration-Interflow).","models/simhyd.py:52-84","VERIFIED_CODE"),
      ("SIMHYD","soil, groundwater and output","soil_available=soil+Infiltration-Interflow-Recharge; ET_soil=min(min(10rho,PET^*-I),soil_available); soil_after=soil_available-ET_soil; Overflow=max(soil_after-SMSC_safe,0); soil_next=soil_after-Overflow; Recharge_total=Recharge+Overflow. K*=clamp(K,0,1); Baseflow=K*groundwater; groundwater_next=groundwater+Recharge_total-Baseflow; Runoff_inst=Direct+Interflow+Baseflow; ET=I+ET_soil.","models/simhyd.py:85-118","VERIFIED_CODE"),
      ("SIMHYD","routing","A finite gamma UH is generated for L=15, normalized by its sum, applied to the concatenation of a 14-sample runoff buffer and Runoff_inst, and the output slice beginning at index 14 is returned. The last 14 input samples become the continuation buffer; pending routing storage is retained diagnostically.","models/simhyd.py:165-200; models/simhyd.py:268-290","VERIFIED_CODE"),
      ("HBV","snow and liquid input","RAIN_t=P_t 1[T_t>=TT]; SNOW_t=P_t 1[T_t<TT]; SNOWPACK*=SNOWPACK+SNOW. Melt_pot=max(CFMAX(T_t-TT),0); Melt=min(Melt_pot,SNOWPACK); MELTWATER*=MELTWATER+Melt; Refreeze_pot=max(CFR CFMAX(TT-T_t),0); Refreeze=min(Refreeze_pot,MELTWATER); SNOWPACK*=SNOWPACK+Refreeze; MELTWATER*=MELTWATER-Refreeze; ToSoil=max(MELTWATER-CWH SNOWPACK,0); MELTWATER*=MELTWATER-ToSoil.","models/hbv.py:30-48","VERIFIED_CODE"),
      ("HBV","soil and response zones","w=clamp((SM/FC)^BETA,0,1); Recharge=(RAIN+ToSoil)w; SM*=SM+RAIN+ToSoil-Recharge; Excess=max(SM-FC,0); SM*=SM-Excess. ET_factor=clamp(SM/(LP FC),0,1); ETact=min(SM,PET_t ET_factor); SM_next=max(SM-ETact,nearzero). SUZ*=SUZ+Recharge+Excess; Perc=min(SUZ,PERC); SUZ*=SUZ-Perc; Q0=K0 max(SUZ-UZL,0); SUZ*=SUZ-Q0; Q1=K1 SUZ; SUZ*=SUZ-Q1; SLZ*=SLZ+Perc; Q2=K2 SLZ; SLZ*=SLZ-Q2; Q=Q0+Q1+Q2.","models/hbv.py:50-71","VERIFIED_CODE"),
      ("CN","basic CemaNeige preprocessing","f_solid=1 if T<=0; f_solid=0 if T>=3; otherwise f_solid=1-(T+1)/4; Snow=P f_solid; Rain=P-Snow. G*=G+Snow; eTG*=CTG eTG_old+(1-CTG)T, then eTG=min(eTG,0). IsMelting=1[eTG=0 and T>0]; Melt_pot=min(1[IsMelting] Kf T,G). g_thresh=0.9 estimate_psol_annual(P,T); SCA=0 if g_thresh<=eps, otherwise clamp(G/(max(g_thresh,eps)+eps),0,1). Melt=min((0.9 SCA+0.1)Melt_pot,G); G_next=G-Melt; effective P=Rain+Melt.","models/cemaneige.py:16-74; models/composed.py:153-185","VERIFIED_CODE"),
      ("TGD","temperature-conditioned generic delay","z_t=(T_t-Tmean_train)/max(Tstd_train,1e-6); h_t=tanh(clamp(z_t,-5,5)); tau_t=clamp(tau exp(-beta h_t),1e-6,3650); f_t=-expm1(-1/tau_t); S_pre=S_old+alpha P_t; Release=f_t S_pre; S_next=S_pre-Release; effective P_t=(1-alpha)P_t+Release. TGD has no phase partition, SWE, or melt state and passes PET unchanged.","models/temperature_delay.py:20-51; models/temperature_delay.py:122-184","VERIFIED_CODE"),
    ]
    write_csv(r/"s2_formula_inventory.csv",[{"model":m,"topic":t,"formula":f,"evidence":e,"status":s} for m,t,f,e,s in formulas])
    states=[
      ("XAJ","WU,WL,WD","tension-water stores","mm","WU=.6UM, WL=.6LM, WD=.6DM; explicit initial_states override","models/xaj.py:467-501"),("XAJ","S","free-water store","mm",".5SM","models/xaj.py:467-501"),("XAJ","FR","runoff coefficient memory","dimensionless","0.1","models/xaj.py:478-500"),("XAJ","QI,QG","interflow/groundwater linear-reservoir output states","model runoff-depth rate","0.1,0.1","models/xaj.py:478-500"),("XAJ","rs_uh_buffer","surface UH continuation buffer","mm","zeros(14)","models/xaj.py:487-500"),
      ("GR4J","S_prod,S_route","production and routing stores","mm",".5X1,.5X3","models/gr4j.py:169-190"),("GR4J","uh1_buf,uh2_buf","UH continuation buffers","mm","zeros(15),zeros(30)","models/gr4j.py:179-190"),
      ("SIMHYD","soil,groundwater","soil and groundwater stores","mm",".5 max(SMSC,1e-6),0","models/simhyd.py:329-353"),("SIMHYD","runoff_uh_buffer","gamma UH continuation buffer","mm","zeros(14)","models/simhyd.py:346-353"),
      ("HBV","SNOWPACK,MELTWATER,SM,SUZ,SLZ","snow, soil, upper and lower response stores","mm","0,0,.5,0,0","models/hbv.py:145-170"),("CN","G,eTG","snow storage and thermal state","mm,degC","0,0","models/cemaneige.py:154-169"),("TGD","S","generic delay storage","mm","0","models/temperature_delay.py:127-136")]
    write_csv(r/"s2_state_inventory.csv",[{"model":a,"state":b,"meaning":c,"unit":d,"initialization":e,"evidence":f,"status":"VERIFIED_CODE"} for a,b,c,d,e,f in states])
    flux=[{"model":m,"flux_or_operation":t,"formula_reference":f,"order":i,"evidence":e,"status":"VERIFIED_CODE"} for i,(m,t,f,e,s) in enumerate(formulas,1)]
    write_csv(r/"s2_flux_inventory.csv",flux)
    inits=[{"model":"XAJ","initialization":"full XAJ default: WU=.6UM, WL=.6LM, WD=.6DM, S=.5SM, FR=QI=QG=.1, rs_uh_buffer=zeros(14)","evidence":"models/xaj.py:467-501"},{"model":"GR4J","initialization":"S_prod=.5X1, S_route=.5X3, UH1/UH2 buffers zero","evidence":"models/gr4j.py:169-190"},{"model":"SIMHYD","initialization":"soil=.5 max(SMSC,1e-6), groundwater=0, runoff buffer zeros(14)","evidence":"models/simhyd.py:329-353"},{"model":"HBV","initialization":"SNOWPACK=0, MELTWATER=0, SM=.5, SUZ=0, SLZ=0","evidence":"models/hbv.py:145-170"},{"model":"CN","initialization":"G=0,eTG=0; host states use host defaults","evidence":"models/cemaneige.py:154-169; models/composed.py:153-164"},{"model":"TGD","initialization":"S=0; host states use host defaults; temperature statistics are required frozen inputs","evidence":"models/temperature_delay.py:122-136; models/composed_temperature_delay.py:344-363"}]
    write_csv(r/"s2_initialization_inventory.csv",inits)
    routes=[{"model":"XAJ","kernel":"hydrodl2 uh_gamma + uh_conv","length":"15 ordinates","normalization":"uh_gamma-generated kernel; local wrapper does not add a separate sum normalization","buffer":"14 previous RS_adj samples","output":"slice beginning at kernel_len-1 plus QI/QG","tail":"future finite-UH tail is not returned; buffer is retained","evidence":"models/xaj.py:297-342,644-674"},{"model":"GR4J","kernel":"S-curve difference UH1/UH2","length":"15/30","normalization":"raw ordinates divided by sum+1e-8","buffer":"15/30 shifted buffers","output":"first buffer entries plus routing-store and direct branches","tail":"remaining buffers are continuation state","evidence":"models/gr4j.py:63-91; models/unit_hydro.py:18-71"},{"model":"SIMHYD","kernel":"hydrodl2 gamma UH + grouped convolution","length":"15 ordinates","normalization":"UH ordinates divided by sum","buffer":"14 previous instantaneous runoff samples","output":"slice beginning at index 14","tail":"future tail represented by buffer/routing_storage","evidence":"models/simhyd.py:178-200"},{"model":"HBV","kernel":"none; Q0/Q1/Q2 are same-day response outflows","length":"not applicable","normalization":"not applicable","buffer":"none","output":"Q0+Q1+Q2","tail":"not applicable","evidence":"models/hbv.py:62-71"}]
    write_csv(r/"s2_routing_inventory.csv",routes)
    coupling=[]
    for host in ("XAJ","GR4J","SIMHYD"):
      for st in ("Base","TGD","CN"):
        coupling.append({"host":host,"structure":st,"active_class":host if st=="Base" else (host+"WithTemperatureConditionedDelay" if st=="TGD" else host+"WithCemaNeige"),"order":"P_t,T_t,PET_t -> preprocessing or bypass -> effective precipitation -> host step -> host routing -> Q_t","module_state":"none" if st=="Base" else "S" if st=="TGD" else "G,eTG","host_state":"host-specific states in s2_state_inventory.csv","PET":"unchanged and passed to host","temperature":"used by TGD/CN; Base host kernel receives validated temp but XAJ/GR4J/SIMHYD Base equations do not use it","evidence":"models/composed.py:46-53; models/composed_temperature_delay.py:32-110; models/xaj.py:373-408; models/gr4j.py:121-167; models/simhyd.py:224-297","status":"VERIFIED_CODE"})
    write_csv(r/"s2_coupling_inventory.csv",coupling)
    ops=[("XAJ","torch.clamp/where/minimum","P/PET floors, branches, stores, fractional-power base floors, free-water limits","models/xaj.py:66-160"),("GR4J","where/clamp/tanh","P-N/E-N split, production ratios, routing ratios, nonnegative flow","models/gr4j.py:33-87"),("SIMHYD","clamp/minimum/exp","forcing and parameter floors, infiltration, ET, overflow, K","models/simhyd.py:52-103"),("HBV","where/clamp/min","rain-snow threshold, melt/refreeze, soil and response limits","models/hbv.py:30-71"),("CN","where/clamp/min","0/3 C partition, thermal cap, SCA, storage-limited melt","models/cemaneige.py:16-74"),("TGD","clamp/tanh/expm1","temperature clipping, tau limits, stable release fraction","models/temperature_delay.py:20-51"),("UH","clamp/where/pow/normalization","GR4J S curves and finite-kernel normalization","models/unit_hydro.py:38-71"),("dPL parameter map","sigmoid","network output constrained to (epsilon,1-epsilon) before physical parameter mapping; outside hydrological daily equations","training/dpl/run_dpl_model.py:148-168"),("Model search","relu/softplus","not used in the active host/module daily equations identified here","models/*.py active classes")]
    write_csv(r/"s2_piecewise_operations.csv",[{"model":a,"operations":b,"mathematical_role":c,"evidence":d,"status":"VERIFIED_CODE"} for a,b,c,d in ops])
    lines=[]
    for m,t,f,e,s in formulas: lines.append({"model":m,"topic":t,"source_lines":e,"formula_status":s})
    write_csv(r/"s2_formula_source_lines.csv",lines)
    unresolved=[{"item":"Exact canonical SIMHYD variant","status":"UNRESOLVED","reason":"The project implements a differentiable SIMHYD variant but does not uniquely identify a canonical literature variant."},{"item":"External hydrodl2 UH internal discretization","status":"UNRESOLVED","reason":"Local code calls hydrodl2 uh_gamma/uh_conv; the external helper implementation is outside this repository scope."},{"item":"Published dPL checkpoint identity","status":"UNRESOLVED","reason":"The active dPL registry and default full class are verified, but no manuscript-specific checkpoint was requested for this extraction."}]
    write_csv(r/"s2_formula_unresolved.csv",unresolved)
    reports.mkdir(parents=True,exist_ok=True); (reports/"S2_formula_extraction_report.md").write_text(report(active))
    (reports/"S2_full_equations_for_writing.md").write_text(full_equations() + expanded_formula_details())
    (reports/"S2_formula_tables_for_manuscript.md").write_text(tables())

def report(active):
    return '''# S2 Formula Extraction Report

## Active full-model path

The active 531 foundation manifest is `ablation/configs/ic_foundation_531_v1.json:2-12`. IC uses `MODEL_CLASSES` in `ablation/ic_core/model_adapter.py:14-28`; `ModelAdapter` selects full classes unless `variant="lite"` is explicitly passed (`ablation/ic_core/model_adapter.py:31-45`). IC evaluation calls this adapter at `ablation/ic_core/runtime.py:104-110`. dPL registers the same full classes at `training/dpl/run_dpl_model.py:84-98` and selects the lite registry only under the explicit `--lite` option at `training/dpl/run_dpl_model.py:613-629`. Thus the default IC and dPL full forward path is shared. HBV is registered as a standalone class and is not a Base/TGD/CN wrapper.

The full active classes are `XAJ`, `GR4J`, `SIMHYD`, `HBV`; `XAJWithCemaNeige`, `GR4JWithCemaNeige`, `SIMHYDWithCemaNeige`; and `XAJWithTemperatureConditionedDelay`, `GR4JWithTemperatureConditionedDelay`, `SIMHYDWithTemperatureConditionedDelay`. The corresponding wrappers call the fused step in the order module first, host second: `models/composed.py:28-93` and `models/composed_temperature_delay.py:32-110`.

## Formula package

The complete ordered formulas are in `results/s2_formula_inventory.csv` and the writing-ready LaTeX version is `S2_full_equations_for_writing.md`. The state, flux, parameter, initialization, routing, coupling and piecewise-operation tables are the companion machine-readable files in `results/`.

## Scope decisions

This extraction contains Base, TGD, CN and the HBV snow-process reference only. PD and GD are excluded by scope. No mass-balance, gradient, training or performance validation was run.

## Implementation distinctions

CN is the active basic two-parameter CemaNeige path calling `_cemaneige_step`, not `_cemaneige_hyst_step`. It uses a 0--3 degC piecewise solid fraction, G/eTG states, a fixed `0.9 * estimated annual solid precipitation` threshold and storage-limited melt. TGD is a three-parameter generic delay with one storage state, frozen training-period temperature statistics and a smooth bounded temperature signal. It has no rain/snow partition, SWE state or melt equation. Both leave PET unchanged.

## Unresolved formula boundaries

The local repository does not uniquely identify a canonical SIMHYD literature variant, and the detailed discretization inside the imported `hydrodl2` UH helper is outside the local source tree. These are explicitly listed in `results/s2_formula_unresolved.csv`; they are not filled with textbook assumptions.
'''.replace(chr(9), "tau").replace(chr(8), "beta").replace("tauau", "tau").replace("betaeta", "beta")

def full_equations():
    return r'''# S2 Formula Source Package

All equations below are the implemented full-model equations. The daily index is (t); (P_t,T_t,E_{p,t}) denote the input precipitation, temperature and PET tensors. `eps` is the model `nearzero` default (10^{-8}). All `clamp`, `min`, `max`, `where`, finite-kernel and normalization operations are retained.

## S2.1 Host model formulations

### S2.1.1 XAJ

#### Inputs and states

The full class accepts `precip`, `pet`, and `temp`; only (P_t) and (E_{p,t}) enter the Base XAJ daily kernel. The state at the beginning of day (t) is (WU_t,WL_t,WD_t,S_t,FR_t,QI_t,QG_t), plus the 14-sample surface-runoff buffer. (WU,WL,WD) are tension-water stores (mm), (S) is the free-water store (mm), (FR) is dimensionless, (QI,QG) are recursive interflow and groundwater output states, and the buffer stores surface-runoff depth. Full-model defaults are (WU_0=0.6UM, WL_0=0.6LM, WD_0=0.6DM, S_0=0.5SM, FR_0=QI_0=QG_0=0.1), and a zero buffer (models/xaj.py:467-501).

#### Daily flux equations in execution order

Define (WM=UM+LM+DM), (WMM=WM(1+B)), (MS=SM(1+EX)). First,

[
P'_t=\max(P_t,0),\qquad E^a_t=\max(kE_{p,t},0).
]

[
EU_t=\min(WU_t+P'_t,E^a_t).
]

Let
[
ED^{raw}_t=\begin{cases}C(E^a_t-EU_t)-WL_t,&WL_t<C,LM\ \text{and}\ WL_t<C(E^a_t-EU_t),\\0,&\text{otherwise},\end{cases}qquad ED_t=\min(ED^{raw}_t,WD_t).
]

[
EL_t=\begin{cases}0,&WU_t+P'_t\ge E^a_t,\\
(E^a_t-EU_t)WL_t/(LM+\varepsilon),&WU_t+P'_t<E^a_t\ \text{and}\ WL_t\ge C,LM,\\
\min(C(E^a_t-EU_t),WL_t),&\text{otherwise}.
\end{cases}
\quad E_t=EU_t+EL_t+ED_t.
]

Set (W_{0,t}=\min(WU_t+WL_t+WD_t,WM-\varepsilon)), (PE_t=\max(P'_t-E_t,0)), (b_t=\max(1-W_{0,t}/(WM+\varepsilon),10^{-6})), and (A_t=WMM[1-b_t^{1/(1+B)}]). Then

[
R^{cal}_t=\begin{cases}
PE_t-(WM-W_{0,t})+WM\left[1-\min\left(\frac{A_t+PE_t}{WMM+\varepsilon},1\right)\right]^{1+B},&PE_t>0,\ PE_t+A_t<WMM,\\
PE_t-(WM-W_{0,t}),&PE_t>0,\ PE_t+A_t\ge WMM,\\
0,&PE_t\le0,
\end{cases}\qquad R_t=\max(R^{cal}_t,0),\quad R^{IM}_t=\max(PE_t IM,0).
]

The tension stores are updated using old-day stores on the right-hand side:

[
WU^*_t=\begin{cases}\min(WU_t+PE_t-R_t,UM),&PE_t>0,\\\max(WU_t+P'_t-E_t,0),&PE_t\le0,\end{cases}
]

[
WD^*_t=\begin{cases}WU_t+WL_t+WD_t+PE_t-R_t-UM-LM,&PE_t>0\ \text{and}\ WU_t+WL_t+PE_t-R_t>UM+LM,\\WD_t,&PE_t>0\ \text{and the condition is false},\\WD_t-ED_t,&PE_t\le0,\end{cases}
]

where the first branch uses the explicit capacity remainder shown above; and

[
WL^*_t=\begin{cases}WU_t+WL_t+WD_t+PE_t-R_t-WU^*_t-WD^*_t,&PE_t>0,\\WL_t-EL_t,&PE_t\le0.\end{cases}
\]

Finally (WU_{t+1}=\operatorname{clamp}(WU^*_t,0,UM)), (WL_{t+1}=\operatorname{clamp}(WL^*_t,0,LM)), and (WD_{t+1}=\operatorname{clamp}(WD^*_t,0,DM)).

For free-water separation, (m_t=1[R_t>0]), (FR^*_t=m_tR_t/(PE_t+\varepsilon)+(1-m_t)FR_t), and (FR_{t+1}=\operatorname{clamp}(FR^*_t,0,1)). Define (SS_t=m_t FR_tS_t/(FR^*_t+\varepsilon)+(1-m_t)S_t), then (SS_t=\min(SS_t,SM-\varepsilon)), (c^f_t=\max(1-SS_t/(SM+\varepsilon),10^{-6})), (AU_t=MS[1-(c^f_t)^{1/(1+EX)}]). Then

[
RS^{fr}_t=\begin{cases}FR_{t+1}\left[PE_t-SM+SS_t+SM\left(1-\min\left(\frac{PE_t+AU_t}{MS+\varepsilon},1\right)\right)^{1+EX}\right],&PE_t+AU_t<MS,\\FR_{t+1}(PE_t+SS_t-SM),&PE_t+AU_t\ge MS,
\end{cases}
]

[RS_t=\operatorname{clamp}(m_t\min(RS^{fr}_t,R_t),0,\infty),quad S^*_t=SS_t+m_t\frac{R_t-RS_t}{FR^*_t+\varepsilon},quad S_{t+1}=\min(S^*_t,SM-\varepsilon).]

[RI_t=KI S_{t+1}FR_{t+1},quad RG_t=KG S_{t+1}FR_{t+1}.]

The prepared XAJ parameters rescale (KI,KG) only when (KI+KG\ge1): they are multiplied by ((1-10^{-5})/\max(KI+KG,10^{-6})); otherwise they are unchanged (models/xaj.py:280-294). The recursive linear reservoirs and surface adjustment are

[QI_{t+1}=CI QI_t+(1-CI)RI_t(1-IM),quad QG_{t+1}=CG QG_t+(1-CG)RG_t(1-IM),quad RS^{adj}_t=RS_t(1-IM)+R^{IM}_t.]

The full path forms a 15-ordinate Gamma UH through `uh_gamma`, applies causal `uh_conv` to the 14-sample buffer concatenated with (RS^{adj}), and returns the current slice plus (QI_{t+1}+QG_{t+1}). The finite future tail is held in the continuation buffer and is not returned in the current finite window (models/xaj.py:297-342, 644-674).

#### Parameters

The active XAJ source specification contains 15 parameters: (k,B,IM,UM,LM,DM,C,SM,EX,KI,KG,CI,CG,a_{UH},\theta_{UH}), named in `models/parameter_specs.py:163-299`. Therefore any manuscript statement that the current active XAJ implementation has 14 parameters conflicts with the active source specification. Their exact code names, bounds and units are in `results/s2_parameter_inventory.csv`.

### S2.1.2 GR4J

States are (S^{prod}_t,S^{route}_t), UH1 buffer length 15 and UH2 buffer length 30. Defaults are (0.5X_1,0.5X_3), and zero buffers (models/gr4j.py:169-190).

[
M_t=1[P_t\ge E_{p,t}],\quad P^N_t=\max(P_t-E_{p,t},0),\quad E^N_t=\max(E_{p,t}-P_t,0),\quad r_t=\operatorname{clamp}(S^{prod}_t/(X_1+\varepsilon),0,1).
]

[
P^S_t=M_t\frac{X_1(1-r_t^2)\tanh(P^N_t/(X_1+\varepsilon))}{1+r_t\tanh(P^N_t/(X_1+\varepsilon))+\varepsilon},
]

[
E^S_t=(1-M_t)\frac{S^{prod}_t(2-r_t)\tanh(E^N_t/(X_1+\varepsilon))}{1+(1-r_t)\tanh(E^N_t/(X_1+\varepsilon))+\varepsilon}.
]

[S^{prod*}_t=S^{prod}_t-E^S_t+P^S_t,quad n_t=\frac49\frac{S^{prod*}_t}{X_1+\varepsilon},quad Perc_t=S^{prod*}_t[1-(1+n_t^4)^{-1/4}],]
[S^{prod}_{t+1}=S^{prod*}_t-Perc_t,quad P^R_t=\max(Perc_t+P^N_t-P^S_t,0),quad P^{R1}_t=.9P^R_t,quad P^{R2}_t=.1P^R_t.]

For (i=1,ldots,L), (L_1=15,L_2=30), (d=\max(X_4,10^{-3})), (u_i=i/d). UH1 uses (S_1(u)=\min(u,1)^{2.5}). UH2 uses (S_2(u)=0.5u^{2.5}) for (u\le1), and (1-0.5\max(2-u,0)^{2.5}) otherwise, followed by (min(S_2,1)). Ordinates are (w_i=S(i)-S(i-1)), divided by (sum_iw_i+10^{-8}) (models/unit_hydro.py:18-71).

Each UH buffer shifts left, appends zero, and adds (w_iP^{Rj}_t). Let (UH1_t,UH2_t) be the first entries. (F_t=X_2[\max(S^{route}_t/(X_3+\varepsilon),0)]^{3.5}). Then (S^{route*}_t=\max(S^{route}_t+UH1_t+F_t,0)), (Q^R_t=S^{route*}_t[1-(1+\max(S^{route*}_t/(X_3+\varepsilon),0)^4)^{-1/4}]), (S^{route}_{t+1}=S^{route*}_t-Q^R_t), (Q^D_t=\max(UH2_t+F_t,0)), and (Q_t=Q^R_t+Q^D_t). Buffers preserve finite-window continuation water (models/gr4j.py:63-91).

The active parameters (X_1,X_2,X_3,X_4) are `x1`--`x4`, with bounds and units in `models/parameter_specs.py:124-160`. Compared with canonical GR4J, the local implementation explicitly uses differentiable tensor S-curves, epsilon terms, finite buffers and local clamps; the formula above is the implemented formulation.

### S2.1.3 SIMHYD

The differentiable SIMHYD implementation used in this study has states (soil_t,groundwater_t) and a 14-sample UH buffer. Its daily equations are exactly the equations in the SIMHYD rows of `results/s2_formula_inventory.csv`: interception, exponential infiltration, direct runoff, interflow, recharge, soil ET, overflow transfer, groundwater recession and instantaneous runoff, followed by the normalized finite gamma UH. Initial states and routing are in `results/s2_initialization_inventory.csv` and `results/s2_routing_inventory.csv`. Parameters are the nine `simhyd_*` entries at `models/parameter_specs.py:432-473`.

### S2.2 Implemented CemaNeige module

The active module is the basic two-parameter variant with (CTG,K_f). Its complete partition, (G,eTG,SCA), melt and effective-input equations are given in the CN row of `results/s2_formula_inventory.csv` and source lines `models/cemaneige.py:16-74`. It is not the hysteresis class. PET is passed unchanged. The wrapper performs this module update first and feeds (P^{eff}_t=Rain_t+Melt_t) to the host on the same day.

### S2.3 Temperature-conditioned generic delay

The active TGD has (alpha,	au,eta), one storage (S_t), frozen training temperature mean and standard deviation, and equations in `models/temperature_delay.py:20-51`. The physical mapping is linear for (alpha,eta) and logarithmic for (	au) in `ablation/ic_core/parameter_adapter.py:56-67`; dPL first constrains network outputs with sigmoid at `training/dpl/run_dpl_model.py:166-168`. TGD uses temperature, conservatively redistributes precipitation through time, leaves PET unchanged, and contains no snow partition, SWE or melt equation.

### S2.4 Structural comparison of Base, TGD and CN

Base bypasses preprocessing; TGD adds one delay storage and three parameters; CN adds (G,eTG) and two parameters. Base/TGD/CN do not have equal state or parameter counts. TGD and CN both use (P,T) and leave PET unchanged, but only CN has explicit solid precipitation, snow storage and melt.

### S2.5 Coupling to host models

For each XAJ, GR4J and SIMHYD wrapper, the order is
[
(P_t,T_t,E_{p,t})\rightarrow\{P_t\ \text{(Base bypass)},\ P^{eff}_t\ \text{(TGD)},\ P^{eff}_t\ \text{(CN)}\}\rightarrow\text{host daily step}\rightarrow\text{host routing}\rightarrow Q_t.
]

The evidence is `models/composed.py:28-93`, `models/composed_temperature_delay.py:32-110`, and the host forward methods cited above. TGD/CN do not modify PET.

### S2.6 HBV snow-process reference

HBV is a standalone explicit snow-process reference. Its complete equations are the HBV rows in `results/s2_formula_inventory.csv`, source `models/hbv.py:30-71`; parameters and initial states are `models/parameter_specs.py:13-122` and `models/hbv.py:145-170`. It does not participate in Base/TGD/CN replacement wrappers.
'''.replace(chr(9), "tau").replace(chr(8), "beta").replace("tauau", "tau").replace("betaeta", "beta")

def expanded_formula_details():
    return r'''

## Expanded daily equations for SIMHYD, HBV, CN and TGD

### SIMHYD daily equations

Let INSC_s=max(INSC,1e-6), SMSC_s=max(SMSC,1e-6), COEFF_s=max(COEFF,1e-6), and rho_t=clamp(soil_t/(SMSC_s+eps),0,1). The exact daily order is:

$$P_s=max(P_t,0),\quad PET_s=max(ETMUL*PET_t,0),\quad I_t=min(INSC_s,PET_s,P_s),\quad T_t=P_s-I_t.$$

$$Icap_t=COEFF_s*exp(-SQ*rho_t),\quad Inf_t=min(Icap_t,T_t),\quad D_t=T_t-Inf_t.$$

$$IF_t=SUB*rho_t*Inf_t,\quad Rec_t=CRAK*rho_t*(Inf_t-IF_t),\quad SoilAvail_t=soil_t+Inf_t-IF_t-Rec_t.$$

$$ETsoil_t=min(10*rho_t,PET_s-I_t,SoilAvail_t),\quad SoilAfter_t=SoilAvail_t-ETsoil_t.$$

$$Overflow_t=max(SoilAfter_t-SMSC_s,0),\quad soil_{t+1}=SoilAfter_t-Overflow_t,$$
$$RecTotal_t=Rec_t+Overflow_t,\quad K_s=clamp(K,0,1),\quad BF_t=K_s*groundwater_t,$$
$$groundwater_{t+1}=groundwater_t+RecTotal_t-BF_t,\quad RunoffInst_t=D_t+IF_t+BF_t,\quad ET_t=I_t+ETsoil_t.$$

RunoffInst is concatenated with the 14-sample buffer, routed by the normalized length-15 gamma kernel, and sliced from index 14. The last 14 samples form the continuation buffer. Evidence: models/simhyd.py:52-118,165-200,224-297,329-353.

### HBV daily equations

$$RAIN_t=P_t*1[T_t>=TT],\quad SNOW_t=P_t*1[T_t<TT],\quad SNOWPACK^*_t=SNOWPACK_t+SNOW_t.$$
$$MeltPot_t=max(CFMAX*(T_t-TT),0),\quad Melt_t=min(MeltPot_t,SNOWPACK^*_t),\quad MELTWATER^*_t=MELTWATER_t+Melt_t.$$
$$RefreezePot_t=max(CFR*CFMAX*(TT-T_t),0),\quad Refreeze_t=min(RefreezePot_t,MELTWATER^*_t),$$
$$SNOWPACK^{**}_t=SNOWPACK^*_t+Refreeze_t,\quad MELTWATER^{**}_t=MELTWATER^*_t-Refreeze_t,$$
$$ToSoil_t=max(MELTWATER^{**}_t-CWH*SNOWPACK^{**}_t,0),\quad MELTWATER_{t+1}=MELTWATER^{**}_t-ToSoil_t.$$

$$w_t=clamp((SM_t/FC)^{BETA},0,1),\quad Recharge_t=(RAIN_t+ToSoil_t)w_t,$$
$$SM^*_t=SM_t+RAIN_t+ToSoil_t-Recharge_t,\quad Excess_t=max(SM^*_t-FC,0),\quad SM^{**}_t=SM^*_t-Excess_t,$$
$$ETfactor_t=clamp(SM^{**}_t/(LP*FC),0,1),\quad ETact_t=min(SM^{**}_t,PET_t*ETfactor_t),\quad SM_{t+1}=max(SM^{**}_t-ETact_t,nearzero).$$

$$SUZ^*_t=SUZ_t+Recharge_t+Excess_t,\quad Perc_t=min(SUZ^*_t,PERC),\quad SUZ^{**}_t=SUZ^*_t-Perc_t,$$
$$Q0_t=K0*max(SUZ^{**}_t-UZL,0),\quad SUZ^{***}_t=SUZ^{**}_t-Q0_t,\quad Q1_t=K1*SUZ^{***}_t,$$
$$SLZ^*_t=SLZ_t+Perc_t,\quad Q2_t=K2*SLZ^*_t,\quad Q_t=Q0_t+Q1_t+Q2_t.$$

Evidence: models/hbv.py:30-71,145-190. There is no convolutional routing kernel or continuation tail in HBV; Q0, Q1 and Q2 are same-day response outflows.

### Basic CemaNeige daily equations

$$fsolid_t=\begin{cases}1,&T_t<=0,\\0,&T_t>=3,\\1-(T_t+1)/4,&0<T_t<3,\end{cases}\quad Snow_t=P_t*fsolid_t,\quad Rain_t=P_t-Snow_t.$$
$$G^*_t=G_t+Snow_t,\quad eTG^*_t=CTG*eTG_t+(1-CTG)*T_t,\quad eTG_{t+1}=min(eTG^*_t,0).$$

With g_thresh=0.9*estimate_psol_annual(P,T), g_safe=max(g_thresh,nearzero),
$$SCA_t=0\quad\text{if }g_thresh<=nearzero,\qquad SCA_t=clamp(G^*_t/(g_safe+nearzero),0,1)\quad\text{otherwise}.$$
$$MeltPot_t=min(1[eTG_{t+1}=0 and T_t>0]*Kf*T_t,G^*_t),$$
$$Melt_t=min((0.9*SCA_t+0.1)*MeltPot_t,G^*_t),\quad G_{t+1}=G^*_t-Melt_t,\quad P^{eff}_t=Rain_t+Melt_t.$$

Evidence: models/cemaneige.py:16-74; the active fused wrappers call this basic step at models/composed.py:46-53.

### Temperature-conditioned generic delay equations

$$z_t=(T_t-Tmean_train)/max(Tstd_train,1e-6),\quad h_t=tanh(clamp(z_t,-5,5)),$$
$$tau_t=clamp(tau*exp(-beta*h_t),1e-6,3650),\quad f_t=-expm1(-1/tau_t).$$
$$Spre_t=S_t+alpha*P_t,\quad Release_t=f_t*Spre_t,\quad S_{t+1}=Spre_t-Release_t,$$
$$P^{eff}_t=(1-alpha)*P_t+Release_t.$$

The default TGD state is S_0=0. alpha and beta use linear physical bounds; tau uses logarithmic interpolation in ablation/ic_core/parameter_adapter.py:56-67. Evidence: models/temperature_delay.py:20-51,122-184 and models/composed_temperature_delay.py:32-110.
'''

def tables():
    return '''# S2 Formula Tables for Manuscript

## State variables

See `results/s2_state_inventory.csv` and `results/s2_initialization_inventory.csv`. They list every host, CN and TGD state, units, defaults and source lines.

## Parameters and bounds

See `results/s2_parameter_inventory.csv`. It contains code name, symbol, description, lower/upper bound, default, unit, scope, transform and source line for every Base/TGD/CN/HBV parameter. `tgd_tau` is log-interpolated; other parameters use linear physical-bound interpolation after the dPL sigmoid output.

## Structural comparison

| Structure | Preprocessing state | Parameters | Temperature use | PET | Explicit snow |
|---|---|---:|---|---|---|
| Base | none | host only | validated; unused by XAJ/GR4J/SIMHYD host kernels | unchanged | no |
| TGD | `S` | host + 3 | standardized, clipped, tanh signal | unchanged | no |
| CN | `G,eTG` | host + 2 | solid fraction and thermal state | unchanged | yes |

## Host coupling

See `results/s2_coupling_inventory.csv`: every row follows raw `P,T,PET`, preprocessing or bypass, effective precipitation, host step, host routing and final discharge.

## Routing

See `results/s2_routing_inventory.csv`. XAJ and SIMHYD use finite gamma UH routing; GR4J uses finite differentiable UH1/UH2 plus a routing store; HBV has no convolutional UH.

## Implementation-specific operations

See `results/s2_piecewise_operations.csv`. It records all locally identified `where`, `min`, `max`, `clamp`, `tanh`, `expm1`, fractional-power floors, epsilon denominators, kernel normalization and finite-window tail handling with file:line evidence.
'''

if __name__ == "__main__": main()
