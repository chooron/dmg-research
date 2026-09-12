#!/usr/bin/env python3
"""Generate the equation/state/threshold source inventories for S2."""
from __future__ import annotations

import argparse
from pathlib import Path
from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv


def generate(root: Path, out: Path) -> None:
    r = out / "results"
    eq = [
        {"model": "XAJ", "section": "evaporation", "equation": "prcp=max(P,0); PETa=max(k*PET,0); EU=min(WU+prcp,PETa); ED=min(ED_raw,WD); EL is the nested XAJ branch; E=EU+EL+ED.", "order": 1, "evidence": "models/xaj.py:42-80 _xaj_step_impl", "status": "VERIFIED_CODE"},
        {"model": "XAJ", "section": "tension runoff", "equation": "WM=UM+LM+DM; W0=min(WU+WL+WD,WM-eps); PE=max(prcp-E,0); A=WMM[1-(1-W0/WM)^(1/(1+B))]; R=max(PE-(WM-W0)+WM[1-(A+PE)/WMM]^(1+B),0) with the code's min/max branches.", "order": 2, "evidence": "models/xaj.py:81-117 _xaj_step_impl", "status": "VERIFIED_CODE"},
        {"model": "XAJ", "section": "free water and linear routing", "equation": "FR=clamp(R/(PE+eps),0,1); SS=min(FR_old*S/(FR+eps),SM-eps); AU=SM[1-(1-SS/SM)^(1/(1+EX))]; RS=clamp(min(FR[PE-SM+SS+SM(1-(PE+AU)/SM)^(1+EX)],R),0); RI=KI*S*FR; RG=KG*S*FR; S_new=S_after*(1-KI-KG); QI=CI*QI_old+(1-CI)RI(1-IM); QG=CG*QG_old+(1-CG)RG(1-IM).", "order": 3, "evidence": "models/xaj.py:118-171 _xaj_step_impl", "status": "VERIFIED_CODE"},
        {"model": "XAJ", "section": "surface routing", "equation": "RS_adj=RS(1-IM)+PE*IM; surface RS_adj is routed by a finite 15-ordinate gamma UH; final Q=RS_routed+QI+QG.", "order": 4, "evidence": "models/xaj.py:308-342, 644-680", "status": "VERIFIED_CODE"},
        {"model": "GR4J", "section": "production store", "equation": "PN=max(P-E,0), EN=max(E-P,0); PS and ES are the tanh production-store equations; S=S-ES+PS; Perc=S[1-(1+(4S/(9X1))^4)^(-1/4)]; S=S-Perc; PR=Perc+PN-PS.", "order": 1, "evidence": "models/gr4j.py:17-56 _gr4j_step", "status": "VERIFIED_CODE"},
        {"model": "GR4J", "section": "unit hydrographs", "equation": "PR is split 0.9 PR to UH1 and 0.1 PR to UH2. UH ordinates are S-curve differences from x4, normalized with eps=1e-8; UH1 max length 15 and UH2 max length 30.", "order": 2, "evidence": "models/gr4j.py:57-77; models/unit_hydro.py:18-71", "status": "VERIFIED_CODE"},
        {"model": "GR4J", "section": "routing store", "equation": "F=X2*(S_route/(X3+eps))^3.5; S_route=max(S_route+UH1_out+F,0); QR=S_route[1-(1+(S_route/X3)^4)^(-1/4)]; QD=max(UH2_out+F,0); Q=QR+QD.", "order": 3, "evidence": "models/gr4j.py:78-108 _gr4j_step", "status": "VERIFIED_CODE"},
        {"model": "SIMHYD", "section": "interception and soil", "equation": "P=max(P,0); PET=max(PET*ETMUL,0); I=min(INSC_safe,PET,P); infiltration=min(COEFF_safe exp(-SQ*soil_ratio),P-I); direct=P-I-infiltration; interflow=SUB*ratio*infiltration; recharge=CRAK*ratio*(infiltration-interflow).", "order": 1, "evidence": "models/simhyd.py:52-84 _simhyd_step_impl", "status": "VERIFIED_CODE"},
        {"model": "SIMHYD", "section": "soil and groundwater", "equation": "soil_available=soil+infiltration-interflow-recharge; soil_ET=min(10 ratio,PET-I,soil_available); overflow=max(soil_after_ET-SMSC_safe,0); soil_new=soil_after_ET-overflow; baseflow=clamp(K,0,1)*groundwater; groundwater_new=groundwater+recharge+overflow-baseflow; runoff=direct+interflow+baseflow.", "order": 2, "evidence": "models/simhyd.py:85-118 _simhyd_step_impl", "status": "VERIFIED_CODE"},
        {"model": "SIMHYD", "section": "routing", "equation": "Instant runoff is passed through a normalized finite gamma UH with SIMHYD_UH_MAX_LEN ordinates and a 14-sample continuation buffer.", "order": 3, "evidence": "models/simhyd.py:165-200, 268-275", "status": "VERIFIED_CODE"},
        {"model": "HBV", "section": "snow", "equation": "RAIN=P 1[T>=TT]; SNOW=P 1[T<TT]; G=G+SNOW; melt=min(max(CFMAX(T-TT),0),G); liquid=liquid+melt; refreeze=min(max(CFR*CFMAX(TT-T),0),liquid); G=G+refreeze; liquid=liquid-refreeze; tosoil=max(liquid-CWH*G,0).", "order": 1, "evidence": "models/hbv.py:30-48 _hbv_step", "status": "VERIFIED_CODE"},
        {"model": "HBV", "section": "soil and response", "equation": "w=(SM/FC)^BETA clipped [0,1]; recharge=(RAIN+tosol)*w; SM=SM+RAIN+tosol-recharge; excess=max(SM-FC,0); ET=min(SM,PET*clip(SM/(LP*FC),0,1)); SM=max(SM-ET,nearzero); SUZ+=recharge+excess; perc=min(SUZ,PERC); Q0=K0 max(SUZ-UZL,0); Q1=K1 SUZ; SLZ+=perc; Q2=K2 SLZ; Q=Q0+Q1+Q2.", "order": 2, "evidence": "models/hbv.py:50-71 _hbv_step", "status": "VERIFIED_CODE"},
        {"model": "CemaNeige", "section": "snow preprocessing", "equation": "f_solid=1 if T<=0, 0 if T>=3, otherwise 1-(T+1)/4; snow=P f_solid; rain=P-snow; G+=snow; eTG=CTG eTG_old+(1-CTG)T then min(eTG,0); melt=min((0.9 SCA+0.1) min(1[eTG=0 and T>0] KF*T,G),G); effective=rain+melt.", "order": 1, "evidence": "models/cemaneige.py:16-74 _solid_liquid_partition/_cemaneige_step", "status": "VERIFIED_CODE"},
        {"model": "TGD", "section": "temperature-conditioned delay", "equation": "z=(T-Tmean_train)/max(Tstd_train,1e-6); h=tanh(clamp(z,-5,5)); tau_t=clamp(tau exp(-beta h),1e-6,3650); f= -expm1(-1/tau_t); S_pre=S+alpha P; release=f S_pre; S_new=S_pre-release; effective=(1-alpha)P+release.", "order": 1, "evidence": "models/temperature_delay.py:26-51 _temperature_conditioned_delay_step", "status": "VERIFIED_CODE"},
    ]
    write_csv(r / "s2_equation_inventory.csv", eq)
    states = [
        {"model": "XAJ", "state": "WU, WL, WD", "unit": "mm", "initialization": "0.5*UM, 0.5*LM, 0.5*DM", "scope": "batch/basin", "evidence": "models/xaj.py:467-493"},
        {"model": "XAJ", "state": "S, FR, QI, QG, rs_uh_buffer", "unit": "mm or mm/day; buffer mm", "initialization": "0.5*SM, 0, 0, 0, zeros(14)", "scope": "batch/basin", "evidence": "models/xaj.py:467-493"},
        {"model": "GR4J", "state": "s_prod, s_route", "unit": "mm", "initialization": "0.5*X1, 0.5*X3", "scope": "batch/basin", "evidence": "models/gr4j.py:144-190"},
        {"model": "GR4J", "state": "uh1_buf, uh2_buf", "unit": "mm", "initialization": "zeros(15), zeros(30)", "scope": "batch/basin", "evidence": "models/gr4j.py:144-190"},
        {"model": "SIMHYD", "state": "soil, groundwater", "unit": "mm", "initialization": "0.5*SMS C_safe, 0", "scope": "batch/basin", "evidence": "models/simhyd.py:329-353"},
        {"model": "SIMHYD", "state": "runoff_uh_buffer", "unit": "mm", "initialization": "zeros(14)", "scope": "batch/basin", "evidence": "models/simhyd.py:329-353"},
        {"model": "HBV", "state": "SNOWPACK, MELTWATER, SM, SUZ, SLZ", "unit": "mm", "initialization": "0, 0, 0.5, 0, 0", "scope": "batch/basin", "evidence": "models/hbv.py:145-170"},
        {"model": "CemaNeige", "state": "G, eTG", "unit": "mm and degC", "initialization": "0, 0", "scope": "batch/basin", "evidence": "models/cemaneige.py:154-169"},
        {"model": "TGD", "state": "S", "unit": "mm", "initialization": "0", "scope": "batch/basin", "evidence": "models/temperature_delay.py:127-136"},
    ]
    write_csv(r / "s2_state_flux_inventory.csv", states)
    thresholds = [
        ("XAJ", "clamp/min/max", "P, PETa, W0, fractional power, stores, FR, S", "nearzero=1e-8; base floor=1e-6", "stability and nonnegative stores", "models/xaj.py:42-171", "VERIFIED_CODE"),
        ("GR4J", "where/clamp", "PN/EN split, ratios, routing store, output", "nearzero=1e-8", "stability and nonnegative output", "models/gr4j.py:17-108", "VERIFIED_CODE"),
        ("SIMHYD", "clamp/minimum", "P, PET, INSC, SMSC, COEFF, K, overflow", "INSC=1e-6, COEFF=1e-6, SMSC=1e-6, nearzero=1e-8", "avoid branch singularities", "models/simhyd.py:52-103", "VERIFIED_CODE"),
        ("CemaNeige", "where/min/clamp", "partition, thermal state, melt, SCA", "nearzero=1e-8", "phase/melt limits", "models/cemaneige.py:16-74", "VERIFIED_CODE"),
        ("TGD", "clamp/tanh/expm1", "alpha, tau, temperature signal, release", "tau [1e-6,3650], z clip +/-5, std floor 1e-6", "smooth temperature control and numerical stability", "models/temperature_delay.py:20-51", "VERIFIED_CODE"),
        ("UH", "clamp/where/normalize", "GR4J S curves and gamma UH", "x4 floor=1e-3; ordinate eps=1e-8; XAJ/SIMHYD gamma implementation", "finite differentiable routing", "models/unit_hydro.py:38-71; models/xaj.py:644-680", "VERIFIED_CODE"),
        ("HBV", "where/clamp/min", "rain/snow, melt/refreeze, soil/recharge/response", "nearzero=1e-8", "physical bounds", "models/hbv.py:30-71", "VERIFIED_CODE"),
    ]
    write_csv(r / "s2_threshold_and_smoothing_inventory.csv", [{"model": a, "operation": b, "location": c, "numeric_constants": d, "purpose": e, "evidence": f, "status": g} for a,b,c,d,e,f,g in thresholds])


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); generate(root,out)
