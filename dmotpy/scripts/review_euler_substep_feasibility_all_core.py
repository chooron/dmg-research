"""
review_euler_substep_feasibility_all_core.py

Review dt-substep feasibility for all runnable core models.
Classifies each model and writes CSV + Markdown outputs.

No hydrological formulas are changed.
No smoothing, clamps, or parameter bounds are changed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.core_model_registry import CORE_MODEL_REGISTRY

OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_all_core"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEASIBILITY_CSV = OUTPUT_DIR / "euler_all_core_substep_feasibility.csv"
FEASIBILITY_MD = OUTPUT_DIR / "euler_all_core_substep_feasibility.md"
INVENTORY_CSV = OUTPUT_DIR / "euler_all_core_model_inventory.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Classification table
# substep_status values:
#   substep_supported
#   substep_supported_with_caveat
#   substep_not_supported_api
#   substep_not_supported_discrete_daily_formula
#   substep_not_supported_unit_hydrograph_dependency
#   manual_review_required
# ─────────────────────────────────────────────────────────────────────────────

FEASIBILITY_TABLE: dict[str, dict] = {
    # ── Already validated representative models ────────────────────────────
    "hbv96": {
        "substep_status": "substep_supported",
        "reason": "Explicit flux partitioning with rate parameters (cfmax, k0, k1, perc, cflux) that scale linearly with dt. Snow/rain threshold avoided by warm forcing. Already validated.",
        "core_state_variables": "S1:snow_water_equivalent, S2:snow_liquid_water, S3:soil_moisture, S4:upper_zone_storage, S5:lower_zone_storage",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "tt,tti,ttm,cfr,cfmax,whc,cflux,fc,lp,beta,k0,alpha,perc,k1,maxbas",
        "dt_scaling_strategy": "Rate parameters (cfmax,k0,k1,perc,cflux) scaled by dt; capacity and shape parameters unchanged",
        "source_file": "hbv96.py",
        "notes": "Representative model; already validated at median p_state=0.954",
    },
    "hymod": {
        "substep_status": "substep_supported",
        "reason": "Storage-scaled flux partitions and linear reservoir rates (kf, ks) scale cleanly with dt. Already validated.",
        "core_state_variables": "S1:soil_moisture, S2:fast_res_1, S3:fast_res_2, S4:fast_res_3, S5:slow_reservoir",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "smax,b_exp,a_split,kf,ks",
        "dt_scaling_strategy": "Rate parameters (kf, ks) scaled by dt",
        "source_file": "hymod.py",
        "notes": "Representative model; already validated at median p_state=1.009",
    },
    "flexb": {
        "substep_status": "substep_supported",
        "reason": "Unsaturated and routing updates admit dt-scaled wrapper; saturation_3 rewrite unchanged. Already validated.",
        "core_state_variables": "S1:unsaturated_zone, S2:fast_reservoir, S3:slow_reservoir",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "s1max,beta,d_split,percmax,lp,nlagf,nlags,kf,ks",
        "dt_scaling_strategy": "Rate parameters (percmax, kf, ks) scaled by dt",
        "source_file": "flexb.py",
        "notes": "Representative model; already validated at median p_state=1.009",
    },
    "vic": {
        "substep_status": "substep_supported",
        "reason": "Interception/soil/groundwater updates with rate parameters (k1, k2) that scale with dt. Already validated.",
        "core_state_variables": "S1:interception_store, S2:soil_moisture, S3:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "ibar,idelta,ishift,stot,fsm,b,k1,c1,k2,c2",
        "dt_scaling_strategy": "Rate parameters (k1, k2) scaled by dt",
        "source_file": "vic.py",
        "notes": "Representative model; already validated at median p_state=1.010",
    },

    # ── FLEX family ────────────────────────────────────────────────────────
    "flexi": {
        "substep_status": "substep_supported",
        "reason": "Same structure as flexb with added interception store. Rate parameters (percmax, kf, ks) scale cleanly with dt.",
        "core_state_variables": "S1:interception, S2:unsaturated_zone, S3:fast_reservoir, S4:slow_reservoir",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "smax,beta,d_split,percmax,lp,nlagf,nlags,kf,ks,imax",
        "dt_scaling_strategy": "Rate parameters (percmax, kf, ks) scaled by dt",
        "source_file": "flexi.py",
        "notes": "FLEX family extension; same dt-wrapping approach as flexb",
    },
    "flexis": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "Same as flexi but adds snow module. Use warm forcing (T >> snow threshold) to stay in rain-only regime for convergence test.",
        "core_state_variables": "S1:snow, S2:interception, S3:unsaturated_zone, S4:fast_reservoir, S5:slow_reservoir",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "smax,beta,d_split,percmax,lp,nlagf,nlags,kf,ks,imax,tt,ddf",
        "dt_scaling_strategy": "Rate parameters scaled by dt; warm forcing avoids snow/rain threshold",
        "source_file": "flexis.py",
        "notes": "Caveat: snow module has temperature threshold; use T >> tt to stay smooth",
    },

    # ── Alpine family ──────────────────────────────────────────────────────
    "alpine1": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "Two-store snow+runoff model. Rate parameter tc scales with dt. Snow threshold at tt; use warm T to stay in rain regime.",
        "core_state_variables": "S1:snow_water_equivalent, S2:catchment_storage",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "tt,ddf,Smax,tc",
        "dt_scaling_strategy": "Rate parameter tc scaled by dt; warm forcing avoids snow/melt threshold",
        "source_file": "alpine1.py",
        "notes": "Caveat: snow threshold at tt; warm-forcing scenario required for smooth convergence",
    },
    "alpine2": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "Extended alpine model with separate interflow and baseflow routing. Rate parameters (tcin, tcbf) scale with dt. Snow threshold at tt.",
        "core_state_variables": "S1:snow_water_equivalent, S2:catchment_storage",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "tt,ddf,Smax,Cfc,tcin,tcbf",
        "dt_scaling_strategy": "Rate parameters (tcin, tcbf) scaled by dt; warm forcing avoids snow/melt threshold",
        "source_file": "alpine2.py",
        "notes": "Caveat: snow threshold at tt; warm-forcing scenario required",
    },

    # ── Australia/Collie family ────────────────────────────────────────────
    "australia": {
        "substep_status": "substep_supported",
        "reason": "Three-store model with power-law drainage (alpha_ss, beta_ss, k_deep, alpha_bf, beta_bf). Power-law fluxes scale with dt. No hard daily-only accumulation.",
        "core_state_variables": "S1:soil_moisture, S2:subsurface_store, S3:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "sb,phi,fc_frac,alpha_ss,beta_ss,k_deep,alpha_bf,beta_bf",
        "dt_scaling_strategy": "Flux rates (alpha_ss, k_deep, alpha_bf) scaled by dt",
        "source_file": "australia.py",
        "notes": "Power-law drainage admits dt scaling; no daily-only empirical rule",
    },
    "collie1": {
        "substep_status": "substep_supported",
        "reason": "Single-store saturation-excess bucket. No rate parameters—flux is proportional fraction of P. dt wrapper applies.",
        "core_state_variables": "S1:catchment_storage",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "Smax",
        "dt_scaling_strategy": "No explicit rate parameters; forcing P scaled by dt per substep",
        "source_file": "collie1.py",
        "notes": "Pure saturation-excess; dt scaling via forcing subdivision",
    },
    "collie2": {
        "substep_status": "substep_supported",
        "reason": "Single-store model with dual evap (bare soil + vegetated). Forcing-proportional fluxes scale with dt.",
        "core_state_variables": "S1:catchment_storage",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "Smax,Sfc_frac,a,M",
        "dt_scaling_strategy": "Forcing P and PET scaled by dt per substep",
        "source_file": "collie2.py",
        "notes": "",
    },
    "collie3": {
        "substep_status": "substep_supported",
        "reason": "Two-store model with topographic wetness index drainage. Power-law flux rates scale with dt.",
        "core_state_variables": "S1:soil_moisture, S2:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "smax,fc,a,m,b,lambda_par",
        "dt_scaling_strategy": "Flux rates scaled by dt",
        "source_file": "collie3.py",
        "notes": "",
    },

    # ── GR4J ──────────────────────────────────────────────────────────────
    "gr4j": {
        "substep_status": "substep_not_supported_discrete_daily_formula",
        "reason": "GR4J uses an analytically integrated daily production store formula (S1 update via tanh/sinh) that cannot be split into substeps without rewriting the formula. The formula is a closed-form daily solution, not a flux-rate equation.",
        "core_state_variables": "S1:production_store, S2:routing_store",
        "fluxes_to_track": "Q (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "x1,x2,x3,x4",
        "dt_scaling_strategy": "N/A - daily closed-form formula",
        "source_file": "gr4j.py",
        "notes": "GR4J production-store update uses tanh(P/x1) / tanh(1/x1) which is analytically exact for daily step; substep scaling would change the formula meaning",
    },

    # ── GSFB ──────────────────────────────────────────────────────────────
    "gsfb": {
        "substep_status": "substep_not_supported_discrete_daily_formula",
        "reason": "GSFB uses threshold-based recharge/interflow partitioning (frate, dpf, sdrmax) that represent daily empirical rules; the fractional parameterisation does not map cleanly to a flux-rate dt interpretation.",
        "core_state_variables": "S1:soil_store, S2:deep_store, S3:routing_store",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "c,ndc,smax,emax,frate,b,dpf,sdrmax",
        "dt_scaling_strategy": "N/A - threshold partitioning parameters are not dt-scalable",
        "source_file": "gsfb.py",
        "notes": "Previously identified as fail_due_to_substep_not_supported in representative suite",
    },

    # ── Hillslope / Wetland / Plateau (FLEX-Topo) ─────────────────────────
    "hillslope": {
        "substep_status": "substep_supported",
        "reason": "FLEX-Topo hillslope with interception, saturation-excess, and linear baseflow (kh). Rate parameter kh scales with dt.",
        "core_state_variables": "S1:hillslope_store, S2:baseflow_store",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "dw,betaw,swmax,a,th,c_rad,kh",
        "dt_scaling_strategy": "Rate parameter kh scaled by dt",
        "source_file": "hillslope.py",
        "notes": "",
    },
    "wetland": {
        "substep_status": "substep_supported",
        "reason": "Single-store FLEX-Topo wetland with interception, saturation-excess, evaporation, and linear baseflow (kw). Rate parameter kw scales with dt.",
        "core_state_variables": "S1:wetland_store",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "dw,betaw,swmax,kw",
        "dt_scaling_strategy": "Rate parameter kw scaled by dt",
        "source_file": "wetland.py",
        "notes": "",
    },
    "plateau": {
        "substep_status": "substep_supported",
        "reason": "Two-store FLEX-Topo plateau with infiltration, capillary rise, and percolation (kp). Rate parameter kp scales with dt.",
        "core_state_variables": "S1:unsaturated_store, S2:percolation_store",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "fmax,dp,sumax,lp,p_coeff,tp,c_rise,kp",
        "dt_scaling_strategy": "Rate parameter kp scaled by dt",
        "source_file": "plateau.py",
        "notes": "",
    },

    # ── IHACRES ───────────────────────────────────────────────────────────
    "ihacres": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "IHACRES uses a deficit store (S1 represents soil moisture deficit, not storage). The effective-rainfall computation (saturation_5) is proportional and dt-scalable. Routing tau parameters represent time constants; scaled by dt. Caveat: S1 has sign inversion (deficit convention).",
        "core_state_variables": "S1:moisture_deficit (negative=wetter)",
        "fluxes_to_track": "flux_u_total (effective_rainfall)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "lp,d,p,alpha,tau_q,tau_s",
        "dt_scaling_strategy": "Routing time constants (tau_q, tau_s) converted to rates; scaled by dt",
        "source_file": "ihacres.py",
        "notes": "Deficit convention: state_sign=-1 applied in registry; routing reservoirs excluded from core test",
    },

    # ── ModHydrolog ───────────────────────────────────────────────────────
    "modhydrolog": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "Five-store model with interception, soil moisture, deep drainage, and groundwater stores. Rate parameters (k1, k2, k3) scale with dt. Caveat: sequential update ordering means some threshold crossings are possible; use interior-state warm scenario.",
        "core_state_variables": "S1:interception, S2:soil_moisture, S3:deep_drainage, S4:groundwater_1, S5:groundwater_2",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "insc,coeff,sq,smsc,sub,crak,em,dsc,ads,md,vcond,dlev,k1,k2,k3",
        "dt_scaling_strategy": "Rate parameters (k1, k2, k3) scaled by dt",
        "source_file": "modhydrolog.py",
        "notes": "Caveat: interception and sequential thresholds; use midpoint parameters and moderate forcing",
    },

    # ── MOPEX family ──────────────────────────────────────────────────────
    "mopex1": {
        "substep_status": "substep_supported",
        "reason": "Four-store model with rate parameters (tw, tu, tc) that scale with dt. Sequential explicit update; no daily-only formula.",
        "core_state_variables": "S1:soil_store, S2:groundwater, Sc1:fast_cascade, Sc2:slow_cascade",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "s1max,tw,tu,se,tc",
        "dt_scaling_strategy": "Rate parameters (tw, tu, tc) scaled by dt",
        "source_file": "mopex1.py",
        "notes": "",
    },
    "mopex2": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "MOPEX-1 with snow module. Rate parameters (tw, tu, tc) scale with dt. Use warm T to stay in rain regime for smooth convergence.",
        "core_state_variables": "Sn:snow, S1:soil_store, S2:groundwater, Sc1:fast_cascade, Sc2:slow_cascade",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "tcrit,ddf,s2max,tw,tu,se,tc",
        "dt_scaling_strategy": "Rate parameters (tw, tu, tc) scaled by dt; warm forcing avoids snow threshold",
        "source_file": "mopex2.py",
        "notes": "Caveat: snow threshold at tcrit; warm-forcing scenario required",
    },
    "mopex3": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "MOPEX-2 with additional deep storage (s3max). Same dt-scaling approach; warm forcing for snow avoidance.",
        "core_state_variables": "Sn:snow, S1:soil_store, S2:groundwater, Sc1:fast_cascade, Sc2:slow_cascade",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "tcrit,ddf,s2max,tw,tu,se,s3max,tc",
        "dt_scaling_strategy": "Rate parameters (tw, tu, tc) scaled by dt; warm forcing avoids snow threshold",
        "source_file": "mopex3.py",
        "notes": "Caveat: snow threshold at tcrit; warm-forcing scenario required",
    },
    "mopex4": {
        "substep_status": "substep_not_supported_api",
        "reason": "mopex4_step requires a 'doy' (day-of-year) keyword argument for seasonally varying interception. The doy is an integer daily index that cannot be meaningfully subdivided into substeps. Providing fractional doy would change model physics.",
        "core_state_variables": "Sn:snow, S1:soil_store, S2:groundwater, Sc1:fast_cascade, Sc2:slow_cascade",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET, doy",
        "parameters_used": "tcrit,ddf,s2max,tw,alpha,is_time,tu,se,s3max,tc",
        "dt_scaling_strategy": "N/A - doy is integer daily index; not substep-divisible",
        "source_file": "mopex4.py",
        "notes": "doy-based interception_4 uses day-of-year as a daily empirical index",
    },
    "mopex5": {
        "substep_status": "substep_not_supported_api",
        "reason": "Same as mopex4: requires 'doy' keyword argument for seasonally varying interception that cannot be substep-divided.",
        "core_state_variables": "Sn:snow, S1:soil_store, S2:groundwater, Sc1:fast_cascade, Sc2:slow_cascade",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET, doy",
        "parameters_used": "tcrit,ddf,s2max,tw,alpha,is_time,tmin,trange,tu,se,s3max,tc",
        "dt_scaling_strategy": "N/A - doy is integer daily index; not substep-divisible",
        "source_file": "mopex5.py",
        "notes": "doy-based interception; same constraint as mopex4",
    },

    # ── NewZealand family ─────────────────────────────────────────────────
    "newzealand1": {
        "substep_status": "substep_supported",
        "reason": "Single-store model with baseflow rate tcbf that scales with dt. Dual evap formulation is forcing-proportional.",
        "core_state_variables": "S1:soil_store",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "s1max,sfc_frac,m,a,b,tcbf",
        "dt_scaling_strategy": "Rate parameter tcbf scaled by dt",
        "source_file": "newzealand1.py",
        "notes": "",
    },
    "newzealand2": {
        "substep_status": "substep_supported",
        "reason": "Two-store extension of NewZealand1 with added groundwater store and delay. Rate parameters (tcbf, d_delay) scale with dt.",
        "core_state_variables": "S1:soil_store, S2:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "s1max,s2max,sfc_frac,m,a,b,tcbf,d_delay",
        "dt_scaling_strategy": "Rate parameters (tcbf, d_delay) scaled by dt",
        "source_file": "newzealand2.py",
        "notes": "",
    },

    # ── Penman ────────────────────────────────────────────────────────────
    "penman": {
        "substep_status": "substep_supported",
        "reason": "Three-store model with linear drainage rate k1 that scales with dt. Storage-partitioned fluxes admit dt scaling.",
        "core_state_variables": "S1:upper_store, S2:lower_store, S3:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "smax,phi,gam,k1",
        "dt_scaling_strategy": "Rate parameter k1 scaled by dt",
        "source_file": "penman.py",
        "notes": "State sign: S2 is deficit store (sign=-1) per registry override",
    },

    # ── SimHyd ────────────────────────────────────────────────────────────
    "simhyd": {
        "substep_status": "substep_supported",
        "reason": "Three-store model with linear baseflow rate k that scales with dt. Sequential explicit update; no daily-only formula.",
        "core_state_variables": "S1:interception, S2:soil_moisture, S3:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "insc,coeff,sq,smsc,sub,crak,k",
        "dt_scaling_strategy": "Rate parameter k scaled by dt",
        "source_file": "simhyd.py",
        "notes": "",
    },

    # ── SMAR ──────────────────────────────────────────────────────────────
    "smar": {
        "substep_status": "substep_supported_with_caveat",
        "reason": "Six-store SMAR model with 5-layer soil stack and groundwater cascade. Rate parameter kg scales with dt. Caveat: n_res and nk_delay are integer-valued routing parameters (Nash cascade); they define cascade length and cannot be substep-divided. Core soil stores are dt-wrappable; routing cascade excluded.",
        "core_state_variables": "S1:soil_layer_1, S2:soil_layer_2, S3:soil_layer_3, S4:soil_layer_4, S5:soil_layer_5, S6:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "h_runoff,y_inf,smax,c_evap,g_rech,kg,n_res,nk_delay",
        "dt_scaling_strategy": "Rate parameter kg scaled by dt; n_res/nk_delay are integer routing descriptors kept fixed",
        "source_file": "smar.py",
        "notes": "Caveat: n_res, nk_delay are Nash-cascade integers; test convergence of core soil+GW states only",
    },

    # ── Susannah family ───────────────────────────────────────────────────
    "susannah1": {
        "substep_status": "substep_supported",
        "reason": "Two-store soil+groundwater model with linear baseflow rate r. Dual evap formulation scales with dt.",
        "core_state_variables": "S1:soil_store, S2:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "sb,sfc_frac,m,a,b,r",
        "dt_scaling_strategy": "Rate parameter r scaled by dt",
        "source_file": "susannah1.py",
        "notes": "",
    },
    "susannah2": {
        "substep_status": "substep_supported",
        "reason": "Two-store variant of Susannah1 with different evap/drainage parameterisation. Rate parameters (r, c, d) scale with dt.",
        "core_state_variables": "S1:soil_store, S2:groundwater",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "sb,phi,fc,r,c,d",
        "dt_scaling_strategy": "Rate parameters (r, c, d) scaled by dt",
        "source_file": "susannah2.py",
        "notes": "",
    },

    # ── Tank ──────────────────────────────────────────────────────────────
    "tank": {
        "substep_status": "substep_not_supported_discrete_daily_formula",
        "reason": "Tank model uses threshold-based overflow rules (st threshold for side-outlet activation) that represent daily empirical accumulation rules. The threshold parameters (st, f1, f2, f3) define discrete daily bucket levels that do not have a clean dt-scaled interpretation. The side-outlet activation is a binary switch on daily storage levels.",
        "core_state_variables": "S1:tank_1, S2:tank_2, S3:tank_3, S4:tank_4",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "a0,b0,c0,a1,fa,fb,fc,fd,st,f2,f1,f3",
        "dt_scaling_strategy": "N/A - threshold st is a daily storage level not a rate",
        "source_file": "tank.py",
        "notes": "Side-outlet activation threshold cannot be substep-divided without changing model physics",
    },

    # ── TCM ───────────────────────────────────────────────────────────────
    "tcm": {
        "substep_status": "substep_not_supported_api",
        "reason": "tcm_step requires a 'mean_P' keyword argument (long-term mean annual precipitation) that is a climatological constant, not a per-step forcing. This parameter defines deficit-accounting ratios and cannot be meaningfully subdivided into substeps. Also has a deficit store (S2) with sign inversion.",
        "core_state_variables": "S1:interception, S2:deficit_store, S3:fast_reservoir, S4:slow_reservoir",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET, mean_P",
        "parameters_used": "phi,rc,gam,k1,fa,k2",
        "dt_scaling_strategy": "N/A - mean_P is a climatological constant; not substep-divisible",
        "source_file": "tcm.py",
        "notes": "Previously identified as fail_due_to_substep_not_supported in representative suite",
    },

    # ── TOPMODEL ──────────────────────────────────────────────────────────
    "topmodel": {
        "substep_status": "substep_not_supported_discrete_daily_formula",
        "reason": "TOPMODEL uses an exponential deficit-discharge relationship (q0*exp(-f*S2)) where S2 is a deficit store with strict sign convention. The threshold/kink in saturation deficit (st) and the exponential activation make clean dt-substep wrapping problematic without formula changes.",
        "core_state_variables": "S1:unsaturated_zone, S2:saturated_deficit",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "suzmax,st,kd,q0,f,chi,phi",
        "dt_scaling_strategy": "N/A - deficit-threshold activation and exponential discharge not dt-wrappable without formula changes",
        "source_file": "topmodel.py",
        "notes": "Previously identified as fail_due_to_substep_not_supported in representative suite",
    },

    # ── US1 ───────────────────────────────────────────────────────────────
    "us1": {
        "substep_status": "substep_supported",
        "reason": "Two-store US1 model with interception and soil moisture. Rate parameter alpha_ss scales with dt.",
        "core_state_variables": "S1:interception_store, S2:soil_store",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "alpha_ei,m,smax,fc,alpha_ss",
        "dt_scaling_strategy": "Rate parameter alpha_ss scaled by dt",
        "source_file": "us1.py",
        "notes": "",
    },

    # ── Xinanjiang ────────────────────────────────────────────────────────
    "xinanjiang": {
        "substep_status": "substep_supported",
        "reason": "Four-store Xinanjiang (Xin'anjiang) model with routing reservoirs. Rate parameters (ki, kg, ci, cg) scale with dt. Sequential explicit update.",
        "core_state_variables": "S1:soil_moisture, S2:interflow_reservoir, S3:groundwater_reservoir, S4:free_water_storage",
        "fluxes_to_track": "flux_qt (total_runoff)",
        "forcing_variables": "P, T, PET",
        "parameters_used": "aim,par_a,par_b,stot,fwm,flm,par_c,ex,ki,kg,ci,cg",
        "dt_scaling_strategy": "Rate parameters (ki, kg, ci, cg) scaled by dt",
        "source_file": "xinanjiang.py",
        "notes": "",
    },

    # ── SHM (disabled) ────────────────────────────────────────────────────
    "shm": {
        "substep_status": "substep_not_supported_api",
        "reason": "shm.py is empty and does not define a runnable core model; disabled in registry.",
        "core_state_variables": "",
        "fluxes_to_track": "",
        "forcing_variables": "",
        "parameters_used": "",
        "dt_scaling_strategy": "N/A",
        "source_file": "shm.py",
        "notes": "Disabled model",
    },
}

# Models that need unit-hydrograph-inseparable check
# None in current API - all routing is either cascade (MOPEX, SMAR) or parameter-based
UH_INSEPARABLE: set[str] = set()


def _has_unit_hydrograph(model_name: str) -> bool:
    """Check if model has unit hydrograph dependency inseparable from core."""
    return model_name in UH_INSEPARABLE


def write_inventory_csv() -> list[dict]:
    rows = []
    for name, entry in sorted(CORE_MODEL_REGISTRY.items()):
        feas = FEASIBILITY_TABLE.get(name, {})
        status = feas.get("substep_status", "manual_review_required")
        uh_insep = _has_unit_hydrograph(name)
        rows.append({
            "model": name,
            "core_file": entry.model_file,
            "runnable_in_registry": str(entry.enabled),
            "has_unit_hydrograph_dependency": str(uh_insep),
            "unit_hydrograph_excluded": "true",
            "candidate_for_core_dt_test": str(entry.enabled and not uh_insep and status not in {
                "substep_not_supported_api",
                "substep_not_supported_discrete_daily_formula",
                "substep_not_supported_unit_hydrograph_dependency",
            }),
            "notes": entry.skip_reason if not entry.enabled else feas.get("reason", "")[:120],
        })

    with open(INVENTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {INVENTORY_CSV}")
    return rows


def write_feasibility_csv() -> list[dict]:
    rows = []
    for name in sorted(FEASIBILITY_TABLE.keys()):
        feas = FEASIBILITY_TABLE[name]
        rows.append({
            "model": name,
            "substep_status": feas["substep_status"],
            "reason": feas["reason"],
            "core_state_variables": feas["core_state_variables"],
            "fluxes_to_track": feas["fluxes_to_track"],
            "forcing_variables": feas["forcing_variables"],
            "parameters_used": feas["parameters_used"],
            "dt_scaling_strategy": feas["dt_scaling_strategy"],
            "source_file": feas["source_file"],
            "notes": feas["notes"],
        })

    with open(FEASIBILITY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {FEASIBILITY_CSV}")
    return rows


def write_feasibility_md(rows: list[dict]) -> None:
    supported = [r for r in rows if r["substep_status"] in {"substep_supported", "substep_supported_with_caveat"}]
    unsupported = [r for r in rows if r["substep_status"] not in {"substep_supported", "substep_supported_with_caveat"}]

    lines = [
        "# Euler Substep Feasibility Review — All Core Models",
        "",
        "Generated by `scripts/review_euler_substep_feasibility_all_core.py`.",
        "",
        "## Summary",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
    ]
    from collections import Counter
    counter = Counter(r["substep_status"] for r in rows)
    for status, count in sorted(counter.items()):
        lines.append(f"| {status} | {count} |")
    lines += [
        "",
        "## Supported Models",
        "",
        "| Model | Status | Key Rate Parameters | Notes |",
        "|-------|--------|---------------------|-------|",
    ]
    for r in supported:
        lines.append(f"| {r['model']} | {r['substep_status']} | {r['parameters_used'][:60]} | {r['notes'][:80]} |")

    lines += [
        "",
        "## Unsupported / Skipped Models",
        "",
        "| Model | Status | Reason |",
        "|-------|--------|--------|",
    ]
    for r in unsupported:
        lines.append(f"| {r['model']} | {r['substep_status']} | {r['reason'][:120]} |")

    lines += [
        "",
        "## Classification Definitions",
        "",
        "- **substep_supported**: Core flux/state update is consistent with Euler dt-scaling; rate parameters scale linearly with dt.",
        "- **substep_supported_with_caveat**: dt-wrapping is valid but requires a specific forcing regime (e.g., warm temperature to avoid snow threshold).",
        "- **substep_not_supported_api**: The model API requires a non-subdivisible argument (e.g., `doy`, `mean_P`) or is disabled.",
        "- **substep_not_supported_discrete_daily_formula**: The core formula is an analytically integrated daily solution or threshold-accumulation rule that cannot be split into substeps without changing the formula meaning.",
        "- **substep_not_supported_unit_hydrograph_dependency**: Unit hydrograph routing is inseparable from the core update in the current API.",
        "",
        "## Unit Hydrograph Routing Exclusion",
        "",
        "Unit hydrograph routing is excluded from this task by design. All models in this suite use either:",
        "- Linear reservoir routing (parameter-based, dt-scalable), or",
        "- Nash cascade routing (integer parameter, kept fixed across substeps)",
        "",
        "No model in the current core set has unit hydrograph routing inseparable from its core state update.",
    ]

    with open(FEASIBILITY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {FEASIBILITY_MD}")


def main() -> int:
    print("=== Euler Substep Feasibility Review — All Core Models ===")
    inventory_rows = write_inventory_csv()
    feasibility_rows = write_feasibility_csv()
    write_feasibility_md(feasibility_rows)

    supported = [r for r in feasibility_rows if r["substep_status"] in {"substep_supported", "substep_supported_with_caveat"}]
    unsupported = [r for r in feasibility_rows if r["substep_status"] not in {"substep_supported", "substep_supported_with_caveat"}]

    print(f"\nTotal models: {len(feasibility_rows)}")
    print(f"substep_supported: {sum(1 for r in feasibility_rows if r['substep_status'] == 'substep_supported')}")
    print(f"substep_supported_with_caveat: {sum(1 for r in feasibility_rows if r['substep_status'] == 'substep_supported_with_caveat')}")
    print(f"substep_not_supported_api: {sum(1 for r in feasibility_rows if r['substep_status'] == 'substep_not_supported_api')}")
    print(f"substep_not_supported_discrete_daily_formula: {sum(1 for r in feasibility_rows if r['substep_status'] == 'substep_not_supported_discrete_daily_formula')}")
    print(f"substep_not_supported_unit_hydrograph_dependency: {sum(1 for r in feasibility_rows if r['substep_status'] == 'substep_not_supported_unit_hydrograph_dependency')}")
    print(f"manual_review_required: {sum(1 for r in feasibility_rows if r['substep_status'] == 'manual_review_required')}")
    print(f"\nCandidate models for dt convergence test: {len(supported)}")
    for r in supported:
        print(f"  {r['model']} ({r['substep_status']})")
    print(f"\nExcluded models: {len(unsupported)}")
    for r in unsupported:
        print(f"  {r['model']} ({r['substep_status']}): {r['reason'][:80]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
