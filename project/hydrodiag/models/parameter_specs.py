"""Parameter specifications for all hydrological models.

Each spec dict maps parameter names to:
    lower: lower bound (physical scale)
    upper: upper bound (physical scale)
    default: default value (physical scale)
    unit: physical unit
    description: human-readable description
    process: which hydrological process this parameter belongs to
    is_snow: whether this is a snow module parameter
"""

import math

HBV_PARAM_SPECS = {
    "parBETA": {
        "lower": 1.0,
        "upper": 6.0,
        "default": 2.0,
        "unit": "-",
        "description": "Exponent of soil moisture control on recharge",
        "process": "soil",
        "is_snow": False,
    },
    "parFC": {
        "lower": 50.0,
        "upper": 1000.0,
        "default": 200.0,
        "unit": "mm",
        "description": "Field capacity of soil moisture storage",
        "process": "soil",
        "is_snow": False,
    },
    "parK0": {
        "lower": 0.05,
        "upper": 0.9,
        "default": 0.3,
        "unit": "1/day",
        "description": "Recession coefficient for surface flow (upper zone)",
        "process": "routing",
        "is_snow": False,
    },
    "parK1": {
        "lower": 0.01,
        "upper": 0.5,
        "default": 0.1,
        "unit": "1/day",
        "description": "Recession coefficient for interflow (upper zone)",
        "process": "routing",
        "is_snow": False,
    },
    "parK2": {
        "lower": 0.001,
        "upper": 0.2,
        "default": 0.05,
        "unit": "1/day",
        "description": "Recession coefficient for baseflow (lower zone)",
        "process": "routing",
        "is_snow": False,
    },
    "parLP": {
        "lower": 0.2,
        "upper": 1.0,
        "default": 0.7,
        "unit": "-",
        "description": "Fraction of FC above which actual ET equals potential ET",
        "process": "soil",
        "is_snow": False,
    },
    "parPERC": {
        "lower": 0.0,
        "upper": 10.0,
        "default": 2.0,
        "unit": "mm/day",
        "description": "Maximum percolation rate from upper to lower zone",
        "process": "groundwater",
        "is_snow": False,
    },
    "parUZL": {
        "lower": 0.0,
        "upper": 100.0,
        "default": 20.0,
        "unit": "mm",
        "description": "Threshold for surface flow generation in upper zone",
        "process": "routing",
        "is_snow": False,
    },
    "parTT": {
        "lower": -2.5,
        "upper": 2.5,
        "default": 0.0,
        "unit": "degC",
        "description": "Threshold temperature for snow/rain partitioning",
        "process": "snow",
        "is_snow": True,
    },
    "parCFMAX": {
        "lower": 0.5,
        "upper": 10.0,
        "default": 3.0,
        "unit": "mm/(degC*day)",
        "description": "Degree-day melt factor",
        "process": "snow",
        "is_snow": True,
    },
    "parCFR": {
        "lower": 0.0,
        "upper": 0.1,
        "default": 0.05,
        "unit": "-",
        "description": "Refreezing coefficient",
        "process": "snow",
        "is_snow": True,
    },
    "parCWH": {
        "lower": 0.0,
        "upper": 0.2,
        "default": 0.1,
        "unit": "1/day",
        "description": "Water holding capacity of snowpack",
        "process": "snow",
        "is_snow": True,
    },
}

GR4J_PARAM_SPECS = {
    "x1": {
        "lower": 10.0,
        "upper": 1200.0,
        "default": 350.0,
        "unit": "mm",
        "description": "Maximum capacity of production store",
        "process": "soil",
        "is_snow": False,
    },
    "x2": {
        "lower": -5.0,
        "upper": 3.0,
        "default": 0.0,
        "unit": "mm/day",
        "description": "Groundwater exchange coefficient",
        "process": "groundwater",
        "is_snow": False,
    },
    "x3": {
        "lower": 20.0,
        "upper": 5000.0,
        "default": 500.0,
        "unit": "mm",
        "description": "Maximum capacity of routing store",
        "process": "routing",
        "is_snow": False,
    },
    "x4": {
        "lower": 1.1,
        "upper": 10.0,
        "default": 2.0,
        "unit": "day",
        "description": "Time base of unit hydrographs",
        "process": "routing",
        "is_snow": False,
    },
}

XAJ_PARAM_SPECS = {
    "xaj_k": {
        "lower": 0.5,
        "upper": 2.0,
        "default": 1.0,
        "unit": "-",
        "description": "Ratio of potential ET to reference crop evaporation",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_b": {
        "lower": 0.1,
        "upper": 2.0,
        "default": 0.3,
        "unit": "-",
        "description": "Exponent of tension water capacity curve",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_im": {
        "lower": 0.0,
        "upper": 0.3,
        "default": 0.01,
        "unit": "-",
        "description": "Impervious area fraction",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_um": {
        "lower": 5.0,
        "upper": 50.0,
        "default": 20.0,
        "unit": "mm",
        "description": "Upper layer tension water capacity",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_lm": {
        "lower": 20.0,
        "upper": 200.0,
        "default": 80.0,
        "unit": "mm",
        "description": "Lower layer tension water capacity",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_dm": {
        "lower": 20.0,
        "upper": 200.0,
        "default": 40.0,
        "unit": "mm",
        "description": "Deep layer tension water capacity",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_c": {
        "lower": 0.05,
        "upper": 0.3,
        "default": 0.15,
        "unit": "-",
        "description": "Deep layer evaporation coefficient",
        "process": "soil",
        "is_snow": False,
    },
    "xaj_sm": {
        "lower": 5.0,
        "upper": 100.0,
        "default": 30.0,
        "unit": "mm",
        "description": "Areal mean free water capacity of surface layer",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_ex": {
        "lower": 0.1,
        "upper": 2.0,
        "default": 1.2,
        "unit": "-",
        "description": "Exponent of free water capacity curve",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_ki": {
        "lower": 0.0,
        "upper": 0.7,
        "default": 0.3,
        "unit": "1/day",
        "description": "Outflow coefficient for interflow",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_kg": {
        "lower": 0.0,
        "upper": 0.7,
        "default": 0.2,
        "unit": "1/day",
        "description": "Outflow coefficient for groundwater",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_ci": {
        "lower": 0.1,
        "upper": 1.0,
        "default": 0.5,
        "unit": "-",
        "description": "Recession constant for interflow reservoir",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_cg": {
        "lower": 0.9,
        "upper": 1.0,
        "default": 0.98,
        "unit": "-",
        "description": "Recession constant for groundwater reservoir",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_a": {
        "lower": 0.0,
        "upper": 2.9,
        "default": 2.5,
        "unit": "-",
        "description": "HBV-compatible Gamma-UH shape parameter",
        "process": "routing",
        "is_snow": False,
    },
    "xaj_theta": {
        "lower": 0.0,
        "upper": 6.5,
        "default": 1.5,
        "unit": "day",
        "description": "HBV-compatible Gamma-UH scale parameter",
        "process": "routing",
        "is_snow": False,
    },
}

# XAJLite deliberately uses the same Gamma-UH routing parameter ranges as
# bettermodel/HBV (rout_a in [0, 2.9], rout_b in [0, 6.5]).  Keep the XAJ
# parameter names for API compatibility while retaining the historical XAJ
# ranges in ``XAJ_PARAM_SPECS``.
XAJ_LITE_PARAM_SPECS = {
    **XAJ_PARAM_SPECS,
    "xaj_a": {
        **XAJ_PARAM_SPECS["xaj_a"],
        "lower": 0.0,
        "upper": 2.9,
        "description": "HBV-compatible Gamma-UH shape parameter",
    },
    "xaj_theta": {
        **XAJ_PARAM_SPECS["xaj_theta"],
        "lower": 0.0,
        "upper": 6.5,
        "description": "HBV-compatible Gamma-UH scale parameter",
    },
}

CEMANEIGE_CORE_PARAM_SPECS = {
    "cn_ctg": {
        "lower": 0.0,
        "upper": 1.0,
        "default": 0.5,
        "unit": "-",
        "description": "Weighting coefficient for thermal state of snowpack",
        "process": "snow",
        "is_snow": True,
    },
    "cn_kf": {
        "lower": 0.0,
        "upper": 10.0,
        "default": 3.0,
        "unit": "mm/(degC*day)",
        "description": "Degree-day melt factor for CemaNeige",
        "process": "snow",
        "is_snow": True,
    },
}

CEMANEIGE_HYST_PARAM_SPECS = {
    **CEMANEIGE_CORE_PARAM_SPECS,
    "cn_thacc": {
        "lower": 1e-6,
        "upper": 1000.0,
        "default": 10.0,
        "unit": "mm",
        "description": "Accumulation threshold for snow-covered area",
        "process": "snow",
        "is_snow": True,
    },
    "cn_rsp": {
        "lower": 0.0,
        "upper": 1.0,
        "default": 0.1,
        "unit": "-",
        "description": "Annual solid precipitation fraction parameter for SCA",
        "process": "snow",
        "is_snow": True,
    },
}

# Public two-parameter specification for the basic CemaNeige model.
CEMANEIGE_PARAM_SPECS = CEMANEIGE_CORE_PARAM_SPECS

PRECIP_DELAY_PARAM_SPECS = {
    "pd_alpha": {
        "lower": 0.0,
        "upper": 1.0,
        "default": 0.5,
        "unit": "-",
        "description": "Fraction of precipitation entering temporary delay storage",
        "process": "precipitation_delay",
        "is_snow": False,
    },
    "pd_tau": {
        "lower": 1e-3,
        "upper": 90.0,
        "default": 5.0,
        "unit": "day",
        "description": "Release time scale of temporary precipitation storage",
        "process": "precipitation_delay",
        "is_snow": False,
    },
}

# TGD2 is a temperature-dependent generic precipitation-memory module, not a
# snow accumulation and melt model.  These are the single authoritative bounds
# used by IC/CMA-ES, XNES, and dPL. Both quantities are log-mapped by adapters.
TGD2_STRUCTURE_VERSION = "temperature_dependent_generic_delay2_v1"
TGD2_T_REF_C = 0.0
TGD2_T_SCALE_C = 2.0
TGD2_EPS_DAYS = 1e-6
TGD2_PARAM_SPECS = {
    "tgd_tau_warm": {
        "lower": 1e-4, "upper": 3.0, "default": 0.25,
        "unit": "day", "description": "Warm-condition generic residence time",
        "process": "temperature_dependent_generic_delay2", "is_snow": False,
    },
    "tgd_delta_tau_cold": {
        "lower": 0.1, "upper": 180.0, "default": 10.0,
        "unit": "day", "description": "Additional cold-condition generic residence time",
        "process": "temperature_dependent_generic_delay2", "is_snow": False,
    },
}

GR4J_CN_PARAM_SPECS = {
    **CEMANEIGE_CORE_PARAM_SPECS,
    **{f"gr4j_{k}": {**v, "is_snow": False} for k, v in GR4J_PARAM_SPECS.items()},
}

XAJ_CN_PARAM_SPECS = {
    **CEMANEIGE_CORE_PARAM_SPECS,
    **XAJ_PARAM_SPECS,
}

# Controlled XAJ structures.  Both public variants include the existing
# two-parameter CemaNeige module by default. XAJ_RWPE replaces only XAJ's
# sequential lower/deep evaporation parameter C with a root-zone stress
# threshold; XAJ_2S replaces the two slow-flow release/recession pairs.
XAJ_RWPE_PARAM_SPECS = {
    **CEMANEIGE_CORE_PARAM_SPECS,
    **{name: spec for name, spec in XAJ_PARAM_SPECS.items() if name != "xaj_c"},
    "xaj_tau_e": {
        "lower": 0.05, "upper": 1.0, "default": 0.5,
        "unit": "-", "description": "Aggregated root-zone evaporation stress threshold",
        "process": "soil", "is_snow": False,
    },
}

XAJ_2S_PARAM_SPECS = {
    **CEMANEIGE_CORE_PARAM_SPECS,
    **{name: spec for name, spec in XAJ_PARAM_SPECS.items()
       if name not in {"xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg"}},
    "xaj_kb": {
        # KI and KG are each [0, .7] and are normalized to a total below one
        # by the baseline.  This open interval therefore covers the baseline
        # attainable total-release range without a forward-pass clamp.
        "lower": 1e-6, "upper": 1.0 - 1e-5, "default": 0.5,
        "unit": "1/day", "description": "Merged slow-flow release coefficient",
        "process": "routing", "is_snow": False,
    },
    "xaj_cb": {
        # Union of the existing CI [.1, 1] and CG [.9, 1] memory ranges,
        # made open at the upper end so the reservoir always has release.
        "lower": 0.1, "upper": 1.0 - 1e-5, "default": 0.8,
        "unit": "-", "description": "Merged slow-flow recession constant",
        "process": "routing", "is_snow": False,
    },
}


SIMHYD_PARAM_SPECS = {
    "simhyd_insc": {
        "lower": 1e-6, "upper": 50.0, "default": 2.0, "unit": "mm",
        "description": "Interception capacity", "process": "interception", "is_snow": False,
    },
    "simhyd_coeff": {
        "lower": 1e-6, "upper": 400.0, "default": 200.0, "unit": "mm/day",
        "description": "Maximum infiltration capacity", "process": "soil", "is_snow": False,
    },
    "simhyd_sq": {
        "lower": 0.0, "upper": 10.0, "default": 2.0, "unit": "-",
        "description": "Infiltration capacity exponent", "process": "soil", "is_snow": False,
    },
    "simhyd_smsc": {
        "lower": 1.0, "upper": 1000.0, "default": 250.0, "unit": "mm",
        "description": "Soil moisture storage capacity", "process": "soil", "is_snow": False,
    },
    "simhyd_sub": {
        "lower": 0.0, "upper": 1.0, "default": 0.4, "unit": "-",
        "description": "Interflow proportionality coefficient", "process": "routing", "is_snow": False,
    },
    "simhyd_crak": {
        "lower": 0.0, "upper": 1.0, "default": 0.1, "unit": "-",
        "description": "Groundwater recharge proportionality coefficient", "process": "groundwater", "is_snow": False,
    },
    "simhyd_k": {
        "lower": 0.0, "upper": 1.0, "default": 0.3, "unit": "1/day",
        "description": "Groundwater recession coefficient", "process": "groundwater", "is_snow": False,
    },
    "simhyd_etmul": {
        "lower": 0.1, "upper": 3.0, "default": 1.0, "unit": "-",
        "description": "Potential evapotranspiration multiplier", "process": "soil", "is_snow": False,
    },
    "simhyd_a": {
        "lower": 0.0, "upper": 2.9, "default": 2.5, "unit": "-",
        "description": "HBV-compatible Gamma-UH shape parameter", "process": "routing", "is_snow": False,
    },
    "simhyd_theta": {
        "lower": 0.0, "upper": 6.5, "default": 1.5, "unit": "day",
        "description": "HBV-compatible Gamma-UH scale parameter", "process": "routing", "is_snow": False,
    },
}

SIMHYD_CN_PARAM_SPECS = {
    **CEMANEIGE_CORE_PARAM_SPECS,
    **SIMHYD_PARAM_SPECS,
}

GR4J_PD_PARAM_SPECS = {
    **PRECIP_DELAY_PARAM_SPECS,
    **{f"gr4j_{k}": {**v, "is_snow": False} for k, v in GR4J_PARAM_SPECS.items()},
}

XAJ_PD_PARAM_SPECS = {
    **PRECIP_DELAY_PARAM_SPECS,
    **XAJ_PARAM_SPECS,
}

SIMHYD_PD_PARAM_SPECS = {
    **PRECIP_DELAY_PARAM_SPECS,
    **SIMHYD_PARAM_SPECS,
}

GR4J_TGD2_PARAM_SPECS = {
    **TGD2_PARAM_SPECS,
    **{f"gr4j_{k}": {**v, "is_snow": False} for k, v in GR4J_PARAM_SPECS.items()},
}
XAJ_TGD2_PARAM_SPECS = {**TGD2_PARAM_SPECS, **XAJ_PARAM_SPECS}
SIMHYD_TGD2_PARAM_SPECS = {**TGD2_PARAM_SPECS, **SIMHYD_PARAM_SPECS}

# Standalone structure-diagnosis process bounds.  These are intentionally not
# added to an existing XAJ configuration: the host composition and registries
# remain unchanged until the scientific variants are reviewed.  Positive
# residence times use the same log-coordinate convention as TGD2 adapters.
EVAPORATION_GAMMA_PARAM_SPECS = {
    "gamma": {
        "lower": 0.2, "upper": 5.0, "default": 1.0,
        "unit": "-", "description": "Generic lower/deep evaporation stress exponent",
        "process": "structure_diagnosis_evaporation", "is_snow": False,
    },
}

# Dissertation-controlled native-response domain.  This changes only the
# admissible parameter domain of the new controlled reference; XAJ_PARAM_SPECS
# above remains the legacy/public contract used by historical experiments.
CONTROLLED_XAJ_CI_LOWER = 0.1
CONTROLLED_XAJ_CI_UPPER = 0.9
CONTROLLED_XAJ_CG_LOWER = 0.9
CONTROLLED_XAJ_CG_UPPER = 0.998
CONTROLLED_XAJ_RESPONSE_DOMAIN = {
    "ci": (CONTROLLED_XAJ_CI_LOWER, CONTROLLED_XAJ_CI_UPPER),
    "cg": (CONTROLLED_XAJ_CG_LOWER, CONTROLLED_XAJ_CG_UPPER),
}

# Dissertation controlled finite-response freeze.  This supersedes the
# previous endpoint-audit operational range; legacy XAJ keeps its own inclusive
# CI/CG specs above.  The exact C=1 legacy endpoint is excluded from the new
# controlled comparison domain.
NATIVE_XAJ_LATENT_Z0 = 3.1553493591016335
NATIVE_XAJ_TAU0_LOWER = -1.0 / math.log(CONTROLLED_XAJ_CI_LOWER)
NATIVE_XAJ_TAU0_UPPER = -1.0 / math.log(CONTROLLED_XAJ_CG_UPPER)
SUBSURFACE_TAU0_PARAM_SPECS = {
    "tau_0": {
        "lower": NATIVE_XAJ_TAU0_LOWER,
        "upper": NATIVE_XAJ_TAU0_UPPER,
        "default": 10.0,
        "unit": "day", "description": "Single subsurface response residence time",
        "process": "structure_diagnosis_subsurface_response", "is_snow": False,
    },
}

# Implementation choice pending Phase 0.  The log-symmetric interval makes
# beta=1 the normalized midpoint and includes both sublinear and superlinear
# recession without introducing a second generic parameter.
SUBSURFACE_BETA_PARAM_SPECS = {
    "beta": {
        "lower": 0.5, "upper": 2.0, "default": 1.0,
        "unit": "-", "description": "Generic subsurface response exponent",
        "process": "structure_diagnosis_subsurface_response", "is_snow": False,
    },
}

XAJ_CONTROLLED_N_PARAM_SPECS = {
    **XAJ_PARAM_SPECS,
    "xaj_ci": {
        **XAJ_PARAM_SPECS["xaj_ci"],
        "lower": CONTROLLED_XAJ_CI_LOWER,
        "upper": CONTROLLED_XAJ_CI_UPPER,
        "description": "Controlled finite-domain interflow recession constant",
    },
    "xaj_cg": {
        **XAJ_PARAM_SPECS["xaj_cg"],
        "lower": CONTROLLED_XAJ_CG_LOWER,
        "upper": CONTROLLED_XAJ_CG_UPPER,
        "description": "Controlled finite-domain groundwater recession constant",
    },
}
# Descriptive alias for callers constructing the common native N reference.
XAJ_CONTROLLED_PARAM_SPECS = XAJ_CONTROLLED_N_PARAM_SPECS

# Controlled XAJ variant specifications.  These are exported for direct
# forward/sensitivity tests only; they are deliberately not added to the
# training registries or existing experiment configurations in this phase.
XAJ_KSS_PARAM_SPEC = {
    "xaj_kss": {
        # Native KI+KG is in [0, 1.4], then _prepare_xaj_parameters maps every
        # sum >= 1 to 1-1e-5.  Hence the effective attainable KSS interval is
        # exactly [0, 1-1e-5].
        "lower": 0.0, "upper": 1.0 - 1e-5, "default": 0.5,
        "unit": "1/day", "description": "Effective total subsurface generation coefficient",
        "process": "structure_diagnosis_subsurface_response", "is_snow": False,
    },
}
XAJ_TAU0_PARAM_SPEC = {
    "xaj_tau0": {
        **SUBSURFACE_TAU0_PARAM_SPECS["tau_0"],
        "description": "Single controlled-XAJ subsurface response time",
    },
}
XAJ_BETA_PARAM_SPEC = {
    "xaj_beta": {
        **SUBSURFACE_BETA_PARAM_SPECS["beta"],
        "description": "Single controlled-XAJ subsurface response exponent",
    },
}
XAJ_GAMMA_PARAM_SPEC = {
    "xaj_gamma": {
        **EVAPORATION_GAMMA_PARAM_SPECS["gamma"],
        "description": "Single controlled-XAJ evaporation stress exponent",
    },
}

XAJ_DE_PARAM_SPECS = {
    **{name: spec for name, spec in XAJ_CONTROLLED_N_PARAM_SPECS.items() if name != "xaj_c"},
}
XAJ_GE_PARAM_SPECS = {**XAJ_DE_PARAM_SPECS, **XAJ_GAMMA_PARAM_SPEC}
XAJ_DR_PARAM_SPECS = {
    **{name: spec for name, spec in XAJ_CONTROLLED_N_PARAM_SPECS.items()
       if name not in {"xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg"}},
    **XAJ_KSS_PARAM_SPEC, **XAJ_TAU0_PARAM_SPEC,
}
XAJ_GR_PARAM_SPECS = {**XAJ_DR_PARAM_SPECS, **XAJ_BETA_PARAM_SPEC}
