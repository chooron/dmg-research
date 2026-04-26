"""Shared metadata and ordering rules for the figure2 suite."""

from __future__ import annotations

from collections.abc import Iterable


MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
MODEL_LABELS = {
    "deterministic": "δdtm",
    "mc_dropout": "δmcd",
    "distributional": "δdtb",
}

LOSS_ORDER = ["NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"]
LOSS_LABELS = {
    "NseBatchLoss": "NSE",
    "LogNseBatchLoss": "LogNSE",
    "HybridNseBatchLoss": "HybridNSE",
}
REFERENCE_LOSS = "HybridNseBatchLoss"

PARAMETER_ORDER = [
    "parFC",
    "parLP",
    "parBETA",
    "parPERC",
    "parK0",
    "parK1",
    "parK2",
    "parUZL",
    "route_a",
    "route_b",
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
]
PARAMETER_LABELS = {
    "parFC": "FC",
    "parLP": "LP",
    "parBETA": "BETA",
    "parPERC": "PERC",
    "parK0": "K0",
    "parK1": "K1",
    "parK2": "K2",
    "parUZL": "UZL",
    "route_a": "route_a",
    "route_b": "route_b",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
}
PARAMETER_FAMILIES = {
    "parFC": "storage_recharge",
    "parLP": "storage_recharge",
    "parBETA": "storage_recharge",
    "parPERC": "storage_recharge",
    "parK0": "routing_runoff",
    "parK1": "routing_runoff",
    "parK2": "routing_runoff",
    "parUZL": "routing_runoff",
    "route_a": "routing_runoff",
    "route_b": "routing_runoff",
    "parTT": "snow_cold",
    "parCFMAX": "snow_cold",
    "parCFR": "snow_cold",
    "parCWH": "snow_cold",
}
PARAMETER_FAMILY_LABELS = {
    "storage_recharge": "Storage / recharge",
    "routing_runoff": "Routing / runoff",
    "snow_cold": "Snow / cold-region",
}

ATTRIBUTE_ORDER = [
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "aridity",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "frac_snow",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "carbonate_rocks_frac",
    "geol_porosity",
    "geol_permeability",
]
FOCUS_ATTRIBUTES = ["aridity", "frac_snow", "slope_mean", "pet_mean", "soil_conductivity", "soil_depth_pelletier"]
FOCUS_PARAMETERS = ["parBETA", "parFC", "parPERC", "parUZL", "parCFR", "parCWH"]
GRADIENT_ATTRIBUTES = ["aridity", "frac_snow", "slope_mean"]

ATTRIBUTE_SHORT_LABELS = {
    "p_mean": "P",
    "pet_mean": "PET",
    "p_seasonality": "P seasonality",
    "aridity": "Aridity",
    "high_prec_freq": "High prec. freq.",
    "high_prec_dur": "High prec. dur.",
    "low_prec_freq": "Low prec. freq.",
    "low_prec_dur": "Low prec. dur.",
    "frac_snow": "Snow fraction",
    "elev_mean": "Elevation",
    "slope_mean": "Slope",
    "area_gages2": "Area",
    "frac_forest": "Forest fraction",
    "lai_max": "LAI max",
    "lai_diff": "LAI diff.",
    "gvf_max": "GVF max",
    "gvf_diff": "GVF diff.",
    "dom_land_cover_frac": "Dom. land cover frac.",
    "dom_land_cover": "Dom. land cover",
    "root_depth_50": "Root depth",
    "soil_depth_pelletier": "Soil depth",
    "soil_depth_statsgo": "Soil depth (STATSGO)",
    "soil_porosity": "Soil porosity",
    "soil_conductivity": "Soil cond.",
    "max_water_content": "Max water content",
    "sand_frac": "Sand fraction",
    "silt_frac": "Silt fraction",
    "clay_frac": "Clay fraction",
    "geol_1st_class": "Geol. class 1",
    "glim_1st_class_frac": "GLiM class 1 frac.",
    "geol_2nd_class": "Geol. class 2",
    "glim_2nd_class_frac": "GLiM class 2 frac.",
    "carbonate_rocks_frac": "Carbonate rocks",
    "geol_porosity": "Geol. porosity",
    "geol_permeability": "Geol. permeability",
}


def attribute_family(name: str) -> str:
    if name in {"p_mean", "pet_mean", "p_seasonality", "aridity", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"}:
        return "climate"
    if name == "frac_snow":
        return "snow"
    if name in {"elev_mean", "slope_mean", "area_gages2"}:
        return "topography"
    if name.startswith("lai") or name.startswith("gvf") or "forest" in name or "land_cover" in name:
        return "vegetation"
    if name.startswith("soil_") or name in {"root_depth_50", "max_water_content", "sand_frac", "silt_frac", "clay_frac"}:
        return "soil"
    if name.startswith("geol_") or name.startswith("glim_") or "carbonate" in name:
        return "geology"
    return "other"


def ordered_items(names: Iterable[str], canonical_order: list[str]) -> list[str]:
    seen = set(names)
    ordered = [name for name in canonical_order if name in seen]
    ordered.extend(sorted(name for name in seen if name not in ordered))
    return ordered


def ordered_parameters(names: Iterable[str]) -> list[str]:
    return ordered_items(names, PARAMETER_ORDER)


def ordered_attributes(names: Iterable[str]) -> list[str]:
    return ordered_items(names, ATTRIBUTE_ORDER)


def parameter_label(name: str) -> str:
    return PARAMETER_LABELS.get(name, name[3:] if isinstance(name, str) and name.startswith("par") else name)


def attribute_label(name: str) -> str:
    return ATTRIBUTE_SHORT_LABELS.get(name, name.replace("_", " "))


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def loss_label(name: str) -> str:
    return LOSS_LABELS.get(name, name)
