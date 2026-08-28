#!/usr/bin/env python3
"""Build final Table S1 from production parameter specifications only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from models import parameter_specs as specs  # noqa: E402

HOSTS = {
    "XAJ": specs.XAJ_PARAM_SPECS,
    "GR4J": specs.GR4J_PARAM_SPECS,
    "SIMHYD": specs.SIMHYD_PARAM_SPECS,
}
HOST_LABELS = {"XAJ": "XAJ", "GR4J": "GR4J", "SIMHYD": "SIMHYD"}
MODULES = {
    "TGD": specs.TGD2_PARAM_SPECS,
    "CN": specs.CEMANEIGE_PARAM_SPECS,
}
SYMBOLS = {
    "tgd_tau_warm": r"\tau_{\mathrm{warm}}",
    "tgd_delta_tau_cold": r"\Delta\tau_{\mathrm{cold}}",
    "cn_ctg": r"C_{\mathrm{TG}}",
    "cn_kf": r"K_f",
}
FIXED_ROWS = [
    {
        "Host": "All hosts",
        "Parameter": "T_ref",
        "Symbol": r"T_{\mathrm{ref}}",
        "Definition": "Fixed TGD gate reference temperature",
        "Lower bound": "—",
        "Upper bound": "—",
        "Unit": "°C",
        "Base membership": "No",
        "TGD membership": "Yes",
        "CN membership": "No",
        "Calibrated / fixed": "Fixed (0.0)",
        "Default / fixed value": 0.0,
        "Process": "temperature gate",
    },
    {
        "Host": "All hosts",
        "Parameter": "s_T",
        "Symbol": r"s_T",
        "Definition": "Fixed TGD gate temperature scale",
        "Lower bound": "—",
        "Upper bound": "—",
        "Unit": "°C",
        "Base membership": "No",
        "TGD membership": "Yes",
        "CN membership": "No",
        "Calibrated / fixed": "Fixed (2.0)",
        "Default / fixed value": 2.0,
        "Process": "temperature gate",
    },
    {
        "Host": "All hosts",
        "Parameter": "epsilon",
        "Symbol": r"\varepsilon",
        "Definition": "Numerical lower bound used in TGD residence-time calculations",
        "Lower bound": "—",
        "Upper bound": "—",
        "Unit": "d",
        "Base membership": "No",
        "TGD membership": "Yes",
        "CN membership": "No",
        "Calibrated / fixed": "Fixed (1e-6)",
        "Default / fixed value": 1e-6,
        "Process": "numerical safeguard",
    },
    {
        "Host": "All hosts",
        "Parameter": "g_thresh",
        "Symbol": r"g_{\mathrm{thresh}}",
        "Definition": "Fixed CN snow-cover threshold, 0.9 times annual solid precipitation",
        "Lower bound": "—",
        "Upper bound": "—",
        "Unit": "mm",
        "Base membership": "No",
        "TGD membership": "No",
        "CN membership": "Yes",
        "Calibrated / fixed": "Fixed (0.9 P_sol^annual)",
        "Default / fixed value": "0.9 P_sol^annual",
        "Process": "snow cover",
    },
    {
        "Host": "All hosts",
        "Parameter": "G_0",
        "Symbol": r"G_0",
        "Definition": "Initial CN snow-water-equivalent state",
        "Lower bound": "—",
        "Upper bound": "—",
        "Unit": "mm",
        "Base membership": "No",
        "TGD membership": "No",
        "CN membership": "Yes",
        "Calibrated / fixed": "Fixed (0)",
        "Default / fixed value": 0.0,
        "Process": "snow state",
    },
    {
        "Host": "All hosts",
        "Parameter": "eTG_0",
        "Symbol": r"eTG_0",
        "Definition": "Initial CN snowpack thermal state",
        "Lower bound": "—",
        "Upper bound": "—",
        "Unit": "°C",
        "Base membership": "No",
        "TGD membership": "No",
        "CN membership": "Yes",
        "Calibrated / fixed": "Fixed (0)",
        "Default / fixed value": 0.0,
        "Process": "snow state",
    },
]


def symbol_for(name: str) -> str:
    if name in SYMBOLS:
        return SYMBOLS[name]
    if name.startswith("xaj_"):
        return name[4:]
    if name.startswith("gr4j_"):
        return name[5:]
    if name.startswith("simhyd_"):
        return name[7:]
    return name


def rows_for_host(host: str, host_specs: dict) -> list[dict]:
    all_names = list(host_specs) + [n for n in specs.TGD2_PARAM_SPECS if n not in host_specs] + [n for n in specs.CEMANEIGE_PARAM_SPECS if n not in host_specs]
    rows = []
    for name in all_names:
        if name in host_specs:
            spec = host_specs[name]
        elif name in specs.TGD2_PARAM_SPECS:
            spec = specs.TGD2_PARAM_SPECS[name]
        else:
            spec = specs.CEMANEIGE_PARAM_SPECS[name]
        is_tgd = name in specs.TGD2_PARAM_SPECS
        is_cn = name in specs.CEMANEIGE_PARAM_SPECS
        rows.append(
            {
                "Host": host,
                "Parameter": name,
                "Symbol": symbol_for(name),
                "Definition": spec.get("description", ""),
                "Lower bound": spec.get("lower", "—"),
                "Upper bound": spec.get("upper", "—"),
                "Unit": spec.get("unit", "—"),
                "Base membership": "Yes" if not is_tgd and not is_cn else "No",
                "TGD membership": "Yes" if not is_cn else "No",
                "CN membership": "Yes" if not is_tgd else "No",
                "Calibrated / fixed": "Calibrated",
                "Default / fixed value": spec.get("default", "—"),
                "Process": spec.get("process", ""),
            }
        )
    return rows


def build() -> pd.DataFrame:
    rows = []
    for host, host_specs in HOSTS.items():
        rows.extend(rows_for_host(host, host_specs))
    rows.extend(FIXED_ROWS)
    return pd.DataFrame(rows)


def simple_markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))
    return "\n".join(lines)


def markdown(df: pd.DataFrame) -> str:
    counts = []
    for host, host_specs in HOSTS.items():
        counts.append((host, len(host_specs), len(host_specs) + len(MODULES["TGD"]), len(host_specs) + len(MODULES["CN"])))
    out = [
        "# Table S1 — Calibrated parameter definitions, bounds, units, and structural membership",
        "",
        "The machine-readable table contains the production parameter specifications for the three host families, plus fixed TGD/CN constants that are not counted as calibrated parameters.",
        "",
        "## Calibrated parameter counts",
        "",
        "| Host | Base | TGD | CN |",
        "|---|---:|---:|---:|",
    ]
    out.extend(f"| {h} | {b} | {t} | {c} |" for h, b, t, c in counts)
    out.extend([
        "",
        "TGD `T_ref`, `s_T`, and numerical `epsilon` are fixed constants. CN `g_thresh`, `G_0`, and `eTG_0` are fixed implementation constants; they are listed for traceability and are not included in the calibrated counts.",
        "",
        "## Parameter rows",
        "",
        simple_markdown_table(df),
        "",
    ])
    return "\n".join(out)


if __name__ == "__main__":
    df = build()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "Table_S1.csv", index=False)
    (OUT / "Table_S1.md").write_text(markdown(df), encoding="utf-8")
    print(f"wrote {len(df)} Table S1 rows")
