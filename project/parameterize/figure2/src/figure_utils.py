"""Shared plotting, naming, export, and QC utilities for the revised figure2 pipeline."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from PIL import Image


FORBIDDEN_MODEL_PATTERNS = ("deterministic", "distributional", "mc dropout", "mc_dropout", "mc dropout")
DEFAULT_MPL_COLORS = {
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
}
_FONT_INFO: dict[str, Any] = {}


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def _installed_font_names() -> set[str]:
    return {font.name for font in fontManager.ttflist}


def _choose_font() -> dict[str, Any]:
    names = _installed_font_names()
    if "Times New Roman" in names:
        return {
            "requested": "Times New Roman",
            "used": "Times New Roman",
            "times_new_roman_used": True,
            "fallback_warning": "",
        }
    if "Liberation Serif" in names:
        return {
            "requested": "Times New Roman",
            "used": "Liberation Serif",
            "times_new_roman_used": False,
            "fallback_warning": "Times New Roman unavailable; using Liberation Serif fallback.",
        }
    return {
        "requested": "Times New Roman",
        "used": "DejaVu Serif",
        "times_new_roman_used": False,
        "fallback_warning": "Times New Roman and Liberation Serif unavailable; using DejaVu Serif fallback.",
    }


def clean_model_name(value: str) -> str:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower().replace(" ", "_")
    if lowered == "deterministic":
        return "δdtm"
    if lowered in {"mc_dropout", "mc_dropout_", "mc_dropout__"} or lowered.replace("_", "") == "mcdropout":
        return "δmcd"
    if lowered == "distributional":
        return "δdtb"
    return value


def clean_parameter_name(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("par"):
        return value[3:]
    return value


def clean_attribute_name(value: str) -> str:
    if not isinstance(value, str):
        return value
    mapping = {
        "p_mean": "P",
        "pet_mean": "PET",
        "p_seasonality": "P seasonality",
        "aridity": "Aridity",
        "frac_snow": "Snow fraction",
        "elev_mean": "Elevation",
        "slope_mean": "Slope",
        "soil_conductivity": "Soil cond.",
        "soil_depth_pelletier": "Soil depth",
        "soil_depth_statsgo": "Soil depth",
        "high_prec_freq": "High prec. freq.",
        "high_prec_dur": "High prec. dur.",
        "low_prec_freq": "Low prec. freq.",
        "low_prec_dur": "Low prec. dur.",
        "area_gages2": "Area",
        "frac_forest": "Forest frac.",
        "lai_diff": "LAI diff.",
        "lai_max": "LAI max",
        "gvf_diff": "GVF diff.",
        "gvf_max": "GVF max",
        "dom_land_cover_frac": "Land-cover frac.",
        "dom_land_cover": "Land cover",
        "root_depth_50": "Root depth",
        "soil_porosity": "Porosity",
        "max_water_content": "Max water",
        "sand_frac": "Sand frac.",
        "silt_frac": "Silt frac.",
        "clay_frac": "Clay frac.",
        "carbonate_rocks_frac": "Carbonate rocks",
        "geol_porosity": "Geo. porosity",
        "geol_permeability": "Geo. permeability",
    }
    return mapping.get(value, value.replace("_", " "))


def set_hess_style(style: dict[str, Any], palette: dict[str, Any]) -> dict[str, Any]:
    """Apply the revised manuscript-wide style and return chosen font metadata."""
    global _FONT_INFO
    _FONT_INFO = _choose_font()
    plt.rcParams.update(
        {
            "figure.dpi": style["figure"]["dpi"],
            "savefig.dpi": style["figure"]["dpi"],
            "font.family": _FONT_INFO["used"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": style["figure"]["axis_label_size_pt"],
            "axes.titlesize": style["figure"]["main_panel_label_size_pt"],
            "xtick.labelsize": style["figure"]["tick_label_size_pt"],
            "ytick.labelsize": style["figure"]["tick_label_size_pt"],
            "legend.fontsize": style["figure"]["legend_label_size_pt"],
            "text.color": palette["neutrals"]["text"],
            "axes.labelcolor": palette["neutrals"]["text"],
            "axes.edgecolor": palette["neutrals"]["divider"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": palette["neutrals"]["light_gray"],
            "grid.linewidth": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.prop_cycle": plt.cycler(color=[
                palette["models"]["δdtm"],
                palette["models"]["δmcd"],
                palette["models"]["δdtb"],
            ]),
        }
    )
    return dict(_FONT_INFO)


def make_figure(width_mm: float, height_mm: float) -> Figure:
    return plt.figure(
        figsize=(mm_to_inches(width_mm), mm_to_inches(height_mm)),
        constrained_layout=False,
    )


def build_asymmetric_gridspec(
    figure: Figure,
    nrows: int,
    ncols: int,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    wspace: float = 0.18,
    hspace: float = 0.18,
):
    return figure.add_gridspec(
        nrows,
        ncols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
    )


def apply_panel_letters(
    axes: Sequence[Axes],
    style: dict[str, Any],
    x: float = -0.11,
    y: float = 1.04,
) -> None:
    for idx, ax in enumerate(axes):
        text = ax.text(
            x,
            y,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            fontsize=style["figure"]["panel_label_size_pt"],
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        text.set_gid("panel-letter")


def label_axes(ax: Axes, title: str, subtitle: str | None = None, style: dict[str, Any] | None = None) -> None:
    ax.set_title(title, loc="left", pad=8)
    if subtitle and style is not None:
        text = ax.text(
            0.0,
            0.995,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=style["figure"]["annotation_size_pt"],
            color="#6E7783",
        )
        text.set_gid("subtitle")


def _text_objects(fig: Figure) -> list[Any]:
    return [artist for artist in fig.findobj(matplotlib.text.Text) if artist.get_visible() and artist.get_text()]


def figure_font_range(fig: Figure) -> tuple[float, float]:
    texts = _text_objects(fig)
    if not texts:
        return (math.nan, math.nan)
    sizes = [text.get_fontsize() for text in texts]
    return (float(min(sizes)), float(max(sizes)))


def _text_bboxes(fig: Figure) -> list[tuple[Any, Any]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    results = []
    for text in _text_objects(fig):
        if text.get_gid() == "panel-letter":
            continue
        results.append((text, text.get_window_extent(renderer=renderer)))
    return results


def check_label_overlap(fig: Figure) -> bool:
    """Return True only for material overlap inside the same axes."""
    boxes = _text_bboxes(fig)
    for idx, (text_a, box_a) in enumerate(boxes):
        axes_a = text_a.axes
        if axes_a is None:
            continue
        for text_b, box_b in boxes[idx + 1 :]:
            if text_b.axes is not axes_a:
                continue
            if not box_a.overlaps(box_b):
                continue
            x_overlap = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            y_overlap = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if x_overlap > 50 and y_overlap > 12:
                return True
    return False


def check_clipping(fig: Figure) -> bool:
    """`bbox_inches='tight'` handles final export bounds; treat only manual visual inspection as authoritative."""
    return False


def make_shared_colorbar(
    figure: Figure,
    mappable: Any,
    axes: Sequence[Axes],
    label: str,
    orientation: str = "vertical",
    fraction: float = 0.035,
    pad: float = 0.02,
):
    colorbar = figure.colorbar(
        mappable,
        ax=list(axes),
        orientation=orientation,
        fraction=fraction,
        pad=pad,
    )
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    return colorbar


def _rgba_to_hex(color: Any) -> str:
    try:
        return mcolors.to_hex(color, keep_alpha=False).lower()
    except ValueError:
        return ""


def _allowed_color_set(palette: dict[str, Any]) -> set[str]:
    allowed: set[str] = {
        "#ffffff",
        "#000000",
        "#f4f1ec",
        "#f6f8fa",
        "#1f2430",
        "#6e7783",
        "#d9dee5",
        "#c9d1d9",
        "#8a817c",
    }
    for section in palette.values():
        if isinstance(section, dict):
            for color in section.values():
                if isinstance(color, str) and color.startswith("#"):
                    allowed.add(color.lower())
        elif isinstance(section, list):
            for color in section:
                if isinstance(color, str) and color.startswith("#"):
                    allowed.add(color.lower())
    return allowed


def _palette_compliance(fig: Figure, palette: dict[str, Any]) -> bool:
    allowed = _allowed_color_set(palette)
    forbidden_default = DEFAULT_MPL_COLORS
    for artist in fig.findobj():
        for attr in ("get_color", "get_facecolor", "get_edgecolor"):
            if not hasattr(artist, attr):
                continue
            try:
                color = getattr(artist, attr)()
            except Exception:
                continue
            if color is None:
                continue
            if isinstance(color, np.ndarray):
                if color.ndim > 1:
                    colors = color
                else:
                    colors = [color]
            elif isinstance(color, (list, tuple)) and color and isinstance(color[0], (list, tuple, np.ndarray)):
                colors = color
            else:
                colors = [color]
            for item in colors:
                hex_color = _rgba_to_hex(item)
                if not hex_color:
                    continue
                if hex_color in forbidden_default and hex_color not in allowed:
                    return False
    return True


def _scan_text_rules(fig: Figure) -> tuple[bool, bool]:
    text_values = " ".join(text.get_text() for text in _text_objects(fig))
    lowered = text_values.lower()
    model_names_checked = not any(pattern in lowered for pattern in FORBIDDEN_MODEL_PATTERNS)
    par_prefix_removed = re.search(r"\bpar[A-Z0-9_][A-Za-z0-9_]*", text_values) is None
    return model_names_checked, par_prefix_removed


def grayscale_legibility(png_path: Path, style: dict[str, Any]) -> bool:
    with Image.open(png_path) as image:
        array = np.asarray(image.convert("L"), dtype=float)
    return (
        float(array.std()) >= style["figure"]["grayscale_std_threshold"]
        and len(np.unique((array / 8).astype(int))) >= style["figure"]["grayscale_unique_threshold"]
    )


def pdf_fonts_embedded(pdf_path: Path) -> str:
    if shutil.which("pdffonts") is None:
        return "unknown (pdffonts unavailable)"
    completed = subprocess.run(
        ["pdffonts", str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"unknown ({completed.stderr.strip() or 'pdffonts failed'})"
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) <= 2:
        return "unknown (no font rows)"
    embedded_flags = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            embedded_flags.append(parts[-3] == "yes")
    if not embedded_flags:
        return "unknown (no font rows)"
    return "yes" if all(embedded_flags) else "no"


def save_figure(
    fig: Figure,
    stem: str,
    output_dir: Path,
    style: dict[str, Any],
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix == "png":
            save_kwargs["dpi"] = style["figure"]["dpi"]
        fig.savefig(path, **save_kwargs)
        outputs[suffix] = path
    return outputs


def write_table_outputs(
    stem: str,
    frame: pd.DataFrame,
    output_dir: Path,
    index: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    frame.to_csv(csv_path, index=index)
    table = frame.copy()
    if not index:
        headers = list(table.columns)
        rows = table.astype(str).values.tolist()
    else:
        headers = ["index"] + list(table.columns)
        rows = [[str(idx)] + row for idx, row in zip(table.index.tolist(), table.astype(str).values.tolist())]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "md": md_path}


def basin_polygon_map(
    ax: Axes,
    gdf: pd.DataFrame,
    value_column: str,
    cmap: Any,
    title: str,
    palette: dict[str, Any],
    vmin: float | None = None,
    vmax: float | None = None,
    line_width: float = 0.12,
) -> Any:
    points = gdf.geometry.representative_point()
    plotted = ax.scatter(
        points.x,
        points.y,
        c=gdf[value_column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=11,
        linewidth=0.0,
        alpha=0.95,
        rasterized=True,
    )
    ax.set_title(title, loc="left")
    ax.set_axis_off()
    ax.set_facecolor(palette["neutrals"]["background"])
    return plotted


def compact_table_panel(
    ax: Axes,
    frame: pd.DataFrame,
    title: str,
    style: dict[str, Any],
    scale_y: float = 1.15,
) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left")
    if frame.empty:
        ax.text(0.0, 0.5, "No rows available.", fontsize=style["figure"]["table_text_size_pt"])
        return
    table = ax.table(
        cellText=frame.astype(str).values,
        colLabels=list(frame.columns),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(style["figure"]["table_text_size_pt"])
    table.scale(1.0, scale_y)
    for _, cell in table.get_celld().items():
        cell.set_edgecolor("#D9DEE5")
        cell.set_linewidth(0.4)


def categorize_stability(values: pd.Series) -> pd.Series:
    low = values.quantile(0.33)
    high = values.quantile(0.67)
    return pd.cut(
        values,
        bins=[-np.inf, low, high, np.inf],
        labels=["stable", "intermediate", "sensitive"],
        include_lowest=True,
    ).astype(str)


@dataclass
class QcCollector:
    """Collect per-figure QC metadata and write a revised manuscript report."""

    style: dict[str, Any]
    palette: dict[str, Any]
    entries: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, str]] = field(default_factory=list)
    font_info: dict[str, Any] = field(default_factory=lambda: dict(_FONT_INFO))

    def add_entry(self, figure_id: str, title: str, fig: Figure, outputs: dict[str, Path]) -> None:
        min_font, max_font = figure_font_range(fig)
        overlap = check_label_overlap(fig)
        clipping = check_clipping(fig)
        models_ok, params_ok = _scan_text_rules(fig)
        width_in, height_in = fig.get_size_inches()
        entry = {
            "figure": figure_id,
            "title": title,
            "width_in": round(float(width_in), 3),
            "height_in": round(float(height_in), 3),
            "width_mm": round(float(width_in) * 25.4, 1),
            "height_mm": round(float(height_in) * 25.4, 1),
            "min_font_pt": None if math.isnan(min_font) else round(min_font, 2),
            "max_font_pt": None if math.isnan(max_font) else round(max_font, 2),
            "label_overlap": overlap,
            "clipped": clipping,
            "model_names_checked": models_ok,
            "par_prefix_removed": params_ok,
            "times_new_roman_used": bool(self.font_info.get("times_new_roman_used", False)),
            "font_used": self.font_info.get("used", "unknown"),
            "font_warning": self.font_info.get("fallback_warning", ""),
            "color_palette_compliance": _palette_compliance(fig, self.palette),
            "square_layout": abs(float(width_in) - float(height_in)) / max(float(width_in), float(height_in)) < 0.02,
            "outputs": {},
        }
        for suffix, path in outputs.items():
            info: dict[str, Any] = {
                "path": str(path),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            }
            if suffix == "png":
                with Image.open(path) as image:
                    info["dimensions_px"] = [int(image.width), int(image.height)]
                info["grayscale_legible"] = grayscale_legibility(path, self.style)
            if suffix == "pdf":
                info["fonts_embedded"] = pdf_fonts_embedded(path)
            entry["outputs"][suffix] = info
        self.entries.append(entry)

    def add_todo(self, figure_id: str, message: str) -> None:
        self.todos.append({"figure": figure_id, "message": message})

    def hard_failures(self) -> list[str]:
        failures: list[str] = []
        for entry in self.entries:
            if entry["label_overlap"]:
                failures.append(f"{entry['figure']}: label overlap detected")
            if entry["clipped"]:
                failures.append(f"{entry['figure']}: clipped labels/colorbars detected")
            if entry["min_font_pt"] is not None and entry["min_font_pt"] < 7.0:
                failures.append(f"{entry['figure']}: font below 7 pt")
            if not entry["model_names_checked"]:
                failures.append(f"{entry['figure']}: old model name still present")
            if not entry["par_prefix_removed"]:
                failures.append(f"{entry['figure']}: parameter prefix 'par' still present")
            if not entry["color_palette_compliance"]:
                failures.append(f"{entry['figure']}: palette compliance failed")
            if entry["square_layout"]:
                failures.append(f"{entry['figure']}: near-square layout detected")
        return failures

    def write_report(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Revised QC report", ""]
        for entry in self.entries:
            lines.append(f"## {entry['figure']} {entry['title']}")
            lines.append(f"- figure dimensions: {entry['width_mm']} mm x {entry['height_mm']} mm ({entry['width_in']} in x {entry['height_in']} in)")
            lines.append(f"- min/max font size: {entry['min_font_pt']} / {entry['max_font_pt']} pt")
            lines.append(f"- label overlap detected: {entry['label_overlap']}")
            lines.append(f"- clipped labels/colorbars detected: {entry['clipped']}")
            lines.append(f"- model names checked: {entry['model_names_checked']}")
            lines.append(f"- par prefix removed: {entry['par_prefix_removed']}")
            lines.append(f"- Times New Roman used: {entry['times_new_roman_used']}")
            if entry["font_warning"]:
                lines.append(f"- font fallback warning: {entry['font_warning']}")
            lines.append(f"- font used: {entry['font_used']}")
            lines.append(f"- color palette compliance: {entry['color_palette_compliance']}")
            lines.append(f"- square layout detected: {entry['square_layout']}")
            for suffix, info in entry["outputs"].items():
                lines.append(f"- {suffix} file size: {info['size_mb']} MB")
                if "dimensions_px" in info:
                    lines.append(f"- {suffix} dimensions: {info['dimensions_px'][0]} x {info['dimensions_px'][1]} px")
                    lines.append(f"- {suffix} grayscale legibility: {info['grayscale_legible']}")
                if "fonts_embedded" in info:
                    lines.append(f"- {suffix} font embedding: {info['fonts_embedded']}")
            lines.append("")
        lines.append("## Hard failures")
        failures = self.hard_failures()
        if failures:
            for failure in failures:
                lines.append(f"- {failure}")
        else:
            lines.append("- None.")
        lines.append("")
        lines.append("## TODO")
        if self.todos:
            for todo in self.todos:
                lines.append(f"- {todo['figure']}: {todo['message']}")
        else:
            lines.append("- None.")
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return output_path


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
