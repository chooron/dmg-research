"""Data loading and registry utilities for the figure2 pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml

from project.parameterize.figure2.src.metadata import (
    ATTRIBUTE_ORDER,
    LOSS_ORDER,
    MODEL_ORDER,
    REFERENCE_LOSS,
    attribute_family,
    ordered_attributes,
    ordered_parameters,
)


class MissingFigureDataError(RuntimeError):
    """Raised when a required figure input is missing."""


@dataclass
class FigureDataRegistry:
    """Single source of truth for all figure2 inputs."""

    config_path: Path
    analysis_root: Path
    style: dict[str, Any]
    palette: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    subset_basin_ids: list[int] = field(default_factory=list)
    spatial: gpd.GeoDataFrame | None = None
    reference_loss: str = REFERENCE_LOSS
    model_order: list[str] = field(default_factory=lambda: list(MODEL_ORDER))
    loss_order: list[str] = field(default_factory=lambda: list(LOSS_ORDER))
    parameter_order: list[str] = field(default_factory=list)
    attribute_order: list[str] = field(default_factory=list)
    focus_parameters: list[str] = field(default_factory=list)
    focus_attributes: list[str] = field(default_factory=list)
    project_root: Path = field(default_factory=Path)

    def table(self, name: str) -> pd.DataFrame:
        frame = self.tables.get(name)
        if frame is None:
            raise MissingFigureDataError(f"Missing table '{name}'.")
        return frame.copy()

    def optional_table(self, name: str) -> pd.DataFrame | None:
        frame = self.tables.get(name)
        return None if frame is None else frame.copy()

    def require_columns(self, table_name: str, columns: list[str]) -> pd.DataFrame:
        frame = self.table(table_name)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise MissingFigureDataError(
                f"Table '{table_name}' is missing columns: {', '.join(missing)}."
            )
        return frame

    def filter_reference(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "loss" not in frame.columns:
            return frame.copy()
        return frame.loc[frame["loss"] == self.reference_loss].copy()

    def focus_corr_table(self, value_column: str) -> pd.DataFrame:
        frame = self.require_columns(
            "correlation_mean_std_var",
            ["method", "model", "loss", "parameter", "attribute", value_column],
        )
        frame = frame.loc[(frame["method"] == "spearman") & (frame["loss"] == self.reference_loss)].copy()
        frame = frame.loc[
            frame["parameter"].isin(self.focus_parameters) & frame["attribute"].isin(self.focus_attributes)
        ].copy()
        return frame

    @classmethod
    def from_paths(
        cls,
        config_path: str,
        analysis_root: str | None = None,
    ) -> "FigureDataRegistry":
        config_file = Path(config_path).resolve()
        project_root = config_file.parents[3]
        figure_root = config_file.parents[1] / "figure2"
        style = yaml.safe_load((figure_root / "config" / "figure_style.yaml").read_text(encoding="utf-8"))
        palette = yaml.safe_load((figure_root / "config" / "figure_palette.yaml").read_text(encoding="utf-8"))

        if analysis_root is None:
            resolved_analysis_root = project_root / "project" / "parameterize" / "outputs" / "analysis" / "stability_stats"
        else:
            resolved_analysis_root = Path(analysis_root).resolve()

        registry = cls(
            config_path=config_file,
            analysis_root=resolved_analysis_root,
            style=style,
            palette=palette,
            project_root=project_root,
        )
        registry._load_tables()
        registry._load_spatial_data()
        registry._finalize_metadata()
        return registry

    def _load_tables(self) -> None:
        table_paths = {
            "metrics_long": self.analysis_root / "tables" / "metrics_long.csv",
            "basin_metrics_long": self.analysis_root / "tables" / "basin_metrics_long.csv",
            "metrics_by_model": self.analysis_root / "tables" / "metrics_by_model.csv",
            "metrics_by_model_loss": self.analysis_root / "tables" / "metrics_by_model_loss.csv",
            "params_long": self.analysis_root / "tables" / "params_long.csv",
            "basin_attributes": self.analysis_root / "tables" / "basin_attributes.csv",
            "relationship_classes": self.analysis_root / "correlation_summaries" / "relationship_classes.csv",
            "pair_seed_stability": self.analysis_root / "correlation_summaries" / "pair_seed_stability.csv",
            "pair_loss_stability": self.analysis_root / "correlation_summaries" / "pair_loss_stability.csv",
            "correlation_mean_std_var": self.analysis_root / "correlation_summaries" / "correlation_mean_std_var.csv",
            "correlation_seed_stability_summary": self.analysis_root / "correlation_summaries" / "correlation_seed_stability_summary.csv",
            "correlation_loss_stability_summary": self.analysis_root / "correlation_summaries" / "correlation_loss_stability_summary.csv",
            "core_relationships_summary": self.analysis_root / "correlation_summaries" / "core_relationships_summary.csv",
            "results331_dominant_attribute_summary": self.analysis_root / "correlation_summaries" / "results331_dominant_attribute_summary.csv",
            "results332_matrix_similarity": self.analysis_root / "correlation_summaries" / "results332_matrix_similarity.csv",
            "results332_matrix_embedding": self.analysis_root / "correlation_summaries" / "results332_matrix_embedding.csv",
            "results333_parameter_feature_importance": self.analysis_root / "correlation_summaries" / "results333_parameter_feature_importance.csv",
            "results333_importance_alignment": self.analysis_root / "correlation_summaries" / "results333_importance_alignment.csv",
            "results333_importance_overlap": self.analysis_root / "correlation_summaries" / "results333_importance_overlap.csv",
            "results341_distributional_mean_relationships": self.analysis_root / "correlation_summaries" / "results341_distributional_mean_relationships.csv",
            "results341_gradient_group_stats": self.analysis_root / "correlation_summaries" / "results341_gradient_group_stats.csv",
            "results342_distributional_std_relationships": self.analysis_root / "correlation_summaries" / "results342_distributional_std_relationships.csv",
            "results342_distributional_vs_dropout_std_compare": self.analysis_root / "correlation_summaries" / "results342_distributional_vs_dropout_std_compare.csv",
            "results342_gradient_std_group_stats": self.analysis_root / "correlation_summaries" / "results342_gradient_std_group_stats.csv",
            "results342_std_plot_data": self.analysis_root / "correlation_summaries" / "results342_std_plot_data.csv",
            "results343_basin_group_summary": self.analysis_root / "correlation_summaries" / "results343_basin_group_summary.csv",
            "results343_representative_basins": self.analysis_root / "correlation_summaries" / "results343_representative_basins.csv",
            "finalcheck_uncertainty_interpretation_flags": self.analysis_root / "correlation_summaries" / "finalcheck_uncertainty_interpretation_flags.csv",
            "finalcheck_mean_std_coupling": self.analysis_root / "correlation_summaries" / "finalcheck_mean_std_coupling.csv",
            "seed_parameter_variance_by_model": self.analysis_root / "parameter_variance" / "seed_parameter_variance_by_model.csv",
            "seed_parameter_variance_by_parameter": self.analysis_root / "parameter_variance" / "seed_parameter_variance_by_parameter.csv",
            "cross_loss_parameter_variance_by_model": self.analysis_root / "parameter_variance" / "cross_loss_parameter_variance_by_model.csv",
            "cross_loss_parameter_variance_by_parameter": self.analysis_root / "parameter_variance" / "cross_loss_parameter_variance_by_parameter.csv",
            "results332_spearman_matrices": self.analysis_root / "correlation_summaries" / "results332_spearman_matrices.parquet",
        }
        for name, path in table_paths.items():
            if path.exists():
                if path.suffix == ".parquet":
                    self.tables[name] = pd.read_parquet(path)
                else:
                    self.tables[name] = pd.read_csv(path)
            else:
                self.tables[name] = None
                self.missing_inputs.append(f"{name}: {path}")

        subset_path = self.project_root / "data" / "531sub_id.txt"
        basin_ids = ast.literal_eval(subset_path.read_text(encoding="utf-8"))
        self.subset_basin_ids = [int(value) for value in basin_ids]

    def _load_spatial_data(self) -> None:
        shape_path = self.project_root / "data" / "camels_loc" / "camels_671_loc.shp"
        if not shape_path.exists():
            self.missing_inputs.append(f"spatial shapefile: {shape_path}")
            self.spatial = None
            return

        spatial = gpd.read_file(shape_path)
        spatial["gage_id"] = spatial["gage_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
        spatial["gage_id_int"] = spatial["gage_id"].astype(int)
        subset = set(self.subset_basin_ids)
        spatial = spatial.loc[spatial["gage_id_int"].isin(subset)].copy()
        if len(spatial) != len(subset):
            raise MissingFigureDataError(
                f"Spatial join mismatch: expected {len(subset)} basins, found {len(spatial)}."
            )
        spatial["basin_id"] = spatial["gage_id_int"]
        spatial["attribute_family"] = spatial["gage_id"].map(lambda _: "other")
        self.spatial = spatial

    def _finalize_metadata(self) -> None:
        params = self.table("params_long")
        attributes = self.table("basin_attributes")
        self.parameter_order = ordered_parameters(params["parameter"].dropna().unique().tolist())
        self.attribute_order = ordered_attributes([column for column in attributes.columns if column != "basin_id"])
        self.focus_parameters = [name for name in ["parBETA", "parFC", "parPERC", "parUZL", "parCFR", "parCWH"] if name in self.parameter_order]
        self.focus_attributes = [name for name in ["aridity", "frac_snow", "slope_mean", "pet_mean", "soil_conductivity", "soil_depth_pelletier"] if name in self.attribute_order]

    def attribute_family_lookup(self) -> dict[str, str]:
        return {name: attribute_family(name) for name in self.attribute_order}
