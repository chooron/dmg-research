"""
Catchment Attribute Loader and Builder for Differentiable Parameter Learning (dPL).
Constructs normalized physical attribute matrices for CAMELS 531 basins.
"""
from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import torch

# Standard 35 CAMELS Catchment Physical Attributes
CAMELS_35_ATTRIBUTES = [
    # Climate Characteristics (9)
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    # Topography & Geometry (3)
    "elev_mean", "slope_mean", "area_gages2",
    # Vegetation & Land Cover (7)
    "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    "dom_land_cover_frac", "dom_land_cover",
    # Soil Properties (9)
    "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo",
    "soil_porosity", "soil_conductivity", "max_water_content",
    "sand_frac", "silt_frac", "clay_frac",
    # Geological Characteristics (7)
    "geol_1st_class", "glim_1st_class_frac", "geol_2nd_class",
    "glim_2nd_class_frac", "carbonate_rocks_frac", "geol_porosity", "geol_permeability"
]


class CatchmentAttributeBuilder:
    """Loads and normalizes physical catchment attributes for 531 basins."""

    def __init__(self, data_root: Path | str | None = None):
        if data_root is None:
            # Try repo root data dir first, then fallback to relative data dir
            candidate_1 = Path(__file__).resolve().parents[3] / "data"
            candidate_2 = Path(__file__).resolve().parents[2] / "data"
            data_root = candidate_1 if candidate_1.exists() else candidate_2
        self.data_root = Path(data_root)

    def load_raw_attributes(self, basin_ids: np.ndarray) -> np.ndarray:
        """Load physical attributes for specified basin IDs from Caravan npy array or dataset pickle."""
        caravan_npy = self.data_root / "caravan_671_attributes.npy"
        if caravan_npy.is_file():
            attributes = np.load(caravan_npy)
        else:
            dataset_path = self.data_root / "camels_dataset"
            if not dataset_path.is_file():
                raise FileNotFoundError(f"CAMELS dataset pickle not found at {dataset_path}")
            with open(dataset_path, "rb") as f:
                _, _, attributes = pickle.load(f)

        reference_ids = np.load(self.data_root / "gage_id.npy")
        indices = np.array([np.where(reference_ids == b)[0][0] for b in basin_ids])
        return attributes[indices]

    def build_normalized_attributes(
        self,
        basin_ids: np.ndarray,
        device: str = "cuda",
        method: str = "zscore",
        log_transform_skewed: bool = True,
    ) -> torch.Tensor:
        """
        Extract and normalize 35 catchment attributes.

        Returns:
            torch.Tensor: Shape (N_basins, 35) on specified device.
        """
        raw_attr = self.load_raw_attributes(basin_ids).copy()

        # Handle NaNs or infinite values if present
        raw_attr = np.nan_to_num(raw_attr, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply log1p transform to highly skewed variables (e.g. area, conductivity, permeability)
        if log_transform_skewed:
            # Indices for area_gages2 (col 11), soil_conductivity (col 23), geol_permeability (col 34)
            skewed_indices = [11, 23, 34]
            for idx in skewed_indices:
                if idx < raw_attr.shape[1]:
                    min_val = np.min(raw_attr[:, idx])
                    shift = abs(min_val) + 1.0 if min_val < 0 else 1.0
                    raw_attr[:, idx] = np.log(raw_attr[:, idx] + shift)

        if method == "zscore":
            mean = np.mean(raw_attr, axis=0, keepdims=True)
            std = np.std(raw_attr, axis=0, keepdims=True) + 1e-6
            norm_attr = (raw_attr - mean) / std
        elif method == "minmax":
            amin = np.min(raw_attr, axis=0, keepdims=True)
            amax = np.max(raw_attr, axis=0, keepdims=True)
            norm_attr = (raw_attr - amin) / (amax - amin + 1e-6)
        else:
            norm_attr = raw_attr

        return torch.as_tensor(norm_attr, dtype=torch.float64, device=device)

