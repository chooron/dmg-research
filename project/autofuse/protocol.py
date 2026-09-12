"""Locked paper protocol, with requested 544-basin run as an explicit subset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class ExperimentProtocol:
    """Experiment contract shared by SCE and dPL evaluators.

    The cloned paper scripts read ``liste_BV_CAMELS_559.txt`` and therefore
    establish 559 as the raw source population.  The requested AutoFuse
    skeleton reserves a 544-catchment analysis subset; its actual IDs must be
    supplied by a manifest rather than guessed here.
    """

    catchment_count: int = 544
    paper_source_catchment_count: int = 559
    structure_count: int = 78
    forcing_product: str = "daymet"
    spatialisation: str = "Lumped"
    dataset: str = "CAMELS"
    metric: str = "KGECOMP"
    transform_calibration: float = 1.0
    warmup_years: int = 2
    forcing_start: date = date(1987, 1, 1)
    simulation_start: date = date(1989, 1, 1)
    simulation_end: date = date(2009, 12, 31)
    calibration_start: date = date(1989, 1, 1)
    calibration_end: date = date(1998, 12, 31)
    evaluation_start: date = date(1999, 1, 1)
    evaluation_end: date = date(2009, 12, 31)
    sce_max_evaluations: int = 10_000
    sce_kstop: int = 3
    sce_pcento: float = 0.001
    default_seed: int = 20260901

    def __post_init__(self) -> None:
        if self.catchment_count != 544:
            raise ValueError("the AutoFuse phase-1 protocol reserves exactly 544 catchments")
        if self.structure_count != 78:
            raise ValueError("the AutoFuse phase-1 protocol reserves exactly 78 structures")
        if self.forcing_start >= self.simulation_start:
            raise ValueError("forcing_start must precede simulation_start by the warm-up period")
        if self.calibration_start < self.simulation_start or self.calibration_end > self.simulation_end:
            raise ValueError("calibration period must be inside the simulation period")
        if self.evaluation_start != self.calibration_end + timedelta(days=1):
            raise ValueError("evaluation must immediately follow calibration in the paper protocol")
        if self.evaluation_end != self.simulation_end:
            raise ValueError("evaluation must end at simulation_end")

    @property
    def periods(self) -> dict[str, tuple[str, str]]:
        return {
            "forcing": (self.forcing_start.isoformat(), self.simulation_end.isoformat()),
            "warmup": (self.forcing_start.isoformat(), (self.simulation_start - timedelta(days=1)).isoformat()),
            "simulation": (self.simulation_start.isoformat(), self.simulation_end.isoformat()),
            "calibration": (self.calibration_start.isoformat(), self.calibration_end.isoformat()),
            "evaluation": (self.evaluation_start.isoformat(), self.evaluation_end.isoformat()),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "catchment_count": self.catchment_count,
            "paper_source_catchment_count": self.paper_source_catchment_count,
            "structure_count": self.structure_count,
            "forcing_product": self.forcing_product,
            "spatialisation": self.spatialisation,
            "dataset": self.dataset,
            "metric": self.metric,
            "transform_calibration": self.transform_calibration,
            "warmup_years": self.warmup_years,
            "periods": self.periods,
            "sce": {
                "max_evaluations": self.sce_max_evaluations,
                "kstop": self.sce_kstop,
                "pcento": self.sce_pcento,
            },
            "default_seed": self.default_seed,
        }
