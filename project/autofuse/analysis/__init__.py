"""Post-evaluation analysis primitives; no model equations live here."""

from .structure_analysis import coverage_set, equivalent_set, rank_correlations, regret

__all__ = ["coverage_set", "equivalent_set", "rank_correlations", "regret"]
