import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import r1_pipeline as pipeline


CUDA = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical statistics require CUDA")
def test_aggregation_order_and_staged_seed_median():
    table = pipeline.read_compact(Path(__file__).parents[3] / "cache/r1_rebuild_audit_staged/r1_basin_level_performance_rebuilt.csv", CUDA)
    index = table.key_index(("basin_id", "paradigm", "structure", "period"))
    key = ("01022500", "dPL-MLP", "Base", "test")
    assert table.rows[index[key]]["seed_or_restart"] == "median_across_seeds"
    assert len(index) == 6372


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical statistics require CUDA")
def test_kge_population_ddof_convention():
    obs = torch.arange(1, 41, dtype=torch.float64, device=CUDA)
    sim = obs * 2
    assert torch.allclose(pipeline.gupta_kge_gpu(obs, sim), torch.tensor(1 - 2**.5, device=CUDA, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical statistics require CUDA")
def test_gpu_spearman_average_ties_nan_and_constant():
    x = torch.tensor([1., 1., 2., 3., float("nan")], device=CUDA)
    y = torch.tensor([3., 2., 1., 1., 4.], device=CUDA)
    ranks = pipeline.average_rank(x)
    assert torch.allclose(ranks[:4], torch.tensor([1.5, 1.5, 3., 4.], device=CUDA, dtype=torch.float64))
    assert torch.isfinite(pipeline.spearman(x, y))
    assert torch.isnan(pipeline.spearman(torch.ones(4, device=CUDA), y[:4]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical statistics require CUDA")
def test_paired_bootstrap_determinism():
    values = torch.arange(12, dtype=torch.float64, device=CUDA).reshape(6, 2)
    a = pipeline._bootstrap_columns(values, 20260730, draws=64)
    b = pipeline._bootstrap_columns(values, 20260730, draws=64)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical statistics require CUDA")
def test_endpoint_and_loro_masks():
    strata = torch.tensor([0, 0, 4, 4, 2], device=CUDA)
    assert torch.equal(strata == 0, torch.tensor([True, True, False, False, False], device=CUDA))
    region_codes = torch.tensor([11, 12, 13, 11], device=CUDA)
    assert torch.equal(region_codes != 11, torch.tensor([False, True, True, False], device=CUDA))
    low, high = torch.tensor([1., 2.], device=CUDA), torch.tensor([4., 5.], device=CUDA)
    assert torch.median(high) - torch.median(low) == 3


def test_snow_counts_and_boundaries():
    assert pipeline.STRATA_COUNTS == {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
    assert pipeline.STRATA == ("S1", "S2", "S3", "S4", "S5")


def test_cuda_refusal(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        pipeline.require_cuda()


def test_no_daily_source_dataflow():
    source = (Path(__file__).parents[1] / "r1_pipeline.py").read_text()
    assert "pyarrow" not in source
    assert "pandas" not in source
    assert "scipy" not in source
    assert ".parquet" not in source
    assert "read_parquet" not in source
