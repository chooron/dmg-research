from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path[:0]=[str(ROOT),str(ROOT/"experiments/cmaes_36models")]
import torch
from src.batched_cmaes import BatchedCMAES
from src.model_registry import audit_registry, get_spec
from src.objective import streaming_kge
from src.parameter_transform import LatentBoundTransform


def test_fixed_registry_and_finite_bounds():
    assert len(audit_registry()) == 36


def test_latent_transform_round_trip():
    t=LatentBoundTransform(get_spec("gr4j").bounds)
    x=torch.tensor([[-2.,0.,2.,4.]],dtype=torch.float64)
    assert torch.allclose(t.normalized_to_latent(t.latent_to_normalized(x)),x,atol=1e-10)


def test_streaming_kge_and_invalid_penalty():
    o=torch.arange(1.,6.).view(5,1); p=o[:,None,:,None]
    score,bad=streaming_kge(p,o)
    # Existing project KGE intentionally uses eps=0.1 in all ratio denominators,
    # so a short perfect series is below one; this checks the inherited convention.
    assert score.item() > .9 and not bad.item()
    p[0]=torch.nan; score,bad=streaming_kge(p,o)
    assert bad.item() and score.item() == -1_000_000.


def test_full_covariance_cma_state_is_independent():
    s=BatchedCMAES(2,4,8,stdev_init=.1,active=True,seed=1,device="cpu")
    z,y,x=s.ask(); fitness=-x.square().sum(-1); s.tell(z,y,x,fitness)
    assert s.state.C.shape == (2,4,4) and not torch.equal(s.state.mean[0],s.state.mean[1])
