import torch

from models.core.tcm import create_initial_state, tcm_step


def test_tcm_runoff_and_parameter_gradients_are_active():
    device = torch.device("cpu")
    n_steps = 30

    P = torch.tensor([[20.0]] * 10 + [[5.0]] * 10 + [[0.5]] * 10, device=device)
    T = torch.zeros_like(P)
    PET = torch.tensor([[3.0]] * n_steps, device=device)
    mean_P = P.mean(dim=0)

    params = {
        "phi": torch.tensor([[0.35]], device=device, requires_grad=True),
        "rc": torch.tensor([[120.0]], device=device, requires_grad=True),
        "gam": torch.tensor([[0.4]], device=device, requires_grad=True),
        "k1": torch.tensor([[0.15]], device=device, requires_grad=True),
        "fa": torch.tensor([[0.02]], device=device, requires_grad=True),
        "k2": torch.tensor([[0.03]], device=device, requires_grad=True),
    }

    states = create_initial_state(1, 1, device)
    runoff = []
    actual_et = []
    abstraction = []
    signed_storage = []

    for t in range(n_steps):
        q_t, ea_t, s1, s2, s3, s4, diagnostics = tcm_step(
            P[t],
            T[t],
            PET[t],
            *(params[name] for name in ("phi", "rc", "gam", "k1", "fa", "k2")),
            *states,
            mean_P=mean_P,
            return_diagnostics=True,
        )
        states = (s1, s2, s3, s4)
        runoff.append(q_t)
        actual_et.append(ea_t)
        abstraction.append(diagnostics["external_losses"])
        signed_storage.append(s1 - s2 + s3 + s4)

    qsum = torch.stack(runoff).sum()
    easum = torch.stack(actual_et).sum()
    asum = torch.stack(abstraction).sum()
    final_storage = signed_storage[-1].sum()
    water_balance_residual = P.sum() - qsum - easum - asum - final_storage

    assert qsum.item() > 0.0
    assert abs(water_balance_residual.item()) < 1e-3

    qsum.backward()

    assert params["phi"].grad.abs().item() > 0.0
    assert params["k1"].grad.abs().item() > 0.0
    assert params["fa"].grad.abs().item() > 0.0
    assert params["k2"].grad.abs().item() > 0.0
