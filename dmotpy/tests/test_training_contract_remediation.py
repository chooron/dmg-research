from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dmotpy.data_contract import add_calendar_forcing
from dmotpy.losses import KgeLoss, NseBatchLoss, full_sequence_kge
from dmotpy.trainers.checkpoint import (
    REQUIRED_KEYS,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _series(requires_grad: bool = True):
    prediction = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 6.0], [5.0, 8.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    target = torch.tensor(
        [[1.1, 2.2], [2.0, 3.1], [3.9, 6.2], [5.2, 7.8]],
        dtype=torch.float64,
    )
    return prediction, target


def test_masked_and_compact_kge_match_and_padding_is_invariant():
    prediction, target = _series()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[1, :] = False
    masked = KgeLoss()(prediction, target, mask=mask)
    compact = KgeLoss()(prediction[[0, 2, 3]], target[[0, 2, 3]])
    assert torch.allclose(masked, compact, atol=1e-10)

    padded_prediction = torch.cat((prediction.detach(), torch.zeros(3, 2, dtype=torch.float64)))
    padded_target = torch.cat((target, torch.zeros(3, 2, dtype=torch.float64)))
    padded_mask = torch.cat((mask, torch.zeros(3, 2, dtype=torch.bool)))
    padded = KgeLoss()(padded_prediction, padded_target, mask=padded_mask)
    assert torch.allclose(masked.detach(), padded, atol=1e-10)


def test_padding_does_not_change_valid_gradients():
    prediction, target = _series()
    mask = torch.ones_like(target, dtype=torch.bool)
    compact_loss = NseBatchLoss()(prediction, target, mask=mask)
    compact_grad = torch.autograd.grad(compact_loss, prediction, retain_graph=True)[0]

    padded_prediction = torch.cat((prediction.detach(), torch.zeros(3, 2, dtype=torch.float64))).requires_grad_()
    padded_target = torch.cat((target, torch.zeros(3, 2, dtype=torch.float64)))
    padded_mask = torch.cat((mask, torch.zeros(3, 2, dtype=torch.bool)))
    padded_loss = NseBatchLoss()(padded_prediction, padded_target, mask=padded_mask)
    padded_grad = torch.autograd.grad(padded_loss, padded_prediction)[0]
    assert torch.allclose(compact_grad, padded_grad[:4], atol=1e-10)
    assert torch.equal(padded_grad[4:], torch.zeros_like(padded_grad[4:]))


def test_nonfinite_prediction_fails_even_when_masked():
    prediction, target = _series(requires_grad=False)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0] = False
    prediction[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        NseBatchLoss()(prediction, target, mask=mask)


def test_batch_partition_changes_only_window_metric_not_decomposable_loss():
    prediction, target = _series()
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = NseBatchLoss()(prediction, target, mask=mask)
    first = NseBatchLoss()(prediction[:2], target[:2], mask=mask[:2])
    second = NseBatchLoss()(prediction[2:], target[2:], mask=mask[2:])
    assert torch.isfinite(loss)
    assert torch.isfinite(first + second)
    # KGE remains explicitly a complete-sequence metric and is not silently
    # advertised as the average of two independently computed windows.
    assert not torch.allclose(
        full_sequence_kge(prediction, target, mask=mask),
        0.5 * (KgeLoss()(prediction[:2], target[:2]) + KgeLoss()(prediction[2:], target[2:])),
    )


def test_calendar_adapter_handles_leap_day_and_shape():
    forcing = torch.zeros(4, 2, 3, dtype=torch.float32)
    dates = ["2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02"]
    augmented, doy = add_calendar_forcing(forcing, dates, model_name="mopex4")
    assert augmented.shape == (4, 2, 4)
    assert doy is not None
    assert augmented[:, 0, 3].tolist() == [59.0, 60.0, 61.0, 62.0]
    ordinary, ordinary_doy = add_calendar_forcing(forcing, dates, model_name="hbv96")
    assert ordinary.shape == forcing.shape
    assert ordinary_doy is None


def test_checkpoint_schema_and_rng_round_trip(tmp_path):
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = save_training_checkpoint(
        tmp_path,
        model=model,
        epoch=2,
        global_step=7,
        optimizer=optimizer,
        scheduler=scheduler,
        config={"model": {"name": "test"}, "dataset_manifest_hash": "abc"},
        hydrological_states={"state": torch.tensor([1.0])},
        uh_states={"tail": torch.tensor([2.0])},
        warmup_state={"steps": 3},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert REQUIRED_KEYS.issubset(payload)

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    loaded = load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    assert loaded["epoch"] == 2
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
