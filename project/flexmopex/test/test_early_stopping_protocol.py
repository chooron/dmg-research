from project.flexmopex.models.early_stopping import EarlyStoppingController


def test_early_stopping_waits_until_min_epoch_and_selects_best():
    controller = EarlyStoppingController(
        enabled=True,
        min_epochs=50,
        patience=3,
        min_delta=1e-4,
    )
    for epoch in range(1, 50):
        assert controller.update(epoch, 1.0) is False
        assert controller.best_epoch is None
    assert controller.update(50, 0.50) is False
    assert controller.best_epoch == 50
    assert controller.update(51, 0.50005) is False
    assert controller.update(52, 0.50005) is False
    assert controller.update(53, 0.50005) is True
    assert controller.stop_epoch == 53
    assert controller.reason == "patience_exhausted"


def test_early_stopping_min_delta_does_not_discard_late_small_improvement():
    controller = EarlyStoppingController(
        enabled=True,
        min_epochs=50,
        patience=20,
        min_delta=1e-4,
    )
    for epoch in range(1, 81):
        controller.update(epoch, 1.0 if epoch < 50 else 0.5000)
    assert controller.update(81, 0.4998) is False
    assert controller.best_epoch == 81
    assert controller.update(82, 0.49975) is False
    assert controller.best_epoch == 81
