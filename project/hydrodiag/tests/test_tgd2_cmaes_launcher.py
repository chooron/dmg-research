from training.ic.launch_tgd2_batched_cmaes_531 import MODELS, PROJECT, RUNNER


def test_formal_tgd2_cmaes_launcher_covers_both_new_models():
    assert MODELS == ("GR4J_TGD2", "SIMHYD_TGD2")
    assert RUNNER == PROJECT / "training/ic/run_tgd2_batched_cmaes_531.py"
