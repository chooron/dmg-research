from training.ic.run_tgd2_batched_cmaes_531 import MODEL_DIMENSIONS, population_for_dimension


def test_tgd2_cmaes_population_scales_from_xaj_reference_dimension():
    assert MODEL_DIMENSIONS == {"GR4J_TGD2": 6, "SIMHYD_TGD2": 12}
    assert population_for_dimension(6) == 12
    assert population_for_dimension(12) == 18
    assert population_for_dimension(17) == 25
