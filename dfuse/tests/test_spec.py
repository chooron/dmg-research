from dfuse.spec import (
    DECISION_CODES,
    DECISION_ORDER,
    PARAMETER_NAMES,
    STATE_NAMES,
    enumerate_structures,
    get_structure,
    validate_catalog,
)


def test_paper_catalog_is_exactly_78_and_unique():
    report = validate_catalog()
    assert report["n_structures"] == 78
    assert report["n_unique_ids"] == 78
    assert report["n_unique_decision_vectors"] == 78
    assert report["n_theoretical_open_combinations"] == 108
    assert report["n_excluded_combinations"] == 30
    assert report["reverse_mapping_ok"]


def test_decision_codes_and_masks_are_total():
    assert DECISION_ORDER == ("RFERR", "ARCH1", "ARCH2", "QSURF", "QPERC", "ESOIL", "QINTF", "Q_TDH", "SNOWM")
    assert len(STATE_NAMES) == 9
    assert len(PARAMETER_NAMES) == 37
    for spec in enumerate_structures():
        assert tuple(spec.state_mask) == STATE_NAMES
        assert tuple(spec.parameter_mask) == PARAMETER_NAMES
        assert len(spec.state_names) == sum(spec.state_mask.values())
        assert len(spec.parameter_names) == sum(spec.parameter_mask.values())
        topology_states = set(spec.topology["upper"]) | set(spec.topology["lower"])
        assert topology_states == set(spec.state_names)
        assert all(code in DECISION_CODES.values() for code in spec.decision_code_vector)
        assert get_structure(spec.model_id).decision_vector == spec.decision_vector


def test_representative_masks_match_upstream_assignments():
    assert get_structure(84).state_names == ("WATR_1", "WATR_2")
    assert get_structure(84).parameter_names == (
        "RFERR_MLT", "FRACTEN", "MAXWATR_1", "MAXWATR_2", "QB_PRMS",
        "PERCRTE", "PERCEXP", "AXV_BEXP", "TIMEDELAY", "MBASE", "MFMAX",
        "MFMIN", "PXTEMP", "OPG", "LAPSE",
    )
    assert get_structure(2).state_names == ("TENS_1", "FREE_1", "TENS_2", "FREE_2A", "FREE_2B")
