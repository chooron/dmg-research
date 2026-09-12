"""
Validate dMoT core model registry metadata consistency.

Ensures STATE_INFO, NPARAM_INFO, PARAM_INFO, INIT_INFO, STFN_INFO
are internally consistent for all 36 core models.
"""
import torch
import inspect
import importlib

import pytest

from models.core import (
    PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO, NPARAM_INFO
)

# 36 canonical core models
EXPECTED_MODEL_COUNT = 36


class TestCoreRegistryMetadata:
    """Validate dMoT core model registry metadata."""

    def test_registry_contains_36_models(self):
        """All four registries must have exactly 36 models."""
        assert len(PARAM_INFO) == EXPECTED_MODEL_COUNT
        assert len(STFN_INFO) == EXPECTED_MODEL_COUNT
        assert len(INIT_INFO) == EXPECTED_MODEL_COUNT
        assert len(STATE_INFO) == EXPECTED_MODEL_COUNT
        assert len(NPARAM_INFO) == EXPECTED_MODEL_COUNT

    def test_registry_keys_are_identical(self):
        """All registries must have the same keys."""
        p_keys = set(PARAM_INFO.keys())
        s_keys = set(STFN_INFO.keys())
        i_keys = set(INIT_INFO.keys())
        st_keys = set(STATE_INFO.keys())
        n_keys = set(NPARAM_INFO.keys())

        assert p_keys == s_keys, f"PARAM_INFO vs STFN_INFO diff: {p_keys ^ s_keys}"
        assert p_keys == i_keys, f"PARAM_INFO vs INIT_INFO diff: {p_keys ^ i_keys}"
        assert p_keys == st_keys, f"PARAM_INFO vs STATE_INFO diff: {p_keys ^ st_keys}"
        assert p_keys == n_keys, f"PARAM_INFO vs NPARAM_INFO diff: {p_keys ^ n_keys}"

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_nparam_info_matches_param_info(self, model_name):
        """NPARAM_INFO must match actual parameter count from PARAM_INFO."""
        actual = len(PARAM_INFO[model_name])
        reported = NPARAM_INFO[model_name]
        assert actual == reported, (
            f"{model_name}: NPARAM_INFO={reported} but PARAM_INFO has {actual} params"
        )

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_state_info_matches_init_fn(self, model_name):
        """STATE_INFO must match actual state count from create_initial_state."""
        init_fn = INIT_INFO[model_name]
        states = init_fn(1, 1, torch.device('cpu'), 1e-6)
        actual = len(states) if isinstance(states, tuple) else 1
        reported = STATE_INFO[model_name]
        assert actual == reported, (
            f"{model_name}: STATE_INFO={reported} but init returns {actual} states"
        )

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_state_info_matches_step_fn_signature(self, model_name):
        """STATE_INFO must match state count from step function signature.

        The step function signature is: P, T, PET, *params, *states, [nearzero], [kwargs]
        State count = len(positional_args) - 3 - len(params)
        """
        step_fn = STFN_INFO[model_name]
        sig = inspect.signature(step_fn)
        positional = [p for p in sig.parameters
                      if sig.parameters[p].default is inspect.Parameter.empty]
        n_params = len(PARAM_INFO[model_name])
        # After P(0) T(1) PET(2) and n_params parameters, remainder is states
        actual_step_states = len(positional) - 3 - n_params
        reported = STATE_INFO[model_name]

        # Some models have non-state positional args (mean_P, doy)
        # If mismatch, check if the extra positional is a known non-state
        if actual_step_states != reported:
            after = positional[3 + n_params:]
            non_state_pos = {'mean_P', 'doy'}
            actual_clean = actual_step_states - sum(1 for a in after if a in non_state_pos)
            assert actual_clean == reported, (
                f"{model_name}: STATE_INFO={reported} but step signature has "
                f"{actual_clean} state arguments (positions: {after})"
            )

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_step_fn_has_required_arguments(self, model_name):
        """Each step function must accept P, T, PET as first three positional args."""
        step_fn = STFN_INFO[model_name]
        sig = inspect.signature(step_fn)
        positional = [p for p in sig.parameters
                      if sig.parameters[p].default is inspect.Parameter.empty]
        assert len(positional) >= 3, f"{model_name}: step_fn has < 3 positional args"
        assert positional[0] == 'P', f"{model_name}: first arg is '{positional[0]}' not 'P'"
        assert positional[1] == 'T', f"{model_name}: second arg is '{positional[1]}' not 'T'"
        assert positional[2] == 'PET', f"{model_name}: third arg is '{positional[2]}' not 'PET'"

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_init_fn_returns_tensors(self, model_name):
        """create_initial_state must return torch.Tensor or tuple of Tensors."""
        init_fn = INIT_INFO[model_name]
        result = init_fn(1, 1, torch.device('cpu'), 1e-6)
        if isinstance(result, tuple):
            for i, t in enumerate(result):
                assert isinstance(t, torch.Tensor), (
                    f"{model_name}: init result[{i}] is {type(t)} not Tensor"
                )
        else:
            assert isinstance(result, torch.Tensor), (
                f"{model_name}: init result is {type(result)} not Tensor"
            )

    def test_no_duplicate_model_keys(self):
        """No duplicate keys in any registry."""
        for name, registry in [('PARAM_INFO', PARAM_INFO), ('STFN_INFO', STFN_INFO),
                                ('INIT_INFO', INIT_INFO), ('STATE_INFO', STATE_INFO),
                                ('NPARAM_INFO', NPARAM_INFO)]:
            keys = list(registry.keys())
            assert len(keys) == len(set(keys)), f"{name} has duplicate keys"

    @pytest.mark.parametrize("model_name", sorted(PARAM_INFO.keys()))
    def test_param_bounds_are_valid(self, model_name):
        """Parameter bounds must be [lower, upper] with lower <= upper."""
        bounds = PARAM_INFO[model_name]
        for pname, (lo, hi) in bounds.items():
            assert lo <= hi, f"{model_name}/{pname}: lower={lo} > upper={hi}"
            assert isinstance(lo, (int, float)), f"{model_name}/{pname}: lo is {type(lo)}"
            assert isinstance(hi, (int, float)), f"{model_name}/{pname}: hi is {type(hi)}"
