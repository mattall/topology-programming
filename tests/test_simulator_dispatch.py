"""Tests for simulator method dispatch."""

import pytest

from onset.handlers import (
    _run_baseline,
    _run_bvt,
    _run_cache,
    _run_cli,
    _run_greylambda,
    _run_milp_method,
    _run_otp,
    _run_tbe,
)
from onset.method_registry import _METHOD_REGISTRY, _resolve_method


class TestDispatchHandlers:
    """Tests that all registered handlers are callable functions."""

    def test_all_handlers_are_callable(self):
        """After wiring in simulator.py, all handlers should be callables."""
        for name, config in _METHOD_REGISTRY.items():
            assert callable(
                config.handler
            ), f"Handler for '{name}' is not callable: {config.handler}"

    def test_milp_methods_use_milp_handler(self):
        for name in ("doppler", "onset_v3", "onset_v2", "onset"):
            config = _METHOD_REGISTRY[name]
            assert (
                config.handler == _run_milp_method
            ), f"{name} should use _run_milp_method"

    def test_heuristic_methods_use_own_handlers(self):
        handler_map = {
            "baseline": _run_baseline,
            "OTP": _run_otp,
            "greylambda": _run_greylambda,
            "cache": _run_cache,
            "BVT": _run_bvt,
            "TBE": _run_tbe,
            "cli": _run_cli,
        }
        for name, expected_handler in handler_map.items():
            config = _METHOD_REGISTRY[name]
            assert (
                config.handler == expected_handler
            ), f"{name} should use its dedicated handler"


class TestResolveMethodDispatch:
    """Tests that _resolve_method returns correctly-configured entries."""

    def test_resolve_onset_v3(self):
        config = _resolve_method("onset_v3")
        assert config.name == "onset_v3"
        assert config.is_milp
        assert config.objective_mode == "mlu"
        assert config.solver_method == "onset_v3"
        assert config.uses_ecmp_multisol is True

    def test_resolve_doppler(self):
        config = _resolve_method("doppler")
        assert config.name == "doppler"
        assert config.objective_mode == "changes_plus_mlu"
        assert config.uses_ecmp_multisol is False

    def test_resolve_otp(self):
        config = _resolve_method("OTP")
        assert config.name == "OTP"
        assert not config.is_milp
        assert config.handler == _run_otp

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            _resolve_method("invalid_method")
