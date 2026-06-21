"""Tests for the method registry module."""

import pytest
import onset.simulator  # noqa: F401 — ensures handler wiring runs before tests
from onset.method_registry import (
    MethodConfig,
    _METHOD_REGISTRY,
    _resolve_method,
    _SOLVER_CALLABLES,
)


class TestMethodRegistry:
    """Tests for _METHOD_REGISTRY and MethodConfig."""

    def test_all_keys_have_handler_field(self):
        """Every entry has a handler field (set directly in method_registry.py)."""
        for name, config in _METHOD_REGISTRY.items():
            assert isinstance(config.handler, str) or callable(config.handler), \
                f"{name}: handler should be str or callable"

    def test_milp_configs_have_solver_callable(self):
        """MILP configs have non-None solve_fn and solver_method."""
        for name in ("doppler", "onset_v3", "onset_v2", "onset"):
            config = _METHOD_REGISTRY[name]
            assert config.is_milp
            assert config.solve_fn is not None
            assert config.solver_method is not None
            assert config.objective_mode is not None

    def test_heuristic_configs_have_no_milp_fields(self):
        """Non-MILP configs have None for MILP-specific fields."""
        for name in ("OTP", "greylambda", "cache", "BVT", "TBE", "cli"):
            config = _METHOD_REGISTRY[name]
            assert not config.is_milp
            assert config.solve_fn is None
            assert config.objective_mode is None

    def test_ten_entries(self):
        """Registry has exactly 10 entries."""
        assert len(_METHOD_REGISTRY) == 10


class TestResolveMethod:
    """Tests for _resolve_method."""

    def test_exact_match(self):
        config = _resolve_method("onset_v3")
        assert config.name == "onset_v3"
        assert config.is_milp

    def test_doppler_substring_match(self):
        """Doppler partial match via case-insensitive substring."""
        config = _resolve_method("Doppler-v2")
        assert config.name == "doppler"

    def test_doppler_case_insensitive(self):
        """'doppler_ecmp' should match 'doppler'."""
        config = _resolve_method("doppler_ecmp")
        assert config.name == "doppler"

    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown topology_programming_method"):
            _resolve_method("nonexistent_method")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _resolve_method("")


class TestSolverCallables:
    """Tests for _SOLVER_CALLABLES."""

    def test_all_four_solver_entries(self):
        assert set(_SOLVER_CALLABLES.keys()) == {"doppler", "onset_v3", "onset_v2", "onset"}

    def test_callables_are_callable(self):
        for name, fn in _SOLVER_CALLABLES.items():
            assert callable(fn), f"{name} solver is not callable"
