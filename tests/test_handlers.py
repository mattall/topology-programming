"""Unit tests for topology-programming method handlers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Resolve method_registry first to break the handlers↔method_registry cycle.
from onset.method_registry import _METHOD_REGISTRY

_run_baseline = _METHOD_REGISTRY["baseline"].handler
_run_bvt = _METHOD_REGISTRY["BVT"].handler
_run_cli = _METHOD_REGISTRY["cli"].handler
_run_greylambda = _METHOD_REGISTRY["greylambda"].handler
_run_milp_method = _METHOD_REGISTRY["doppler"].handler
_run_otp = _METHOD_REGISTRY["OTP"].handler
_run_tbe = _METHOD_REGISTRY["TBE"].handler


def _write_congestion_file(
    path: str, edges: list[tuple[str, str, float]]
) -> str:
    """Write a minimal EdgeCongestionVsIterations.dat file.

    Each edge is (node_u, node_v, congestion_float).
    """
    filepath = os.path.join(path, "EdgeCongestionVsIterations.dat")
    with open(filepath, "w") as f:
        f.write("Iteration 0\n")
        f.write("============\n")
        for u, v, cong in edges:
            f.write(f"\t\t(s{u},s{v}) : {cong}\n")
    return filepath


class TestRunBaseline:
    def test_does_nothing(self):
        sim = MagicMock()
        config = MagicMock()
        _run_baseline(sim, config)
        sim.assert_not_called()


class TestRunBVT:
    def test_clears_sig_add_circuits(self, tmp_path):
        _write_congestion_file(str(tmp_path), [("1", "2", 1.0)])
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.sig_add_circuits = True
        config = MagicMock()
        _run_bvt(sim, config)
        assert sim.sig_add_circuits is False

    def test_no_congestion_file_raises(self, tmp_path):
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.sig_add_circuits = True
        config = MagicMock()
        with pytest.raises(FileNotFoundError):
            _run_bvt(sim, config)


class TestRunTBE:
    def test_no_flashcrowd_does_nothing(self):
        sim = MagicMock()
        sim.traffic_file = "background_tm"
        sim.demand_factor = 1.0
        config = MagicMock()
        _run_tbe(sim, config)
        sim.wolf.relax_restricted_bandwidth.assert_not_called()

    def test_flashcrowd_high_demand_relaxes_bandwidth(self):
        sim = MagicMock()
        sim.traffic_file = "flashcrowd_tm"
        sim.demand_factor = 0.95
        config = MagicMock()
        _run_tbe(sim, config)
        sim.wolf.relax_restricted_bandwidth.assert_called_once()

    def test_flashcrowd_low_demand_does_not_relax(self):
        sim = MagicMock()
        sim.traffic_file = "flashcrowd_tm"
        sim.demand_factor = 0.5
        config = MagicMock()
        _run_tbe(sim, config)
        sim.wolf.relax_restricted_bandwidth.assert_not_called()


class TestRunCLI:
    def test_calls_wolf_cli(self):
        sim = MagicMock()
        config = MagicMock()
        _run_cli(sim, config)
        sim.wolf.cli.assert_called_once()


class TestRunOTP:
    def test_adds_shortcut_for_congested_edge(self, tmp_path):
        _write_congestion_file(
            str(tmp_path),
            [("1", "2", 0.90), ("1", "3", 0.85)],
        )
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.wolf.logical_graph.edges.return_value = []
        sim.circuits = 1
        sim.flux_circuits = []
        sim.sig_add_circuits = True
        config = MagicMock()
        _run_otp(sim, config)
        assert sim.sig_add_circuits is False
        sim.wolf.add_circuit.assert_called()

    def test_no_congested_edges_skips(self, tmp_path):
        _write_congestion_file(str(tmp_path), [("1", "2", 0.50)])
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.wolf.logical_graph.edges.return_value = [("1", "2")]
        sim.flux_circuits = []
        sim.sig_add_circuits = True
        config = MagicMock()
        with patch("onset.handlers.find_shortcut_link", return_value=[]):
            _run_otp(sim, config)
        sim.wolf.add_circuit.assert_not_called()
        assert sim.sig_add_circuits is False


class TestRunGreylambda:
    def test_adds_circuit_for_fully_congested_edge(self, tmp_path):
        _write_congestion_file(str(tmp_path), [("1", "2", 1.0)])
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.wolf.add_circuit.return_value = 1
        sim.circuits = 1
        sim.flux_circuits = []
        sim.sig_add_circuits = True
        sim.circuits_added = False
        config = MagicMock()
        _run_greylambda(sim, config)
        sim.wolf.add_circuit.assert_called()
        assert sim.sig_add_circuits is False
        assert sim.circuits_added is False

    def test_marks_circuits_added_when_add_circuit_returns_zero(self, tmp_path):
        _write_congestion_file(str(tmp_path), [("1", "2", 1.0)])
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.wolf.add_circuit.return_value = 0
        sim.circuits = 2
        sim.flux_circuits = []
        sim.sig_add_circuits = True
        sim.circuits_added = False
        config = MagicMock()
        _run_greylambda(sim, config)
        assert sim.circuits_added is True

    def test_no_fully_congested_edges_skips(self, tmp_path):
        _write_congestion_file(str(tmp_path), [("1", "2", 0.80)])
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.flux_circuits = []
        sim.sig_add_circuits = True
        config = MagicMock()
        _run_greylambda(sim, config)
        sim.wolf.add_circuit.assert_not_called()
        assert sim.sig_add_circuits is False

    def test_tuple_edge_format(self, tmp_path):
        _write_congestion_file(str(tmp_path), [("1", "2", 1.0), ("2", "3", 1.0)])
        sim = MagicMock()
        sim.PREV_ITER_ABS_PATH = str(tmp_path)
        sim.wolf.add_circuit.return_value = 1
        sim.circuits = 1
        sim.flux_circuits = []
        sim.sig_add_circuits = True
        sim.circuits_added = False
        config = MagicMock()
        _run_greylambda(sim, config)
        assert sim.wolf.add_circuit.call_count == 2


class TestRunMILPMethod:
    def test_sets_max_load(self):
        sim = MagicMock()
        sim.te_method = "-mcf"
        sim.top_k = 100
        result = MagicMock()
        result.has_solutions = False
        sim._run_topology_optimization.return_value = result
        config = MagicMock()
        config.objective_mode = "changes_plus_mlu"
        config.solver_method = "doppler"
        config.uses_ecmp_multisol = False
        _run_milp_method(sim, config)
        assert sim.max_load == 0.9
        sim._run_topology_optimization.assert_called_once()

    def test_mcf_single_solution_applied(self):
        sim = MagicMock()
        sim.te_method = "-mcf"
        sim.top_k = 100
        result = MagicMock()
        result.has_solutions = True
        sim._run_topology_optimization.return_value = result
        config = MagicMock()
        config.objective_mode = "changes_plus_mlu"
        config.solver_method = "doppler"
        config.uses_ecmp_multisol = False
        _run_milp_method(sim, config)
        sim.apply_solution.assert_called_once_with(result.selected_solution)

    def test_none_result_skips(self):
        sim = MagicMock()
        sim.te_method = "-ecmp"
        sim.top_k = 100
        sim._run_topology_optimization.return_value = None
        config = MagicMock()
        config.objective_mode = "mlu"
        config.solver_method = "onset_v3"
        config.uses_ecmp_multisol = True
        _run_milp_method(sim, config)
        sim.apply_solution.assert_not_called()
