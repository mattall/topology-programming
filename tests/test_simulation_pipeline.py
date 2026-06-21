"""Deterministic end-to-end tests for the maintained simulation pipeline."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

import onset.preprocessing as preprocessing
import onset.simulator as simulator_module
import onset.te.engine as te_engine
import onset.validation as validation
from onset.simulator import Simulation


@pytest.fixture
def simulation_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a complete three-router experiment without repository data files."""
    for module in (preprocessing, simulator_module, te_engine, validation):
        monkeypatch.setattr(module, "SCRIPT_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    graph_dir = tmp_path / "data" / "graphs" / "gml"
    graph_dir.mkdir(parents=True)
    graph = nx.Graph()
    graph.add_node("1", Longitude=0.0, Latitude=0.0)
    graph.add_node("2", Longitude=1.0, Latitude=0.0)
    graph.add_node("3", Longitude=2.0, Latitude=0.0)
    graph.add_edge("1", "2", capacity=100)
    graph.add_edge("2", "3", capacity=100)
    nx.write_gml(graph, graph_dir / "tiny.gml")

    traffic = tmp_path / "tiny.tm"
    traffic.write_text("0 0 50000000000 0 0 0 0 0 0\n", encoding="utf-8")
    return traffic


def make_simulation(traffic: Path, method: str) -> Simulation:
    return Simulation(
        "tiny",
        3,
        f"pipeline_{method}",
        iterations=1,
        te_method="-ecmp",
        traffic_file=str(traffic),
        topology_programming_method=method,
        fallow_transponders=1,
        candidate_link_choice_method="max",
        optimizer_time_limit_minutes=0.1,
        parallel_path_computation=False,
    )


def test_perform_sim_internal_ecmp_pipeline(simulation_inputs: Path) -> None:
    simulation = make_simulation(simulation_inputs, "baseline")
    result = simulation.perform_sim(start_iter=1, end_iter=2)

    assert isinstance(result, dict)
    assert result["Routing"] == ["ECMP"]
    assert result["Defense"] == ["baseline"]
    assert result["Congestion"][0] > 0
    assert result["Throughput"] == [1.0]
    iteration_dir = Path(simulation.ITERATION_ABS_PATH)
    assert (iteration_dir / "MaxExpCongestionVsIterations.dat").is_file()


def test_perform_sim_highs_topology_programming_pipeline(
    simulation_inputs: Path,
) -> None:
    simulation_inputs.write_text("0 0 90000000000 0 0 0 0 0 0\n", encoding="utf-8")
    simulation = make_simulation(simulation_inputs, "doppler")
    result = simulation.perform_sim(start_iter=1, end_iter=2)

    assert isinstance(result, dict)
    assert simulation.optimization_result is not None
    assert simulation.optimization_result.has_solutions
    assert simulation.optimization_result.backend.value == "open"
    assert simulation._applied_solution is not None
    assert ("1", "3") in simulation._applied_solution.added
    assert result["Defense"] == ["doppler"]
    assert result["Optimization Time"][0] >= 0
    iteration_dir = Path(simulation.ITERATION_ABS_PATH)
    assert (iteration_dir / "MaxExpCongestionVsIterations.dat").is_file()
    assert (iteration_dir / "TotalSolutions.dat").is_file()
    assert (iteration_dir / "OptimalTopoID.dat").is_file()
