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


def _monkeypatch_script_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch SCRIPT_HOME across all relevant modules to point to *tmp_path*."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for module in (preprocessing, simulator_module, te_engine, validation):
        monkeypatch.setattr(module, "SCRIPT_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "graphs" / "gml").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".temp").mkdir(exist_ok=True)


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


def test_perform_sim_mcf_te_method(simulation_inputs: Path) -> None:
    """MCF te_method uses top_k=1 and single-solve path."""
    simulation_inputs.write_text("0 0 90000000000 0 0 0 0 0 0\n", encoding="utf-8")
    simulation = Simulation(
        "tiny",
        3,
        "pipeline_mcf",
        iterations=1,
        te_method="-mcf",
        traffic_file=str(simulation_inputs),
        topology_programming_method="doppler",
        fallow_transponders=1,
        candidate_link_choice_method="max",
        optimizer_time_limit_minutes=0.1,
        parallel_path_computation=False,
    )
    result = simulation.perform_sim(start_iter=1, end_iter=2)
    assert isinstance(result, dict)
    assert result["Routing"] == ["MCF"]
    assert simulation.optimization_result is not None


def test_perform_sim_conservative_candidate_links(
    simulation_inputs: Path,
) -> None:
    """Conservative candidate link set exercises a different code path."""
    simulation_inputs.write_text("0 0 70000000000 0 0 0 0 0 0\n", encoding="utf-8")
    simulation = Simulation(
        "tiny",
        3,
        "pipeline_consv",
        iterations=1,
        te_method="-ecmp",
        traffic_file=str(simulation_inputs),
        topology_programming_method="baseline",
        fallow_transponders=1,
        candidate_link_choice_method="conservative",
        optimizer_time_limit_minutes=0.1,
        parallel_path_computation=False,
    )
    result = simulation.perform_sim(start_iter=1, end_iter=2)
    assert isinstance(result, dict)
    assert result["CandidateLinkSet"] == ["conservative"]


def test_perform_sim_dry_run(simulation_inputs: Path) -> None:
    """dry=True skips TE evaluation and returns the result path."""
    simulation_inputs.write_text("0 0 50000000000 0 0 0 0 0 0\n", encoding="utf-8")
    simulation = Simulation(
        "tiny",
        3,
        "pipeline_dry",
        iterations=1,
        te_method="-ecmp",
        traffic_file=str(simulation_inputs),
        topology_programming_method="baseline",
        fallow_transponders=1,
        candidate_link_choice_method="max",
        optimizer_time_limit_minutes=0.1,
        parallel_path_computation=False,
    )
    result = simulation.perform_sim(start_iter=1, end_iter=2, dry=True)
    assert isinstance(result, str)
    assert simulation.ITERATION_ABS_PATH in result


def test_apply_and_revert_solution(simulation_inputs: Path) -> None:
    """apply_solution and revert_solution round-trip correctly."""
    simulation_inputs.write_text("0 0 90000000000 0 0 0 0 0 0\n", encoding="utf-8")
    simulation = make_simulation(simulation_inputs, "doppler")
    simulation.perform_sim(start_iter=1, end_iter=2)
    assert simulation._applied_solution is not None

    sol = simulation._applied_solution
    edges_after = set(simulation.wolf.logical_graph.edges())
    assert len(sol.added) > 0
    for u, v in sol.added:
        assert (u, v) in edges_after

    simulation.revert_solution()
    assert simulation._applied_solution is None
    assert not simulation.circuits_added
    edges_reverted = set(simulation.wolf.logical_graph.edges())
    for u, v in sol.added:
        assert (u, v) not in edges_reverted

    simulation.apply_solution(sol)
    assert simulation._applied_solution is not None
    assert simulation.circuits_added
    edges_restored = set(simulation.wolf.logical_graph.edges())
    assert edges_restored == edges_after


# ---------------------------------------------------------------------------
# SMORE (-semimcfraeke / -semimcfraekeft) integration via perform_sim
# ---------------------------------------------------------------------------


def _diamond_gml(path: Path) -> None:
    """Write an undirected 4-switch diamond GML (1-2-4, 1-3-4)."""
    g = nx.Graph()
    for n in ("1", "2", "3", "4"):
        g.add_node(n, Longitude=0.0, Latitude=float(n))
    for a, b in (("1", "2"), ("1", "3"), ("2", "4"), ("3", "4")):
        g.add_edge(a, b, capacity=100)
    nx.write_gml(g, path / "data" / "graphs" / "gml" / "diamond.gml")


def _diamond_traffic(path: Path) -> Path:
    """4-host TM: 100 Gbps from h1 -> h4, zero elsewhere."""
    tm = path / "diamond.tm"
    # 4 hosts → 16 entries; h1→h4 = 100e9, everything else 0
    tm.write_text(
        "0 0 0 100000000000 " "0 0 0 0 " "0 0 0 0 " "0 0 0 0\n",
        encoding="utf-8",
    )
    return tm


def _make_smore_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    te_method: str,
    te_seed: int | None = None,
) -> Simulation:
    _monkeypatch_script_home(tmp_path, monkeypatch)
    _diamond_gml(tmp_path)
    traffic = _diamond_traffic(tmp_path)
    return Simulation(
        "diamond",
        4,
        f"pipeline_{te_method.lstrip('-')}",
        iterations=1,
        te_method=te_method,
        traffic_file=str(traffic),
        topology_programming_method="baseline",
        fallow_transponders=1,
        candidate_link_choice_method="max",
        optimizer_time_limit_minutes=0.1,
        parallel_path_computation=False,
        te_seed=te_seed,
    )


@pytest.mark.parametrize(
    ("te_method", "routing_label"),
    [
        ("-semimcfraeke", "SEMIMCFRAEKE"),
        ("-semimcfraekeft", "SEMIMCFRAEKEFT"),
    ],
)
def test_perform_sim_smore_pipeline_is_seeded_and_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    te_method: str,
    routing_label: str,
) -> None:
    """Both SMORE methods run end-to-end and reproduce seeded results."""
    sim1 = _make_smore_simulation(tmp_path / "run1", monkeypatch, te_method, te_seed=42)
    result1 = sim1.perform_sim(start_iter=1, end_iter=2)

    assert isinstance(result1, dict)
    assert result1["Routing"] == [routing_label]
    assert result1["Throughput"] == [1.0]
    assert result1["Loss"] == [0.0]
    assert "seed42" in sim1.EXPERIMENT_ID
    assert "seed42" in sim1.ITERATION_ID

    # Check result files exist
    iter_dir = Path(sim1.ITERATION_ABS_PATH)
    assert (iter_dir / "MaxExpCongestionVsIterations.dat").is_file()
    assert (iter_dir / "TotalThroughputVsIterations.dat").is_file()
    assert (iter_dir / "NumPathsVsIterations.dat").is_file()
    paths_dir = iter_dir / "paths"
    assert paths_dir.is_dir()
    path_snapshot1 = {
        path.relative_to(paths_dir): path.read_text(encoding="utf-8")
        for path in paths_dir.rglob("*")
        if path.is_file()
    }
    assert path_snapshot1

    # Use a separate root so this cannot pass by rereading overwritten files.
    sim2 = _make_smore_simulation(tmp_path / "run2", monkeypatch, te_method, te_seed=42)
    result2 = sim2.perform_sim(start_iter=1, end_iter=2)
    assert isinstance(result2, dict)
    assert result2["Throughput"] == result1["Throughput"]
    assert result2["Loss"] == result1["Loss"]
    assert result2["Congestion"] == result1["Congestion"]

    paths_dir2 = Path(sim2.ITERATION_ABS_PATH) / "paths"
    path_snapshot2 = {
        path.relative_to(paths_dir2): path.read_text(encoding="utf-8")
        for path in paths_dir2.rglob("*")
        if path.is_file()
    }
    assert path_snapshot2 == path_snapshot1
