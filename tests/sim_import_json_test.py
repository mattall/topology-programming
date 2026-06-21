"""Self-contained JSON import coverage for ``Simulation``."""

import json
from pathlib import Path

import networkx as nx
import pytest

import onset.simulator as simulator_module
import onset.validation as validation
from onset.simulator import Simulation


def test_simulation_imports_json_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in (simulator_module, validation):
        monkeypatch.setattr(module, "SCRIPT_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    graph_dir = tmp_path / "data" / "graphs" / "json"
    graph_dir.mkdir(parents=True)
    (graph_dir / "tiny.json").write_text(
        json.dumps(nx.adjacency_data(nx.path_graph(["1", "2"]))),
        encoding="utf-8",
    )
    traffic = tmp_path / "tiny.tm"
    traffic.write_text("0 1 1 0\n", encoding="utf-8")

    simulation = Simulation(
        "tiny",
        2,
        "json_import",
        iterations=1,
        traffic_file=str(traffic),
        topology_programming_method="baseline",
    )

    assert set(simulation.wolf.logical_graph) == {"1", "2"}
