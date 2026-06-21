"""Self-contained tests for flow generation."""

import json
from pathlib import Path

import networkx as nx

from onset.utilities.flows import generate_flows


def test_generate_flows_uses_modeled_clients(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    topology.write_text(
        json.dumps(nx.adjacency_data(nx.path_graph(["1", "2", "3"]))),
        encoding="utf-8",
    )

    graph, flows = generate_flows(str(topology), 7, 7)

    assert {"client_1", "client_2", "client_3"}.issubset(graph)
    assert len(flows) == 6
    assert all(source.startswith("client_") for source, _, _ in flows)
    assert all(target.startswith("client_") for _, target, _ in flows)
    assert {volume for _, _, volume in flows} == {7}
