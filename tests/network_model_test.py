"""Self-contained tests for the modeled network graph."""

import json
from pathlib import Path

import networkx as nx

from onset.network_model import Network


def test_network_adds_router_and_client_metadata(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    topology.write_text(
        json.dumps(nx.adjacency_data(nx.path_graph(["1", "2"]))),
        encoding="utf-8",
    )

    graph = Network(str(topology)).graph

    for node in ("1", "2"):
        assert graph.nodes[node]["node_type"] == "Router"
        assert graph.nodes[node]["router_id"] == f"router_{node}"
        assert graph.nodes[node]["client_id"] == f"client_{node}"
        assert graph.nodes[f"client_{node}"]["router_id"] == f"router_{node}"
