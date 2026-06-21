#!/usr/bin/env python3
"""Run a generated, nonzero ECMP experiment through ``Simulation.perform_sim``."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx

import onset.simulator as simulator_module
import onset.te.engine as te_engine
import onset.validation as validation
from onset.simulator import Simulation


def main() -> None:
    with TemporaryDirectory(prefix="topology-programming-smoke-") as directory:
        root = Path(directory)
        for module in (simulator_module, te_engine, validation):
            module.SCRIPT_HOME = str(root)
        graph_dir = root / "data" / "graphs" / "gml"
        graph_dir.mkdir(parents=True)
        graph = nx.path_graph(["1", "2", "3"])
        nx.set_node_attributes(graph, 0.0, "Longitude")
        nx.set_node_attributes(graph, 0.0, "Latitude")
        nx.set_edge_attributes(graph, 100, "capacity")
        nx.write_gml(graph, graph_dir / "smoke.gml")
        traffic = root / "smoke.tm"
        traffic.write_text("0 0 50000000000 0 0 0 0 0 0\n", encoding="utf-8")

        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            result = Simulation(
                "smoke",
                3,
                "internal_ecmp",
                iterations=1,
                te_method="-ecmp",
                traffic_file=str(traffic),
                topology_programming_method="baseline",
            ).perform_sim(start_iter=1, end_iter=2)
        finally:
            os.chdir(previous_directory)

    assert isinstance(result, dict)
    assert result["Congestion"][0] > 0
    assert result["Loss"] == [0.0]
    assert result["Throughput"] == [1.0]
    print(f"Internal ECMP smoke passed: MLU={result['Congestion'][0]}")


if __name__ == "__main__":
    main()
