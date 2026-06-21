"""Lightweight geographic helpers for optical reconfiguration timing."""

from __future__ import annotations

import math

import networkx as nx


def calc_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres."""
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    lat1_radians = math.radians(lat1)
    lat2_radians = math.radians(lat2)
    a = math.sin(delta_lat / 2) ** 2 + (
        math.sin(delta_lon / 2) ** 2 * math.cos(lat1_radians) * math.cos(lat2_radians)
    )
    return 6371 * 2 * math.asin(math.sqrt(a))


def get_reconfig_time(gml_file: str, circuits: list[object]) -> float:
    """Estimate the longest optical-link reconfiguration in seconds."""
    if not gml_file or not circuits:
        return 0.0
    graph = nx.read_gml(gml_file)
    reconfig_time = 0.0
    for circuit in circuits:
        if not isinstance(circuit, tuple | list) or len(circuit) != 2:
            continue
        source, target = (f"s{node}" for node in circuit)
        distance = calc_haversine(
            graph.nodes[source]["Latitude"],
            graph.nodes[source]["Longitude"],
            graph.nodes[target]["Latitude"],
            graph.nodes[target]["Longitude"],
        )
        reconfig_time = max(reconfig_time, math.ceil(distance / 80) / 10)
    return reconfig_time
