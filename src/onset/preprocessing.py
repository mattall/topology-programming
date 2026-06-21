"""
Gurobi-free preprocessing for topology optimization.

Extracted from optimization_two.py, this module handles all candidate
link selection, tunnel/path computation, and demand loading.  It never
imports gurobipy and can be safely imported in any environment.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from itertools import permutations
from math import floor, log10
from multiprocessing import Manager, Pool
from time import time
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from onset.base_types import DopplerProblem, _PathProblemData
from onset.constants import SCRIPT_HOME
from onset.utilities.graph import (
    astar_path_generator,
    link_on_path,
)
from onset.utilities.logger import logger
from onset.utilities.plot_reconfig_time import calc_haversine
from onset.utilities.sysUtils import file_writer


# ---------------------------------------------------------------------------
# Worker functions (multiprocessing-compatible, must be at module level)
# ---------------------------------------------------------------------------


def _shortest_path_worker(
    source: str,
    target: str,
    G: nx.Graph,
    original_tunnel_list: List[List[str]],
    tunnel_list: List[List[str]],
    is_done,
):
    """Worker for parallel shortest-path enumeration on a super-graph."""
    if not original_tunnel_list:
        return (source, target)
    core_length = len(original_tunnel_list[0])
    cutoff = max(core_length, 4)
    max_paths = core_length**2

    paths: List[List[str]] = []
    path_generator = nx.shortest_simple_paths(G, source, target)
    for path in path_generator:
        if len(path) > cutoff:
            break
        paths.append(path)
        paths.append(list(reversed(path)))
        if len(paths) >= max_paths:
            break

    # Ensure original tunnels are included
    for path in original_tunnel_list:
        if path not in paths:
            paths.append(path)
            paths.append(list(reversed(path)))

    tunnel_list.extend(paths)
    is_done.value = True
    return (source, target)


def _astar_path_worker(
    source: str,
    target: str,
    G: nx.Graph,
    original_tunnel_list: List[List[str]],
    tunnel_list: List[List[str]],
    is_done,
):
    """Worker for parallel A*-based path enumeration on a super-graph."""
    if not original_tunnel_list:
        return (source, target)
    core_length = len(original_tunnel_list[0])
    cutoff = max(core_length, 4)
    max_paths = core_length**2

    def dist(u: str, v: str) -> float:
        u_lat = G.nodes[u]["Latitude"]
        u_long = G.nodes[u]["Longitude"]
        v_lat = G.nodes[v]["Latitude"]
        v_long = G.nodes[v]["Longitude"]
        return calc_haversine(u_lat, u_long, v_lat, v_long)

    paths: List[List[str]] = []
    path_generator = astar_path_generator(G, source, target, dist)
    for path in path_generator:
        if len(path) > cutoff:
            break
        paths.append(path)
        paths.append(list(reversed(path)))
        if len(paths) >= max_paths:
            break

    for path in original_tunnel_list:
        if path not in paths:
            paths.append(path)
            paths.append(list(reversed(path)))

    tunnel_list.extend(paths)
    is_done.value = True
    return (source, target)


# ---------------------------------------------------------------------------
# Demand loading
# ---------------------------------------------------------------------------


def load_demand_from_file(
    demand_matrix_file: str,
    scale_down_factor: float = 1.0,
    dynamic_scale_down: bool = False,
) -> Dict[Tuple[str, str], float]:
    """Load a demand matrix from a text file.

    Returns a dict mapping (source, target) -> float demand value.
    """
    matrix = np.loadtxt(demand_matrix_file)
    n = len(matrix)
    demand: Dict[Tuple[str, str], float] = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                val = float(matrix[i][j]) / scale_down_factor
                if dynamic_scale_down:
                    # overridden below
                    pass
                demand[(str(i), str(j))] = val

    if dynamic_scale_down:
        nonzero = [v for v in demand.values() if v > 0]
        if nonzero:
            dynamic_factor = 10 ** floor(log10(min(nonzero)))
            demand = {k: v / dynamic_factor for k, v in demand.items()}

    return demand


# ---------------------------------------------------------------------------
# Candidate link selection
# ---------------------------------------------------------------------------


def compute_candidate_links(
    G: nx.Graph,
    candidate_set: str = "max",
    max_distance: float = float("inf"),
    liberal_p: float = 0.1,
) -> List[Tuple[str, str]]:
    """Compute a list of candidate (undirected) shortcut links.

    Parameters
    ----------
    G : nx.Graph
        The physical/core graph (nodes have 'Longitude'/'Latitude' attrs).
    candidate_set : str
        One of 'max', 'liberal', or 'conservative'.
    max_distance : float
        Maximum geographic distance (km) for a candidate link (only 'max').
    liberal_p : float
        Fraction of top-ranked edges to consider (only 'liberal'/'conservative').

    Returns
    -------
    List of (u, v) tuples, each in sorted order.
    """
    candidates: List[Tuple[str, str]] = []
    edges = set(G.edges)

    if candidate_set == "max":
        for source, target in permutations(G.nodes, 2):
            if (source, target) in edges or (target, source) in edges:
                continue
            try:
                shortest_paths = list(
                    nx.all_shortest_paths(G, source=source, target=target)
                )
            except nx.NetworkXNoPath:
                continue
            if not shortest_paths:
                continue
            shortest_len = len(shortest_paths[0])
            if shortest_len != 3:  # exactly 2 hops, i.e., 3 nodes on path
                continue
            # Check geographic distance
            lat_s = G.nodes[source]["Latitude"]
            lon_s = G.nodes[source]["Longitude"]
            lat_t = G.nodes[target]["Latitude"]
            lon_t = G.nodes[target]["Longitude"]
            d = calc_haversine(lat_s, lon_s, lat_t, lon_t)
            if d <= max_distance:
                candidates.append(
                    tuple(sorted((source, target)))
                )

    elif candidate_set == "liberal":
        bc = nx.edge_betweenness_centrality(G)
        if bc:
            sorted_edges = sorted(bc, key=bc.get, reverse=True)
            top_n = max(1, int(len(sorted_edges) * liberal_p))
            for u, v in sorted_edges[:top_n]:
                u_neighbors = set(G.neighbors(u)) - {v}
                v_neighbors = set(G.neighbors(v)) - {u}
                for n_u in u_neighbors:
                    for n_v in v_neighbors:
                        if n_u != n_v:
                            cand = tuple(sorted((n_u, n_v)))
                            if cand not in edges:
                                candidates.append(cand)

    elif candidate_set == "conservative":
        bc = nx.edge_betweenness_centrality(G)
        if bc:
            sorted_edges = sorted(bc, key=bc.get, reverse=True)
            top_n = max(1, int(len(sorted_edges) * liberal_p))
            for u, v in sorted_edges[:top_n]:
                for n_u in set(G.neighbors(u)) - {v}:
                    cand = tuple(sorted((n_u, v)))
                    if cand not in edges and cand not in candidates:
                        candidates.append(cand)
                for n_v in set(G.neighbors(v)) - {u}:
                    cand = tuple(sorted((u, n_v)))
                    if cand not in edges and cand not in candidates:
                        candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# Super-graph construction
# ---------------------------------------------------------------------------


def build_super_graph(
    core_G: nx.Graph,
    candidate_links: List[Tuple[str, str]],
) -> nx.Graph:
    """Build the super-graph: core graph + candidate shortcut edges.

    Nodes retain 'Longitude' and 'Latitude' attributes from core_G.
    """
    super_graph = nx.Graph()
    for node in core_G.nodes:
        super_graph.add_node(
            node,
            Longitude=core_G.nodes[node].get("Longitude"),
            Latitude=core_G.nodes[node].get("Latitude"),
        )
    for u, v in core_G.edges:
        super_graph.add_edge(u, v)
    for u, v in candidate_links:
        super_graph.add_edge(u, v)
    return super_graph


# ---------------------------------------------------------------------------
# Path / tunnel computation
# ---------------------------------------------------------------------------


def find_original_tunnels(
    G: nx.Graph,
    ordered_node_pairs: Set[Tuple[str, str]],
) -> Tuple[List[List[str]], Dict[Tuple[str, str], List[List[str]]]]:
    """Compute shortest paths on the original graph (without candidate links).

    Returns (tunnel_list, tunnel_dict).
    """
    tunnel_list: List[List[str]] = []
    tunnel_dict: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)

    for s, t in ordered_node_pairs:
        try:
            gen = nx.shortest_simple_paths(G, s, t)
            count = 0
            for path in gen:
                tunnel_list.append(path)
                tunnel_dict[(s, t)].append(path)
                count += 1
                if count >= 6:
                    break
        except nx.NetworkXNoPath:
            pass
    return tunnel_list, tunnel_dict


def compute_tunnels(
    super_graph: nx.Graph,
    ordered_node_pairs: Set[Tuple[str, str]],
    original_tunnel_list: List[List[str]],
    original_tunnel_dict: Dict[Tuple[str, str], List[List[str]]],
    *,
    parallel: bool = False,
    use_astar: bool = False,
    network_name: str = "",
    use_cache: bool = False,
) -> Tuple[
    List[List[str]],
    Dict[Tuple[str, str], List[List[str]]],
    Dict[Tuple[str, str], List[List[str]]],
]:
    """Compute tunnels (paths) on the super-graph for all node pairs.

    Returns (tunnel_list, tunnel_dict, tunnel_tuple_dict).
    """
    # Try cache first
    if use_cache:
        cached = _load_cached_paths(network_name)
        if cached is not None:
            tunnel_list, tunnel_dict, tunnel_tuple_dict = cached
            return tunnel_list, tunnel_dict, tunnel_tuple_dict

    tunnel_list: List[List[str]] = []
    tunnel_dict: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)

    if parallel:
        manager = Manager()
        shared_list = manager.list()
        is_done_flags: Dict[Tuple[str, str], object] = {}

        work = []
        worker_fn = _astar_path_worker if use_astar else _shortest_path_worker
        for s, t in ordered_node_pairs:
            flag = manager.Value("b", False)
            is_done_flags[(s, t)] = flag
            original = original_tunnel_dict.get((s, t), [])
            work.append((s, t, super_graph, original, shared_list, flag))

        with Pool() as pool:
            start = time()
            results = pool.starmap_async(worker_fn, work)
            while not results.ready():
                results.wait(timeout=10)
                elapsed = time() - start
                done = sum(1 for f in is_done_flags.values() if f.value)
                logger.info(
                    "Path computation: %d/%d pairs done (%.0fs)",
                    done,
                    len(work),
                    elapsed,
                )
            results.get()

        for s, t in ordered_node_pairs:
            flag = is_done_flags[(s, t)]
            if flag.value:
                tunnel_dict[(s, t)] = [
                    p for p in shared_list
                    if p and p[0] == s and p[-1] == t
                ]
                tunnel_list.extend(tunnel_dict[(s, t)])

    else:
        shortest_path_lengths: Dict[Tuple[str, str], int] = {}
        for s, t in ordered_node_pairs:
            try:
                gen = nx.shortest_simple_paths(super_graph, s, t)
                first_path = next(gen)
                shortest_path_lengths[(s, t)] = len(first_path)
            except (StopIteration, nx.NetworkXNoPath):
                shortest_path_lengths[(s, t)] = 4

        for s, t in ordered_node_pairs:
            cutoff = max(shortest_path_lengths.get((s, t), 4), 4)
            paths_s_t: List[List[str]] = []
            try:
                gen = nx.shortest_simple_paths(super_graph, s, t)
                for path in gen:
                    if len(path) > cutoff:
                        break
                    paths_s_t.append(path)
                    paths_s_t.append(list(reversed(path)))
            except nx.NetworkXNoPath:
                pass

            for path in original_tunnel_dict.get((s, t), []):
                if path not in paths_s_t:
                    paths_s_t.append(path)
                    paths_s_t.append(list(reversed(path)))

            tunnel_dict[(s, t)] = paths_s_t
            tunnel_list.extend(paths_s_t)

    # Build tunnel_tuple_dict (same content, plain dict)
    tunnel_tuple_dict: Dict[Tuple[str, str], List[List[str]]] = {}
    for (s, t), paths in tunnel_dict.items():
        tunnel_tuple_dict[(s, t)] = paths

    # Cache if requested
    if use_cache and network_name:
        _save_cached_paths(
            network_name, tunnel_list, tunnel_dict, tunnel_tuple_dict
        )

    return tunnel_list, tunnel_dict, tunnel_tuple_dict


def _load_cached_paths(network_name: str):
    """Load cached tunnel data from disk. Returns None if not found."""
    cache_dir = os.path.join(
        SCRIPT_HOME, "data", "paths", "optimization"
    )
    tunnel_list_file = os.path.join(
        cache_dir, f"{network_name}_tunnel_list.pickle"
    )
    tunnel_dict_file = os.path.join(
        cache_dir, f"{network_name}_tunnel_dict.pickle"
    )
    tunnel_tuple_file = os.path.join(
        cache_dir, f"{network_name}_tunnel_tuple_dict.pickle"
    )

    if os.path.exists(tunnel_list_file) and os.path.exists(
        tunnel_dict_file
    ):
        with open(tunnel_list_file, "rb") as f:
            tunnel_list = pickle.load(f)
        with open(tunnel_dict_file, "rb") as f:
            tunnel_dict = pickle.load(f)
        if os.path.exists(tunnel_tuple_file):
            with open(tunnel_tuple_file, "rb") as f:
                tunnel_tuple_dict = pickle.load(f)
        else:
            tunnel_tuple_dict = {
                k: v for k, v in tunnel_dict.items()
            }
        logger.info("Loaded cached paths for %s", network_name)
        return tunnel_list, tunnel_dict, tunnel_tuple_dict
    return None


def _save_cached_paths(
    network_name: str,
    tunnel_list: List[List[str]],
    tunnel_dict: Dict[Tuple[str, str], List[List[str]]],
    tunnel_tuple_dict: Dict[Tuple[str, str], List[List[str]]],
):
    """Save tunnel data to disk for future reuse."""
    cache_dir = os.path.join(
        SCRIPT_HOME, "data", "paths", "optimization"
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(
        os.path.join(cache_dir, f"{network_name}_tunnel_list.pickle"), "wb"
    ) as f:
        pickle.dump(list(tunnel_list), f)
    with open(
        os.path.join(cache_dir, f"{network_name}_tunnel_dict.pickle"), "wb"
    ) as f:
        pickle.dump(dict(tunnel_dict), f)
    with open(
        os.path.join(
            cache_dir, f"{network_name}_tunnel_tuple_dict.pickle"
        ),
        "wb",
    ) as f:
        pickle.dump(dict(tunnel_tuple_dict), f)
    logger.info("Saved cached paths for %s", network_name)


def save_original_paths(
    network: str,
    tunnel_list: List[List[str]],
    tunnel_dict: Dict[Tuple[str, str], List[List[str]]],
):
    """Save original (pre-candidate) tunnel data to JSON on disk."""
    import json as _json

    cache_dir = os.path.join(
        SCRIPT_HOME, "data", "paths", "optimization"
    )
    os.makedirs(cache_dir, exist_ok=True)

    serializable_list = [
        list(p) for p in tunnel_list
    ]
    serializable_dict = {
        f"{s}->{t}": [list(p) for p in paths]
        for (s, t), paths in tunnel_dict.items()
    }

    list_file = os.path.join(cache_dir, f"{network}_original.json")
    dict_file = os.path.join(cache_dir, f"{network}_original_dict.json")

    with open(list_file, "w") as f:
        _json.dump({"list": serializable_list}, f, indent=2)
    with open(dict_file, "w") as f:
        _json.dump(serializable_dict, f, indent=2)


def load_original_paths(
    network: str,
) -> Tuple[List[List[str]], Dict[Tuple[str, str], List[List[str]]]]:
    """Load original tunnel data from disk. Returns ([], {}) if missing."""
    import json as _json

    list_file = os.path.join(
        SCRIPT_HOME, "data", "paths", "optimization",
        f"{network}_original.json",
    )
    dict_file = os.path.join(
        SCRIPT_HOME, "data", "paths", "optimization",
        f"{network}_original_dict.json",
    )

    tunnel_list: List[List[str]] = []
    tunnel_dict: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)

    if os.path.exists(list_file):
        with open(list_file) as f:
            data = _json.load(f)
            tunnel_list = data.get("list", [])
    if os.path.exists(dict_file):
        with open(dict_file) as f:
            data = _json.load(f)
            for key, paths in data.items():
                s, t = key.split("->")
                tunnel_dict[(s, t)] = paths

    return tunnel_list, tunnel_dict


# ---------------------------------------------------------------------------
# Top-level preprocessing convenience
# ---------------------------------------------------------------------------


def preprocess_doppler(
    logical_graph: nx.Graph,
    base_graph: nx.Graph,
    demand_matrix_file: str,
    network_name: str,
    txp_count: Optional[Dict[str, int]] = None,
    *,
    candidate_set: str = "max",
    scale_down_factor: float = 1.0,
    dynamic_scale_down: bool = False,
    use_cache: bool = False,
    parallel_execution: bool = False,
    compute_paths: bool = True,
) -> Dict:
    """Run all Gurobi-free preprocessing and return a data dict.

    This is the main entry point for both the open and legacy backends.
    Returns a dictionary with all the data needed to construct a MILP
    model (super graph, candidate edges, tunnels, demand, etc.).

    The caller is responsible for constructing a DopplerProblem from
    these values plus any per-call overrides (top_k, time_limit, etc.).
    """
    # Demand
    demand_dict = load_demand_from_file(
        demand_matrix_file,
        scale_down_factor=scale_down_factor,
        dynamic_scale_down=dynamic_scale_down,
    )

    # Node ordering
    all_pairs = list(permutations(base_graph.nodes, 2))
    ordered_pairs = {
        tuple(sorted((s, t))) for (s, t) in all_pairs
    }

    # Transponder counts
    if txp_count is None:
        txp_count = {
            node: (len(list(base_graph.neighbors(node))) + 1)
            for node in base_graph.nodes
        }
    if isinstance(txp_count, dict):
        pass
    elif isinstance(txp_count, list):
        txp_count = dict(zip(base_graph.nodes, txp_count))

    # Candidate links
    candidate_links = compute_candidate_links(
        base_graph, candidate_set=candidate_set
    )

    # Super graph
    super_graph = build_super_graph(base_graph, candidate_links)

    # Original tunnels (on base graph, for fallback)
    orig_tunnel_list, orig_tunnel_dict = load_original_paths(network_name)
    if not orig_tunnel_list:
        orig_tunnel_list, orig_tunnel_dict = find_original_tunnels(
            base_graph, ordered_pairs
        )
        save_original_paths(network_name, orig_tunnel_list, orig_tunnel_dict)

    # Tunnels on super graph
    if compute_paths:
        tunnel_list, tunnel_dict, tunnel_tuple_dict = compute_tunnels(
            super_graph,
            ordered_pairs,
            orig_tunnel_list,
            orig_tunnel_dict,
            parallel=parallel_execution,
            network_name=network_name,
            use_cache=use_cache,
        )
    else:
        tunnel_list, tunnel_dict, tunnel_tuple_dict = [], {}, {}

    # Build canonical edge order (undirected, sorted)
    all_undirected_edges = sorted(
        set(
            tuple(sorted(e))
            for e in set(base_graph.edges) | set(candidate_links)
        )
    )

    return {
        "super_graph": super_graph,
        "candidate_links": candidate_links,
        "canonical_edge_order": all_undirected_edges,
        "legacy_edge_order": all_undirected_edges,
        "current_edges": frozenset(
            tuple(sorted(e)) for e in logical_graph.edges
        ),
        "demand_dict": demand_dict,
        "txp_count": txp_count,
        "tunnel_tuple_dict": tunnel_tuple_dict,
        "tunnel_dict": tunnel_dict,
        "tunnel_list": tunnel_list,
        "original_tunnel_list": orig_tunnel_list,
        "original_tunnel_dict": orig_tunnel_dict,
        "ordered_node_pairs": ordered_pairs,
        "all_node_pairs": all_pairs,
    }


# ---------------------------------------------------------------------------
# DopplerProblem factory
# ---------------------------------------------------------------------------


def build_doppler_problem(
    logical_graph,
    base_graph,
    demand_matrix_file: str,
    network_name: str,
    txp_count: Optional[Dict[str, int]] = None,
    *,
    candidate_set: str = "max",
    scale_down_factor: float = 1.0,
    congestion_threshold_upper_bound: float = 0.8,
    top_k: int = 100,
    optimizer_time_limit: float = 60.0,
    link_capacity: float = 100e9,
    use_cache: bool = False,
    parallel_execution: bool = False,
    compute_paths: bool = True,
    retain_commodity_flows: bool = False,
    method: str = "doppler",
) -> DopplerProblem:
    """Build a DopplerProblem from AlpWolf / simulator state.

    This is the single factory for constructing a fully-validated
    DopplerProblem from the raw graph, demand, and transponder data
    available at the simulation layer.  Used by Doppler, onset_v3,
    onset_v2, and onset optimization paths.

    Parameters
    ----------
    logical_graph : nx.Graph
        The current logical (IP-layer) topology.
    base_graph : nx.Graph
        The physical fiber topology (super-graph source).
    demand_matrix_file : str
        Path to the demand matrix text file.
    network_name : str
        Network identifier (used for cache keys).
    txp_count : dict, optional
        Per-node transponder counts.  Computed from base_graph if None.
    candidate_set : str
        Candidate link selection strategy.
    scale_down_factor : float
        Demand/capacity scaling divisor.
    congestion_threshold_upper_bound : float
        Maximum allowed MLU.
    top_k : int
        Number of solutions to enumerate.
    optimizer_time_limit : float
        Solver time budget in seconds.
    link_capacity : float
        Raw link capacity (bps).
    use_cache : bool
        Load cached tunnel data if available.
    parallel_execution : bool
        Compute tunnels in parallel.
    compute_paths : bool
        Whether to compute tunnel paths.
    retain_commodity_flows : bool
        If True, per-commodity flows are retained in solutions.
    method : str
        Optimization method: "doppler", "onset_v3", "onset_v2", "onset".
        Determines which solver and data structures are used.

    Returns
    -------
    DopplerProblem
        Fully validated, immutable problem description.
    """
    data = preprocess_doppler(
        logical_graph=logical_graph,
        base_graph=base_graph,
        demand_matrix_file=demand_matrix_file,
        network_name=network_name,
        txp_count=txp_count,
        candidate_set=candidate_set,
        scale_down_factor=scale_down_factor,
        use_cache=use_cache,
        parallel_execution=parallel_execution,
        compute_paths=compute_paths,
    )

    nodes = tuple(sorted(
        base_graph.nodes, key=lambda s: str(s).encode("utf-8")
    ))
    canonical_edges = tuple(
        tuple(sorted(e)) for e in data["canonical_edge_order"]
    )

    # Build tunnel_edge_sets: frozenset of directed edges per commodity
    tunnel_edge_sets = {}
    tunnel_tuple_dict = data.get("tunnel_tuple_dict", {})
    for (s, t), paths in tunnel_tuple_dict.items():
        directed_set = set()
        for path in paths:
            for i in range(len(path) - 1):
                directed_set.add((path[i], path[i + 1]))
        if directed_set:
            tunnel_edge_sets[(s, t)] = frozenset(directed_set)

    # Build path-based data for onset_v2 / onset formulations
    path_data = None
    if method in ("onset", "onset_v2"):
        tunnel_list = data["tunnel_list"]
        tunnel_dict = data["tunnel_dict"]
        candidate_links = data["candidate_links"]
        current_edges = data["current_edges"]
        super_graph = data["super_graph"]

        # path_list: tuple of paths, each path is a tuple of node IDs
        path_list = tuple(tuple(p) for p in tunnel_list)

        # commodity_to_paths: (s,t) -> tuple of path indices
        commodity_to_paths: Dict[Tuple[str, str], Tuple[int, ...]] = {}
        ordered_pairs = data["ordered_node_pairs"]
        for s, t in ordered_pairs:
            if (s, t) in data["demand_dict"] and data["demand_dict"][(s, t)] > 0:
                idxs = tuple(tunnel_dict.get((s, t), []))
                if idxs:
                    commodity_to_paths[(s, t)] = idxs

        # candidate_edge_indices: indices into canonical_edges for edges NOT in current
        cand_link_set = {tuple(sorted(e)) for e in candidate_links}
        candidate_edge_indices: Tuple[int, ...] = tuple(
            i for i, e in enumerate(canonical_edges) if e in cand_link_set
        )

        # path_candidate_map: per path, which local candidate indices (into
        # candidate_edge_indices, NOT global canonical_edges) are in this path
        cand_global_to_local = {gi: li for li, gi in enumerate(candidate_edge_indices)}
        path_candidate_map: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(
                cand_global_to_local[ci]
                for ci in candidate_edge_indices
                if any(
                    canonical_edges[ci] == tuple(sorted((path[j], path[j + 1])))
                    for j in range(len(path) - 1)
                )
            )
            for path in path_list
        )

        # supergraph_directed_edges: both directions of all canonical candidate edges
        supergraph_directed_edges: Tuple[Tuple[str, str], ...] = tuple(
            (u, v) for (u, v) in canonical_edges
        ) + tuple((v, u) for (u, v) in canonical_edges)

        dir_to_idx = {e: i for i, e in enumerate(supergraph_directed_edges)}

        # link_path_map: per supergraph directed edge, which path indices traverse it
        link_path_map: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(
                pi for pi, path in enumerate(path_list)
                if len(path) >= 2 and any(
                    (path[j], path[j + 1]) == de
                    for j in range(len(path) - 1)
                )
            )
            for de in supergraph_directed_edges
        )

        # core_edge_set: undirected physical-graph edges (onset_v2 only)
        if method == "onset_v2":
            core_edge_set: FrozenSet[Tuple[str, str]] = frozenset(
                tuple(sorted(e)) for e in base_graph.edges
            )
        else:
            core_edge_set = frozenset()

        path_data = _PathProblemData(
            path_list=path_list,
            commodity_to_paths=commodity_to_paths,
            candidate_edge_indices=candidate_edge_indices,
            path_candidate_map=path_candidate_map,
            supergraph_directed_edges=supergraph_directed_edges,
            link_path_map=link_path_map,
            core_edge_set=core_edge_set,
        )

    return DopplerProblem(
        canonical_node_order=nodes,
        canonical_candidate_edges=canonical_edges,
        legacy_candidate_edge_order=canonical_edges,
        current_edges=data["current_edges"],
        txp_count=data["txp_count"],
        demand=data["demand_dict"],
        tunnel_edge_sets=tunnel_edge_sets,
        link_capacity=link_capacity,
        scale_factor=scale_down_factor,
        congestion_threshold_upper_bound=congestion_threshold_upper_bound,
        top_k=top_k,
        optimizer_time_limit=optimizer_time_limit,
        retain_commodity_flows=retain_commodity_flows,
        path_data=path_data,
    )
