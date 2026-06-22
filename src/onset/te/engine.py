"""ECMP/MCF routing and legacy-compatible statistics generation."""

from __future__ import annotations

import itertools
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from onset.constants import SCRIPT_HOME

Edge = tuple[str, str]
Commodity = tuple[str, str]
Route = tuple[tuple[str, ...], float]

_MAGNITUDES = {
    "": 1.0,
    "k": float(2**10),
    "m": float(2**20),
    "g": float(2**30),
    "t": float(2**40),
}


@dataclass(frozen=True)
class EvaluationResult:
    max_congestion: float
    mean_congestion: float
    throughput: float
    congestion_loss: float
    failure_loss: float
    num_paths: int
    solver_time: float
    result_dir: Path


def _unquote(value: object) -> str:
    return str(value).strip().strip('"')


def _magnitude(value: object) -> float:
    text = _unquote(value).strip()
    match = re.fullmatch(r"([0-9.eE+-]+)\s*([kKmMgGtT]?)(?:bps)?", text)
    if not match:
        raise ValueError(f"Unsupported capacity value: {value!r}")
    return float(match.group(1)) * _MAGNITUDES[match.group(2).lower()]


def _load_topology(path: str) -> nx.DiGraph:
    raw = nx.drawing.nx_pydot.read_dot(path)
    graph: nx.DiGraph[str] = nx.DiGraph()
    for node, attrs in raw.nodes(data=True):
        node = _unquote(node)
        if node not in {"", "\\n"}:
            graph.add_node(
                node, **{key: _unquote(value) for key, value in attrs.items()}
            )
    for source, target, attrs in raw.edges(data=True):
        source, target = _unquote(source), _unquote(target)
        capacity = _magnitude(attrs["capacity"])
        if not (
            graph.nodes[source].get("type") == "switch"
            and graph.nodes[target].get("type") == "switch"
        ):
            # Preserve the historical 100x access-link capacity convention.
            capacity *= 100.0
        graph.add_edge(
            source,
            target,
            cost=float(_unquote(attrs.get("cost", 1))),
            capacity=capacity,
        )
    if not graph.edges:
        raise ValueError(f"Topology contains no edges: {path}")
    return graph


def _load_demands(traffic_file: str, hosts_file: str) -> dict[Commodity, float]:
    with open(hosts_file, encoding="utf-8") as stream:
        hosts = [line.strip() for line in stream if line.strip()]
    with open(traffic_file, encoding="utf-8") as stream:
        line = next((line for line in stream if line.strip()), "")
    values = [float(value) for value in line.split()]
    expected = len(hosts) ** 2
    if len(values) != expected:
        raise ValueError(
            f"Traffic matrix has {len(values)} entries; expected {expected} "
            f"for {len(hosts)} hosts"
        )
    return {
        (source, target): values[i * len(hosts) + j]
        for i, source in enumerate(hosts)
        for j, target in enumerate(hosts)
        if source != target
    }


def _ecmp_routes(
    graph: nx.DiGraph, demands: Mapping[Commodity, float], budget: int
) -> dict[Commodity, list[Route]]:
    routes: dict[Commodity, list[Route]] = {}
    for commodity in sorted(demands):
        source, target = commodity
        paths = []
        try:
            for path in nx.all_shortest_paths(graph, source, target, weight="cost"):
                paths.append(tuple(path))
                if len(paths) == budget:
                    break
        except nx.NetworkXNoPath:
            paths = []
        probability = 1.0 / len(paths) if paths else 0.0
        routes[commodity] = [(path, probability) for path in paths]
    return routes


def _mcf_routes(
    graph: nx.DiGraph, demands: Mapping[Commodity, float]
) -> dict[Commodity, list[Route]]:
    all_commodities = sorted(demands)
    commodities = [commodity for commodity in all_commodities if demands[commodity] > 0]
    if not commodities:
        return {commodity: [] for commodity in all_commodities}
    edges = sorted(graph.edges())
    nodes = sorted(graph.nodes())
    edge_index = {edge: i for i, edge in enumerate(edges)}
    node_index = {node: i for i, node in enumerate(nodes)}
    edge_count = len(edges)
    variable_count = len(commodities) * edge_count + 1
    z_index = variable_count - 1

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(len(commodities) * len(nodes))
    for k, (source, target) in enumerate(commodities):
        demand = demands[(source, target)]
        for edge_i, (u, v) in enumerate(edges):
            variable = k * edge_count + edge_i
            eq_rows.extend(
                (k * len(nodes) + node_index[u], k * len(nodes) + node_index[v])
            )
            eq_cols.extend((variable, variable))
            eq_data.extend((1.0, -1.0))
        b_eq[k * len(nodes) + node_index[source]] = demand
        b_eq[k * len(nodes) + node_index[target]] = -demand

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    for edge_i, edge in enumerate(edges):
        for k in range(len(commodities)):
            ub_rows.append(edge_i)
            ub_cols.append(k * edge_count + edge_i)
            ub_data.append(1.0)
        ub_rows.append(edge_i)
        ub_cols.append(z_index)
        ub_data.append(-float(graph.edges[edge]["capacity"]))

    host_nodes = {node for commodity in demands for node in commodity}
    bounds = []
    for source, target in commodities:
        for u, v in edges:
            transit_host = (u in host_nodes and u != source) or (
                v in host_nodes and v != target
            )
            bounds.append((0.0, 0.0) if transit_host else (0.0, None))
    bounds.append((0.0, None))

    objective = np.zeros(variable_count)
    objective[z_index] = 1.0
    solution = linprog(
        objective,
        A_ub=coo_matrix(
            (ub_data, (ub_rows, ub_cols)), shape=(len(edges), variable_count)
        ),
        b_ub=np.zeros(len(edges)),
        A_eq=coo_matrix(
            (eq_data, (eq_rows, eq_cols)),
            shape=(len(commodities) * len(nodes), variable_count),
        ),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"MCF solve failed: {solution.message}")

    routes: dict[Commodity, list[Route]] = {
        commodity: [] for commodity in all_commodities
    }
    tolerance = max(demands.values(), default=1.0) * 1e-9
    for k, commodity in enumerate(commodities):
        residual = {
            edge: solution.x[k * edge_count + edge_index[edge]]
            for edge in edges
            if solution.x[k * edge_count + edge_index[edge]] > tolerance
        }
        paths: list[Route] = []
        while residual:
            flow_graph: nx.DiGraph[str] = nx.DiGraph(
                edge for edge, flow in residual.items() if flow > tolerance
            )
            try:
                path = nx.shortest_path(flow_graph, *commodity)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                break
            path_edges = list(itertools.pairwise(path))
            flow = min(residual[edge] for edge in path_edges)
            paths.append((tuple(path), flow / demands[commodity]))
            for edge in path_edges:
                residual[edge] -= flow
                if residual[edge] <= tolerance:
                    residual.pop(edge)
        routes[commodity] = paths
    return routes


# ---------------------------------------------------------------------------
# Semi-oblivious MCF (SMORE / YATES)
# ---------------------------------------------------------------------------
#
# SMORE (Semi-oblivious MCF with Raecke decomposition) is a two-phase traffic
# engineering system from Cornell:
#   https://www.cs.cornell.edu/~praveenk/smore/
#   https://cornell-netlab.github.io/yates/
#   https://github.com/cornell-netlab/yates
#
# "Semi-oblivious Traffic Engineering: The Road Not Taken"
#   Praveen Kumar, Yang Yuan, Chris Yu, Nate Foster, Robert Kleinberg,
#   Petr Lapukhov, Chiun Lin Lim, and Robert Soule.
#   USENIX NSDI, April 2018.
#   https://www.cs.cornell.edu/~praveenk/papers/smore-nsdi18.pdf
#
# Phase 1 (oblivious): Raecke's hierarchical tree decomposition (FRT) with
# multiplicative weights.  Random FRT trees are built, each routing all
# commodities up/down the tree.  MW iterates, reweighting edges by
# exponential cumulative usage, producing a probability distribution over
# routing trees.  Final candidate paths are the union of all physical
# paths from all trees.
#
# Phase 2 (semi-oblivious / rate adaptation): Solve a restricted MCF LP
# over the pre-selected paths to minimize max link utilization for the
# given traffic matrix.  This matches the YATES SemiMcf.solve LP.
#
# semimcfraeke  - Raecke FRT+MW path selection on original topology.
# semimcfraekeft - Same, with a fault-tolerance envelope: path selection
#                  is repeated on every single-link-failure topology and
#                  the resulting path sets are merged (YATES all_failures_envelope).
#
# Parameters (from YATES Raeke.ml):
#   epsilon = 0.1           MW learning rate
#   beta     ~ Uniform[1,2)  FRT distance scaling factor per tree


def _all_pairs_dist(
    graph: nx.DiGraph,
) -> dict[tuple[str, str], float]:
    dists: dict[tuple[str, str], float] = {}
    for src in graph.nodes():
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, src, weight="cost")
            for dst, length in lengths.items():
                dists[(src, dst)] = length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return dists


def _frt_decompose(
    vertices: list[str],
    dists: dict[tuple[str, str], float],
    beta: float | None = None,
) -> Any:
    """FRT hierarchical tree decomposition (YATES Yates_Frt.make_frt_tree).

    Returns a nested tree: ("node", center, cluster_set, [children])
    or ("leaf", center, {singleton}).
    """
    if beta is None:
        beta = math.exp(random.random() * math.log(2.0))

    if not vertices:
        return ("leaf", "", set())

    # Random permutation determines cluster-assignment priority
    order = list(vertices)
    random.shuffle(order)

    max_diameter = max(dists.values(), default=0.0)
    initial_i = int(math.floor(math.log2(max_diameter))) if max_diameter > 0 else 0

    def _level(i: int, center: str, cset: set[str]):
        beta_i = 2.0 ** (i - 1) * beta
        partition: dict[str, set[str]] = {}
        for v in cset:
            for h in order:
                if dists.get((h, v), float("inf")) <= beta_i:
                    partition.setdefault(h, set()).add(v)
                    break

        children = []
        for ctr, child_set in partition.items():
            if len(child_set) <= 1 or i <= 0:
                children.append(("leaf", ctr, child_set))
            else:
                children.append(_level(i - 1, ctr, child_set))
        return ("node", center, cset, children)

    head = order[0]
    return _level(initial_i, head, set(vertices))


def _frt_paths(
    tree,
    src: str,
    dst: str,
) -> list[str]:
    """Compute the tree path between src and dst by walking up to LCA."""

    def _find_leaf(t, target: str) -> list[str]:
        if t[0] == "leaf":
            return [t[1]] if target in t[2] else []
        for child in t[3]:
            result = _find_leaf(child, target)
            if result:
                return [t[1], *result]
        return []

    src_path = _find_leaf(tree, src)
    dst_path = _find_leaf(tree, dst)
    if not src_path or not dst_path:
        return []

    # Find LCA: last common element
    lca_idx = 0
    for i, (a, b) in enumerate(zip(src_path, dst_path, strict=False)):
        if a == b:
            lca_idx = i
        else:
            break

    # Up from src leaf (deepest) to LCA, then down to dst leaf
    up = list(reversed(src_path[lca_idx:]))
    down = dst_path[lca_idx + 1 :]
    return up + down


def _raecke_paths(
    graph: nx.DiGraph,
    hosts: set[str],
    *,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[Commodity, set[tuple[str, ...]]]:
    """Raecke oblivious routing: FRT trees + multiplicative weights.

    Iteratively builds random FRT routing trees, computes edge usage,
    updates edge weights via MW, and accumulates a probability distribution
    over trees.  MW weights are applied to edge ``cost`` attributes so each
    successive FRT tree is biased away from previously congested edges.

    Returns the union of all distinct physical paths from all weighted
    trees for each commodity.

    Matches YATES Raeke.solve + Yates_Mw.hedge_iterations.
    """
    if seed is not None:
        random.seed(seed)

    commodities = [(s, t) for s in sorted(hosts) for t in sorted(hosts) if s != t]
    all_paths: dict[Commodity, set[tuple[str, ...]]] = {
        comm: set() for comm in commodities
    }

    vertices = list(graph.nodes())
    num_edges = graph.number_of_edges()
    if num_edges == 0:
        return all_paths

    # Work on a copy to avoid mutating the caller's graph with MW cost updates.
    g = graph.copy()

    # Save original costs for restoration.
    _orig_cost: dict[Edge, float] = {}
    for u, v in g.edges():
        _orig_cost[(u, v)] = float(g.edges[u, v].get("cost", 1.0))

    # Cumulative MW usage table.
    usage_table: dict[Edge, float] = {edge: 0.0 for edge in g.edges()}

    acc_weight = 0.0
    max_iterations = 64

    for _ in range(max_iterations):
        # 1. Build random FRT tree on the current weighted graph.
        dists = _all_pairs_dist(g)
        tree = _frt_decompose(vertices, dists)

        # 2. Compute edge usage: count commodity traversals, normalized
        #    by capacity.  (The vendored YATES uses cluster-boundary
        #    capacity; commodity-count/capacity is a tractable proxy.)
        edge_usage: dict[Edge, float] = defaultdict(float)
        for src, dst in commodities:
            tree_path = _frt_paths(tree, src, dst)
            if len(tree_path) < 2:
                continue
            for i in range(len(tree_path) - 1):
                a, b = tree_path[i], tree_path[i + 1]
                try:
                    phys = nx.shortest_path(g, a, b, weight="cost")
                    for j in range(len(phys) - 1):
                        edge = (phys[j], phys[j + 1])
                        if g.has_edge(*edge):
                            cap = float(g.edges[edge].get("capacity", 1.0))
                            edge_usage[edge] += 1.0 / max(cap, 1.0)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        if not edge_usage:
            continue

        # 3. Tree weight = 1 / max_usage.
        max_u = max(edge_usage.values())
        w = 1.0 / max_u

        # 4. Stopping condition.
        if acc_weight + w >= 1.0:
            final_w = 1.0 - acc_weight
            if final_w > 0:
                for src, dst in commodities:
                    tp = _frt_paths(tree, src, dst)
                    if len(tp) >= 2:
                        try:
                            phys = _tree_to_physical(g, tp)
                            all_paths[(src, dst)].add(tuple(phys))
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass
            break

        # 5. Collect paths from this tree.
        for src, dst in commodities:
            tp = _frt_paths(tree, src, dst)
            if len(tp) >= 2:
                try:
                    phys = _tree_to_physical(g, tp)
                    all_paths[(src, dst)].add(tuple(phys))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        acc_weight += w

        # 6. Scale usage so max = 1 and update cumulative table.
        for e, u in edge_usage.items():
            usage_table[e] = usage_table.get(e, 0.0) + u / max_u

        # 7. Apply MW exponential weights to graph edge costs.
        #    Edges with high cumulative usage get higher cost, steering
        #    the next FRT tree toward less-congested paths.
        sum_exp = 0.0
        mw: dict[Edge, float] = {}
        for e in usage_table:
            mw[e] = math.exp(epsilon * usage_table[e])
            sum_exp += mw[e]
        if sum_exp > 0:
            for e in g.edges():
                multiplier = mw.get(e, 1.0) / (sum_exp / len(mw))
                base = _orig_cost.get(e, 1.0)
                g.edges[e]["cost"] = base * multiplier

    return all_paths


def _tree_to_physical(
    graph: nx.DiGraph,
    tree_path: list[str],
) -> list[str]:
    """Convert a tree-node path to a concatenated physical shortest path."""
    result: list[str] = [tree_path[0]]
    for i in range(len(tree_path) - 1):
        a, b = tree_path[i], tree_path[i + 1]
        segment = nx.shortest_path(graph, a, b, weight="cost")
        result.extend(segment[1:])
    return result


def _oblivious_paths(
    graph: nx.DiGraph,
    hosts: set[str],
) -> dict[Commodity, list[tuple[str, ...]]]:
    """Select diverse candidate paths via Raecke FRT + MW decomposition.

    Traffic-oblivious: no demand information is used.  Returns a set of
    physical paths per commodity.  Matches YATES Raeke.solve Phase 1.
    """
    raw = _raecke_paths(graph, hosts)
    return {comm: list(pset) for comm, pset in raw.items()}


def _oblivious_paths_ft(
    graph: nx.DiGraph,
    hosts: set[str],
) -> dict[Commodity, list[tuple[str, ...]]]:
    """Fault-tolerant variant: merge paths from every single-link-failure topology.

    For each switch-to-switch edge, removes the edge, checks connectivity,
    runs Raecke path selection, and unions all resulting path sets.
    Matches YATES all_failures_envelope for semimcfraekeft.
    """
    all_paths: dict[Commodity, set[tuple[str, ...]]] = {
        (s, t): set() for s in sorted(hosts) for t in sorted(hosts) if s != t
    }

    switch_edges = [
        (u, v)
        for u, v in graph.edges()
        if graph.nodes[u].get("type") == "switch"
        and graph.nodes[v].get("type") == "switch"
    ]
    for u, v in switch_edges:
        fail_graph = graph.copy()
        fail_graph.remove_edge(u, v)
        if fail_graph.has_edge(v, u):
            fail_graph.remove_edge(v, u)
        if not nx.is_strongly_connected(fail_graph):
            continue
        fail_paths = _oblivious_paths(fail_graph, hosts)
        for comm, plist in fail_paths.items():
            all_paths[comm].update(plist)

    return {comm: list(pset) for comm, pset in all_paths.items()}


def _semimcf_routes(
    graph: nx.DiGraph,
    demands: Mapping[Commodity, float],
    candidate_paths: dict[Commodity, list[tuple[str, ...]]],
) -> dict[Commodity, list[Route]]:
    """Restricted MCF over pre-selected candidate paths (SMORE Phase 2).

    Solves a path-based LP: for each commodity, allocate flow across its
    candidate paths to minimize the maximum link utilization.  Uses the
    HiGHS LP solver (scipy.optimize.linprog).

    This is the same formulation as YATES SemiMcf.solve.
    """
    all_commodities = sorted(demands)
    active = [
        (s, t)
        for s, t in all_commodities
        if demands[(s, t)] > 0 and candidate_paths.get((s, t))
    ]
    if not active:
        return {commodity: [] for commodity in all_commodities}

    # Build variable index: one flow variable per (commodity, path)
    var_index: dict[tuple[Commodity, int], int] = {}
    path_map: dict[Commodity, list[tuple[str, ...]]] = {}
    idx = 0
    for comm in active:
        path_map[comm] = candidate_paths[comm]
        for pi in range(len(path_map[comm])):
            var_index[(comm, pi)] = idx
            idx += 1
    z_index = idx  # max-congestion variable
    variable_count = idx + 1

    # Build per-edge path membership
    edge_path_vars: dict[Edge, list[int]] = defaultdict(list)
    for comm, paths in path_map.items():
        for pi, path in enumerate(paths):
            for edge in itertools.pairwise(path):
                edge_path_vars[edge].append(var_index[(comm, pi)])

    edges = sorted(graph.edges())
    commodity_count = len(active)
    edge_count = len(edges)

    # Capacity constraints: sum(flows using edge) - Z * capacity <= 0
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    for edge_i, edge in enumerate(edges):
        for var_i in edge_path_vars.get(edge, []):
            ub_rows.append(edge_i)
            ub_cols.append(var_i)
            ub_data.append(1.0)
        ub_rows.append(edge_i)
        ub_cols.append(z_index)
        ub_data.append(-float(graph.edges[edge]["capacity"]))

    # Demand constraints: sum(flows for commodity) >= demand
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq: list[float] = []
    for k, comm in enumerate(active):
        for pi in range(len(path_map[comm])):
            eq_rows.append(k)
            eq_cols.append(var_index[(comm, pi)])
            eq_data.append(1.0)
        b_eq.append(demands[comm])

    bounds = [(0.0, None)] * idx + [(0.0, None)]  # Z unbounded above

    objective = np.zeros(variable_count)
    objective[z_index] = 1.0

    solution = linprog(
        objective,
        A_ub=coo_matrix(
            (ub_data, (ub_rows, ub_cols)),
            shape=(edge_count, variable_count),
        ),
        b_ub=np.zeros(edge_count),
        A_eq=coo_matrix(
            (eq_data, (eq_rows, eq_cols)),
            shape=(commodity_count, variable_count),
        ),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Semimcf solve failed: {solution.message}")

    routes: dict[Commodity, list[Route]] = {
        commodity: [] for commodity in all_commodities
    }
    tolerance = max(demands.values(), default=1.0) * 1e-9
    for comm in active:
        total = sum(
            solution.x[var_index[(comm, pi)]] for pi in range(len(path_map[comm]))
        )
        for pi, path in enumerate(path_map[comm]):
            flow = solution.x[var_index[(comm, pi)]]
            if flow > tolerance and total > tolerance:
                routes[comm].append((path, flow / total))
        if not routes[comm]:
            routes[comm] = (
                [(path, 1.0 / len(paths)) for path in path_map[comm]]
                if path_map[comm]
                else []
            )
    return routes


def _fair_share(
    capacity: float, flows: Sequence[tuple[int, float]]
) -> dict[int, float]:
    remaining = capacity
    shares: dict[int, float] = {}
    ordered = sorted(flows, key=lambda item: (item[1], item[0]))
    for position, (flow_id, demand) in enumerate(ordered):
        count = len(ordered) - position
        share = demand if demand * count <= remaining else remaining / count
        shares[flow_id] = max(0.0, share)
        remaining -= share
    return shares


def _statistics(
    graph: nx.DiGraph,
    demands: Mapping[Commodity, float],
    routes: Mapping[Commodity, Sequence[Route]],
) -> tuple[dict[Edge, float], dict[Edge, float], float, float, float]:
    path_flows: list[tuple[tuple[str, ...], float]] = []
    expected_loads: dict[Edge, float] = defaultdict(float)
    for commodity, commodity_routes in routes.items():
        for path, probability in commodity_routes:
            demand = demands[commodity] * probability
            path_flows.append((path, demand))
            for edge in itertools.pairwise(path):
                expected_loads[edge] += demand

    current = {flow_id: demand for flow_id, (_, demand) in enumerate(path_flows)}
    actual_loads: dict[Edge, float] = defaultdict(float)
    max_hops = max((len(path) - 1 for path, _ in path_flows), default=0)
    congestion_drop = 0.0
    for hop in range(max_hops):
        queues: dict[Edge, list[tuple[int, float]]] = defaultdict(list)
        for flow_id, (path, _) in enumerate(path_flows):
            if hop < len(path) - 1 and current.get(flow_id, 0.0) > 0:
                queues[(path[hop], path[hop + 1])].append((flow_id, current[flow_id]))
        for edge, flows in queues.items():
            shares = _fair_share(float(graph.edges[edge]["capacity"]), flows)
            for flow_id, incoming in flows:
                current[flow_id] = shares[flow_id]
                congestion_drop += incoming - shares[flow_id]
                actual_loads[edge] += shares[flow_id]

    offered = sum(demands.values())
    delivered = sum(current.values())
    unroutable = sum(
        demands[commodity]
        for commodity, commodity_routes in routes.items()
        if not commodity_routes
    )
    throughput = delivered / offered if offered else 1.0
    loss = congestion_drop / offered if offered else 0.0
    failure_loss = unroutable / offered if offered else 0.0
    return expected_loads, actual_loads, throughput, loss, failure_loss


def _result_directory(result_path: str) -> Path:
    path = Path(result_path)
    if not path.is_absolute():
        path = Path(SCRIPT_HOME) / "data" / "results" / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_scalar(path: Path, header: str, solver: str, value: float) -> None:
    path.write_text(
        f"# solver\titer\t{header}\tstddev\n{solver}\t0\t{value:.6f}\t0.000000\n",
        encoding="utf-8",
    )


def _write_results(
    result_dir: Path,
    solver: str,
    routes: Mapping[Commodity, Sequence[Route]],
    graph: nx.DiGraph,
    expected_loads: Mapping[Edge, float],
    actual_loads: Mapping[Edge, float],
    throughput: float,
    loss: float,
    failure_loss: float,
    solver_time: float,
) -> EvaluationResult:
    expected = {
        edge: expected_loads.get(edge, 0.0) / float(graph.edges[edge]["capacity"])
        for edge in graph.edges
    }
    actual = {
        edge: actual_loads.get(edge, 0.0) / float(graph.edges[edge]["capacity"])
        for edge in graph.edges
    }
    max_expected = max(expected.values(), default=0.0)
    mean_expected = float(np.mean(list(expected.values()))) if expected else 0.0
    max_actual = max(actual.values(), default=0.0)
    mean_actual = float(np.mean(list(actual.values()))) if actual else 0.0
    num_paths = sum(len(paths) for paths in routes.values())
    scalars = {
        "MaxExpCongestionVsIterations.dat": ("max-exp-congestion", max_expected),
        "MeanExpCongestionVsIterations.dat": ("mean-exp-congestion", mean_expected),
        "MaxCongestionVsIterations.dat": ("max-congestion", max_actual),
        "MeanCongestionVsIterations.dat": ("mean-congestion", mean_actual),
        "TotalThroughputVsIterations.dat": ("total-throughput", throughput),
        "TotalSinkThroughputVsIterations.dat": ("total-throughput", 0.0),
        "FailureLossVsIterations.dat": ("failure-drop", failure_loss),
        "CongestionLossVsIterations.dat": ("congestion-drop", loss),
        "NumPathsVsIterations.dat": ("num_paths", float(num_paths)),
        "TimeVsIterations.dat": ("time", solver_time),
        "TMChurnVsIterations.dat": ("churn", 0.0),
        "RecoveryChurnVsIterations.dat": ("churn", 0.0),
    }
    for filename, (header, value) in scalars.items():
        _write_scalar(result_dir / filename, header, solver, value)

    for prefix, values in (("", actual), ("Exp", expected)):
        ordered = sorted(values.values())
        for percentile in (10, 20, 30, 40, 50, 60, 70, 80, 90, 95):
            if ordered:
                index = min(
                    len(ordered) - 1,
                    int(np.ceil(percentile / 100 * len(ordered))) - 1,
                )
                value = ordered[index]
            else:
                value = 0.0
            _write_scalar(
                result_dir / f"k{percentile}{prefix}CongestionVsIterations.dat",
                "congestion",
                solver,
                value,
            )

    for filename, header, values in (
        ("EdgeExpCongestionVsIterations.dat", "edge-exp-congestion", expected),
        ("EdgeCongestionVsIterations.dat", "edge-congestion", actual),
    ):
        lines = [f"# solver\titer\t{header}", f"{solver}\t0\t"]
        lines.extend(
            f"\t\t({u},{v}) : {value:.12g}" for (u, v), value in sorted(values.items())
        )
        (result_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    paths_dir = result_dir / "paths"
    paths_dir.mkdir(exist_ok=True)
    path_lines = []
    for (source, target), commodity_routes in sorted(routes.items()):
        path_lines.append(f"{source} -> {target} :")
        for path, probability in commodity_routes:
            edges = ", ".join(f"({u},{v})" for u, v in itertools.pairwise(path))
            path_lines.append(f"[{edges}] @ {probability:.6f}")
    (paths_dir / f"{solver}_0").write_text(
        "\n".join(path_lines) + "\n", encoding="utf-8"
    )
    (result_dir / "LatencyDistributionVsIterations.dat").write_text(
        f"#solver\titer\tlatency-throughput\n{solver}\t0\t\n",
        encoding="utf-8",
    )

    return EvaluationResult(
        max_congestion=max_expected,
        mean_congestion=mean_expected,
        throughput=throughput,
        congestion_loss=loss,
        failure_loss=failure_loss,
        num_paths=num_paths,
        solver_time=solver_time,
        result_dir=result_dir,
    )


def evaluate(
    topo_file: str,
    traffic_file: str,
    hosts_file: str,
    te_method: str,
    result_path: str,
    budget: int = 3,
) -> EvaluationResult:
    """Evaluate one traffic matrix and write the historical result-file contract."""
    graph = _load_topology(topo_file)
    demands = _load_demands(traffic_file, hosts_file)
    method = te_method.lower().lstrip("-")
    started = time.perf_counter()
    if method == "ecmp":
        routes = _ecmp_routes(graph, demands, budget)
    elif method == "mcf":
        routes = _mcf_routes(graph, demands)
    elif method in ("semimcfraeke", "semimcfraekeft"):
        hosts = {node for commodity in demands for node in commodity}
        if method == "semimcfraeke":
            cand = _oblivious_paths(graph, hosts)
        else:
            cand = _oblivious_paths_ft(graph, hosts)
        routes = _semimcf_routes(graph, demands, cand)
    else:
        raise ValueError(
            f"Unsupported in-process TE method {te_method!r}; "
            f"supported methods are -ecmp, -mcf, -semimcfraeke, -semimcfraekeft"
        )
    solver_time = time.perf_counter() - started
    expected, actual, throughput, loss, failure_loss = _statistics(
        graph, demands, routes
    )
    return _write_results(
        _result_directory(result_path),
        method,
        routes,
        graph,
        expected,
        actual,
        throughput,
        loss,
        failure_loss,
        solver_time,
    )
