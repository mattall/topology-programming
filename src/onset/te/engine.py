"""TE routing: ECMP, MCF, and SMORE/Raecke semi-oblivious MCF."""

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
# SMORE (Semi-oblivious MCF with Raecke decomposition):
#   "Semi-oblivious Traffic Engineering: The Road Not Taken"
#   Praveen Kumar et al., USENIX NSDI, April 2018.
#   https://www.cs.cornell.edu/~praveenk/papers/smore-nsdi18.pdf
#
# Phase 1 (oblivious): Raecke's FRT tree decomposition with multiplicative
# weights.  Each MW iteration builds a random FRT tree, computes
# boundary-capacity edge usage (dimensionless, scale-invariant), and
# reweights graph edges via normalized exponential cumulative usage.
# Produces a probability-weighted scheme: each commodity gets a
# distribution over candidate physical paths.
#
# Phase 2 (semi-oblivious / rate adaptation): Restricted MCF LP over the
# pruned path set (top-k by probability) to minimize max link
# utilization for the current traffic matrix.
#
# The implementation targets YATES compatibility:
#   external/yates/lib/routing/Yates_Frt.ml
#   external/yates/lib/routing/Yates_Mw.ml
#   external/yates/lib/routing/Raeke.ml
#   external/yates/lib/routing/Util.ml
#   external/yates/lib/solvers/Helper.ml
#
# Parameters:
#   epsilon = 0.1           MW learning rate
#   beta     ~ Uniform[1,2)  FRT distance scaling factor per tree
#
# Type aliases
# ------------
# FRTNode = tuple[str, str, set[str], list]   # ("node"|"leaf", center, set, [children])
# WeightedScheme = dict[Commodity, dict[tuple[str, ...], float]]
#   commodity -> {physical_path: probability}


# Weighted scheme: commodity -> {physical_path: probability}
WeightedScheme = dict[tuple[str, str], dict[tuple[str, ...], float]]


def _all_pairs_paths(
    graph: nx.DiGraph,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], list[str]],
]:
    """All-pairs shortest-path distances and physical paths on *graph*."""
    dists: dict[tuple[str, str], float] = {}
    paths: dict[tuple[str, str], list[str]] = {}
    for src in graph.nodes():
        try:
            result: tuple[dict[str, float], dict[str, list[str]]] = (
                nx.single_source_dijkstra(  # type: ignore[assignment]
                    graph, src, weight="cost"
                )
            )
            for dst in result[1]:
                dists[(src, dst)] = result[0][dst]
                paths[(src, dst)] = result[1][dst]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return dists, paths


def _all_pairs_dist(
    graph: nx.DiGraph,
) -> dict[tuple[str, str], float]:
    dists, _ = _all_pairs_paths(graph)
    return dists


def _frt_decompose(
    vertices: list[str],
    dists: dict[tuple[str, str], float],
    beta: float | None = None,
    rng: random.Random | None = None,
) -> tuple:
    """FRT hierarchical tree decomposition (YATES Yates_Frt.make_frt_tree).

    Returns a nested tree: ("node", center, cluster_set, [children])
    or ("leaf", center, {singleton}).
    """
    if rng is None:
        rng = random.Random()

    if beta is None:
        beta = math.exp(rng.random() * math.log(2.0))

    if not vertices:
        return ("leaf", "", set())

    # Random permutation determines cluster-assignment priority
    order = list(vertices)
    rng.shuffle(order)

    max_diameter = max(dists.values(), default=0.0)
    initial_i = int(math.floor(math.log2(max_diameter))) if max_diameter > 0 else 0

    def _level(i: int, center: str, cset: set[str]) -> tuple:
        beta_i = 2.0 ** (i - 1) * beta
        partition: dict[str, set[str]] = {}
        for v in cset:
            for h in order:
                if dists.get((h, v), float("inf")) <= beta_i:
                    partition.setdefault(h, set()).add(v)
                    break

        children = []
        for ctr, child_set in partition.items():
            if len(child_set) == 1:
                leaf_center = next(iter(child_set))
                children.append(("leaf", leaf_center, child_set))
            else:
                children.append(_level(i - 1, ctr, child_set))
        return ("node", center, cset, children)

    head = order[0]
    return _level(initial_i, head, set(vertices))


def _frt_paths(
    tree: tuple,
    src: str,
    dst: str,
) -> list[str]:
    """Compute the tree path between src and dst by walking up to LCA."""

    def _find_leaf(t: tuple, target: str) -> list[str]:
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


def _boundary_capacity(
    graph: nx.DiGraph,
    cluster: set[str],
) -> float:
    """Sum of capacities of directed edges from inside *cluster* to outside."""
    total = 0.0
    for u in cluster:
        for v in graph.successors(u):
            if v not in cluster:
                total += float(graph.edges[u, v]["capacity"])
    return total


def _build_routing_usage(
    graph: nx.DiGraph,
    frt_tree: tuple,
    endpoints: set[str],
    phys_paths: dict[tuple[str, str], list[str]],
) -> tuple[dict[Edge, float], tuple | None, dict[tuple[str, str], list[str]]]:
    """Walk the FRT tree and compute YATES-compatible boundary-capacity usage.

    Two passes (matching Yates_Frt.ml generate_rt):
      1. Prune: keep only FRT subtrees that contain at least one endpoint.
      2. Charge: for each retained parent→child edge, compute the boundary
         capacity of the *pruned* child set, then charge it to every physical
         edge on the child→parent shortest path and its inverse (once each).

    Returns (usage_dimensionless, pruned_tree, path_table) where
    *usage_dimensionless* is accumulated B(C) / edge capacity.
    """
    # ---- pass 1: endpoint pruning ------------------------------------------

    def _prune(node: tuple) -> tuple | None:
        tag, center, cset = node[0], node[1], node[2]
        if tag == "leaf":
            my_eps = cset & endpoints
            return ("leaf", center, my_eps) if my_eps else None
        children = node[3]
        kept: list[tuple] = []
        for child in children:
            pruned = _prune(child)
            if pruned is not None:
                kept.append(pruned)
        my_eps = cset & endpoints
        if not my_eps and not kept:
            return None
        return ("node", center, my_eps, kept)

    pruned_root = _prune(frt_tree)
    if pruned_root is None:
        return {}, None, {}

    # ---- pass 2: boundary-capacity usage -----------------------------------

    usage_abs: dict[Edge, float] = defaultdict(float)
    path_table: dict[tuple[str, str], list[str]] = {}

    def _charge(node: tuple) -> None:
        tag, center, _cset = node[0], node[1], node[2]
        if tag == "leaf":
            return
        for child in node[3]:
            child_center = child[1]
            child_set = child[2]  # pruned endpoint set (YATES c_set)

            # Store both tree-edge directions in the path table
            # (needed for _frt_paths which walks up and down).
            for key in ((child_center, center), (center, child_center)):
                seg = phys_paths.get(key)
                if seg is not None:
                    path_table[key] = seg

            # Boundary capacity of the *pruned* child set
            bcap = _boundary_capacity(graph, child_set)

            # Charge only path_up (child → parent) — one mirrored charge
            path_up = phys_paths.get((child_center, center))
            if path_up is not None:
                for i in range(len(path_up) - 1):
                    e = (path_up[i], path_up[i + 1])
                    usage_abs[e] += bcap
                    inv = (path_up[i + 1], path_up[i])
                    if graph.has_edge(*inv):
                        usage_abs[inv] += bcap

            _charge(child)

    _charge(pruned_root)

    # Convert absolute boundary capacity to dimensionless usage
    usage: dict[Edge, float] = {}
    for e, abs_u in usage_abs.items():
        cap = float(graph.edges[e]["capacity"])
        usage[e] = abs_u / cap

    return usage, pruned_root, path_table


def _tree_to_physical(
    tree_path: list[str],
    path_table: dict[tuple[str, str], list[str]],
    graph: nx.DiGraph,
) -> list[str]:
    """Convert a tree-centre path to a physical path using *path_table*.

    Returns an empty list when any segment is unroutable (e.g. disconnected
    failure topology).
    """
    if not tree_path:
        return []
    result = [tree_path[0]]
    for i in range(len(tree_path) - 1):
        a, b = tree_path[i], tree_path[i + 1]
        seg = path_table.get((a, b))
        if seg is None:
            try:
                seg = nx.shortest_path(graph, a, b, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        result.extend(seg[1:])
    return result


def _normalize_scheme(scheme: WeightedScheme) -> WeightedScheme:
    """Ensure each commodity's path probabilities sum to 1."""
    result: WeightedScheme = {}
    for comm, pps in scheme.items():
        total = sum(pps.values())
        if total <= 0:
            result[comm] = dict(pps)
        else:
            result[comm] = {path: prob / total for path, prob in pps.items()}
    return result


def _prune_scheme(
    scheme: WeightedScheme,
    budget: int,
) -> dict[Commodity, list[tuple[str, ...]]]:
    """Keep at most *budget* highest-probability paths per commodity.

    Returns a mapping from commodity to a list of path tuples suitable
    for the restricted-MCF phase.
    """
    result: dict[Commodity, list[tuple[str, ...]]] = {}
    for comm, pps in scheme.items():
        sorted_paths = sorted(pps.items(), key=lambda kv: kv[1], reverse=True)
        top = sorted_paths[:budget]
        result[comm] = [path for path, _ in top]
    return result


def _raecke_paths(
    graph: nx.DiGraph,
    hosts: set[str],
    *,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> WeightedScheme:
    """Raecke oblivious routing: FRT trees + multiplicative weights.

    Iteratively builds random FRT routing trees, computes boundary-capacity
    edge usage (dimensionless, scale-invariant), and reweights graph edges
    via normalised exponential cumulative usage.  Produces a probability
    distribution over trees.

    Returns a weighted scheme: commodity → {physical_path: probability}.

    Matches YATES Raeke.solve + Yates_Mw.hedge_iterations.
    """
    rng = random.Random(seed)

    commodities = [(s, t) for s in sorted(hosts) for t in sorted(hosts) if s != t]
    scheme: WeightedScheme = {comm: {} for comm in commodities}

    vertices = list(graph.nodes())
    if graph.number_of_edges() == 0:
        return scheme

    # Work on a copy whose edge costs will be modified via MW.
    g = graph.copy()

    # Cumulative MW usage — initialised with zero for every edge so
    # the first weight vector sums to one (YATES Yates_Mw.ml lines 106-108).
    cumulative: dict[Edge, float] = {e: 0.0 for e in g.edges()}

    acc_weight = 0.0

    # YATES iterates until accumulated tree weight reaches one
    # (Yates_Mw.ml lines 98-105).  A generous safety bound guards
    # against a non-converging pathological topology.
    _safety = 1024
    for _iter in range(_safety):
        # 1. Build random FRT tree on the current weighted graph.
        #    Both distances *and* physical paths come from the reweighted
        #    topology so the next routing tree reflects MW cost updates
        #    (YATES Raeke.ml select_structure → make_frt_tree → generate_rt).
        dists, phys_paths = _all_pairs_paths(g)
        frt = _frt_decompose(vertices, dists, rng=rng)

        # 2. Build routing tree and compute boundary-capacity usage.
        edge_usage, pruned_tree, path_table = _build_routing_usage(
            graph, frt, hosts, phys_paths
        )

        if not edge_usage:
            continue

        # 3. Tree weight = 1 / max_usage.
        max_u = max(edge_usage.values())
        if max_u <= 0:
            continue
        w = 1.0 / max_u

        def _host_padded(phys: list[str], src: str, dst: str) -> tuple[str, ...] | None:
            """Extend physical path to include host access links."""
            p = list(phys)
            if p[0] != src and g.has_edge(src, p[0]):
                p.insert(0, src)
            if p[-1] != dst and g.has_edge(p[-1], dst):
                p.append(dst)
            if len(p) >= 2 and p[0] == src and p[-1] == dst:
                return tuple(p)
            return None

        # 4. Stopping condition — clip final weight so sum of weights = 1.
        if acc_weight + w >= 1.0:
            final_w = 1.0 - acc_weight
            if final_w > 0 and pruned_tree is not None:
                for src, dst in commodities:
                    tp = _frt_paths(pruned_tree, src, dst)
                    if len(tp) >= 2:
                        phys = _tree_to_physical(tp, path_table, g)
                        if phys:
                            padded = _host_padded(phys, src, dst)
                            if padded is not None:
                                scheme[(src, dst)][padded] = (
                                    scheme[(src, dst)].get(padded, 0.0) + final_w
                                )
            break

        # 5. Collect weighted paths from this tree.
        if pruned_tree is not None:
            for src, dst in commodities:
                tp = _frt_paths(pruned_tree, src, dst)
                if len(tp) >= 2:
                    phys = _tree_to_physical(tp, path_table, g)
                    if phys:
                        padded = _host_padded(phys, src, dst)
                        if padded is not None:
                            scheme[(src, dst)][padded] = (
                                scheme[(src, dst)].get(padded, 0.0) + w
                            )

        acc_weight += w

        # 6. Scale usage so max = 1, add to cumulative.
        for e, u in edge_usage.items():
            cumulative[e] += u / max_u

        # 7. MW exponential reweighting — every edge participates so the
        #    weight vector sums to one (YATES Yates_Mw.ml lines 67-73).
        mw: dict[Edge, float] = {
            e: math.exp(epsilon * cumulative[e]) for e in g.edges()
        }
        sum_exp = sum(mw.values())
        if sum_exp > 0:
            for e in g.edges():
                g.edges[e]["cost"] = mw[e] / sum_exp
    else:
        # Loop exhausted without convergence — YATES always converges
        # under the MW guarantee; this signals a defect or pathology.
        raise RuntimeError(
            f"Raecke MW did not converge: accumulated weight "
            f"{acc_weight:.4f} after {_safety} iterations"
        )

    for (src, dst), paths in scheme.items():
        total = sum(paths.values())
        if nx.has_path(graph, src, dst):
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(
                    f"Raecke scheme for {src}->{dst} has probability mass "
                    f"{total:.12g}, expected 1"
                )
        elif paths:
            raise RuntimeError(
                f"Raecke scheme for disconnected commodity {src}->{dst} is nonempty"
            )

    return scheme


def _all_hosts_connected(
    graph: nx.DiGraph,
    hosts: set[str],
) -> bool:
    """Check all ordered host-pair directed paths exist (YATES
    all_pairs_connectivity).  Only host-pair reachability is considered;
    isolated switch-only vertices do not disqualify a failure scenario."""
    for src in hosts:
        for dst in hosts:
            if src != dst and not nx.has_path(graph, src, dst):
                return False
    return True


def _oblivious_paths_ft(
    graph: nx.DiGraph,
    hosts: set[str],
    *,
    seed: int | None = None,
) -> WeightedScheme:
    """Fault-tolerant envelope: merge weighted schemes from single-link failures.

    For each unique bidirectional switch-switch physical link,
    runs Raecke path selection (with a derived reproducible child seed)
    and merges the resulting weighted schemes.
    Matches YATES all_failures_envelope for semimcfraekeft.
    """
    commodities = [(s, t) for s in sorted(hosts) for t in sorted(hosts) if s != t]
    merged: WeightedScheme = {comm: {} for comm in commodities}

    switch_links: list[Edge] = sorted(
        {
            tuple(sorted((u, v)))
            for u, v in graph.edges()
            if graph.nodes[u].get("type") == "switch"
            and graph.nodes[v].get("type") == "switch"
            and graph.has_edge(v, u)
        }
    )

    rng = random.Random(seed)
    used_child_seeds: set[int] = set()
    for u, v in switch_links:
        fail_graph = graph.copy()
        fail_graph.remove_edge(u, v)
        fail_graph.remove_edge(v, u)
        if not _all_hosts_connected(fail_graph, hosts):
            continue

        # Restrict to the strongly connected component shared by the hosts.
        # Vertices outside it cannot participate in any host-to-host path and
        # can prevent the FRT/MW decomposition from reaching complete mass.
        anchor = min(hosts)
        relevant = {anchor} | (
            nx.descendants(fail_graph, anchor) & nx.ancestors(fail_graph, anchor)
        )
        fail_graph = fail_graph.subgraph(relevant)

        # Resolve the vanishingly unlikely RNG collision deterministically so
        # every surviving failure is guaranteed a distinct child stream.
        child_seed = rng.getrandbits(64)
        while child_seed in used_child_seeds:
            child_seed = (child_seed + 1) % (2**64)
        used_child_seeds.add(child_seed)
        fail_scheme = _raecke_paths(fail_graph, hosts, seed=child_seed)
        for comm, pps in fail_scheme.items():
            for path, prob in pps.items():
                merged[comm][path] = merged[comm].get(path, 0.0) + prob

    return _normalize_scheme(merged)


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
    *,
    seed: int | None = None,
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
            scheme = _raecke_paths(graph, hosts, seed=seed)
        else:
            scheme = _oblivious_paths_ft(graph, hosts, seed=seed)
        cand = _prune_scheme(scheme, budget)
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
