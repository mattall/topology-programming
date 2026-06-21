"""
Open-source Doppler MILP solver using HiGHS (via highspy).

Implements the corrected tunnel-restricted Doppler formulation from the
refactor plan (§3) with:
  - Binary edge variables per canonical undirected candidate edge
  - Directed flow variables only for commodity+edge pairs in tunnel-allowed sets
  - Root-flow connectivity constraint
  - Corrected objective: 2 * change_count + max_link_utilization
  - Deterministic variable indexing via canonical node/edge order

Never imports gurobipy.
"""

from __future__ import annotations

import logging
from time import perf_counter


import numpy as np
from scipy.sparse import csr_matrix, eye

from onset.base_types import (
    BackendProvenance,
    OptimizationProblem,
    OptimizerStatus,
    TopologySolution,
    OptimizationResult,
    BINARY_TOLERANCE,
    FLOW_TOLERANCE_ABSOLUTE,
    FLOW_TOLERANCE_RELATIVE,
    compute_stable_topology_id,
    compute_legacy_topology_id,
    map_milp_status,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MILP builder
# ---------------------------------------------------------------------------


def _build_edge_flow_milp(problem: OptimizationProblem, objective_mode: str = "changes_plus_mlu"):
    """Build a HiGHS-compatible LP model for the corrected Doppler formulation.

    Parameters
    ----------
    problem : OptimizationProblem
    objective_mode : str
        "changes_plus_mlu" (Doppler) or "mlu" (onset_v3).

    Returns (lp, index_maps) where lp is a highspy.HighsLp and index_maps
    contains dictionaries mapping variable/index names to column indices.
    """
    import highspy

    nodes = list(problem.canonical_node_order)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Undirected candidate edges (canonical order)
    undirected_edges = list(problem.canonical_candidate_edges)
    n_undirected = len(undirected_edges)
    edge_to_idx = {e: i for i, e in enumerate(undirected_edges)}

    # Directed edges: both directions for each undirected edge
    directed_edges = [(u, v) for (u, v) in undirected_edges] + [
        (v, u) for (u, v) in undirected_edges
    ]
    n_directed = len(directed_edges)
    dir_to_idx = {e: i for i, e in enumerate(directed_edges)}

    # Commodities: demands with tunnel-allowed directed edge sets
    commodities = list(problem.demand.keys())
    n_commodities = len(commodities)
    comm_to_idx = {c: i for i, c in enumerate(commodities)}

    # Normalized demand
    sc = problem.scaled_capacity
    norm_demand = {c: problem.demand[c] / sc for c in commodities}

    # Current edges (for change count)
    current_set = problem.current_edges
    current_idx = {e: i for i, e in enumerate(undirected_edges)
                   if e in current_set}

    # Variable layout (deterministic order):
    #   x[0..E-1]: binary edge vars (undirected, canonical order)
    #   f[E..E+F-1]: continuous flow vars, F = sum(len(tunnel_edges[c])) per commodity
    #   mlu: scalar (last var)
    x_offset = 0
    x_count = n_undirected
    f_offset = x_count
    flow_var_index: dict[tuple, int] = {}  # (comm_idx, u, v) -> col
    f_idx = f_offset
    for ci, comm in enumerate(commodities):
        allowed = problem.tunnel_edge_sets.get(comm, frozenset())
        for (u, v) in sorted(allowed):
            flow_var_index[(ci, u, v)] = f_idx
            f_idx += 1
    n_flow = f_idx - f_offset
    mlu_idx = f_idx
    n_vars = mlu_idx + 1

    # ---- Constraints ----
    # We build constraint rows as lists of (col, coeff) pairs, then assemble into CSR

    row_entries: list[dict[int, float]] = []  # list of {col: coeff}
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_row(lower: float, upper: float, entries: dict[int, float]):
        row_entries.append(entries)
        row_lower.append(lower)
        row_upper.append(upper)

    # 1. Transponder constraints: sum(x for edges incident to node) <= txp(node)
    for node in nodes:
        entries = {}
        for (u, v), ei in edge_to_idx.items():
            if u == node or v == node:
                entries[ei] = 1.0
        if entries:
            txp = min(problem.txp_count.get(node, 0), len(undirected_edges))
            add_row(-np.inf, float(txp), entries)

    # 2. Flow conservation per commodity per node
    for ci, (s, t) in enumerate(commodities):
        d = norm_demand[(s, t)]
        if d <= 0:
            continue
        allowed = problem.tunnel_edge_sets.get((s, t), frozenset())
        for node in nodes:
            inflow = {}
            outflow = {}
            for (u, v) in sorted(allowed):
                fcol = flow_var_index.get((ci, u, v))
                if fcol is None:
                    continue
                if v == node:  # flows INTO node
                    inflow[fcol] = 1.0
                if u == node:  # flows OUT OF node
                    outflow[fcol] = -1.0
            combined: dict[int, float] = {}
            for col, coeff in inflow.items():
                combined[col] = combined.get(col, 0.0) + coeff
            for col, coeff in outflow.items():
                combined[col] = combined.get(col, 0.0) + coeff
            if combined:
                if node == s:
                    add_row(-d, -d, combined)  # in - out = -d at source
                elif node == t:
                    add_row(d, d, combined)    # in - out = +d at target
                else:
                    add_row(0.0, 0.0, combined)  # flow conservation

    # 3. Activation / capacity: f <= x for each commodity+edge pair
    for ci, (s, t) in enumerate(commodities):
        allowed = problem.tunnel_edge_sets.get((s, t), frozenset())
        for (u, v) in sorted(allowed):
            fcol = flow_var_index.get((ci, u, v))
            if fcol is None:
                continue
            undir = (u, v) if (u, v) in edge_to_idx else (v, u)
            xcol = edge_to_idx.get(undir)
            if xcol is None:
                continue
            entries = {fcol: 1.0, xcol: -1.0}
            add_row(-np.inf, 0.0, entries)

    # 4. MLU: sum of normalized flows per directed edge <= mlu
    for de, (u, v) in enumerate(directed_edges):
        entries = {mlu_idx: -1.0}
        for ci, (s, t) in enumerate(commodities):
            fcol = flow_var_index.get((ci, u, v))
            if fcol is not None:
                entries[fcol] = 1.0
        if len(entries) > 1:
            add_row(-np.inf, 0.0, entries)

    # 5. Connectivity: root-flow formulation
    root = nodes[0]
    root_flow_offset = n_vars
    # Add artificial root-flow variables (one per directed edge)
    rf_vars: dict[tuple[str, str], int] = {}
    for (u, v) in directed_edges:
        rf_vars[(u, v)] = root_flow_offset
        root_flow_offset += 1
    n_vars_with_rf = root_flow_offset

    # Root-flow lower <= 0, upper bounded by (n_nodes-1) * x
    # Root node supplies (n_nodes-1), other nodes consume 1
    supply = n_nodes - 1
    for (u, v), rfi in rf_vars.items():
        undir = (u, v) if (u, v) in edge_to_idx else (v, u)
        xcol = edge_to_idx[undir]
        entries = {rfi: 1.0, xcol: -(supply)}
        add_row(-np.inf, 0.0, entries)

    for node in nodes:
        entries_in = {}
        entries_out = {}
        for (u, v), rfi in rf_vars.items():
            if v == node:
                entries_in[rfi] = 1.0
            if u == node:
                entries_out[rfi] = -1.0
        combined = {}
        for col, coeff in entries_in.items():
            combined[col] = combined.get(col, 0.0) + coeff
        for col, coeff in entries_out.items():
            combined[col] = combined.get(col, 0.0) + coeff
        if combined:
            if node == root:
                add_row(-supply, -supply, combined)  # outflow - inflow = supply
            else:
                add_row(1.0, 1.0, combined)

    # Assemble CSR matrix
    n_con = len(row_entries)
    row_indices = []
    col_indices = []
    values = []
    for i, entries_dict in enumerate(row_entries):
        for col, val in entries_dict.items():
            row_indices.append(i)
            col_indices.append(col)
            values.append(val)

    A = csr_matrix((values, (row_indices, col_indices)),
                   shape=(n_con, n_vars_with_rf))

    # Objective
    obj = np.zeros(n_vars_with_rf, dtype=np.float64)
    if objective_mode == "mlu":
        obj[mlu_idx] = 1.0
        change_offset = 0.0
    else:
        # changes_plus_mlu: 2 * change_count + MLU
        # change_count = sum(1 - x[e] for e in current) + sum(x[e] for e not in current)
        obj[mlu_idx] = 1.0
        for ei, e in enumerate(undirected_edges):
            if e in current_set:
                obj[ei] = -2.0
            else:
                obj[ei] = 2.0
        change_offset = 2.0 * len(current_set)

    # Variable bounds
    lb = np.zeros(n_vars_with_rf, dtype=np.float64)
    ub = np.full(n_vars_with_rf, np.inf, dtype=np.float64)
    ub[:x_count] = 1.0      # binary
    ub[mlu_idx] = 1.0        # MLU in [0, 1]
    ub[mlu_idx] = min(ub[mlu_idx], problem.congestion_threshold_upper_bound)
    # Root-flow variables: lower bound is 0 (already set), upper is supply (handled by constraints)

    # Integrality
    integrality = np.zeros(n_vars_with_rf, dtype=np.int32)
    integrality[:x_count] = 1

    # Build HiGHS LP
    lp = highspy.HighsLp()
    lp.num_col_ = n_vars_with_rf
    lp.num_row_ = n_con
    lp.col_cost_ = obj
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.array(row_lower, dtype=np.float64)
    lp.row_upper_ = np.array(row_upper, dtype=np.float64)
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if i < x_count
        else highspy.HighsVarType.kContinuous
        for i in range(n_vars_with_rf)
    ]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.offset_ = change_offset  # objective constant term

    index_maps = {
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "undirected_edges": undirected_edges,
        "edge_to_idx": edge_to_idx,
        "directed_edges": directed_edges,
        "dir_to_idx": dir_to_idx,
        "commodities": commodities,
        "comm_to_idx": comm_to_idx,
        "flow_var_index": flow_var_index,
        "x_offset": x_offset,
        "x_count": x_count,
        "f_offset": f_offset,
        "n_flow": n_flow,
        "mlu_idx": mlu_idx,
        "n_vars": n_vars,  # original var count (excluding root-flow)
        "n_vars_rf": n_vars_with_rf,
        "change_offset": change_offset,
        "rf_vars": rf_vars,
        "n_con": n_con,
        "objective_mode": objective_mode,
    }
    return lp, index_maps


# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------


def _extract_solution(
    problem: OptimizationProblem,
    index_maps: dict,
    solution: np.ndarray,
    proven_optimal: bool,
) -> TopologySolution:
    """Extract and validate a HiGHS solution into a TopologySolution."""
    im = index_maps
    edges = im["undirected_edges"]

    # Round binary variables
    x_vals = solution[:im["x_count"]]
    selected = frozenset(
        edges[i] for i in range(im["x_count"])
        if x_vals[i] > 0.5 - BINARY_TOLERANCE
    )

    # Derived changes
    current = problem.current_edges
    added = frozenset(selected - current)
    dropped = frozenset(current - selected)
    change_count = len(added) + len(dropped)

    # Aggregate flows per directed edge
    agg_loads: dict[tuple[str, str], float] = {}
    total_flow_on_directed: dict[tuple[str, str], float] = {}
    for (ci, u, v), fcol in im["flow_var_index"].items():
        val = solution[fcol]
        if val > FLOW_TOLERANCE_ABSOLUTE:
            total_flow_on_directed[(u, v)] = (
                total_flow_on_directed.get((u, v), 0.0) + val
            )

    # Convert to original capacity-scaled units
    sc = problem.scaled_capacity
    for (u, v), val in total_flow_on_directed.items():
        agg_loads[(u, v)] = val * sc

    # Commodity flows (if requested)
    commodity_flows = None
    if problem.retain_commodity_flows:
        commodity_flows = {}
        for (ci, u, v), fcol in im["flow_var_index"].items():
            val = solution[fcol]
            if val > FLOW_TOLERANCE_ABSOLUTE:
                commodity_flows[(im["commodities"][ci][0],
                                im["commodities"][ci][1],
                                u, v)] = val * sc

    # MLU
    solver_mlu = float(solution[im["mlu_idx"]])

    # Independent validation
    validated_mlu = _validate_mlu_independently(
        problem, selected, agg_loads
    )

    # Objective
    if im["objective_mode"] == "mlu":
        obj = solver_mlu
    else:
        obj = 2.0 * change_count + solver_mlu

    # IDs
    nodes = list(problem.canonical_node_order)
    stable_id = compute_stable_topology_id(nodes, list(selected))
    legacy_id = compute_legacy_topology_id(
        problem.legacy_candidate_edge_order, selected
    )

    return TopologySolution(
        selected_edges=selected,
        added=added,
        dropped=dropped,
        commodity_flows=commodity_flows,
        aggregate_edge_loads=agg_loads,
        solver_mlu=solver_mlu,
        validated_mlu=validated_mlu,
        change_count=change_count,
        objective_value=obj,
        stable_topology_id=stable_id,
        legacy_topology_id=legacy_id,
        provenance=BackendProvenance.OPEN,
        proven_optimal=proven_optimal,
    )


def _validate_mlu_independently(
    problem: OptimizationProblem,
    selected: frozenset[tuple[str, str]],
    agg_loads: dict[tuple[str, str], float],
) -> float:
    """Recompute MLU from aggregate loads and capacity."""
    sc = problem.scaled_capacity
    max_util = 0.0
    for (u, v), load in agg_loads.items():
        undir = (u, v) if (u, v) in selected else (v, u)
        if undir not in selected:
            logger.warning(f"Flow on inactive edge {(u, v)} — validation issue")
        util = load / (sc * 1.0)  # normalized capacity is 1.0 in model units
        if util > max_util:
            max_util = util
    return max_util


# ---------------------------------------------------------------------------
# Single-solve entry point
# ---------------------------------------------------------------------------


def solve_edge_flow_changes_mlu_single(
    problem: OptimizationProblem,
    time_limit: float,
    *,
    additional_cuts: list[dict[int, float]] | None = None,
    objective_mode: str = "changes_plus_mlu",
) -> tuple[TopologySolution | None, OptimizerStatus, float]:
    """Solve the corrected Doppler MILP once using HiGHS.

    Parameters
    ----------
    problem : OptimizationProblem
        The fully specified problem.
    time_limit : float
        Solver time limit in seconds.
    additional_cuts : list of dict, optional
        Additional constraint rows to add (no-good cuts from prior solves).
    objective_mode : str
        "changes_plus_mlu" or "mlu".

    Returns
    -------
    (solution, status, wall_time) where solution is None if no incumbent.
    """
    lp, im = _build_edge_flow_milp(problem, objective_mode=objective_mode)
    result = _solve_single_milp(problem, lp, im, _extract_solution)
    if result.solutions:
        return result.solutions[0], result.status, result.wall_time
    return None, result.status, result.wall_time


# ---------------------------------------------------------------------------
# No-good cut construction
# ---------------------------------------------------------------------------


def make_no_good_cut(
    selected_edges: frozenset[tuple[str, str]],
    undirected_edges: list[tuple[str, str]],
    edge_to_idx: dict[tuple[str, str], int],
    n_vars: int,
) -> tuple[np.ndarray, float, float]:
    """Build a no-good cut excluding exactly this topology.

    Constraint: sum_{e in S} (1 - x_e) + sum_{e not in S} x_e >= 1
    => sum_{e not in S} x_e - sum_{e in S} x_e >= 1 - |S|

    Returns (coeffs, lower, upper) where coeffs is a length-n_vars array.
    """
    coeffs = np.zeros(n_vars, dtype=np.float64)
    selected_set = set(selected_edges)
    for e, ei in edge_to_idx.items():
        if e in selected_set:
            coeffs[ei] = -1.0  # -x_e for selected edges
        else:
            coeffs[ei] = 1.0   # +x_e for unselected edges
    lower = 1.0 - len(selected_set)
    upper = np.inf
    return coeffs, lower, upper


# ---------------------------------------------------------------------------
# LP baseline evaluation
# ---------------------------------------------------------------------------


def solve_baseline(
    problem: OptimizationProblem,
) -> tuple[bool, float | None]:
    """Solve a continuous LP on the CURRENT topology (no optimization).

    Returns (feasible, mlu) where mlu is None if infeasible.
    """
    import highspy
    from scipy.sparse import csr_matrix as csr

    nodes = list(problem.canonical_node_order)
    n_nodes = len(nodes)
    undirected_edges = list(problem.canonical_candidate_edges)
    edge_to_idx = {e: i for i, e in enumerate(undirected_edges)}
    current_set = problem.current_edges
    directed_edges = [(u, v) for (u, v) in undirected_edges] + [
        (v, u) for (u, v) in undirected_edges
    ]
    commodities = list(problem.demand.keys())
    sc = problem.scaled_capacity
    norm_demand = {c: problem.demand[c] / sc for c in commodities}

    flow_var_index: dict[tuple, int] = {}
    f_idx = 0
    for ci, comm in enumerate(commodities):
        allowed = problem.tunnel_edge_sets.get(comm, frozenset())
        for (u, v) in sorted(allowed):
            flow_var_index[(ci, u, v)] = f_idx
            f_idx += 1
    n_flow = f_idx
    mlu_idx = f_idx
    n_vars = mlu_idx + 1

    obj = np.zeros(n_vars, dtype=np.float64)
    obj[mlu_idx] = 1.0

    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    ub[mlu_idx] = 1.0

    rows = []
    row_lb = []
    row_ub = []

    # Flow conservation, activation (for inactive edges only), and MLU
    for ci, (s, t) in enumerate(commodities):
        d = norm_demand[(s, t)]
        if d <= 0:
            continue
        allowed = problem.tunnel_edge_sets.get((s, t), frozenset())
        for node in nodes:
            entries: dict[int, float] = {}
            for (u, v) in sorted(allowed):
                fcol = flow_var_index.get((ci, u, v))
                if fcol is None:
                    continue
                if v == node:
                    entries[fcol] = entries.get(fcol, 0.0) + 1.0
                if u == node:
                    entries[fcol] = entries.get(fcol, 0.0) - 1.0
            if entries:
                if node == s:
                    rows.append(dict(entries))
                    row_lb.append(-d)
                    row_ub.append(-d)
                elif node == t:
                    rows.append(dict(entries))
                    row_lb.append(d)
                    row_ub.append(d)
                else:
                    rows.append(dict(entries))
                    row_lb.append(0.0)
                    row_ub.append(0.0)

    for ci, (s, t) in enumerate(commodities):
        allowed = problem.tunnel_edge_sets.get((s, t), frozenset())
        for (u, v) in sorted(allowed):
            fcol = flow_var_index.get((ci, u, v))
            if fcol is None:
                continue
            undir = (u, v) if (u, v) in edge_to_idx else (v, u)
            if undir not in current_set:
                entries = {fcol: 1.0}
                rows.append(entries)
                row_lb.append(0.0)
                row_ub.append(0.0)

    for de, (u, v) in enumerate(directed_edges):
        entries = {mlu_idx: -1.0}
        for ci, (s, t) in enumerate(commodities):
            fcol = flow_var_index.get((ci, u, v))
            if fcol is not None:
                entries[fcol] = 1.0
        if len(entries) > 1:
            rows.append(entries)
            row_lb.append(-np.inf)
            row_ub.append(0.0)

    n_con = len(rows)
    ri, cj, vi = [], [], []
    for i, r in enumerate(rows):
        for col, val in r.items():
            ri.append(i); cj.append(col); vi.append(val)
    A = csr((vi, (ri, cj)), shape=(n_con, n_vars))

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_con
    lp.col_cost_ = obj
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.array(row_lb, dtype=np.float64)
    lp.row_upper_ = np.array(row_ub, dtype=np.float64)
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * n_vars
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    feasible = status == highspy.HighsModelStatus.kOptimal
    mlu = float(h.getSolution().col_value[mlu_idx]) if feasible else None
    return feasible, mlu


# ---------------------------------------------------------------------------
# Full enumeration entry point
# ---------------------------------------------------------------------------


def solve_edge_flow_changes_mlu(
    problem: OptimizationProblem,
    objective_mode: str = "changes_plus_mlu",
) -> OptimizationResult:
    """Run the full open-backend optimization with solution enumeration.

    Parameters
    ----------
    problem : OptimizationProblem
    objective_mode : str
        "changes_plus_mlu" (Doppler) or "mlu" (onset_v3).

    This is the main entry point. It:
    1. Runs baseline LP evaluation (outside budget)
    2. Iteratively solves MILP with no-good cuts until top_k, exhaustion, or time
    3. Deduplicates by stable topology ID
    4. Returns an immutable OptimizationResult
    """
    import highspy

    t_start = perf_counter()
    budget = problem.optimizer_time_limit

    # Baseline
    baseline_feasible, baseline_mlu = solve_baseline(problem)

    # Build model once
    lp, im = _build_edge_flow_milp(problem, objective_mode=objective_mode)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.passModel(lp)

    solutions: list[TopologySolution] = []
    seen_ids: set[str] = set()
    solve_count = 0

    undirected_edges = im["undirected_edges"]
    edge_to_idx = im["edge_to_idx"]

    while True:
        # Check budget
        elapsed = perf_counter() - t_start
        remaining = budget - elapsed
        if remaining <= 0:
            break

        h.setOptionValue("time_limit", max(remaining, 0.01))

        # Add no-good cuts to exclude previously found topologies
        for sol in solutions:
            coeffs, lower, upper = make_no_good_cut(
                sol.selected_edges, undirected_edges, edge_to_idx,
                im["n_vars_rf"]
            )
            nz = np.nonzero(coeffs)[0]
            if nz.size > 0:
                h.addRow(
                    lower, upper,
                    nz.size,
                    nz.astype(np.int32),
                    coeffs[nz].astype(np.float64),
                )

        h.run()
        solve_count += 1
        status_code = h.getModelStatus()
        elapsed = perf_counter() - t_start

        has_sol = (
            h.getSolution().col_value is not None
            and len(h.getSolution().col_value) > 0
        )

        if status_code == highspy.HighsModelStatus.kOptimal:
            if not has_sol:
                break
            solution = np.array(h.getSolution().col_value)
            try:
                ts = _extract_solution(problem, im, solution, proven_optimal=True)
            except Exception:
                logger.exception("Solution extraction failed at solve %d", solve_count)
                return OptimizationResult(
                    solutions=tuple(solutions),
                    status=OptimizerStatus.VALIDATION_FAILED,
                    wall_time=elapsed,
                    backend=BackendProvenance.OPEN,
                    solve_count=solve_count,
                    baseline_feasible=baseline_feasible,
                    baseline_mlu=baseline_mlu,
                )

            sid = ts.stable_topology_id
            if sid not in seen_ids:
                seen_ids.add(sid)
                solutions.append(ts)

                if len(solutions) >= problem.top_k:
                    sorted_sols = tuple(sorted(
                        solutions,
                        key=lambda s: (
                            s.objective_value,
                            s.change_count,
                            s.validated_mlu,
                            s.stable_topology_id,
                        ),
                    ))
                    return OptimizationResult(
                        solutions=sorted_sols,
                        status=OptimizerStatus.TOP_K_REACHED,
                        wall_time=elapsed,
                        backend=BackendProvenance.OPEN,
                        solve_count=solve_count,
                        baseline_feasible=baseline_feasible,
                        baseline_mlu=baseline_mlu,
                    )

        elif status_code in (
            highspy.HighsModelStatus.kTimeLimit,
            highspy.HighsModelStatus.kIterationLimit,
        ):
            if has_sol:
                solution = np.array(h.getSolution().col_value)
                try:
                    ts = _extract_solution(
                        problem, im, solution, proven_optimal=False,
                    )
                except Exception:
                    logger.exception("Timed-out solution extraction failed")
                    break

                sid = ts.stable_topology_id
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    solutions.append(ts)

                if solutions:
                    sorted_sols = tuple(sorted(
                        solutions,
                        key=lambda s: (
                            s.objective_value,
                            s.change_count,
                            s.validated_mlu,
                            s.stable_topology_id,
                        ),
                    ))
                    return OptimizationResult(
                        solutions=sorted_sols,
                        status=OptimizerStatus.TIME_LIMIT_WITH_SOLUTION,
                        wall_time=elapsed,
                        backend=BackendProvenance.OPEN,
                        solve_count=solve_count,
                        baseline_feasible=baseline_feasible,
                        baseline_mlu=baseline_mlu,
                    )
                else:
                    return OptimizationResult(
                        solutions=(),
                        status=OptimizerStatus.TIME_LIMIT_WITHOUT_SOLUTION,
                        wall_time=elapsed,
                        backend=BackendProvenance.OPEN,
                        solve_count=solve_count,
                        baseline_feasible=baseline_feasible,
                        baseline_mlu=baseline_mlu,
                    )

        elif status_code == highspy.HighsModelStatus.kInfeasible:
            if not solutions:
                return OptimizationResult(
                    solutions=(),
                    status=OptimizerStatus.INFEASIBLE,
                    wall_time=elapsed,
                    backend=BackendProvenance.OPEN,
                    solve_count=solve_count,
                    baseline_feasible=baseline_feasible,
                    baseline_mlu=baseline_mlu,
                )
            else:
                sorted_sols = tuple(sorted(
                    solutions,
                    key=lambda s: (
                        s.objective_value,
                        s.change_count,
                        s.validated_mlu,
                        s.stable_topology_id,
                    ),
                ))
                return OptimizationResult(
                    solutions=sorted_sols,
                    status=OptimizerStatus.EXHAUSTED,
                    wall_time=elapsed,
                    backend=BackendProvenance.OPEN,
                    solve_count=solve_count,
                    baseline_feasible=baseline_feasible,
                    baseline_mlu=baseline_mlu,
                )

        elif status_code == highspy.HighsModelStatus.kUnbounded:
            return OptimizationResult(
                solutions=tuple(solutions),
                status=OptimizerStatus.UNBOUNDED,
                wall_time=elapsed,
                backend=BackendProvenance.OPEN,
                solve_count=solve_count,
                baseline_feasible=baseline_feasible,
                baseline_mlu=baseline_mlu,
            )

        else:
            return OptimizationResult(
                solutions=tuple(solutions),
                status=OptimizerStatus.SOLVER_ERROR,
                wall_time=elapsed,
                backend=BackendProvenance.OPEN,
                solve_count=solve_count,
                baseline_feasible=baseline_feasible,
                baseline_mlu=baseline_mlu,
            )

    # Budget exhausted
    if solutions:
        sorted_sols = tuple(sorted(
            solutions,
            key=lambda s: (
                s.objective_value,
                s.change_count,
                s.validated_mlu,
                s.stable_topology_id,
            ),
        ))
        return OptimizationResult(
            solutions=sorted_sols,
            status=OptimizerStatus.TIME_LIMIT_WITH_SOLUTION,
            wall_time=elapsed,
            backend=BackendProvenance.OPEN,
            solve_count=solve_count,
            baseline_feasible=baseline_feasible,
            baseline_mlu=baseline_mlu,
        )
    else:
        return OptimizationResult(
            solutions=(),
            status=OptimizerStatus.TIME_LIMIT_WITHOUT_SOLUTION,
            wall_time=elapsed,
            backend=BackendProvenance.OPEN,
            solve_count=solve_count,
            baseline_feasible=baseline_feasible,
            baseline_mlu=baseline_mlu,
        )


def solve_edge_flow_mlu(problem: OptimizationProblem) -> OptimizationResult:
    """Run onset_v3 open-backend optimization (MLU-only objective).

    Convenience wrapper around solve_edge_flow_changes_mlu with
    objective_mode="mlu".  The caller should set top_k on the problem
    (top_k=1 for MCF single-solve, top_k=N for ECMP enumeration).
    """
    return solve_edge_flow_changes_mlu(problem, objective_mode="mlu")


# ---------------------------------------------------------------------------
# Linearized min_() over binary variables (AND constraint)
# ---------------------------------------------------------------------------


def _add_and_constraints(
    row_entries: list[dict[int, float]],
    row_lower: list[float],
    row_upper: list[float],
    path_var_col: int,
    edge_var_cols: list[int],
) -> None:
    """Add linear constraints encoding path_var == min(edge_vars).

    Linearized as (n = len(edge_var_cols)):
      path_var <= edge_var_i     for all i
      path_var >= sum(edge_vars) - (n - 1)
      0 <= path_var <= 1

    When all edge_vars are binary, this is exact AND logic.
    """
    n = len(edge_var_cols)
    if n == 0:
        return

    for ei in edge_var_cols:
        row_entries.append({path_var_col: 1.0, ei: -1.0})
        row_lower.append(-np.inf)
        row_upper.append(0.0)

    entries = {path_var_col: 1.0}
    for ei in edge_var_cols:
        entries[ei] = -1.0
    row_entries.append(entries)
    row_lower.append(-(n - 1))
    row_upper.append(np.inf)


# ---------------------------------------------------------------------------
# Single-solve helper (no enumeration)
# ---------------------------------------------------------------------------


def _solve_single_milp(
    problem: OptimizationProblem,
    lp,
    index_maps: dict,
    extract_fn,
) -> OptimizationResult:
    """Solve a pre-built HiGHS LP once and extract the single solution.

    Parameters
    ----------
    problem : OptimizationProblem
    lp : highspy.HighsLp
    index_maps : dict
    extract_fn : callable
        Function(problem, index_maps, solution_array, proven_optimal) -> TopologySolution.
        Must match the builder that produced index_maps.
    """
    import highspy

    t_start = perf_counter()
    budget = problem.optimizer_time_limit

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("time_limit", budget)
    h.passModel(lp)
    h.run()

    elapsed = perf_counter() - t_start
    status = h.getModelStatus()
    info = h.getInfo()

    if status == highspy.HighsModelStatus.kOptimal:
        sol = h.getSolution()
        solution = extract_fn(
            problem, index_maps, np.array(sol.col_value),
            proven_optimal=True,
        )
        return OptimizationResult(
            solutions=(solution,),
            status=OptimizerStatus.TOP_K_REACHED,
            wall_time=elapsed,
            backend=BackendProvenance.OPEN,
            solve_count=1,
        )

    if status == highspy.HighsModelStatus.kInfeasible:
        return OptimizationResult(
            solutions=(),
            status=OptimizerStatus.INFEASIBLE,
            wall_time=elapsed,
            backend=BackendProvenance.OPEN,
            solve_count=1,
        )

    if status == highspy.HighsModelStatus.kUnbounded:
        return OptimizationResult(
            solutions=(),
            status=OptimizerStatus.UNBOUNDED,
            wall_time=elapsed,
            backend=BackendProvenance.OPEN,
            solve_count=1,
        )

    if status == highspy.HighsModelStatus.kTimeLimit:
        if info.primal_solution_status == 1:
            sol = h.getSolution()
            try:
                solution = extract_fn(
                    problem, index_maps, np.array(sol.col_value),
                    proven_optimal=False,
                )
                return OptimizationResult(
                    solutions=(solution,),
                    status=OptimizerStatus.TIME_LIMIT_WITH_SOLUTION,
                    wall_time=elapsed,
                    backend=BackendProvenance.OPEN,
                    solve_count=1,
                )
            except Exception:
                pass
        return OptimizationResult(
            solutions=(),
            status=OptimizerStatus.TIME_LIMIT_WITHOUT_SOLUTION,
            wall_time=elapsed,
            backend=BackendProvenance.OPEN,
            solve_count=1,
        )

    return OptimizationResult(
        solutions=(),
        status=OptimizerStatus.SOLVER_ERROR,
        wall_time=elapsed,
        backend=BackendProvenance.OPEN,
        solve_count=1,
    )


# ---------------------------------------------------------------------------
# onset_v1 builder (edge-based path flow, budget, candidate-link penalty)
# ---------------------------------------------------------------------------


def _build_path_flow_budget_milp(problem: OptimizationProblem):
    """Build a HiGHS LP for the onset_v1 formulation.

    Variables (deterministic order):
      x_cand[0..C-1]: binary, per candidate (not-current) undirected edge
      x_path[0..P-1]: binary, per path
      flow[0..P-1]: continuous, normalized flow per path
      util[0..E-1]: continuous, normalized utilization per directed edge
      mlu: scalar [0, 1]

    Returns (lp, index_maps).
    """
    import highspy

    pd = problem.path_data
    if pd is None:
        raise ValueError("onset_v1 requires path_data on OptimizationProblem")

    n_cand = len(pd.candidate_edge_indices)
    n_paths = len(pd.path_list)
    n_dir = len(pd.supergraph_directed_edges)

    sc = problem.scaled_capacity
    norm_demand = {c: problem.demand[c] / sc for c in problem.demand}

    var_idx = 0
    x_cand_start = var_idx
    var_idx += n_cand
    x_path_start = var_idx
    var_idx += n_paths
    flow_start = var_idx
    var_idx += n_paths
    util_start = var_idx
    var_idx += n_dir
    mlu_idx = var_idx
    n_vars = var_idx + 1

    rows: list[dict[int, float]] = []
    rlb: list[float] = []
    rub: list[float] = []

    def add_row(lo, hi, entries):
        rows.append(entries)
        rlb.append(lo)
        rub.append(hi)

    # 1. Budget: sum(x_cand) <= n_cand (non-binding; matches legacy BUDGET=len(candidates))
    if n_cand > 0:
        entries = {x_cand_start + i: 1.0 for i in range(n_cand)}
        add_row(-np.inf, float(n_cand), entries)

    # 2. Path availability: x_path[p] == AND(x_cand[c] for c in path)
    for pi in range(n_paths):
        cand_indices = list(pd.path_candidate_map[pi])
        if cand_indices:
            _add_and_constraints(
                rows, rlb, rub,
                x_path_start + pi,
                [x_cand_start + ci for ci in cand_indices],
            )
        else:
            add_row(1.0, 1.0, {x_path_start + pi: 1.0})

    # 3. Flow activation: flow[p] <= x_path[p]
    for pi in range(n_paths):
        add_row(-np.inf, 0.0, {
            flow_start + pi: 1.0,
            x_path_start + pi: -1.0,
        })

    # 4. Demand satisfaction per commodity
    for (s, t), path_idxs in pd.commodity_to_paths.items():
        d = norm_demand.get((s, t), 0.0)
        if d <= 0 or not path_idxs:
            continue
        entries = {flow_start + pi: 1.0 for pi in path_idxs}
        add_row(d, np.inf, entries)

    # 5. Link utilization: util[ei] >= sum(flow[p] for p on edge)
    for ei, path_idxs in enumerate(pd.link_path_map):
        if not path_idxs:
            continue
        entries = {util_start + ei: 1.0}
        for pi in path_idxs:
            entries[flow_start + pi] = -1.0
        add_row(0.0, 0.0, entries)

    # 6. Capacity: util[ei] <= 1.0
    for ei in range(n_dir):
        add_row(-np.inf, 1.0, {util_start + ei: 1.0})

    # 7. MLU: mlu >= util[ei] for all ei
    for ei in range(n_dir):
        add_row(-np.inf, 0.0, {
            mlu_idx: -1.0,
            util_start + ei: 1.0,
        })

    # Assemble CSR
    n_con = len(rows)
    ri, ci, vi = [], [], []
    for i, r in enumerate(rows):
        for col, val in r.items():
            ri.append(i); ci.append(col); vi.append(val)
    A = csr_matrix((vi, (ri, ci)), shape=(n_con, n_vars))

    # Objective: sum(x_cand) / n_cand + mlu (mlu is tiebreaker, edge count primary)
    obj = np.zeros(n_vars, dtype=np.float64)
    if n_cand > 0:
        for i in range(n_cand):
            obj[x_cand_start + i] = 1.0 / float(n_cand)
    obj[mlu_idx] = 0.01  # small MLU tiebreaker

    # Bounds
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    ub[x_cand_start:x_cand_start + n_cand] = 1.0
    ub[x_path_start:x_path_start + n_paths] = 1.0
    ub[util_start:util_start + n_dir] = 1.0
    ub[mlu_idx] = min(1.0, problem.congestion_threshold_upper_bound)

    # Integrality
    integrality = [highspy.HighsVarType.kContinuous] * n_vars
    for i in range(x_cand_start, x_cand_start + n_cand):
        integrality[i] = highspy.HighsVarType.kInteger
    for i in range(x_path_start, x_path_start + n_paths):
        integrality[i] = highspy.HighsVarType.kInteger

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_con
    lp.col_cost_ = obj
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.array(rlb, dtype=np.float64)
    lp.row_upper_ = np.array(rub, dtype=np.float64)
    lp.integrality_ = integrality
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.offset_ = 0.0

    index_maps = {
        "x_cand_start": x_cand_start,
        "n_cand": n_cand,
        "candidate_edge_indices": pd.candidate_edge_indices,
        "undirected_edges": problem.canonical_candidate_edges,
        "current_edges": problem.current_edges,
        "x_path_start": x_path_start,
        "n_paths": n_paths,
        "flow_start": flow_start,
        "util_start": util_start,
        "n_dir": n_dir,
        "mlu_idx": mlu_idx,
        "n_vars": n_vars,
        "commodities": list(problem.demand.keys()),
        "comm_to_idx": {c: i for i, c in enumerate(problem.demand)},
        "flow_var_index": {(pi,): flow_start + pi for pi in range(n_paths)},
    }
    return lp, index_maps


# ---------------------------------------------------------------------------
# onset_v1_1 builder (path-based flow, core-edge mandate, transponders)
# ---------------------------------------------------------------------------


def _build_path_flow_core_milp(problem: OptimizationProblem):
    """Build a HiGHS LP for the onset_v1_1 formulation.

    Variables (deterministic order):
      x_edge[0..E-1]: binary, per directed edge
      node_deg[0..V-1]: integer, per node
      x_path[N_paths]: binary, one per (s,t,i)
      flow[N_paths]: continuous, normalized flow per path
      util[0..E-1]: continuous, normalized utilization per directed edge
      mlu: scalar [0, 1]

    Returns (lp, index_maps).
    """
    import highspy

    pd = problem.path_data
    if pd is None:
        raise ValueError("onset_v1_1 requires path_data on OptimizationProblem")

    nodes = list(problem.canonical_node_order)
    n_nodes = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_dir = len(pd.supergraph_directed_edges)
    dir_to_idx = {e: i for i, e in enumerate(pd.supergraph_directed_edges)}

    sc = problem.scaled_capacity
    norm_demand = {c: problem.demand[c] / sc for c in problem.demand}

    # Enumerate path variables: flat list of (s,t,i) -> col
    path_vars: dict[tuple, int] = {}
    path_idx_to_edge_cols: dict[int, list[int]] = {}
    next_path = 0
    for (s, t), path_idxs in pd.commodity_to_paths.items():
        for li, pi in enumerate(path_idxs):
            key = (s, t, li)
            path_vars[key] = next_path
            edges = pd.path_list[pi]
            edge_cols = []
            for j in range(len(edges) - 1):
                de = (edges[j], edges[j + 1])
                if de in dir_to_idx:
                    edge_cols.append(dir_to_idx[de])
            path_idx_to_edge_cols[next_path] = edge_cols
            next_path += 1
    n_path_vars = next_path

    # Build link_path_map from computed path_idx_to_edge_cols
    link_path_map: dict[int, list[int]] = {}
    for edge_dir in range(n_dir):
        link_path_map[edge_dir] = []
    for pi, edge_cols in path_idx_to_edge_cols.items():
        for ei in edge_cols:
            link_path_map[ei].append(pi)

    var_idx = 0
    x_edge_start = var_idx
    var_idx += n_dir
    node_deg_start = var_idx
    var_idx += n_nodes
    x_path_start = var_idx
    var_idx += n_path_vars
    flow_start = var_idx
    var_idx += n_path_vars
    util_start = var_idx
    var_idx += n_dir
    mlu_idx = var_idx
    n_vars = var_idx + 1

    rows: list[dict[int, float]] = []
    rlb: list[float] = []
    rub: list[float] = []

    def add_row(lo, hi, entries):
        rows.append(entries)
        rlb.append(lo)
        rub.append(hi)

    # 1. Symmetric edges: x_edge[u,v] == x_edge[v,u]
    undir_seen = set()
    for ei, (u, v) in enumerate(pd.supergraph_directed_edges):
        u_canon, v_canon = _canonical_edge(u, v)
        if (u_canon, v_canon) in undir_seen:
            continue
        undir_seen.add((u_canon, v_canon))
        rev = (v, u)
        if rev in dir_to_idx:
            rev_ei = dir_to_idx[rev]
            add_row(0.0, 0.0, {
                x_edge_start + ei: 1.0,
                x_edge_start + rev_ei: -1.0,
            })

    # 2. Transponder degree constraints
    for ni, node in enumerate(nodes):
        entries = {node_deg_start + ni: 1.0}
        for ei, (u, v) in enumerate(pd.supergraph_directed_edges):
            if u == node:
                entries[x_edge_start + ei] = -1.0
        add_row(0.0, 0.0, entries)

        txp = problem.txp_count.get(node, len(pd.supergraph_directed_edges))
        add_row(-np.inf, float(txp), {node_deg_start + ni: 1.0})

    # 3. Core edge mandate
    for (u, v) in pd.core_edge_set:
        if (u, v) in dir_to_idx:
            add_row(1.0, 1.0, {x_edge_start + dir_to_idx[(u, v)]: 1.0})
        if (v, u) in dir_to_idx:
            add_row(1.0, 1.0, {x_edge_start + dir_to_idx[(v, u)]: 1.0})

    # 4. Path availability: x_path[p] == AND(x_edge[e] for e in path)
    for pi in range(n_path_vars):
        edge_cols = path_idx_to_edge_cols.get(pi, [])
        if edge_cols:
            _add_and_constraints(
                rows, rlb, rub,
                x_path_start + pi,
                [x_edge_start + ei for ei in edge_cols],
            )
        else:
            add_row(1.0, 1.0, {x_path_start + pi: 1.0})

    # 5. Flow activation: flow[p] <= x_path[p]
    for pi in range(n_path_vars):
        add_row(-np.inf, 0.0, {
            flow_start + pi: 1.0,
            x_path_start + pi: -1.0,
        })

    # 6. Demand satisfaction per commodity
    for (s, t), path_idxs in pd.commodity_to_paths.items():
        d = norm_demand.get((s, t), 0.0)
        if d <= 0 or not path_idxs:
            continue
        entries = {}
        for li, pi in enumerate(path_idxs):
            key = (s, t, li)
            if key in path_vars:
                entries[flow_start + path_vars[key]] = 1.0
        if entries:
            add_row(d, np.inf, entries)

    # 7. Link utilization: util[ei] == sum(flow[p] for p on edge)
    #    Equality (not <=) so that mlu = max(util) correctly measures
    #    the maximum edge utilization rather than going to zero.
    for ei in range(n_dir):
        path_cols = link_path_map.get(ei, [])
        entries = {util_start + ei: 1.0}
        for pi in path_cols:
            entries[flow_start + pi] = -1.0
        add_row(0.0, 0.0, entries)

    # 8. Capacity: sum(flow on edge) <= x_edge[ei] (normalized)
    for ei in range(n_dir):
        path_cols = link_path_map.get(ei, [])
        entries = {x_edge_start + ei: -1.0}
        for pi in path_cols:
            entries[flow_start + pi] = 1.0
        if len(entries) > 1:
            add_row(-np.inf, 0.0, entries)

    # 9. MLU: mlu >= util[ei] for all ei
    for ei in range(n_dir):
        add_row(-np.inf, 0.0, {
            mlu_idx: -1.0,
            util_start + ei: 1.0,
        })

    # Assemble CSR
    n_con = len(rows)
    ri, ci, vi = [], [], []
    for i, r in enumerate(rows):
        for col, val in r.items():
            ri.append(i); ci.append(col); vi.append(val)
    A = csr_matrix((vi, (ri, ci)), shape=(n_con, n_vars))

    # Objective: sum(x_edge) + epsilon * mlu
    # Matches legacy: edge count dominates (link_util -> 0, MLU is tiebreaker)
    obj = np.zeros(n_vars, dtype=np.float64)
    for ei in range(n_dir):
        obj[x_edge_start + ei] = 1.0
    obj[mlu_idx] = 0.01

    # Bounds
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    ub[x_edge_start:x_edge_start + n_dir] = 1.0
    ub[x_path_start:x_path_start + n_path_vars] = 1.0
    ub[util_start:util_start + n_dir] = 1.0
    ub[mlu_idx] = min(1.0, problem.congestion_threshold_upper_bound)

    # Integrality
    integrality = [highspy.HighsVarType.kContinuous] * n_vars
    for i in range(x_edge_start, x_edge_start + n_dir):
        integrality[i] = highspy.HighsVarType.kInteger
    for i in range(node_deg_start, node_deg_start + n_nodes):
        integrality[i] = highspy.HighsVarType.kInteger
    for i in range(x_path_start, x_path_start + n_path_vars):
        integrality[i] = highspy.HighsVarType.kInteger

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_con
    lp.col_cost_ = obj
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.array(rlb, dtype=np.float64)
    lp.row_upper_ = np.array(rub, dtype=np.float64)
    lp.integrality_ = integrality
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.offset_ = 0.0

    index_maps = {
        "x_edge_start": x_edge_start,
        "n_dir": n_dir,
        "supergraph_directed_edges": pd.supergraph_directed_edges,
        "dir_to_idx": dir_to_idx,
        "undirected_edges": problem.canonical_candidate_edges,
        "current_edges": problem.current_edges,
        "node_deg_start": node_deg_start,
        "n_nodes": n_nodes,
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "x_path_start": x_path_start,
        "n_path_vars": n_path_vars,
        "path_vars": path_vars,
        "path_idx_to_edge_cols": path_idx_to_edge_cols,
        "flow_start": flow_start,
        "util_start": util_start,
        "mlu_idx": mlu_idx,
        "n_vars": n_vars,
        "commodities": list(problem.demand.keys()),
        "comm_to_idx": {c: i for i, c in enumerate(problem.demand)},
        "link_path_map": link_path_map,
        "core_edge_set": pd.core_edge_set,
    }
    return lp, index_maps


# ---------------------------------------------------------------------------
# Solution extraction for path-based formulations
# ---------------------------------------------------------------------------


def _canonical_edge(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _extract_path_flow_budget(
    problem: OptimizationProblem,
    index_maps: dict,
    solution: np.ndarray,
    proven_optimal: bool,
) -> TopologySolution:
    """Extract onset_v1 solution from HiGHS output."""
    im = index_maps
    n_cand = im["n_cand"]
    cand_indices = im["candidate_edge_indices"]
    undirected_edges = im["undirected_edges"]

    x_cand = solution[im["x_cand_start"]:im["x_cand_start"] + n_cand]
    selected = set(problem.current_edges)
    for ci in range(n_cand):
        if x_cand[ci] > 0.5 - BINARY_TOLERANCE:
            ei = cand_indices[ci]
            selected.add(undirected_edges[ei])

    selected_frozen = frozenset(selected)
    current = problem.current_edges
    added = frozenset(selected_frozen - current)
    dropped = frozenset(current - selected_frozen)
    change_count = len(added) + len(dropped)

    solver_mlu = float(solution[im["mlu_idx"]])

    pd = problem.path_data
    assert pd is not None
    sc = problem.scaled_capacity
    n_dir = im.get("n_dir", len(pd.supergraph_directed_edges))

    agg_loads: dict[tuple[str, str], float] = {}
    for ei in range(n_dir):
        total = 0.0
        for pi in pd.link_path_map[ei]:
            total += solution[im["flow_start"] + pi]
        if total > FLOW_TOLERANCE_ABSOLUTE:
            u, v = pd.supergraph_directed_edges[ei]
            agg_loads[(u, v)] = total * sc

    validated_mlu = _validate_mlu_independently(
        problem, selected_frozen, agg_loads
    )

    obj = solver_mlu + change_count

    nodes = list(problem.canonical_node_order)
    stable_id = compute_stable_topology_id(nodes, list(selected_frozen))
    legacy_id = compute_legacy_topology_id(
        problem.legacy_candidate_edge_order, selected_frozen
    )

    return TopologySolution(
        selected_edges=selected_frozen,
        added=added,
        dropped=dropped,
        commodity_flows=None,
        aggregate_edge_loads=agg_loads,
        solver_mlu=solver_mlu,
        validated_mlu=validated_mlu,
        change_count=change_count,
        objective_value=obj,
        stable_topology_id=stable_id,
        legacy_topology_id=legacy_id,
        provenance=BackendProvenance.OPEN,
        proven_optimal=proven_optimal,
    )


def _extract_path_flow_core(
    problem: OptimizationProblem,
    index_maps: dict,
    solution: np.ndarray,
    proven_optimal: bool,
) -> TopologySolution:
    """Extract onset_v1_1 solution from HiGHS output."""
    im = index_maps
    n_dir = im["n_dir"]
    directed_edges = im["supergraph_directed_edges"]

    x_edge = solution[im["x_edge_start"]:im["x_edge_start"] + n_dir]
    selected_undir: set[tuple[str, str]] = set()
    for ei in range(n_dir):
        if x_edge[ei] > 0.5 - BINARY_TOLERANCE:
            u, v = directed_edges[ei]
            selected_undir.add(_canonical_edge(u, v))

    selected_frozen = frozenset(selected_undir)
    current = problem.current_edges
    added = frozenset(selected_frozen - current)
    dropped = frozenset(current - selected_frozen)
    change_count = len(added) + len(dropped)

    solver_mlu = float(solution[im["mlu_idx"]])

    sc = problem.scaled_capacity
    link_path_map = im["link_path_map"]
    agg_loads: dict[tuple[str, str], float] = {}
    for ei in range(n_dir):
        total = 0.0
        for pi in link_path_map.get(ei, []):
            total += solution[im["flow_start"] + pi]
        if total > FLOW_TOLERANCE_ABSOLUTE:
            u, v = directed_edges[ei]
            agg_loads[(u, v)] = total * sc

    validated_mlu = _validate_mlu_independently(
        problem, selected_frozen, agg_loads
    )

    obj = solver_mlu + change_count

    nodes = list(problem.canonical_node_order)
    stable_id = compute_stable_topology_id(nodes, list(selected_frozen))
    legacy_id = compute_legacy_topology_id(
        problem.legacy_candidate_edge_order, selected_frozen
    )

    return TopologySolution(
        selected_edges=selected_frozen,
        added=added,
        dropped=dropped,
        commodity_flows=None,
        aggregate_edge_loads=agg_loads,
        solver_mlu=solver_mlu,
        validated_mlu=validated_mlu,
        change_count=change_count,
        objective_value=obj,
        stable_topology_id=stable_id,
        legacy_topology_id=legacy_id,
        provenance=BackendProvenance.OPEN,
        proven_optimal=proven_optimal,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def solve_path_flow_budget(problem: OptimizationProblem) -> OptimizationResult:
    """Run onset_v1 optimization (edge-based path flow with budget).

    Single-solve method. Returns one solution or infeasible.
    """
    lp, im = _build_path_flow_budget_milp(problem)
    return _solve_single_milp(problem, lp, im, _extract_path_flow_budget)


def solve_path_flow_core(problem: OptimizationProblem) -> OptimizationResult:
    """Run onset_v1_1 optimization (path-based flow, core-edge mandate).

    Single-solve method. Returns one solution or infeasible.
    """
    lp, im = _build_path_flow_core_milp(problem)
    return _solve_single_milp(problem, lp, im, _extract_path_flow_core)
