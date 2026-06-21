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
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix, eye

from onset.base_types import (
    BackendProvenance,
    DopplerProblem,
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


def _build_milp(problem: DopplerProblem):
    """Build a HiGHS-compatible LP model for the corrected Doppler formulation.

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
    flow_var_index: Dict[Tuple, int] = {}  # (comm_idx, u, v) -> col
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

    row_entries: List[Dict[int, float]] = []  # list of {col: coeff}
    row_lower: List[float] = []
    row_upper: List[float] = []

    def add_row(lower: float, upper: float, entries: Dict[int, float]):
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
            combined = {}
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
    rf_vars: Dict[Tuple[str, str], int] = {}
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

    # Objective: 2 * change_count + MLU
    # change_count = sum(1 - x[e] for e in current) + sum(x[e] for e not in current)
    obj = np.zeros(n_vars_with_rf, dtype=np.float64)
    obj[mlu_idx] = 1.0
    for ei, e in enumerate(undirected_edges):
        if e in current_set:
            obj[ei] = -2.0  # coefficient for x[e]: removal contributes 1-x[e]
        else:
            obj[ei] = 2.0   # coefficient for x[e]: addition contributes x[e]

    # Offset: 2 * |current_edges| for the constant part of removal counting
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
    }
    return lp, index_maps


# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------


def _extract_solution(
    problem: DopplerProblem,
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
    agg_loads: Dict[Tuple[str, str], float] = {}
    total_flow_on_directed: Dict[Tuple[str, str], float] = {}
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
    problem: DopplerProblem,
    selected: FrozenSet[Tuple[str, str]],
    agg_loads: Dict[Tuple[str, str], float],
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


def solve_doppler(
    problem: DopplerProblem,
    time_limit: float,
    *,
    additional_cuts: Optional[List[Dict[int, float]]] = None,
) -> Tuple[Optional[TopologySolution], OptimizerStatus, float]:
    """Solve the corrected Doppler MILP once using HiGHS.

    Parameters
    ----------
    problem : DopplerProblem
        The fully specified problem.
    time_limit : float
        Solver time limit in seconds.
    additional_cuts : list of dict, optional
        Additional constraint rows to add (no-good cuts from prior solves).
        Each dict maps column index -> coefficient.

    Returns
    -------
    (solution, status, wall_time) where solution is None if no incumbent.
    """
    import highspy

    t_start = perf_counter()

    # Build model
    lp, im = _build_milp(problem)

    h = highspy.Highs()
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)  # prove optimality
    h.passModel(lp)

    # Add any additional cuts (no-good cuts)
    if additional_cuts:
        for cut in additional_cuts:
            cols = []
            coeffs = []
            for col, coeff in cut.items():
                if col < im["n_vars_rf"]:
                    cols.append(int(col))
                    coeffs.append(float(coeff))
            if cols:
                indices = np.array(cols, dtype=np.int32)
                values_arr = np.array(coeffs, dtype=np.float64)
                h.addRow(1.0, 1e10, len(indices), indices, values_arr)

    h.run()
    elapsed = perf_counter() - t_start

    status_code = h.getModelStatus()
    has_solution = h.getSolution().col_value is not None and len(
        h.getSolution().col_value
    ) > 0

    # Map HiGHS status to our status
    if status_code == highspy.HighsModelStatus.kOptimal:
        if not has_solution:
            return None, OptimizerStatus.SOLVER_ERROR, elapsed
        solution = np.array(h.getSolution().col_value)
        try:
            ts = _extract_solution(problem, im, solution, proven_optimal=True)
            return ts, OptimizerStatus.TOP_K_REACHED, elapsed
        except Exception:
            logger.exception("Solution extraction failed")
            return None, OptimizerStatus.VALIDATION_FAILED, elapsed

    elif status_code in (
        highspy.HighsModelStatus.kTimeLimit,
        highspy.HighsModelStatus.kIterationLimit,
    ):
        if has_solution:
            solution = np.array(h.getSolution().col_value)
            try:
                ts = _extract_solution(
                    problem, im, solution, proven_optimal=False
                )
                return ts, OptimizerStatus.TIME_LIMIT_WITH_SOLUTION, elapsed
            except Exception:
                logger.exception("Timed-out solution extraction failed")
                return None, OptimizerStatus.VALIDATION_FAILED, elapsed
        return None, OptimizerStatus.TIME_LIMIT_WITHOUT_SOLUTION, elapsed

    elif status_code == highspy.HighsModelStatus.kInfeasible:
        return None, OptimizerStatus.INFEASIBLE, elapsed

    elif status_code == highspy.HighsModelStatus.kUnbounded:
        return None, OptimizerStatus.UNBOUNDED, elapsed

    else:
        return None, OptimizerStatus.SOLVER_ERROR, elapsed


# ---------------------------------------------------------------------------
# No-good cut construction
# ---------------------------------------------------------------------------


def make_no_good_cut(
    selected_edges: FrozenSet[Tuple[str, str]],
    undirected_edges: List[Tuple[str, str]],
    edge_to_idx: Dict[Tuple[str, str], int],
    n_vars: int,
) -> Tuple[np.ndarray, float, float]:
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
    problem: DopplerProblem,
) -> Tuple[bool, Optional[float]]:
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

    flow_var_index: Dict[Tuple, int] = {}
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
            entries = {}
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
    ri, ci, vi = [], [], []
    for i, r in enumerate(rows):
        for col, val in r.items():
            ri.append(i); ci.append(col); vi.append(val)
    A = csr((vi, (ri, ci)), shape=(n_con, n_vars))

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


def solve_doppler_with_enumeration(
    problem: DopplerProblem,
) -> OptimizationResult:
    """Run the full open-backend Doppler optimization with solution enumeration.

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
    lp, im = _build_milp(problem)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.passModel(lp)

    solutions: List[TopologySolution] = []
    seen_ids: Set[str] = set()
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
                        problem, im, solution, proven_optimal=False
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
