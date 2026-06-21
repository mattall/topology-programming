"""Tests for open_doppler.py — correctness via exhaustive oracle and smoke."""

import itertools
import pytest
import numpy as np

from onset.base_types import (
    DopplerProblem,
    TopologySolution,
    OptimizationResult,
    OptimizerStatus,
    BackendProvenance,
    compute_stable_topology_id,
)
from onset.open_doppler import (
    solve_doppler_with_enumeration,
    solve_baseline,
    make_no_good_cut,
)


def _build_triangle_problem():
    """A minimal 3-node triangle: a-b, b-c, a-c candidates. Demand a->c."""
    nodes = ("a", "b", "c")
    candidates = (("a", "b"), ("b", "c"), ("a", "c"))
    return DopplerProblem(
        canonical_node_order=nodes,
        canonical_candidate_edges=candidates,
        legacy_candidate_edge_order=candidates,
        current_edges=frozenset({("a", "b")}),
        txp_count={"a": 2, "b": 2, "c": 1},
        demand={("a", "c"): 50.0},
        tunnel_edge_sets={
            ("a", "c"): frozenset({("a", "b"), ("b", "c"), ("a", "c")})
        },
        link_capacity=100.0,
        scale_factor=1.0,
        congestion_threshold_upper_bound=1.0,
        top_k=10,
        optimizer_time_limit=30.0,
    )


def _build_square_problem():
    """4-node square: a-b-c-d with candidate diagonal a-c."""
    nodes = ("a", "b", "c", "d")
    candidates = (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c"))
    current = frozenset({("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")})
    return DopplerProblem(
        canonical_node_order=nodes,
        canonical_candidate_edges=candidates,
        legacy_candidate_edge_order=candidates,
        current_edges=current,
        txp_count={"a": 3, "b": 3, "c": 3, "d": 3},
        demand={("a", "c"): 30.0, ("b", "d"): 30.0},
        tunnel_edge_sets={
            ("a", "c"): frozenset({("a", "b"), ("b", "c"), ("a", "c")}),
            ("b", "d"): frozenset({("b", "c"), ("c", "d"), ("a", "d"), ("a", "b")}),
        },
        link_capacity=100.0,
        scale_factor=1.0,
        congestion_threshold_upper_bound=1.0,
        top_k=10,
        optimizer_time_limit=30.0,
    )


class TestBaselineEvaluation:
    def test_baseline_feasible(self):
        """Current topology can route demand: a-b and a-c active, tunnel allows both."""
        nodes = ("a", "b", "c")
        candidates = (("a", "b"), ("b", "c"), ("a", "c"))
        prob = DopplerProblem(
            canonical_node_order=nodes,
            canonical_candidate_edges=candidates,
            legacy_candidate_edge_order=candidates,
            current_edges=frozenset({("a", "b"), ("b", "c")}),
            txp_count={"a": 2, "b": 2, "c": 2},
            demand={("a", "c"): 50.0},
            tunnel_edge_sets={
                ("a", "c"): frozenset({("a", "b"), ("b", "c")})
            },
            link_capacity=100.0,
            scale_factor=1.0,
            congestion_threshold_upper_bound=1.0,
            top_k=1,
            optimizer_time_limit=10.0,
        )
        feasible, mlu = solve_baseline(prob)
        assert feasible
        assert mlu is not None
        assert 0.0 <= mlu <= 1.0

    def test_baseline_infeasible(self):
        # Disconnected current topology
        nodes = ("a", "b", "c")
        candidates = (("a", "b"),)
        prob = DopplerProblem(
            canonical_node_order=nodes,
            canonical_candidate_edges=candidates,
            legacy_candidate_edge_order=candidates,
            current_edges=frozenset(),
            txp_count={"a": 1, "b": 1, "c": 1},
            demand={("a", "c"): 50.0},
            tunnel_edge_sets={
                ("a", "c"): frozenset({("a", "b")})
            },
            link_capacity=100.0,
            scale_factor=1.0,
            congestion_threshold_upper_bound=1.0,
            top_k=1,
            optimizer_time_limit=10.0,
        )
        feasible, mlu = solve_baseline(prob)
        # With no edges at all, can't route from a to c
        assert not feasible or mlu is None


class TestSingleSolve:
    def test_triangle_feasible(self):
        prob = _build_triangle_problem()
        result = solve_doppler_with_enumeration(prob)
        assert result.has_solutions
        assert result.status in (
            OptimizerStatus.TOP_K_REACHED,
            OptimizerStatus.EXHAUSTED,
        )
        assert result.solve_count >= 1
        sol = result.solutions[0]
        assert len(sol.selected_edges) >= 2  # need path a->c
        assert 0.0 <= sol.validated_mlu <= 1.0

    def test_square_feasible(self):
        prob = _build_square_problem()
        result = solve_doppler_with_enumeration(prob)
        assert result.has_solutions
        assert result.solve_count >= 1

    def test_solution_ids_unique(self):
        prob = _build_square_problem()
        result = solve_doppler_with_enumeration(prob)
        if len(result.solutions) > 1:
            ids = [s.stable_topology_id for s in result.solutions]
            assert len(ids) == len(set(ids))

    def test_selected_in_candidates(self):
        prob = _build_square_problem()
        result = solve_doppler_with_enumeration(prob)
        for sol in result.solutions:
            for e in sol.selected_edges:
                assert e in set(prob.canonical_candidate_edges)

    def test_connectivity(self):
        """Every returned topology must be connected (validation check)."""
        prob = _build_square_problem()
        result = solve_doppler_with_enumeration(prob)
        for sol in result.solutions:
            edges = list(sol.selected_edges)
            # Build undirected graph from selected edges
            import networkx as nx
            g = nx.Graph()
            g.add_nodes_from(prob.canonical_node_order)
            g.add_edges_from(edges)
            assert nx.is_connected(g), f"Disconnected: {edges}"


class TestExhaustiveOracle:
    """Tiny exhaustive enumeration: compare MILP against brute-force."""

    def test_triangle_exhaustive(self):
        """3 nodes, 3 candidate edges, 2^3 = 8 possible topologies."""
        prob = _build_triangle_problem()

        # Enumerate all undirected edge subsets
        candidates = list(prob.canonical_candidate_edges)
        nodes = list(prob.canonical_node_order)

        # For each subset, check transponders, compute MLU via LP
        import networkx as nx

        feasible_topologies = []
        for bits in itertools.product([0, 1], repeat=len(candidates)):
            selected = frozenset(
                candidates[i] for i, b in enumerate(bits) if b
            )
            # Transponder check
            ok = True
            for node in nodes:
                deg = sum(1 for (u, v) in selected if u == node or v == node)
                if deg > prob.txp_count[node]:
                    ok = False
                    break
            if not ok:
                continue
            # Connectivity
            g = nx.Graph()
            g.add_nodes_from(nodes)
            g.add_edges_from(selected)
            if not nx.is_connected(g):
                continue
            # Compute MLU via LP
            try:
                import highspy
                from scipy.sparse import csr_matrix as csr
            except ImportError:
                pytest.skip("highspy not available")

            # Build LP with fixed topology
            node_list = list(nodes)
            n_flow = 0
            flow_map = {}
            for (s, t), d in prob.demand.items():
                if d <= 0:
                    continue
                allowed = prob.tunnel_edge_sets.get((s, t), frozenset())
                for (u, v) in sorted(allowed):
                    flow_map[(s, t, u, v)] = n_flow
                    n_flow += 1
            mlu_idx = n_flow
            n_vars = mlu_idx + 1

            obj = np.zeros(n_vars)
            obj[mlu_idx] = 1.0

            rows = []
            row_lb = []
            row_ub = []

            sc = prob.scaled_capacity
            nd = {k: v / sc for k, v in prob.demand.items()}

            for (s, t), d in nd.items():
                if d <= 0:
                    continue
                allowed = prob.tunnel_edge_sets.get((s, t), frozenset())
                for node in node_list:
                    entries = {}
                    for (u, v) in sorted(allowed):
                        fcol = flow_map.get((s, t, u, v))
                        if fcol is None:
                            continue
                        if v == node:
                            entries[fcol] = entries.get(fcol, 0.0) + 1.0
                        if u == node:
                            entries[fcol] = entries.get(fcol, 0.0) - 1.0
                    if entries:
                        if node == s:
                            rows.append(dict(entries)); row_lb.append(-d); row_ub.append(-d)
                        elif node == t:
                            rows.append(dict(entries)); row_lb.append(d); row_ub.append(d)
                        else:
                            rows.append(dict(entries)); row_lb.append(0.0); row_ub.append(0.0)

            for (s, t), d in nd.items():
                if d <= 0:
                    continue
                allowed = prob.tunnel_edge_sets.get((s, t), frozenset())
                for (u, v) in sorted(allowed):
                    fcol = flow_map.get((s, t, u, v))
                    if fcol is None:
                        continue
                    undir = (u, v) if (u, v) in set(candidates) else (v, u)
                    if undir not in selected:
                        entries = {fcol: 1.0}
                        rows.append(entries); row_lb.append(-np.inf); row_ub.append(0.0)
                    else:
                        # Active edge: flow <= capacity
                        pass

            for de in [(u, v) for (u, v) in candidates] + [(v, u) for (u, v) in candidates]:
                entries = {mlu_idx: -1.0}
                for (s, t) in prob.demand:
                    fcol = flow_map.get((s, t, de[0], de[1]))
                    if fcol is not None:
                        entries[fcol] = 1.0
                if len(entries) > 1:
                    rows.append(entries); row_lb.append(-np.inf); row_ub.append(0.0)

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
            lp.col_lower_ = np.zeros(n_vars)
            lp.col_upper_ = np.full(n_vars, np.inf)
            lp.col_upper_[mlu_idx] = 1.0
            lp.row_lower_ = np.array(row_lb)
            lp.row_upper_ = np.array(row_ub)
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
            if status == highspy.HighsModelStatus.kOptimal:
                mlu = float(h.getSolution().col_value[mlu_idx])
                change_count = len(selected - prob.current_edges) + len(
                    prob.current_edges - selected
                )
                obj_val = 2.0 * change_count + mlu
                sid = compute_stable_topology_id(nodes, list(selected))
                feasible_topologies.append((obj_val, change_count, mlu, sid, selected))

        # Sort by objective (as MILP does)
        feasible_topologies.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        # Run MILP
        result = solve_doppler_with_enumeration(prob)
        assert result.has_solutions

        # MILP's best objective should match exhaustive best objective
        milp_best_obj = result.objective_best.objective_value
        exhausive_best_obj = feasible_topologies[0][0] if feasible_topologies else float("inf")
        assert abs(milp_best_obj - exhausive_best_obj) < 1e-6, (
            f"MILP best obj {milp_best_obj} != exhaustive best obj {exhausive_best_obj}"
        )

        # MILP's best MLU should be within tolerance
        milp_best_mlu = result.objective_best.validated_mlu
        exhaustive_best_mlu = feasible_topologies[0][2]
        assert abs(milp_best_mlu - exhaustive_best_mlu) < 1e-5


class TestNoGoodCut:
    def test_basic(self):
        edges = [("a", "b"), ("b", "c"), ("a", "c")]
        idx = {e: i for i, e in enumerate(edges)}
        coeffs, lower, upper = make_no_good_cut(
            frozenset({("a", "b")}), edges, idx, 3
        )
        assert lower == 0.0  # 1 - |S| = 1 - 1 = 0
        assert coeffs[0] == -1.0  # selected: -x
        assert coeffs[1] == 1.0   # unselected: +x
        assert coeffs[2] == 1.0   # unselected: +x

    def test_all_selected(self):
        edges = [("a", "b"), ("b", "c")]
        idx = {e: i for i, e in enumerate(edges)}
        coeffs, lower, upper = make_no_good_cut(
            frozenset({("a", "b"), ("b", "c")}), edges, idx, 2
        )
        assert lower == -1.0  # 1 - 2 = -1
        # All selected means all -x terms
        assert all(c == -1.0 for c in coeffs)
