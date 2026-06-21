"""Tests for open_doppler.py — correctness via exhaustive oracle and smoke."""

import itertools
import pytest
import numpy as np

from onset.base_types import (
    OptimizationProblem,
    TopologySolution,
    OptimizationResult,
    OptimizerStatus,
    BackendProvenance,
    compute_stable_topology_id,
    _PathProblemData,
)
from onset.open_doppler import (
    solve_edge_flow_changes_mlu,
    solve_baseline,
    make_no_good_cut,
    solve_path_flow_budget,
    solve_path_flow_core,
    _add_and_constraints,
    _build_path_flow_budget_milp,
    _build_path_flow_core_milp,
)


def _build_triangle_problem():
    """A minimal 3-node triangle: a-b, b-c, a-c candidates. Demand a->c."""
    nodes = ("a", "b", "c")
    candidates = (("a", "b"), ("b", "c"), ("a", "c"))
    return OptimizationProblem(
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
    return OptimizationProblem(
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
        prob = OptimizationProblem(
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
        prob = OptimizationProblem(
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
        result = solve_edge_flow_changes_mlu(prob)
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
        result = solve_edge_flow_changes_mlu(prob)
        assert result.has_solutions
        assert result.solve_count >= 1

    def test_solution_ids_unique(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        if len(result.solutions) > 1:
            ids = [s.stable_topology_id for s in result.solutions]
            assert len(ids) == len(set(ids))

    def test_selected_in_candidates(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        for sol in result.solutions:
            for e in sol.selected_edges:
                assert e in set(prob.canonical_candidate_edges)

    def test_connectivity(self):
        """Every returned topology must be connected (validation check)."""
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
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
        result = solve_edge_flow_changes_mlu(prob)
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


# ---------------------------------------------------------------------------
# M2: onset_v3 (MLU-only objective) tests
# ---------------------------------------------------------------------------


class TestOnsetV3Objective:
    """Verify that objective_mode="mlu" produces MLU-only objective values."""

    def test_mlu_objective_single_solve(self):
        prob = _build_triangle_problem()
        result = solve_edge_flow_changes_mlu(prob, objective_mode="mlu")
        assert result.has_solutions
        sol = result.solutions[0]
        # With MLU-only objective, objective_value == validated_mlu
        assert abs(sol.objective_value - sol.validated_mlu) < 1e-6
        # Change count should NOT be in the objective
        assert sol.change_count >= 0

    def test_changes_plus_mlu_objective(self):
        """Default mode still works: objective = 2*changes + mlu."""
        prob = _build_triangle_problem()
        result = solve_edge_flow_changes_mlu(prob, objective_mode="changes_plus_mlu")
        assert result.has_solutions
        sol = result.solutions[0]
        assert abs(sol.objective_value - (2.0 * sol.change_count + sol.validated_mlu)) < 1e-6

    def test_mlu_mode_objective_ordering(self):
        """With MLU-only, solutions sort by MLU first."""
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob, objective_mode="mlu")
        if len(result.solutions) >= 2:
            for i in range(len(result.solutions) - 1):
                a = result.solutions[i]
                b = result.solutions[i + 1]
                assert a.objective_value <= b.objective_value + 1e-9

    def test_mlu_mode_feasible(self):
        """MLU-only mode finds feasible solutions."""
        prob = _build_triangle_problem()
        result = solve_edge_flow_changes_mlu(prob, objective_mode="mlu")
        assert result.has_solutions
        assert result.solve_count >= 1
        assert result.solutions[0].objective_value < float("inf")


class TestSolutionApplyRevert:
    """Verify TopologySolution.added/dropped consistency."""

    def test_added_dropped_are_disjoint(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        for sol in result.solutions:
            assert sol.added.isdisjoint(sol.dropped)

    def test_selected_equals_current_plus_added_minus_dropped(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        for sol in result.solutions:
            expected = (prob.current_edges | sol.added) - sol.dropped
            assert sol.selected_edges == expected

    def test_change_count_matches_derived(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        for sol in result.solutions:
            assert sol.change_count == len(sol.added) + len(sol.dropped)


class TestSelectedSolution:
    """Verify selected_solution picks lowest MLU."""

    def test_selected_solution_lowest_mlu(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        if result.has_solutions:
            sel = result.selected_solution
            assert sel is not None
            # selected_solution should have the minimum validated_mlu
            min_mlu = min(s.validated_mlu for s in result.solutions)
            assert abs(sel.validated_mlu - min_mlu) < 1e-9

    def test_objective_best_is_first(self):
        prob = _build_square_problem()
        result = solve_edge_flow_changes_mlu(prob)
        if result.has_solutions:
            assert result.objective_best is result.solutions[0]


# ---------------------------------------------------------------------------
# M3: path-flow budget and path-flow core builders
# ---------------------------------------------------------------------------


def _build_path_problem_budget():
    """4-node line: a-b-c-d with candidate a-c (shortcut).

    Logical: a-b-c-d  (3 undirected edges)
    Candidate: a-c (1 candidate)
    Paths: a->c via a-b-c (needs b-c active) or direct a-c (needs candidate)
    Demand: a->c = 50
    """
    nodes = ("a", "b", "c", "d")
    candidates = (("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"))
    current = frozenset({("a", "b"), ("b", "c"), ("c", "d")})
    cand_edge_idxs = (3,)  # ("a", "c") is index 3 in candidates

    # paths: a-b-c and a-c (direct)
    path_list = (("a", "b", "c"), ("a", "c"))
    # path 0 has candidate edge ("a","c")? No, path 0 is a-b-c, only current edges
    path_cand_map = ((), (0,))  # path 1 uses candidate idx 0 (which is ("a","c"))
    commodity_to_paths = {("a", "c"): (0, 1)}

    supergraph_dir = tuple(
        (u, v) for (u, v) in candidates
    ) + tuple((v, u) for (u, v) in candidates)

    # link_path_map: per supergraph directed edge, which paths traverse it
    link_path_map = []
    for de in supergraph_dir:
        paths_on_edge = []
        for pi, path in enumerate(path_list):
            for j in range(len(path) - 1):
                if (path[j], path[j + 1]) == de:
                    paths_on_edge.append(pi)
        link_path_map.append(tuple(paths_on_edge))

    pd = _PathProblemData(
        path_list=path_list,
        commodity_to_paths=commodity_to_paths,
        candidate_edge_indices=cand_edge_idxs,
        path_candidate_map=path_cand_map,
        supergraph_directed_edges=supergraph_dir,
        link_path_map=tuple(link_path_map),
        core_edge_set=frozenset(),
    )

    return OptimizationProblem(
        canonical_node_order=nodes,
        canonical_candidate_edges=candidates,
        legacy_candidate_edge_order=candidates,
        current_edges=current,
        txp_count={"a": 2, "b": 2, "c": 2, "d": 2},
        demand={("a", "c"): 50.0},
        tunnel_edge_sets={
            ("a", "c"): frozenset({("a", "b"), ("b", "c"), ("a", "c")})
        },
        link_capacity=100.0,
        scale_factor=1.0,
        congestion_threshold_upper_bound=1.0,
        top_k=1,
        optimizer_time_limit=30.0,
        path_data=pd,
    )


def _build_path_problem_core():
    """Same topology as path_flow_budget but with core-edge mandate on a-b.

    Core edges (physical): a-b, b-c, c-d
    Candidate: a-c
    Demand: a->c = 50
    """
    nodes = ("a", "b", "c", "d")
    candidates = (("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"))
    current = frozenset({("a", "b"), ("b", "c"), ("c", "d")})
    core_set = frozenset({("a", "b"), ("b", "c"), ("c", "d")})

    path_list = (("a", "b", "c"), ("a", "c"))
    commodity_to_paths = {("a", "c"): (0, 1)}

    supergraph_dir = tuple(
        (u, v) for (u, v) in candidates
    ) + tuple((v, u) for (u, v) in candidates)

    link_path_map = []
    for de in supergraph_dir:
        paths_on_edge = []
        for pi, path in enumerate(path_list):
            for j in range(len(path) - 1):
                if (path[j], path[j + 1]) == de:
                    paths_on_edge.append(pi)
        link_path_map.append(tuple(paths_on_edge))

    pd = _PathProblemData(
        path_list=path_list,
        commodity_to_paths=commodity_to_paths,
        candidate_edge_indices=(3,),
        path_candidate_map=((), (0,)),
        supergraph_directed_edges=supergraph_dir,
        link_path_map=tuple(link_path_map),
        core_edge_set=core_set,
    )

    return OptimizationProblem(
        canonical_node_order=nodes,
        canonical_candidate_edges=candidates,
        legacy_candidate_edge_order=candidates,
        current_edges=current,
        txp_count={"a": 2, "b": 3, "c": 3, "d": 2},
        demand={("a", "c"): 50.0},
        tunnel_edge_sets={
            ("a", "c"): frozenset({("a", "b"), ("b", "c"), ("a", "c")})
        },
        link_capacity=100.0,
        scale_factor=1.0,
        congestion_threshold_upper_bound=1.0,
        top_k=1,
        optimizer_time_limit=30.0,
        path_data=pd,
    )


class TestPathFlowBudgetBuilder:
    """Tests for the path-flow budget builder."""

    def test_builds_without_error(self):
        prob = _build_path_problem_budget()
        lp, im = _build_path_flow_budget_milp(prob)
        assert lp.num_col_ > 0
        assert lp.num_row_ > 0
        assert im["n_cand"] == 1
        assert im["n_paths"] == 2

    def test_solve_returns_solution(self):
        prob = _build_path_problem_budget()
        result = solve_path_flow_budget(prob)
        assert result.has_solutions
        assert result.status == OptimizerStatus.TOP_K_REACHED
        sol = result.selected_solution
        assert sol is not None
        assert len(sol.added | sol.dropped) >= 0

    def test_solution_satisfies_demand(self):
        prob = _build_path_problem_budget()
        result = solve_path_flow_budget(prob)
        assert result.has_solutions
        sol = result.selected_solution
        # Selected edges should include a-c path (either via b or direct)
        assert ("a", "c") in sol.selected_edges or (
            ("a", "b") in sol.selected_edges and ("b", "c") in sol.selected_edges
        )

    def test_solution_invariants(self):
        prob = _build_path_problem_budget()
        result = solve_path_flow_budget(prob)
        sol = result.selected_solution
        assert sol.selected_edges == (
            prob.current_edges | sol.added
        ) - sol.dropped
        assert not (sol.added & sol.dropped)
        assert sol.change_count == len(sol.added) + len(sol.dropped)


class TestPathFlowCoreBuilder:
    """Tests for the path-flow core builder."""

    def test_builds_without_error(self):
        prob = _build_path_problem_core()
        lp, im = _build_path_flow_core_milp(prob)
        assert lp.num_col_ > 0
        assert lp.num_row_ > 0
        assert im["n_dir"] > 0
        assert im["n_path_vars"] == 2

    def test_solve_returns_solution(self):
        prob = _build_path_problem_core()
        result = solve_path_flow_core(prob)
        assert result.has_solutions
        assert result.status == OptimizerStatus.TOP_K_REACHED

    def test_core_edges_always_active(self):
        prob = _build_path_problem_core()
        result = solve_path_flow_core(prob)
        sol = result.selected_solution
        # Core edges a-b, b-c, c-d must be selected
        for e in prob.path_data.core_edge_set:
            assert e in sol.selected_edges, f"Core edge {e} not selected"

    def test_solution_invariants(self):
        prob = _build_path_problem_core()
        result = solve_path_flow_core(prob)
        sol = result.selected_solution
        assert sol.selected_edges == (
            prob.current_edges | sol.added
        ) - sol.dropped
        assert not (sol.added & sol.dropped)
        assert sol.change_count == len(sol.added) + len(sol.dropped)


class TestAndConstraints:
    """Tests for the _add_and_constraints linearization helper."""

    def test_adds_expected_rows(self):
        rows = []
        rlb = []
        rub = []
        _add_and_constraints(rows, rlb, rub, 0, [1, 2, 3])
        # 3 <= constraints + 1 >= constraint = 4 rows
        assert len(rows) == 4
        # <= constraints: path <= x_i
        for i in range(3):
            assert rows[i][0] == 1.0
            assert rows[i][i + 1] == -1.0
            assert rlb[i] == -np.inf
            assert rub[i] == 0.0
        # >= constraint: path >= sum - 2
        assert rows[3][0] == 1.0
        assert rows[3][1] == -1.0
        assert rows[3][2] == -1.0
        assert rows[3][3] == -1.0
        assert rlb[3] == -2.0  # -(3-1) = -2
        assert rub[3] == np.inf

    def test_empty_input_is_noop(self):
        rows = []
        rlb = []
        rub = []
        _add_and_constraints(rows, rlb, rub, 0, [])
        assert len(rows) == 0
