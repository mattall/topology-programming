"""Unit tests for canonical types (base_types.py)."""

from dataclasses import FrozenInstanceError

import pytest

from onset.base_types import (
    BackendProvenance,
    OptimizationProblem,
    OptimizationResult,
    OptimizerStatus,
    TopologySolution,
    compute_legacy_topology_id,
    compute_stable_topology_id,
    map_milp_status,
)


class TestStableTopologyID:
    def test_deterministic(self):
        a = compute_stable_topology_id(["a", "b", "c"], [("a", "b"), ("b", "c")])
        b = compute_stable_topology_id(["c", "b", "a"], [("b", "c"), ("a", "b")])
        assert a == b
        assert len(a) == 64  # SHA-256 hex

    def test_order_independent(self):
        sid = compute_stable_topology_id(["x", "y"], [("x", "y")])
        sid2 = compute_stable_topology_id(["y", "x"], [("y", "x")])
        assert sid == sid2

    def test_different_topologies(self):
        a = compute_stable_topology_id(["a", "b"], [("a", "b")])
        b = compute_stable_topology_id(["a", "b", "c"], [("a", "b"), ("b", "c")])
        assert a != b


class TestLegacyID:
    def test_basic(self):
        lid = compute_legacy_topology_id(
            [("a", "b"), ("b", "c")], frozenset([("a", "b")])
        )
        assert isinstance(lid, str)

    def test_empty(self):
        lid = compute_legacy_topology_id([], frozenset())
        assert lid == "0.0"


class TestOptimizationProblem:
    def test_valid_construction(self):
        p = OptimizationProblem(
            canonical_node_order=("a", "b", "c"),
            canonical_candidate_edges=(("a", "b"), ("b", "c")),
            legacy_candidate_edge_order=(("a", "b"), ("b", "c")),
            current_edges=frozenset([("a", "b")]),
            txp_count={"a": 2, "b": 2, "c": 1},
            demand={("a", "c"): 50.0},
            tunnel_edge_sets={("a", "c"): frozenset([("a", "b"), ("b", "c")])},
            link_capacity=100.0,
            scale_factor=10.0,
            congestion_threshold_upper_bound=1.0,
            top_k=5,
            optimizer_time_limit=60.0,
        )
        assert p.scaled_capacity == 10.0
        assert p.normalized_demand[("a", "c")] == 5.0
        assert p.node_count == 3
        assert p.candidate_edge_count == 2

    def test_rejects_unknown_node_in_edge(self):
        with pytest.raises(ValueError, match="unknown node"):
            OptimizationProblem(
                canonical_node_order=("a", "b"),
                canonical_candidate_edges=(("a", "b"), ("a", "c")),
                legacy_candidate_edge_order=(("a", "b"), ("a", "c")),
                current_edges=frozenset(),
                txp_count={"a": 1, "b": 1, "c": 1},
                demand={("a", "b"): 1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=100.0,
                scale_factor=10.0,
                congestion_threshold_upper_bound=1.0,
                top_k=1,
                optimizer_time_limit=60.0,
            )

    def test_rejects_current_edge_not_candidate(self):
        with pytest.raises(ValueError, match="not a candidate"):
            OptimizationProblem(
                canonical_node_order=("a", "b", "c"),
                canonical_candidate_edges=(("a", "b"),),
                legacy_candidate_edge_order=(("a", "b"),),
                current_edges=frozenset([("b", "c")]),
                txp_count={"a": 1, "b": 1, "c": 1},
                demand={("a", "b"): 1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=100.0,
                scale_factor=10.0,
                congestion_threshold_upper_bound=1.0,
                top_k=1,
                optimizer_time_limit=60.0,
            )

    def test_rejects_negative_scale(self):
        with pytest.raises(ValueError, match="scale_factor"):
            OptimizationProblem(
                canonical_node_order=("a", "b"),
                canonical_candidate_edges=(("a", "b"),),
                legacy_candidate_edge_order=(("a", "b"),),
                current_edges=frozenset(),
                txp_count={"a": 1, "b": 1},
                demand={("a", "b"): 1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=100.0,
                scale_factor=-1.0,
                congestion_threshold_upper_bound=1.0,
                top_k=1,
                optimizer_time_limit=60.0,
            )

    def test_rejects_zero_capacity_after_scale(self):
        with pytest.raises(ValueError, match="Scaled capacity"):
            OptimizationProblem(
                canonical_node_order=("a", "b"),
                canonical_candidate_edges=(("a", "b"),),
                legacy_candidate_edge_order=(("a", "b"),),
                current_edges=frozenset(),
                txp_count={"a": 1, "b": 1},
                demand={("a", "b"): 1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=0.0,
                scale_factor=1.0,
                congestion_threshold_upper_bound=1.0,
                top_k=1,
                optimizer_time_limit=60.0,
            )
        with pytest.raises(ValueError, match="scale_factor"):
            OptimizationProblem(
                canonical_node_order=("a", "b"),
                canonical_candidate_edges=(("a", "b"),),
                legacy_candidate_edge_order=(("a", "b"),),
                current_edges=frozenset(),
                txp_count={"a": 1, "b": 1},
                demand={("a", "b"): 1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=100.0,
                scale_factor=0.0,
                congestion_threshold_upper_bound=1.0,
                top_k=1,
                optimizer_time_limit=60.0,
            )

    def test_rejects_congestion_out_of_range(self):
        with pytest.raises(ValueError, match="congestion_threshold"):
            OptimizationProblem(
                canonical_node_order=("a", "b"),
                canonical_candidate_edges=(("a", "b"),),
                legacy_candidate_edge_order=(("a", "b"),),
                current_edges=frozenset(),
                txp_count={"a": 1, "b": 1},
                demand={("a", "b"): 1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=100.0,
                scale_factor=10.0,
                congestion_threshold_upper_bound=1.5,
                top_k=1,
                optimizer_time_limit=60.0,
            )

    def test_rejects_negative_demand(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            OptimizationProblem(
                canonical_node_order=("a", "b"),
                canonical_candidate_edges=(("a", "b"),),
                legacy_candidate_edge_order=(("a", "b"),),
                current_edges=frozenset(),
                txp_count={"a": 1, "b": 1},
                demand={("a", "b"): -1.0},
                tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
                link_capacity=100.0,
                scale_factor=10.0,
                congestion_threshold_upper_bound=1.0,
                top_k=1,
                optimizer_time_limit=60.0,
            )

    def test_immutable(self):
        p = OptimizationProblem(
            canonical_node_order=("a", "b"),
            canonical_candidate_edges=(("a", "b"),),
            legacy_candidate_edge_order=(("a", "b"),),
            current_edges=frozenset(),
            txp_count={"a": 1, "b": 1},
            demand={("a", "b"): 1.0},
            tunnel_edge_sets={("a", "b"): frozenset([("a", "b")])},
            link_capacity=100.0,
            scale_factor=10.0,
            congestion_threshold_upper_bound=1.0,
            top_k=1,
            optimizer_time_limit=60.0,
        )
        with pytest.raises(FrozenInstanceError):
            p.top_k = 5  # frozen dataclass


class TestTopologySolution:
    def test_construction(self):
        sol = TopologySolution(
            selected_edges=frozenset([("a", "b"), ("b", "c")]),
            added=frozenset([("b", "c")]),
            dropped=frozenset(),
            commodity_flows=None,
            aggregate_edge_loads={("a", "b"): 5.0, ("b", "c"): 5.0},
            solver_mlu=0.5,
            validated_mlu=0.5,
            change_count=1,
            objective_value=2.5,
            stable_topology_id="abc123",
            legacy_topology_id="1.0",
            provenance=BackendProvenance.OPEN,
            proven_optimal=True,
        )
        assert sol.change_count == 1

    def test_empty_placeholder(self):
        sol = TopologySolution.empty(BackendProvenance.OPEN)
        assert sol.stable_topology_id == ""
        assert sol.validated_mlu == 0.0

    def test_rejects_out_of_range_mlu(self):
        with pytest.raises(ValueError):
            TopologySolution(
                selected_edges=frozenset(),
                added=frozenset(),
                dropped=frozenset(),
                commodity_flows=None,
                aggregate_edge_loads={},
                solver_mlu=2.0,
                validated_mlu=0.5,
                change_count=0,
                objective_value=2.0,
                stable_topology_id="x",
                legacy_topology_id="x",
                provenance=BackendProvenance.OPEN,
                proven_optimal=False,
            )


class TestOptimizationResult:
    def test_empty_result(self):
        r = OptimizationResult.empty(
            OptimizerStatus.INFEASIBLE, 0.5, BackendProvenance.OPEN, 1
        )
        assert not r.has_solutions
        assert r.objective_best is None
        assert r.selected_solution is None

    def test_selection_rule(self):
        sol1 = TopologySolution(
            selected_edges=frozenset([("a", "b")]),
            added=frozenset(),
            dropped=frozenset(),
            commodity_flows=None,
            aggregate_edge_loads={},
            solver_mlu=0.3,
            validated_mlu=0.3,
            change_count=1,
            objective_value=2.3,
            stable_topology_id="a",
            legacy_topology_id="x",
            provenance=BackendProvenance.OPEN,
            proven_optimal=True,
        )
        sol2 = TopologySolution(
            selected_edges=frozenset([("a", "b"), ("b", "c")]),
            added=frozenset([("b", "c")]),
            dropped=frozenset(),
            commodity_flows=None,
            aggregate_edge_loads={},
            solver_mlu=0.2,
            validated_mlu=0.2,
            change_count=2,
            objective_value=4.2,
            stable_topology_id="b",
            legacy_topology_id="y",
            provenance=BackendProvenance.OPEN,
            proven_optimal=True,
        )
        r = OptimizationResult(
            solutions=(sol1, sol2),
            status=OptimizerStatus.TOP_K_REACHED,
            wall_time=1.0,
            backend=BackendProvenance.OPEN,
            solve_count=2,
        )
        # Selected = lowest MLU (sol2 has 0.2 vs sol1 0.3)
        assert r.selected_solution.stable_topology_id == "b"
        # Objective best = lowest objective (sol1 has 2.3 vs sol2 4.2)
        assert r.objective_best.stable_topology_id == "a"

    def test_rejects_negative_wall_time(self):
        with pytest.raises(ValueError):
            OptimizationResult(
                solutions=(),
                status=OptimizerStatus.INFEASIBLE,
                wall_time=-1.0,
                backend=BackendProvenance.OPEN,
                solve_count=0,
            )


class TestStatusMapping:
    def test_optimal_top_k(self):
        s = map_milp_status(0, True, True, True, True, True)
        assert s == OptimizerStatus.TOP_K_REACHED

    def test_time_limit_with_solution(self):
        s = map_milp_status(2, True, True, False, True, False)
        assert s == OptimizerStatus.TIME_LIMIT_WITH_SOLUTION

    def test_time_limit_without_solution(self):
        s = map_milp_status(2, False, False, False, False, False)
        assert s == OptimizerStatus.TIME_LIMIT_WITHOUT_SOLUTION

    def test_first_infeasible(self):
        s = map_milp_status(3, False, False, True, False, False)
        assert s == OptimizerStatus.INFEASIBLE

    def test_later_infeasible_exhausted(self):
        s = map_milp_status(3, False, False, False, True, False)
        assert s == OptimizerStatus.EXHAUSTED

    def test_unbounded(self):
        s = map_milp_status(4, False, False, True, False, False)
        assert s == OptimizerStatus.UNBOUNDED

    def test_unexpected_status(self):
        s = map_milp_status(1, False, False, True, False, False)
        assert s == OptimizerStatus.SOLVER_ERROR
