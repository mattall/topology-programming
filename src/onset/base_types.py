"""
Canonical, backend-neutral types for topology optimization.

These types are immutable value objects that never import gurobipy.
They are the single source of truth for data flowing between the
simulator, backend solvers, validation, and reporting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from struct import pack
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OptimizerStatus(Enum):
    """Overall termination status of an optimization run.

    Describes enumeration outcome, not optimality.
    """

    TOP_K_REACHED = auto()
    EXHAUSTED = auto()
    TIME_LIMIT_WITH_SOLUTION = auto()
    TIME_LIMIT_WITHOUT_SOLUTION = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    SOLVER_ERROR = auto()
    VALIDATION_FAILED = auto()


class BackendProvenance(Enum):
    OPEN = "open"
    GUROBI_LEGACY = "gurobi-legacy"


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

BINARY_TOLERANCE = 1e-6
FLOW_TOLERANCE_ABSOLUTE = 1e-7
FLOW_TOLERANCE_RELATIVE = 1e-7


# ---------------------------------------------------------------------------
# Helper: topology identity
# ---------------------------------------------------------------------------


def _canonical_node_order(nodes: Sequence[str]) -> List[str]:
    """Return nodes sorted by UTF-8 byte order (deterministic)."""
    return sorted(nodes, key=lambda s: s.encode("utf-8"))


def _canonical_edge_tuple(u: str, v: str) -> Tuple[str, str]:
    """Return (lower, higher) in canonical order."""
    return (u, v) if u <= v else (v, u)


def compute_stable_topology_id(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
) -> str:
    """SHA-256 hex digest of canonical JSON.

    JSON shape: {"nodes": [...], "edges": [[u,v], ...]}
    Both arrays in canonical UTF-8 byte order, edges with lower endpoint first.
    """
    ordered_nodes = _canonical_node_order(nodes)
    ordered_edges = sorted(
        _canonical_edge_tuple(u, v) for (u, v) in edges
    )
    payload = json.dumps(
        {"nodes": ordered_nodes, "edges": ordered_edges},
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().lower()


def compute_legacy_topology_id(
    candidate_edge_order: Sequence[Tuple[str, str]],
    selected: FrozenSet[Tuple[str, str]],
) -> str:
    """Legacy bit-vector ID for report compatibility.

    Packs a bit per candidate edge (in legacy order) into a double-precision
    float.  Only 53 mantissa bits are available; IDs are unreliable beyond
    that threshold.
    """
    bits = "".join(
        "1" if (u, v) in selected or (v, u) in selected else "0"
        for (u, v) in candidate_edge_order
    )
    return str(
        np.frombuffer(
            pack(">Q", int(bits[::-1].ljust(64, "0"), 2)),
            dtype=np.float64,
        )[0]
    )


# ---------------------------------------------------------------------------
# OptimizationProblem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PathProblemData:
    """Precomputed path-flow data used by path_flow_budget / path_flow_core builders.

    Built once by build_optimization_problem when solver="onset" or
    solver="onset_v2", then stored on OptimizationProblem.path_data.
    """

    path_list: Tuple[Tuple[str, ...], ...]
    commodity_to_paths: Dict[Tuple[str, str], Tuple[int, ...]]
    candidate_edge_indices: Tuple[int, ...]
    path_candidate_map: Tuple[Tuple[int, ...], ...]
    supergraph_directed_edges: Tuple[Tuple[str, str], ...]
    link_path_map: Tuple[Tuple[int, ...], ...]
    core_edge_set: FrozenSet[Tuple[str, str]]


@dataclass(frozen=True)
class OptimizationProblem:
    """Immutable description of a topology optimization problem.

    All validation is performed at construction time via __post_init__.
    The object is immutable after creation.
    """

    canonical_node_order: Tuple[str, ...]
    canonical_candidate_edges: Tuple[Tuple[str, str], ...]
    legacy_candidate_edge_order: Tuple[Tuple[str, str], ...]
    current_edges: FrozenSet[Tuple[str, str]]
    txp_count: Dict[str, int]
    demand: Dict[Tuple[str, str], float]
    tunnel_edge_sets: Dict[
        Tuple[str, str], FrozenSet[Tuple[str, str]]
    ]
    link_capacity: float
    scale_factor: float
    congestion_threshold_upper_bound: float
    top_k: int
    optimizer_time_limit: float
    retain_commodity_flows: bool = False
    path_data: Optional[_PathProblemData] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        nodes = set(self.canonical_node_order)
        # Node consistency
        if len(nodes) != len(self.canonical_node_order):
            raise ValueError("Duplicate node IDs in canonical_node_order")
        if any(not isinstance(n, str) for n in nodes):
            raise ValueError("All node IDs must be strings")

        # Edge consistency
        for (u, v) in self.canonical_candidate_edges:
            if u not in nodes or v not in nodes:
                raise ValueError(
                    f"Candidate edge ({u},{v}) references unknown node"
                )
            if u != _canonical_edge_tuple(u, v)[0]:
                raise ValueError(
                    f"Candidate edge ({u},{v}) not in canonical order"
                )

        # Current edges are subset of candidates
        candidate_set = set(self.canonical_candidate_edges)
        for (u, v) in self.current_edges:
            canon = _canonical_edge_tuple(u, v)
            if canon not in candidate_set:
                raise ValueError(
                    f"Current edge ({u},{v}) is not a candidate edge"
                )

        # Transponder keys cover all nodes
        for n in nodes:
            if n not in self.txp_count:
                raise ValueError(f"Node {n} missing from txp_count")

        # Demands finite, non-negative
        for (s, t), d in self.demand.items():
            if not np.isfinite(d) or d < 0:
                raise ValueError(
                    f"Demand ({s},{t}) = {d} must be finite and non-negative"
                )

        # Capacity and scale
        if not np.isfinite(self.link_capacity) or self.link_capacity < 0:
            raise ValueError("link_capacity must be finite and non-negative")
        if not np.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")
        if self.link_capacity / self.scale_factor <= 0:
            raise ValueError(
                "Scaled capacity <= 0; scale_factor may exceed raw capacity"
            )

        # Congestion bound
        if not (0.0 <= self.congestion_threshold_upper_bound <= 1.0):
            raise ValueError(
                "congestion_threshold_upper_bound must be in [0, 1]"
            )

        # Top-K and time
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.optimizer_time_limit <= 0:
            raise ValueError("optimizer_time_limit must be positive")

        # Tunnel edge consistency: every tunnel edge is a candidate
        for (s, t), allowed in self.tunnel_edge_sets.items():
            for (u, v) in allowed:
                canon = _canonical_edge_tuple(u, v)
                if canon not in candidate_set:
                    raise ValueError(
                        f"Tunnel edge ({u},{v}) for demand ({s},{t}) "
                        "is not in candidate set"
                    )

        # Legacy order consistency
        if set(self.legacy_candidate_edge_order) != candidate_set:
            raise ValueError(
                "legacy_candidate_edge_order must have same elements "
                "as canonical_candidate_edges"
            )

        # Path-data consistency (if present)
        if self.path_data is not None:
            pd = self.path_data
            if len(pd.path_candidate_map) != len(pd.path_list):
                raise ValueError(
                    "path_candidate_map length must equal path_list length"
                )
            if len(pd.link_path_map) != len(pd.supergraph_directed_edges):
                raise ValueError(
                    "link_path_map length must equal supergraph_directed_edges length"
                )
            for (s, t), idxs in pd.commodity_to_paths.items():
                if (s, t) not in self.demand:
                    raise ValueError(
                        f"commodity_to_paths key ({s},{t}) not in demand"
                    )
                for pi in idxs:
                    if pi >= len(pd.path_list) or pi < 0:
                        raise ValueError(
                            f"Path index {pi} for ({s},{t}) out of range"
                        )

        # Warning: legacy ID precision limit
        if len(self.legacy_candidate_edge_order) > 53:
            import logging
            logging.getLogger(__name__).warning(
                "Legacy topology ID has >53 candidate edges (%d); "
                "float-precision ID is unreliable.",
                len(self.legacy_candidate_edge_order),
            )

    @property
    def node_count(self) -> int:
        return len(self.canonical_node_order)

    @property
    def candidate_edge_count(self) -> int:
        return len(self.canonical_candidate_edges)

    @property
    def scaled_capacity(self) -> float:
        """Capacity after applying scale_factor (used as model capacity)."""
        return self.link_capacity / self.scale_factor

    @property
    def normalized_demand(self) -> Dict[Tuple[str, str], float]:
        """Demand normalized by scaled_capacity."""
        sc = self.scaled_capacity
        return {k: v / sc for k, v in self.demand.items()}


# ---------------------------------------------------------------------------
# TopologySolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologySolution:
    """A single validated topology solution from an optimization run.

    Immutable.  All fields are set at construction and never change.
    """

    selected_edges: FrozenSet[Tuple[str, str]]
    added: FrozenSet[Tuple[str, str]]
    dropped: FrozenSet[Tuple[str, str]]
    commodity_flows: Optional[
        Dict[Tuple[str, str, str, str], float]
    ]
    aggregate_edge_loads: Dict[Tuple[str, str], float]
    solver_mlu: float
    validated_mlu: float
    change_count: int
    objective_value: float
    stable_topology_id: str
    legacy_topology_id: str
    provenance: BackendProvenance
    proven_optimal: bool

    def __post_init__(self) -> None:
        # Basic sanity
        if not (0.0 <= self.solver_mlu <= 1.0 + 1e-9):
            raise ValueError("solver_mlu out of range")
        if not (0.0 <= self.validated_mlu <= 1.0 + 1e-9):
            raise ValueError("validated_mlu out of range")

    @classmethod
    def empty(cls, provenance: BackendProvenance) -> "TopologySolution":
        """Create a placeholder (used when no solutions are found)."""
        return cls(
            selected_edges=frozenset(),
            added=frozenset(),
            dropped=frozenset(),
            commodity_flows=None,
            aggregate_edge_loads={},
            solver_mlu=0.0,
            validated_mlu=0.0,
            change_count=0,
            objective_value=float("inf"),
            stable_topology_id="",
            legacy_topology_id="",
            provenance=provenance,
            proven_optimal=False,
        )


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable collection of solutions from a single optimization run."""

    solutions: Tuple[TopologySolution, ...]
    status: OptimizerStatus
    wall_time: float
    backend: BackendProvenance
    solve_count: int
    backend_diagnostics: Optional[str] = None
    baseline_feasible: bool = False
    baseline_mlu: Optional[float] = None

    def __post_init__(self) -> None:
        if self.solve_count < 0:
            raise ValueError("solve_count must be non-negative")
        if self.wall_time < 0:
            raise ValueError("wall_time must be non-negative")

    @property
    def has_solutions(self) -> bool:
        return len(self.solutions) > 0

    @property
    def objective_best(self) -> Optional[TopologySolution]:
        """Repository index 0 after deterministic sort."""
        return self.solutions[0] if self.solutions else None

    @property
    def selected_solution(self) -> Optional[TopologySolution]:
        """Lowest MLU, then fewer changes, then lower objective, then stable ID."""
        if not self.solutions:
            return None
        return min(
            self.solutions,
            key=lambda s: (
                s.validated_mlu,
                s.change_count,
                s.objective_value,
                s.stable_topology_id,
            ),
        )

    @classmethod
    def empty(
        cls,
        status: OptimizerStatus,
        wall_time: float,
        backend: BackendProvenance,
        solve_count: int,
        diagnostics: Optional[str] = None,
    ) -> "OptimizationResult":
        return cls(
            solutions=(),
            status=status,
            wall_time=wall_time,
            backend=backend,
            solve_count=solve_count,
            backend_diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Status mapping (SciPy milp status codes -> OptimizerStatus)
# ---------------------------------------------------------------------------


def map_milp_status(
    scipy_status: int,
    has_incumbent: bool,
    incumbent_valid: bool,
    is_first_solve: bool,
    has_prior_solutions: bool,
    top_k_reached: bool,
) -> OptimizerStatus:
    """Map a SciPy milp status code to the canonical OptimizerStatus.

    Parameters
    ----------
    scipy_status : int
        Status code from scipy.optimize.milp (0=optimal, 1=limit, 2=time,
        3=infeasible, 4=unbounded).
    has_incumbent : bool
        Whether the solve returned an incumbent solution.
    incumbent_valid : bool
        Whether the incumbent passed independent validation.
    is_first_solve : bool
        True if this was the first solve (no prior no-good cuts).
    has_prior_solutions : bool
        True if any prior solves contributed validated solutions.
    top_k_reached : bool
        True if the number of unique validated solutions equals top_k.

    Returns
    -------
    OptimizerStatus
    """
    if scipy_status == 0:  # success / optimal
        if top_k_reached:
            return OptimizerStatus.TOP_K_REACHED
        # Continue enumeration - caller handles this
        return OptimizerStatus.TOP_K_REACHED  # will be corrected by caller

    if scipy_status == 2:  # time limit
        if has_incumbent and incumbent_valid:
            return OptimizerStatus.TIME_LIMIT_WITH_SOLUTION
        return OptimizerStatus.TIME_LIMIT_WITHOUT_SOLUTION

    if scipy_status == 3:  # infeasible
        if is_first_solve:
            return OptimizerStatus.INFEASIBLE
        return OptimizerStatus.EXHAUSTED

    if scipy_status == 4:  # unbounded
        return OptimizerStatus.UNBOUNDED

    # Status 1 (iteration/other limit) or unexpected
    return OptimizerStatus.SOLVER_ERROR
