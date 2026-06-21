"""Method registry for topology programming dispatch.

Maps topology_programming_method strings to MethodConfig entries,
replacing the old if/elif dispatch chain in simulator.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from onset.open_doppler import (
    solve_edge_flow_changes_mlu,
    solve_path_flow_budget,
    solve_path_flow_core,
)


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for a topology programming method."""

    name: str
    is_milp: bool
    objective_mode: str | None  # "changes_plus_mlu" or "mlu"
    solver_method: str | None  # "doppler", "onset_v3", "onset_v2", "onset"
    solve_fn: Callable | None  # solver callable for MILP methods
    uses_ecmp_multisol: bool
    description: str
    handler: Callable[..., None] | None = (
        None  # (Simulation, MethodConfig) -> None; resolved at import time
    )


# ---------------------------------------------------------------------------
# Solver callables: maps solver_method string → solver function
# ---------------------------------------------------------------------------

_SOLVER_CALLABLES: dict[str, Callable] = {
    "doppler": solve_edge_flow_changes_mlu,
    "onset_v3": lambda p: solve_edge_flow_changes_mlu(p, objective_mode="mlu"),
    "onset_v2": solve_path_flow_core,
    "onset": solve_path_flow_budget,
}


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

from onset.handlers import (  # noqa: E402 -- imported after MethodConfig to break a cycle
    _run_baseline,
    _run_bvt,
    _run_cache,
    _run_cli,
    _run_greylambda,
    _run_milp_method,
    _run_otp,
    _run_tbe,
)

_METHOD_REGISTRY: dict[str, MethodConfig] = {
    "baseline": MethodConfig(
        name="baseline",
        handler=_run_baseline,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Evaluate the input topology without reconfiguration",
    ),
    "doppler": MethodConfig(
        name="doppler",
        handler=_run_milp_method,
        is_milp=True,
        objective_mode="changes_plus_mlu",
        solver_method="doppler",
        solve_fn=_SOLVER_CALLABLES["doppler"],
        uses_ecmp_multisol=False,
        description="Doppler reconnaissance defense (TNSM 2024)",
    ),
    "onset_v3": MethodConfig(
        name="onset_v3",
        handler=_run_milp_method,
        is_milp=True,
        objective_mode="mlu",
        solver_method="onset_v3",
        solve_fn=_SOLVER_CALLABLES["onset_v3"],
        uses_ecmp_multisol=True,
        description="ONSET DDoS defense — post major revision (TDSC 2025)",
    ),
    "onset_v2": MethodConfig(
        name="onset_v2",
        handler=_run_milp_method,
        is_milp=True,
        objective_mode="mlu",
        solver_method="onset_v2",
        solve_fn=_SOLVER_CALLABLES["onset_v2"],
        uses_ecmp_multisol=False,
        description="ONSET DDoS defense — path-based formulation (TDSC 2025)",
    ),
    "onset": MethodConfig(
        name="onset",
        handler=_run_milp_method,
        is_milp=True,
        objective_mode="mlu",
        solver_method="onset",
        solve_fn=_SOLVER_CALLABLES["onset"],
        uses_ecmp_multisol=False,
        description="Original topology programming formulation",
    ),
    "OTP": MethodConfig(
        name="OTP",
        handler=_run_otp,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Offline Traffic Provisioning — shortcut-link heuristic",
    ),
    "greylambda": MethodConfig(
        name="greylambda",
        handler=_run_greylambda,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Greylambda — add circuits on fully-congested edges",
    ),
    "cache": MethodConfig(
        name="cache",
        handler=_run_cache,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Cache-based defense (Defender module)",
    ),
    "BVT": MethodConfig(
        name="BVT",
        handler=_run_bvt,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Bandwidth-variable transceiver emulation",
    ),
    "TBE": MethodConfig(
        name="TBE",
        handler=_run_tbe,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Temporary bandwidth expansion during flashcrowd",
    ),
    "cli": MethodConfig(
        name="cli",
        handler=_run_cli,
        is_milp=False,
        objective_mode=None,
        solver_method=None,
        solve_fn=None,
        uses_ecmp_multisol=False,
        description="Interactive CLI mode",
    ),
}


def _resolve_method(name: str) -> MethodConfig:
    """Resolve a topology_programming_method string to a MethodConfig.

    Performs exact match first, then falls back to case-insensitive
    substring match (e.g., "Doppler-v2" matches "doppler").
    """
    if name in _METHOD_REGISTRY:
        return _METHOD_REGISTRY[name]

    name_lower = name.lower()
    for key, config in _METHOD_REGISTRY.items():
        if key.lower() in name_lower:
            return config

    raise ValueError(
        f"Unknown topology_programming_method: {name!r}. "
        f"Valid: {sorted(_METHOD_REGISTRY.keys())}"
    )
