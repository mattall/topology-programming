from __future__ import annotations

import os
import logging
from itertools import combinations
from collections import Counter

from onset.method_registry import MethodConfig
from onset.utilities.logger import logger
from onset.utilities.post_process import read_link_congestion_to_dict


def _run_milp_method(sim, config: MethodConfig) -> None:
    """Unified handler for all four MILP methods (doppler, onset_v3, onset_v2, onset)."""
    import logging
    logger = logging.getLogger(__name__)

    sim.max_load = 0.9

    # Determine top_k: 1 for MCF, configurable for ECMP
    if sim.te_method == "-mcf":
        top_k = 1
    else:
        top_k = getattr(sim, 'top_k', 100)

    result = sim._run_topology_optimization(
        objective_mode=config.objective_mode,
        solver=config.solver_method,
        top_k=top_k,
    )

    if result is None:
        return

    if sim.te_method == "-mcf":
        if result.has_solutions:
            sim.apply_solution(result.selected_solution)
    elif sim.te_method == "-ecmp" and config.uses_ecmp_multisol:
        if result.has_solutions:
            from onset.reporter import evaluate_candidate_topologies
            best_sol, multi_time, best_idx, best_mlu = evaluate_candidate_topologies(
                solutions=result.solutions,
                wolf=sim.wolf,
                iteration_abs_path=sim.ITERATION_ABS_PATH,
                iteration_rel_path=sim.ITERATION_REL_PATH,
                hosts_file=sim.hosts_file,
                te_method=sim.te_method,
                temp_tm_file=sim.temp_tm_i_file,
                unit=sim.unit,
            )
            sim.multi_sol_time = multi_time
            sim.multi_sol_number_best_sol = best_idx
            sim.multi_sol_best_mlu = best_mlu
            if best_sol is not None:
                sim.apply_solution(best_sol)
    else:
        if result.has_solutions:
            sim.apply_solution(result.selected_solution)


def find_shortcut_link(congested_edges, existing_edges):
    node_counter = Counter()
    message = "Looking for a shortcut link among: "
    congested_edges = [
        e.strip("()").replace("s", "").split(",")
        for e in congested_edges
    ]
    for e in congested_edges:
        u, v = e
        node_counter.update((u, v))
        message += f"({u}, {v}) "
    logger.info(message)
    midpoint = max(node_counter, key=node_counter.get)
    terminals = []
    for c in congested_edges:
        this = c[:]
        if midpoint in this:
            this.remove(midpoint)
            terminals.append(this[0])
    shortcuts = [
        c
        for c in combinations(terminals, 2)
        if c[0] != c[1]
        and c not in existing_edges
    ]
    logger.info(
        f"Found the following shortcut: {shortcuts}"
    )
    return shortcuts


def _run_otp(sim, config: MethodConfig) -> None:
    edge_congestion_file = os.path.join(
        sim.PREV_ITER_ABS_PATH,
        "EdgeCongestionVsIterations.dat",
    )
    edge_congestion_d = read_link_congestion_to_dict(
        edge_congestion_file
    )
    congested_edges = [
        k
        for k in edge_congestion_d
        if edge_congestion_d[k] > 0.80
    ]

    shortcuts = find_shortcut_link(congested_edges, sim.wolf.logical_graph.edges())
    for edge in shortcuts:
        u, v = edge
        for _ in range(sim.circuits):
            sim.wolf.add_circuit(u, v, 100)
            sim.flux_circuits.append((u, v))
    # flux_circuits.extend(congested_edges)
    sim.sig_add_circuits = False
    return


def _run_greylambda(sim, config: MethodConfig) -> None:
    edge_congestion_file = os.path.join(
        sim.PREV_ITER_ABS_PATH,
        "EdgeCongestionVsIterations.dat",
    )
    edge_congestion_d = read_link_congestion_to_dict(
        edge_congestion_file
    )
    congested_edges = [
        k
        for k in edge_congestion_d
        if edge_congestion_d[k] == 1
    ]
    for edge in congested_edges:
        if isinstance(edge, str):
            u, v = edge.strip("()").replace("s", "").split(",")
        elif isinstance(edge, tuple) and len(edge) == 2:
            u, v = edge
        else:
            raise ValueError(f"Unexpected edge type: {type(edge)}")
        for _ in range(sim.circuits):
            added = sim.wolf.add_circuit(u, v)
            if added == 0:
                sim.circuits_added = True

    sim.flux_circuits.extend(congested_edges)
    sim.sig_add_circuits = False
    return


def _run_cache(sim, config: MethodConfig) -> None:
    from onset.defender import Defender
    defender = Defender(
        sim.network_name,
        sim.circuits,
        sim.candidate_link_choice_method,
        sim.use_heuristic,
        sim.PREV_ITER_ABS_PATH,
        sim.attack_proportion,
    )
    # TODO: Pass get_strategic_circuit the paths file from the previous iteration.
    sim.new_circuit = defender.get_strategic_circuit()
    if (
        type(sim.new_circuit) == tuple
        and len(sim.new_circuit) == 2
    ):
        logger.debug(
            "Adding {} ({}, {}) circuits.".format(
                sim.circuits, *sim.new_circuit
            )
        )
        for _ in range(sim.circuits):
            u, v = sim.new_circuit
            sim.wolf.add_circuit(u, v)
    return


def _run_bvt(sim, config: MethodConfig) -> None:
    edge_congestion_file = os.path.join(
        sim.PREV_ITER_ABS_PATH,
        "EdgeCongestionVsIterations.dat",
    )
    edge_congestion_d = read_link_congestion_to_dict(
        edge_congestion_file
    )
    congested_edges = [
        k
        for k in edge_congestion_d
        if edge_congestion_d[k] == 1
    ]

    sim.sig_add_circuits = False
    return


def _run_tbe(sim, config: MethodConfig) -> None:
    if "flashcrowd" in sim.traffic_file \
        and sim.demand_factor > 0.9:

        sim.wolf.relax_restricted_bandwidth()
    # sig_add_circuits = False
    return


def _run_cli(sim, config: MethodConfig) -> None:
    sim.wolf.cli()
