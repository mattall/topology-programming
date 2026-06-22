"""Pure functions for writing optimization reports and evaluating candidate
topologies via parallel ECMP evaluation.

Extracted from ``Simulation.save_doppler_results`` and
``Simulation._evaluate_candidate_topologies``.
"""

from __future__ import annotations

import os
from time import time
from typing import Any

from onset.base_types import OptimizationResult, TopologySolution

# ── local file writers (simple fmt, no dependency on post_process) ──────────


def _write_result_val(file_path: str, label: str, value: Any) -> None:
    with open(file_path, "a") as f:
        f.write(f"{label} {value}\n")


def _write_result_vals(
    file_path: str,
    id_label: str,
    val_label: str,
    val_dict: dict[str, float],
) -> None:
    with open(file_path, "a") as f:
        for k, v in val_dict.items():
            f.write(f"{id_label} {k} {val_label} {v}\n")


# ── write_optimization_reports ──────────────────────────────────────────────


def write_optimization_reports(
    result: OptimizationResult | None,
    return_data: dict[str, list[Any]],
    iteration_abs_path: str,
    iteration_id: str,
    te_method: str,
    opt_time: float,
    multi_sol_time: Any,
    multi_sol_number_best_sol: Any,
    multi_sol_best_mlu: Any,
) -> None:
    """Write the six Doppler report files and append to ``return_data``.

    Parameters
    ----------
    result:
        Optimization result from the open backend, or ``None`` when no
        optimization was run.
    return_data:
        Dictionary keyed by metric name whose values are lists to which
        per-iteration results are appended (mutated in place).
    iteration_abs_path:
        Absolute path to the iteration output directory.
    iteration_id:
        Identifier for the current iteration (used in legacy path only; kept
        for signature compatibility).
    te_method:
        TE method label (used in legacy path only; kept for sig compat).
    opt_time:
        Wall-clock time in seconds for the optimization run.
    multi_sol_time:
        Time spent in multi-solution parallel TE evaluation.
    multi_sol_number_best_sol:
        Index of the best solution found during multi-solution evaluation.
    multi_sol_best_mlu:
        MLU of the best solution found during multi-solution evaluation.
    """
    if result is None:
        return

    has_sol: bool = result.has_solutions

    if has_sol:
        total: Any = len(result.solutions)
        sel: TopologySolution | None = result.selected_solution
        obj_best: TopologySolution | None = result.objective_best

        optimal_id: Any = obj_best.stable_topology_id if obj_best else "NaN"
        curr_id: Any = sel.stable_topology_id if sel else "NaN"
        min_mlu: Any = round(sel.validated_mlu, 3) if sel else "NaN"
        mlu_dict: dict[str, float] = {
            str(i): sol.validated_mlu for i, sol in enumerate(result.solutions)
        }
    else:
        total = 0
        optimal_id = "NaN"
        curr_id = "NaN"
        min_mlu = "NaN"
        mlu_dict = {}

    # -- TotalSolutions.dat (count file) --
    _write_result_val(
        os.path.join(iteration_abs_path, "TotalSolutions.dat"),
        "Total Solutions",
        str(total),
    )

    # -- OptTime.dat --
    return_data.setdefault("Doppler Optimization Time", []).append(opt_time)
    _write_result_val(
        os.path.join(iteration_abs_path, "OptTime.dat"),
        "Optimization Time",
        str(opt_time),
    )

    # -- per-field report files --
    return_data.setdefault("Total Solutions", []).append(total)
    return_data.setdefault("Optimal Topology ID", []).append(optimal_id)
    return_data.setdefault("Current Topology ID", []).append(curr_id)
    return_data.setdefault("Doppler Min MLU", []).append(min_mlu)

    _write_result_val(
        os.path.join(iteration_abs_path, "OptimalTopoID.dat"),
        "Optimal Topology ID",
        optimal_id,
    )
    _write_result_val(
        os.path.join(iteration_abs_path, "CurrTopoID.dat"),
        "Current Topology ID",
        curr_id,
    )
    _write_result_val(
        os.path.join(iteration_abs_path, "DopplerMinMLU.dat"),
        "Doppler Min MLU",
        min_mlu,
    )

    if mlu_dict:
        _write_result_vals(
            os.path.join(iteration_abs_path, "DopplerMLU.dat"),
            "Solution ID",
            "Max Link Util",
            mlu_dict,
        )

    # -- multi-solution evaluation fields --
    return_data.setdefault("Doppler Multi-Sol Time", []).append(multi_sol_time)
    return_data.setdefault("Multi-sol Best Solution", []).append(
        multi_sol_number_best_sol
    )
    return_data.setdefault("Multi-sol Min MLU", []).append(multi_sol_best_mlu)


# ── evaluate_candidate_topologies ───────────────────────────────────────────


def evaluate_candidate_topologies(
    solutions: tuple[TopologySolution, ...],
    wolf: Any,
    iteration_abs_path: str,
    iteration_rel_path: str,
    hosts_file: str,
    te_method: str,
    temp_tm_file: str,
    unit: str,
    n_workers: int | None = None,
) -> tuple[TopologySolution | None, float, int | None, float | None]:
    """Apply, export, evaluate (ECMP in parallel), revert each solution.

    Returns the solution with the lowest validated MLU.

    Parameters
    ----------
    solutions:
        Sequence of ``TopologySolution`` objects from an ``OptimizationResult``.
    wolf:
        An ``AlpWolf`` instance whose logical graph is mutated and restored.
    iteration_abs_path:
        Absolute directory for writing per-solution ``.dot`` / ``.gml`` files.
    iteration_rel_path:
        Relative path prefix for TE result output.
    hosts_file:
        Path to the hosts file for TE evaluation.
    te_method:
        TE method label passed to ``evaluate_te``.
    temp_tm_file:
        Temporary traffic-matrix file path passed to ``evaluate_te``.
    unit:
        Unit string for ``Gml_to_dot`` (e.g. ``"Gbps"``).
    n_workers:
        Number of parallel ECMP evaluation workers.  Defaults to
        ``os.cpu_count()``, capped at the number of distinct solutions
        (max 32).

    Returns
    -------
    ``(best_solution, multi_sol_time, best_sol_index, best_mlu)``.
    ``best_solution`` is ``None`` when no TE evaluation produced a valid MLU.
    """
    # -- lazy imports (multiprocessing is heavy, graph deps are optional) --
    from multiprocessing import Manager, Pool

    from onset.utilities.graph import Gml_to_dot, write_gml
    from onset.utilities.post_process import evaluate_te

    solution_set: set = set()
    dot_files: dict[int, str] = {}
    sol_paths: dict[int, str] = {}
    id_to_solution: dict[int, TopologySolution] = {}

    for i, sol in enumerate(solutions):
        if i >= 32:
            break
        tid: str = sol.stable_topology_id
        if tid not in solution_set:
            # apply
            for u, v in sol.dropped:
                wolf.drop_circuit(u, v)
            for u, v in sol.added:
                wolf.add_circuit(u, v)

            sol_topo: str = iteration_abs_path + f"_sol_{i}.dot"
            Gml_to_dot(wolf.logical_graph, sol_topo, unit=unit)
            write_gml(wolf.logical_graph, iteration_abs_path + f"_sol_{i}.gml")

            # revert
            for u, v in sol.added:
                wolf.drop_circuit(u, v)
            for u, v in sol.dropped:
                wolf.add_circuit(u, v)

            solution_set.add(tid)
            id_to_solution[i] = sol
            dot_files[i] = sol_topo
            sol_paths[i] = iteration_rel_path + f"_sol_{i}"

    sol_ids: list[int] = sorted(dot_files.keys())
    manager = Manager()
    mlu_container = manager.dict({sid: "NaN" for sid in sol_ids})

    work: list[tuple] = []
    for sid in sol_ids:
        work.append(
            (
                dot_files[sid],
                hosts_file,
                te_method,
                sol_paths[sid],
                temp_tm_file,
                sid,
                mlu_container,
            )
        )

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, len(sol_ids))
    p = Pool(n_workers)
    start: float = time()
    p.starmap(evaluate_te, work)
    p.close()
    p.join()
    end: float = time()
    multi_sol_time: float = end - start

    valid_items: list[tuple[int, Any, float]] = []
    for k, v in mlu_container.items():
        pair: Any = v
        if isinstance(pair, tuple) and len(pair) == 2:
            m = pair[1]
            if isinstance(m, int | float):
                valid_items.append((k, pair[0], float(m)))

    if valid_items:
        (best_i, best_topo, best_mlu) = min(valid_items, key=lambda x: x[2])
        best_sol: TopologySolution | None = id_to_solution.get(best_i)
        return (best_sol, multi_sol_time, best_i, best_mlu)

    return (None, multi_sol_time, None, None)
