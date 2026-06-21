import os
from collections import defaultdict
from hashlib import sha1
from os import makedirs
from typing import cast

from onset.alpwolf import AlpWolf
from onset.base_types import OptimizationResult, TopologySolution
from onset.constants import SCRIPT_HOME
from onset.method_registry import _METHOD_REGISTRY, _resolve_method
from onset.preprocessing import build_optimization_problem
from onset.reporter import write_optimization_reports
from onset.utilities.config import CROSSFIRE
from onset.utilities.gml_to_dot import Gml_to_dot
from onset.utilities.logger import logger
from onset.utilities.reconfiguration import get_reconfig_time
from onset.utilities.result_io import read_result_val
from onset.validation import (
    _evaluate_te,
    _system,
    evaluate_performance_from_adding_link,
    export_logical_topo_to_gml,
    validate_simulation_inputs,
)


class Simulation:
    def __init__(
        self,
        network_name: str,
        num_hosts: int,
        test_name: str,
        iterations=0,
        te_method="-ecmp",
        start_clean=False,
        magnitude=100 * 10**10,
        traffic_file="",
        topology_programming_method="",
        fallow_transponders=0,
        use_heuristic="",
        candidate_link_choice_method="max",
        congestion_threshold_upper_bound=0.8,
        congestion_threshold_lower_bound=0.3,
        attack_proportion="",
        shakeroute=False,
        net_dir=False,
        fallow_tx_allocation_strategy="static",
        fallow_tx_allocation_file="",
        line_code="fixed",
        scale_down_factor=1,
        salt="",
        top_k=100,
        optimizer_time_limit_minutes=1,
        parallel_path_computation=True,
    ):
        """Simulation initializer

        Args:
            network_name (str):
            num_hosts (int): number of hosts in the network

            test_name (str): Unique string, identifier for the simulation results

            iterations (int, optional): Number of simulated epochs. Defaults to 0.

            te_method (str, optional): Internal traffic engineering method.
                Supported values are '-ecmp' and '-mcf'. Defaults to '-ecmp'.

            start_clean (bool, optional): Overwrites simulation traffic matrices with new data. Defaults to False.

            magnitude (int, optional): Mean value for traffic demand between nodes
                - used only when start_clean is True. Defaults to 100*10**10.

            traffic_file (str, optional): Path to traffic matrix file. Ignored when start_clean is True. Defaults to "".

            topology_programming_method (str, optional): Topology programming method to use
                Options: "cli", "cache", "onset", or "greylambda". Defaults to "".

            fallow_transponders (int, optional): total fallow transponders per node. Defaults to 5.

            use_heuristic (str, optional): Used when `topology_programming_method` is cache.
                Limits the number of cached solutions to generate. Defaults to "".

            candidate_link_choice_method (str, optional): "all" or "heuristic". Determines how to choose candidate links
                when `topology_programming_method` is set to "cache". Defaults to "all".

            congestion_threshold_upper_bound (float, optional): Congestion at or above this level triggers a topology
                programming response from the network. Defaults to 0.8.

            congestion_threshold_lower_bound (float, optional): Congestion at or below this level reverse any topology
                programming update to the network. Defaults to 0.3.

            attack_proportion (str, optional): Percent of total traffic that is from attackers.
                Only used for file naming. Defaults to "".

            shakeroute (bool, optional): shakeroute experiments simulate fiber cut failure scenarios. Defaults to False.

            net_dir (bool, optional): director for network topology files. Defaults to False.

            fallow_tx_allocation_strategy (str, optional): Describes how to allocate transponders. Defaults to 'static'.
                can also be "dynamic" or "file". If it is "dynamic" then the top 90th percentile nodes get 'fallow_transponders' and others get 'fallow_transponders/2'.

            fallow_tx_allocation_file (str, optional): _description_. Defaults to "".
                Used when fallow_tx_allocation_strategy = "file", contains a path to a file that explicitly states the number of fallow transponders per node.

            salt (str, optional): Used to generate unique file name for temp files when experiments with similar parameters are running simultaneously. Defaults to "".
        """
        makedirs(".temp", exist_ok=True)
        self.nonce = (
            "./.temp/"
            + sha1(
                "".join(
                    [
                        str(x)
                        for x in [
                            network_name,
                            test_name,
                            iterations,
                            te_method,
                            start_clean,
                            magnitude,
                            traffic_file,
                            topology_programming_method,
                            fallow_transponders,
                            use_heuristic,
                            candidate_link_choice_method,
                            congestion_threshold_upper_bound,
                            congestion_threshold_lower_bound,
                            fallow_tx_allocation_strategy,
                            fallow_tx_allocation_file,
                            attack_proportion,
                            scale_down_factor,
                            top_k,
                            optimizer_time_limit_minutes,
                            salt,
                        ]
                    ]
                ).encode()
            ).hexdigest()
        )
        logger.info(f"Nonce: {self.nonce}")
        logger.info(f"Initializing simulator: {network_name} {test_name} {iterations}")
        self.network_name = network_name
        self.num_hosts = int(num_hosts)
        self.test_name = test_name
        self.iterations = iterations
        self.te_method = te_method
        self.traffic_file = traffic_file
        self.start_clean = start_clean
        self.magnitude = magnitude
        self.topology_programming_method = topology_programming_method
        self.topo_file = ""
        self.hosts_file = ""
        self.fallow_transponders = fallow_transponders
        self.use_heuristic = use_heuristic
        self.candidate_link_choice_method = candidate_link_choice_method
        self.congestion_threshold_upper_bound = congestion_threshold_upper_bound
        self.congestion_threshold_lower_bound = congestion_threshold_lower_bound
        self.line_code = line_code
        self.exit_early = False
        self.attack_proportion = attack_proportion
        self.shakeroute = shakeroute
        self.net_dir = net_dir
        self.fallow_tx_allocation_strategy = fallow_tx_allocation_strategy
        self.fallow_tx_allocation_file = fallow_tx_allocation_file
        self.scale_down_factor = scale_down_factor
        self.top_k = top_k
        self.optimizer_time_limit_minutes = optimizer_time_limit_minutes
        self.parallel_path_computation = parallel_path_computation
        self.topo_solved = None
        self.optimization_result = None
        self._applied_solution: TopologySolution | None = None
        self.multi_sol_time = "NaN"
        self.multi_sol_number_best_sol = "NaN"
        self.multi_sol_best_mlu = "NaN"
        # Set Experiment ID
        if self.use_heuristic.isdigit():
            self.EXPERIMENT_ID = "_".join(
                [
                    network_name,
                    test_name,
                    str(fallow_transponders),
                    self.attack_proportion,
                    self.te_method,
                    str(self.top_k),
                ]
            ).replace("heuristic", f"heuristic_{self.use_heuristic}")
        else:
            self.EXPERIMENT_ID = "_".join(
                [
                    network_name,
                    test_name,
                    fallow_tx_allocation_strategy,
                    str(fallow_transponders),
                    topology_programming_method,
                    candidate_link_choice_method,
                    str(optimizer_time_limit_minutes),
                    self.attack_proportion,
                    self.te_method,
                    str(top_k),
                ]
            )

        # Set Experiment absolute path
        if self.net_dir:
            self.EXPERIMENT_ABSOLUTE_PATH = os.path.join(
                SCRIPT_HOME,
                "data",
                "results",
                self.network_name,
                self.EXPERIMENT_ID,
            )

        else:
            self.EXPERIMENT_ABSOLUTE_PATH = os.path.join(
                SCRIPT_HOME, "data", "results", self.EXPERIMENT_ID
            )
        logger.info(f"Saving experiment results to: {self.EXPERIMENT_ABSOLUTE_PATH}")
        # The following three commands must be ordered as follows.
        # self.base_graph = FiberGraph(self.name)
        self.validate_simulation_inputs()
        self.wolf = AlpWolf(
            self.topo_file,
            self.fallow_transponders,
            fallow_tx_allocation_strategy=self.fallow_tx_allocation_strategy,
            fallow_tx_allocation_file=self.fallow_tx_allocation_file,
            top_k=self.top_k,
        )
        if self.topology_programming_method == "TBE":
            self.wolf.restrict_bandwidth(0.8)
        makedirs("data/graphs/img", exist_ok=True)

    def _system(self, command: str):
        return _system(command)

    def _evaluate_te(self, topo_file, result_path, traffic_file=""):
        if traffic_file == "":
            traffic_file = self.traffic_file
        return _evaluate_te(
            topo_file=topo_file,
            result_path=result_path,
            traffic_file=traffic_file,
            shakeroute=self.shakeroute,
            hosts_file=self.hosts_file,
            te_method=self.te_method,
            exit_early=self.exit_early,
            network_name=self.network_name,
        )

    def evaluate_performance_from_adding_link(self, circuits_to_add=1):
        return evaluate_performance_from_adding_link(
            wolf=self.wolf,
            network_name=self.network_name,
            experiment_absolute_path=self.EXPERIMENT_ABSOLUTE_PATH,
            experiment_id=self.EXPERIMENT_ID,
            use_heuristic=self.use_heuristic,
            te_method=self.te_method,
            traffic_file=self.traffic_file,
            shakeroute=self.shakeroute,
            hosts_file=self.hosts_file,
            exit_early=self.exit_early,
            circuits_to_add=circuits_to_add,
        )

    def export_logical_topo_to_gml(self, name, G=None):
        if G is None:
            G = self.wolf.logical_graph
        return export_logical_topo_to_gml(name=name, G=G)

    def perform_sim(
        self,
        circuits: int = 1,
        start_iter: int = 0,
        end_iter: int | None = None,
        repeat: bool = False,
        unit: str = "Gbps",
        demand_factor: int = 1,
        dry: bool = False,
    ) -> dict[str, list[object]] | str | None:
        self.unit = unit
        self.circuits = circuits
        self.demand_factor = demand_factor
        sim_param_tag = f"{self.circuits}_{start_iter}_{end_iter}_{int(repeat)}_{unit}_{self.demand_factor:.1f}"
        self.new_circuit: list[object] = []
        self.chaff: list[object] = []
        if end_iter is None:
            end_iter = self.iterations

        return_data: defaultdict[str, list[object]] = defaultdict(list)
        dry_path: str | None = None
        self.return_data = return_data
        self.circuits_added = False

        if CROSSFIRE:
            self.sig_add_circuits = True
        else:
            self.sig_add_circuits = False

        self.sig_drop_circuits = False
        logger.info("Performing simulation ")
        name = self.network_name
        iterations = self.iterations
        traffic = self.traffic_file

        EXPERIMENT_ID = self.EXPERIMENT_ID
        EXPERIMENT_ABSOLUTE_PATH = self.EXPERIMENT_ABSOLUTE_PATH

        # Open traffic file and pass a new line from the file for every iteration.
        with open(traffic, "rb") as fob:
            tm_data = fob.readlines()

        self.PREV_ITER_ABS_PATH = ""

        # Make iteration range
        if repeat:
            iter_range = []
            for i in range(start_iter, end_iter + 1):
                iter_range.extend([i, i])
        else:
            iter_range = list(range(start_iter, end_iter))

        for i, iter_i in enumerate(iter_range):
            j = i % 2
            if self.shakeroute:
                ITERATION_ID = self.topology_programming_method.replace(".", "")

            elif repeat:
                ITERATION_ID = (
                    f"{name}_{iter_i}-{j}-{iterations}_{sim_param_tag}".replace(".", "")
                )
            else:
                ITERATION_ID = (
                    f"{name}_{iter_i}-0-{iterations}_{sim_param_tag}".replace(".", "")
                )

            self.ITERATION_ID = ITERATION_ID.replace(".", "")
            self.ITERATION_REL_PATH = ITERATION_REL_PATH = os.path.join(
                EXPERIMENT_ID, ITERATION_ID
            ).replace(".", "")
            self.ITERATION_ABS_PATH = ITERATION_ABS_PATH = os.path.join(
                EXPERIMENT_ABSOLUTE_PATH, ITERATION_ID
            ).replace(".", "")

            logger.info(f"Initializing Traffic Matrix ({i}, {iter_i})")
            tm_i_data = [
                str(float(demand_val) * self.demand_factor)
                for demand_val in tm_data[iter_i - 1].split()
            ]
            tm_i_data_to_temp_file = " ".join(tm_i_data)

            if dry:
                dry_path = ITERATION_ABS_PATH
                continue

            self.temp_tm_i_file = self.nonce + "-" + ITERATION_ID
            with open(self.temp_tm_i_file, "w") as temp_fob:
                temp_fob.write(tm_i_data_to_temp_file)
            logger.debug("Initializing Traffic Matrix --- Complete")
            reconfig_time = 0.0
            self.new_circuit = []
            self.chaff = []
            self.flux_circuits: list[object] = []
            self.optimization_result = None
            self._applied_solution = None
            self.multi_sol_time = "NaN"
            self.multi_sol_number_best_sol = "NaN"
            self.multi_sol_best_mlu = "NaN"
            makedirs(ITERATION_ABS_PATH, exist_ok=True)

            return_data["Experiment"].append(EXPERIMENT_ID)
            return_data["Iteration"].append(f"{iter_i}-{j}")
            self.opt_time = float("NaN")
            self.max_load = 1
            # try:
            # create dot graph and put it into the appropriate file for this run.
            iteration_topo = ITERATION_ABS_PATH
            # logger.debug("Drawing initial topology")
            # initial_topo_img_file = f"data/graphs/img/0-{ITERATION_ID}"
            # draw_graph(self.wolf.logical_graph,
            #            name=initial_topo_img_file)
            # logger.debug(f"Drawing initial topology --- Complete: {initial_topo_img_file}")

            # Dispatch to the registered method handler
            config = _resolve_method(self.topology_programming_method)
            if config.handler is None:
                raise ValueError(
                    f"No handler for topology method {self.topology_programming_method!r}"
                )
            config.handler(self, config)

            # self.base_graph.G = Graph.copy(self.wolf.logical_graph)

            # Save this iteration graph to GML and Dot
            if self.circuits_added:
                pass  # circuit_tag was updated above.
            #     circuit_tag = "circuit-{}-{}".format(u,v)
            #     circuit_tag = "circuit-{}".format(".".joint())
            else:
                pass
            # updated_topology_file = iteration_topo + circuit_tag

            if self.line_code == "BVT":
                self.wolf.logical_graph.edges[("63", "133")]["capacity"] *= 0.75

            updated_topology_file = iteration_topo
            self.export_logical_topo_to_gml(updated_topology_file + ".gml")
            Gml_to_dot(self.wolf.logical_graph, iteration_topo + ".dot", unit=unit)

            iter_congestion = self._evaluate_te(
                iteration_topo + ".dot",
                ITERATION_REL_PATH,
                traffic_file=self.temp_tm_i_file,
            )
            if len(self.new_circuit) > 0:
                if self.topology_programming_method == "doppler":
                    reconfig_time = 1.0
                else:
                    reconfig_time = get_reconfig_time(
                        updated_topology_file + ".gml", self.new_circuit
                    )
            else:
                reconfig_time = 0.0

            return_data["ReconfigTime"].append(reconfig_time)
            return_data["Strategy"].append(
                f"{self.te_method} {self.topology_programming_method}"
            )
            return_data["CandidateLinkSet"].append(self.candidate_link_choice_method)

            return_data["Routing"].append(f"{self.te_method}".strip("-").upper())
            return_data["Defense"].append(f"{self.topology_programming_method}")

            return_data["Congestion"].append(
                read_result_val(
                    os.path.join(
                        ITERATION_ABS_PATH,
                        "MaxExpCongestionVsIterations.dat",
                    )
                )
            )
            return_data["Loss"].append(
                read_result_val(
                    os.path.join(
                        ITERATION_ABS_PATH,
                        "CongestionLossVsIterations.dat",
                    )
                )
            )
            return_data["Throughput"].append(
                read_result_val(
                    os.path.join(
                        ITERATION_ABS_PATH,
                        "TotalThroughputVsIterations.dat",
                    )
                )
            )
            return_data["Total Links Added"].append(len(self.new_circuit))
            return_data["Links Added"].append(self.new_circuit)
            # return_data["Total Flux Links"].append(len(self.flux_circuits))
            return_data["Total Links Dropped"].append(len(self.chaff))
            return_data["Links Dropped"].append(self.chaff)
            return_data["Link Bandwidth Coefficient"].append(self.max_load)
            return_data["Demand Factor"].append(self.demand_factor)
            return_data["Optimization Time"].append(self.opt_time)
            write_optimization_reports(
                result=self.optimization_result,
                return_data=return_data,
                iteration_abs_path=self.ITERATION_ABS_PATH,
                iteration_id=ITERATION_ID,
                te_method=self.te_method,
                opt_time=self.opt_time,
                multi_sol_time=getattr(self, "multi_sol_time", "NaN"),
                multi_sol_number_best_sol=getattr(
                    self, "multi_sol_number_best_sol", "NaN"
                ),
                multi_sol_best_mlu=getattr(self, "multi_sol_best_mlu", "NaN"),
            )
            if iter_congestion == "SIG_EXIT":
                return None

            congestion = float(iter_congestion)

            if congestion >= self.congestion_threshold_upper_bound:
                self.sig_add_circuits = True

            elif (
                congestion <= self.congestion_threshold_lower_bound
                and len(self.chaff) > 0
            ):
                self.sig_drop_circuits = True

            if CROSSFIRE:
                self.sig_drop_circuits = True

            if self.sig_drop_circuits:
                logger.info(
                    f"Max link util, {congestion}, below threshold, {self.congestion_threshold_lower_bound}. Reverting changes."
                )
                self.revert_solution()
                self.new_circuit = []
                self.chaff = []
                # self.drop_circuits = self.flux_circuits[:]
                # if len(self.drop_circuits) > 0:
                #     for dc in self.drop_circuits:
                #         u, v = dc
                #         for _ in range(self.circuits):
                #             self.wolf.drop_circuit(u, v)
                #     self.flux_circuits = []
                #     self.circuits_added = False
                #     self.sig_drop_circuits = False

            # except BaseException as e:
            #     logger.error("Unknown Error", exc_info=True, stack_info=True)
            #     # self._system("rm %s" % temp_tm_i)
            #     if self.topology_programming_method == "greylambda":
            #         return -1
            #     return return_data

            # finally:
            # # Remove the temp file.
            # self._system("rm %s" % temp_tm_i)
            self.PREV_ITER_ABS_PATH = ITERATION_ABS_PATH
            # self.base_graph.set_weights(CONGESTION_PATH)
            # self.base_graph.draw_graphs(ITERATION_ABS_PATH)
            # edge_congestion = get_edge_congestion(CONGESTION_PATH)
            # PLOT_DIR = os.path.join(ITERATION_ABS_PATH, "plot_dir")
            # makedirs(PLOT_DIR, exist_ok=True)
            # try:
            #     congestion_heatmap(edge_congestion, os.path.join(
            #         PLOT_DIR, "edge_congestion_heatmap"))
            # except:
            #     logger.warning("Did not make heatmap for this iteration.")
            #     pass
        return dry_path if dry_path is not None else dict(return_data)

    def validate_simulation_inputs(self):
        """Validate the files implied by ``network_name``.
        If the topo file is not found, then the program halts.
        traffic and hosts files are generated if they are needed.
        Side effect: assigns object variables to verified filesystem paths:
        self.topo_file, self.hosts_file, and self.traffic_file.
        """
        self.topo_file, self.hosts_file, self.traffic_file = validate_simulation_inputs(
            network_name=self.network_name,
            shakeroute=self.shakeroute,
            topology_programming_method=self.topology_programming_method,
            traffic_file=self.traffic_file,
            start_clean=self.start_clean,
            num_hosts=self.num_hosts,
            iterations=self.iterations,
            magnitude=self.magnitude,
        )

    # ------------------------------------------------------------------
    # Open-backend AlpWolf integration
    # ------------------------------------------------------------------
    # Open-backend AlpWolf integration
    # ------------------------------------------------------------------

    def apply_solution(self, solution: TopologySolution):
        """Apply a TopologySolution to AlpWolf.  Call revert_solution to undo."""
        self._applied_solution = solution
        self.new_circuit = list(solution.added)
        self.chaff = list(solution.dropped)
        for u, v in solution.dropped:
            self.wolf.drop_circuit(u, v)
        for u, v in solution.added:
            self.wolf.add_circuit(u, v)
        self.circuits_added = bool(solution.added or solution.dropped)

    def revert_solution(self):
        """Undo the last applied TopologySolution."""
        sol = self._applied_solution
        if sol is None:
            return
        for u, v in sol.added:
            self.wolf.drop_circuit(u, v)
        for u, v in sol.dropped:
            self.wolf.add_circuit(u, v)
        self._applied_solution = None
        self.circuits_added = False

    # ------------------------------------------------------------------
    # Unified topology optimization entry point
    # ------------------------------------------------------------------

    def _run_topology_optimization(
        self,
        objective_mode: str = "changes_plus_mlu",
        solver: str = "doppler",
        top_k: int | None = None,
    ) -> OptimizationResult:
        """Build a OptimizationProblem from AlpWolf state, solve, store result.

        Parameters
        ----------
        objective_mode : str
            "changes_plus_mlu" (Doppler) or "mlu" (onset_v3/onset_v2/onset).
        solver : str
            Solver method: "doppler", "onset_v3", "onset_v2", "onset".
            Controls solver selection and problem construction.
        top_k : int, optional
            Override self.top_k (e.g. 1 for MCF single-solve).
        """
        if top_k is None:
            top_k = self.top_k

        problem = build_optimization_problem(
            logical_graph=self.wolf.logical_graph,
            base_graph=self.wolf.base_graph,
            demand_matrix_file=self.temp_tm_i_file,
            network_name=self.network_name,
            txp_count=self.wolf.txp_count,
            candidate_set=self.candidate_link_choice_method,
            scale_down_factor=self.scale_down_factor,
            congestion_threshold_upper_bound=self.congestion_threshold_upper_bound,
            top_k=top_k,
            optimizer_time_limit=self.optimizer_time_limit_minutes * 60.0,
            use_cache=True,
            parallel_execution=self.parallel_path_computation,
            solver=solver,
        )

        method_config = _METHOD_REGISTRY.get(solver)
        if method_config is None or method_config.solve_fn is None:
            raise ValueError(f"No solver for method: {solver}")
        solve_fn = method_config.solve_fn
        result = solve_fn(problem)
        self.optimization_result = result
        self.opt_time = result.wall_time
        return cast(OptimizationResult, result)
