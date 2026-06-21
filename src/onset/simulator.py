import sys
import os
from time import time
from itertools import combinations
from collections import Counter, defaultdict
from typing import Optional

from onset.alpwolf import AlpWolf
from onset.constants import SCRIPT_HOME
from onset.preprocessing import build_optimization_problem
from onset.open_doppler import solve_edge_flow_changes_mlu, solve_path_flow_budget, solve_path_flow_core
from onset.base_types import TopologySolution, OptimizationResult
from onset.method_registry import _METHOD_REGISTRY, _resolve_method, MethodConfig
from onset.reporter import write_optimization_reports
from onset.utilities.config import CROSSFIRE

# Lazy import: gurobipy must not be loaded at module level.
# The open backend (default) does not require it; only legacy methods do.



from onset.utilities.diff_compare import diff_compare
from onset.utilities.gml_to_dot import Gml_to_dot
# from onset.utilities.logger import NewLogger
# logger = NewLogger().get_logger()
from onset.utilities.logger import logger
from onset.te import evaluate
from onset.utilities.plot_reconfig_time import get_reconfig_time
from onset.utilities.plotters import (
    cdf_average_congestion,
    cdf_churn,
    draw_graph,
    plot_points,
)
from onset.utilities.post_process import (
    read_link_congestion_to_dict,
    read_result_val,
)
from onset.utilities.sysUtils import count_lines, percent_diff
from onset.utilities.tmg import rand_gravity_matrix
from onset.utilities.graph_utils import write_gml
from networkx.relabel import relabel_nodes

from copy import deepcopy
from hashlib import sha1

from os import makedirs, system
from os.path import dirname, isfile

DRAW = False


# ------------------------------------------------------------------
# Handler functions for topology programming method dispatch
# ------------------------------------------------------------------

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

    def find_shortcut_link(congested_edges):
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
            and c not in sim.wolf.logical_graph.edges()
        ]
        logger.info(
            f"Found the following shortcut: {shortcuts}"
        )
        return shortcuts

    shortcuts = find_shortcut_link(congested_edges)
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
                sim.circuits
            ),
            *sim.new_circuit,
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

    # for edge in congested_edges:
    #     u, v = edge.strip("()").replace("s", "").split(",")
    #     u = int(u)
    #     v = int(v)
    #     for _ in range(circuits):
    #         sim.wolf.add_circuit(u, v)
    # flux_circuits.extend(congested_edges)
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
        scale_down_factor = 1,
        salt="",
        top_k=100,
        optimizer_time_limit_minutes=1,
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
        logger.info(
            f"Initializing simulator: {network_name} {test_name} {iterations}"
        )        
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
        self.congestion_threshold_upper_bound = (
            congestion_threshold_upper_bound
        )
        self.congestion_threshold_lower_bound = (
            congestion_threshold_lower_bound
        )
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
        self.topo_solved = None
        self.optimization_result = None
        self._applied_solution = None
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
                    str(self.top_k)
                ]
            ).replace("heuristic", "heuristic_{}".format(self.use_heuristic))
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
            top_k=self.top_k
        )
        if self.topology_programming_method == "TBE": 
            self.wolf.restrict_bandwidth(0.8)
        makedirs("data/graphs/img", exist_ok=True)

    def _system(self, command: str):
        logger.info("Calling system command: {}".format(command))
        return system(command)

    def _evaluate_te(self, topo_file, result_path, traffic_file=""):
        if traffic_file == "":
            traffic_file = self.traffic_file
        if self.shakeroute:
            result_path = os.path.join(self.network_name, result_path)
        result = evaluate(
            topo_file=topo_file,
            traffic_file=traffic_file,
            hosts_file=self.hosts_file,
            te_method=self.te_method,
            result_path=result_path,
            budget=3,
        )
        max_congestion = result.max_congestion
        logger.info("Max congestion: {}".format(max_congestion))
        if self.exit_early and float(max_congestion) == 1.0:
            logger.info("Max Congestion has reached 1. Ending simulation.")
            return "SIG_EXIT"
        return max_congestion

    def evaluate_performance_from_adding_link(self, circuits_to_add=1):
        # self.EXPERIMENT_ABSOLUTE_PATH += (
        #     "_circuits_{}".format(circuits_to_add))
        # self.EXPERIMENT_ID += ("_circuits_{}".format(circuits_to_add))
        path_churn = []
        congestion_change = []

        # prep directory
        GRAPHS_PATH = os.path.join(self.EXPERIMENT_ABSOLUTE_PATH, "graphs")
        makedirs(GRAPHS_PATH, exist_ok=True)

        # prep initial file
        initial_graph = self.wolf.logical_graph
        INITIAL_GRAPH_PATH = os.path.join(
            GRAPHS_PATH, self.network_name + "_0.dot"
        )
        INITIAL_RESULTS_REL_PATH = os.path.join(self.EXPERIMENT_ID, "__0")

        # Write Graphs to files
        self.export_logical_topo_to_gml(
            INITIAL_GRAPH_PATH.replace(".dot", ".gml"), G=initial_graph
        )
        Gml_to_dot(initial_graph, INITIAL_GRAPH_PATH)

        if (
            self._evaluate_te(INITIAL_GRAPH_PATH, INITIAL_RESULTS_REL_PATH)
            == "SIG_EXIT"
        ):
            return

        PATH_DIFF_FOLDER = os.path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "path_diff"
        )
        CONGESTION_DIFF_FOLDER = os.path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "congestion_diff"
        )
        makedirs(PATH_DIFF_FOLDER, exist_ok=True)
        makedirs(CONGESTION_DIFF_FOLDER, exist_ok=True)

        INITIAL_PATHS = os.path.join(
            self.EXPERIMENT_ABSOLUTE_PATH,
            "__0",
            "paths",
            self.te_method.strip("-") + "_0",
        )
        INITIAL_CONGESTION = os.path.join(
            self.EXPERIMENT_ABSOLUTE_PATH,
            "__0",
            "MaxExpCongestionVsIterations.dat",
        )
        # Run experiment
        if self.use_heuristic != "":
            # FIXME CHANGE BACK TO `candid_set='ranked'`
            # candidate_circuits = self.wolf.get_candidate_circuits(candid_set='ranked', k=5, l=5)
            candidate_circuits = self.wolf.get_candidate_circuits(
                candid_set="all"
            )
        else:
            candidate_circuits = self.wolf.get_candidate_circuits(
                candid_set="all"
            )

        for u, v in candidate_circuits:
            TEST_RESULTS_REL_PATH = os.path.join(
                self.EXPERIMENT_ID, "{}_{}".format(u, v)
            )
            test_alpwolf = deepcopy(self.wolf)
            for _ in range(circuits_to_add):
                test_alpwolf.add_circuit(u, v)

            TEST_GRAPH_PATH = os.path.join(
                GRAPHS_PATH, self.network_name + "_{}_{}.dot".format(u, v)
            )

            # Write Graphs to files
            Gml_to_dot(test_alpwolf.logical_graph, TEST_GRAPH_PATH)
            self.export_logical_topo_to_gml(
                TEST_GRAPH_PATH.replace(".dot", ".gml"),
                G=test_alpwolf.logical_graph,
            )

            if (
                self._evaluate_te(TEST_GRAPH_PATH, TEST_RESULTS_REL_PATH)
                == "SIG_EXIT"
            ):
                return

            TEST_PATHS = os.path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "{}_{}".format(u, v),
                "paths",
                self.te_method.strip("-") + "_0",
            )
            TEST_CONGESTION = os.path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "{}_{}".format(u, v),
                "MaxExpCongestionVsIterations.dat",
            )

            PATH_DIFF = os.path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "path_diff",
                "{}_{}.txt".format(u, v),
            )
            CONGESTION_DIFF = os.path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "congestion_diff",
                "{}_{}.txt".format(u, v),
            )

            self._system(
                "diff {} {} > {}".format(INITIAL_PATHS, TEST_PATHS, PATH_DIFF)
            )
            self._system(
                "diff {} {} > {}".format(
                    INITIAL_CONGESTION, TEST_CONGESTION, CONGESTION_DIFF
                )
            )
            path_churn.append(diff_compare(PATH_DIFF, "path"))
            congestion_change.append(diff_compare(CONGESTION_DIFF))
            if DRAW: 
                draw_graph( test_alpwolf.logical_graph, os.path.join(
                        GRAPHS_PATH, self.network_name + "_{}_{}".format(u, v)))

        PLOT_DIR = os.path.join(self.EXPERIMENT_ABSOLUTE_PATH, "plot_dir")
        CONGESTION_VS_PATHCHURN = os.path.join(
            PLOT_DIR, "congestion_vs_pathChurn"
        )
        makedirs(PLOT_DIR, exist_ok=True)
        plot_points(
            path_churn,
            congestion_change,
            "Path Churn",
            "Congestion Change",
            CONGESTION_VS_PATHCHURN,
        )
        CONGESTION_CDF = os.path.join(PLOT_DIR, "congestion_cdf")
        PATHCHURN_CDF = os.path.join(PLOT_DIR, "pathChurn_cdf")
        cdf_average_congestion(congestion_change, CONGESTION_CDF)
        cdf_churn(path_churn, PATHCHURN_CDF)

        return self.EXPERIMENT_ABSOLUTE_PATH

    def export_logical_topo_to_gml(self, name, G=None):
        if G == None:
            G = self.wolf.logical_graph

        # Relabels to be consistent with naming in Ripple.
        if 0:  # to make consistent with ripple examples.
            te_to_ripple_map = {
                node: ("sw" + str(int(node) - 1)) for (node) in G
            }

        te_to_ripple_map = {node: ("s{}".format(node)) for (node) in G}
        gml_view = relabel_nodes(G, te_to_ripple_map, copy=True)
        write_gml(gml_view, name)
        del gml_view

    def perform_sim(
        self,
        circuits=1,
        start_iter=0,
        end_iter=None,
        repeat=False,
        unit="Gbps",
        demand_factor=1,
        dry=False
    ):
        self.unit = unit
        self.circuits = circuits
        self.demand_factor = demand_factor
        sim_param_tag = f"{self.circuits}_{start_iter}_{end_iter}_{int(repeat)}_{unit}_{self.demand_factor:.1f}"
        self.new_circuit = []
        self.chaff = []
        if end_iter is None:
            end_iter = self.iterations

        return_data = self.return_data = defaultdict(list)
        self.circuits_added = False

        if CROSSFIRE == True:
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
            for i in range(start_iter, end_iter+1):
                iter_range.extend([i, i])
        else:
            iter_range = [i for i in range(start_iter, end_iter)]
                        
        for i, iter_i in enumerate(iter_range):
            
            j = i % 2 
            if self.shakeroute:
                ITERATION_ID = self.topology_programming_method.replace('.','')
            
            elif repeat:
                ITERATION_ID = f"{name}_{iter_i}-{j}-{iterations}_{sim_param_tag}".replace('.','')
            else:
                ITERATION_ID = f"{name}_{iter_i}-0-{iterations}_{sim_param_tag}".replace('.','')

            self.ITERATION_ID = ITERATION_ID.replace('.','')
            self.ITERATION_REL_PATH = ITERATION_REL_PATH = os.path.join(
                EXPERIMENT_ID, ITERATION_ID
            ).replace('.','')
            self.ITERATION_ABS_PATH = ITERATION_ABS_PATH = os.path.join(
                EXPERIMENT_ABSOLUTE_PATH, ITERATION_ID
            ).replace('.','')
            if False: #dry:
                return_data = ITERATION_ABS_PATH
                if i == end_iter: 
                    if repeat and j == 2:
                        return return_data
                    elif repeat and j == 1:
                        j += 1
                    elif repeat is False:
                        pass
                    else:
                        logger.error(f"Impossible path. i = {i}, j={j}, repeat = {repeat}, end_iter = {end_iter}, ITERATION_ABS_PATH = {ITERATION_ABS_PATH}")
                
                    if i < start_iter:
                        continue
                    if i > end_iter:
                        if repeat and i == (end_iter + 1) and j == 2:
                            pass
                        else:
                            continue
                    if repeat:
                        j += 1

                
                continue                    

            # CONGESTION_PATH = os.path.join(
            #     EXPERIMENT_ABSOLUTE_PATH,
            #     ITERATION_ID,
            #     "EdgeCongestionVsIterations.dat",
            # )
            # MAX_CONGESTION_PATH = os.path.join(
            #     EXPERIMENT_ABSOLUTE_PATH,
            #     ITERATION_ID,
            #     "MaxCongestionVsIterations.dat",
            # )
            logger.info(f"Initializing Traffic Matrix ({i}, {iter_i})")
            tm_i_data = [
                str(float(demand_val) * self.demand_factor)
                for demand_val in tm_data[iter_i-1].split()
            ]
            tm_i_data_to_temp_file = " ".join(tm_i_data)

            if dry:
                return_data = ITERATION_ABS_PATH
                continue
            
            self.temp_tm_i_file = self.nonce + "-" + ITERATION_ID
            with open(self.temp_tm_i_file, "w") as temp_fob:
                temp_fob.write(tm_i_data_to_temp_file)
            logger.debug("Initializing Traffic Matrix --- Complete")
            reconfig_time = 0
            self.new_circuit = []
            self.chaff = []
            self.flux_circuits = []
            self.optimization_result = None
            self._applied_solution = None
            self.multi_sol_time = "NaN"
            self.multi_sol_number_best_sol = "NaN"
            self.multi_sol_best_mlu = "NaN"
            makedirs(ITERATION_ABS_PATH, exist_ok=True)

            # if i == 360:
            #     print("BREAK")
            # if 300 <= i: #<= 335 :
            #     pass
            # else:
            #     continue

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
            config.handler(self, config)

            # self.base_graph.G = Graph.copy(self.wolf.logical_graph)

            # Save this iteration graph to GML and Dot
            if self.circuits_added:
                pass  # circuit_tag was updated above.
            #     circuit_tag = "circuit-{}-{}".format(u,v)
            #     circuit_tag = "circuit-{}".format(".".joint())
            else:
                circuit_tag = ""
            # updated_topology_file = iteration_topo + circuit_tag
            
            if self.line_code == "BVT": 
                self.wolf.logical_graph.edges[('63', '133')]["capacity"] *= 0.75
            
            
            # if self.topo_solved: 
            #     os.copy_file_range(self.topo_solved, iteration_topo + ".dot")
            if not self.topo_solved: 
                updated_topology_file = iteration_topo            
                self.export_logical_topo_to_gml( updated_topology_file + ".gml" )                
                Gml_to_dot( self.wolf.logical_graph, iteration_topo + ".dot", unit=unit )
            else:                 
                system(f"cp {self.topo_solved} {iteration_topo}.dot")
            self.topo_solved = None

            # Draw the link graph for the instanced topology.
            if DRAW: 
                draw_graph(self.wolf.logical_graph, 
                       name=f"data/graphs/img/1-{ITERATION_ID}")

            # self.base_graph._init_link_graph()
            if (self.PREV_ITER_ABS_PATH and ( iter_congestion > 0 )
                and percent_diff(tm_i_data, PREV_ITER_TM_DATA) + iter_congestion < self.congestion_threshold_upper_bound                  
                and ( len(self.new_circuit) == 0 ) and ( len(self.chaff) == 0 )):
                logger.debug("skipping computation")
                logger.debug(f"Prev: {self.PREV_ITER_ABS_PATH}\t{iter_congestion}\tPercent diff {percent_diff(tm_i_data, PREV_ITER_TM_DATA) + iter_congestion}\t Threshold{self.congestion_threshold_upper_bound}")
                # Prevents us from running the simulation if the topology has not changed
                # TODO: Check on that percent diff heuristic.  
                system(f"cp -r {self.PREV_ITER_ABS_PATH}/* {ITERATION_ABS_PATH}/")
            else:
                iter_congestion = self._evaluate_te(
                        iteration_topo + ".dot",
                        ITERATION_REL_PATH,
                        traffic_file=self.temp_tm_i_file                        
                )                    
            if len(self.new_circuit) > 0:
                if self.topology_programming_method == "doppler":
                    reconfig_time = 1
                else:
                    reconfig_time = get_reconfig_time(
                        updated_topology_file + ".gml", self.new_circuit
                    )
            else:
                reconfig_time = 0

            return_data["ReconfigTime"].append(reconfig_time)
            return_data["Strategy"].append(
                "{} {}".format(
                    self.te_method, self.topology_programming_method
                )
            )
            return_data["CandidateLinkSet"].append(
                self.candidate_link_choice_method
            )

            return_data["Routing"].append(
                "{}".format(self.te_method).strip("-").upper()
            )
            return_data["Defense"].append(
                "{}".format(self.topology_programming_method)
            )

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
            return_data["Total Links Dropped"].append(
                len(self.chaff)
            )
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
                opt_time=self.opt_time if hasattr(self, 'opt_time') else "NaN",
                multi_sol_time=getattr(self, 'multi_sol_time', "NaN"),
                multi_sol_number_best_sol=getattr(self, 'multi_sol_number_best_sol', "NaN"),
                multi_sol_best_mlu=getattr(self, 'multi_sol_best_mlu', "NaN"),
            )
            if iter_congestion == "SIG_EXIT":
                return

            if (
                iter_congestion
                >= self.congestion_threshold_upper_bound
            ):
                self.sig_add_circuits = True

            elif (
                iter_congestion
                <= self.congestion_threshold_lower_bound
            ):
                if len(self.chaff) > 0:
                    self.sig_drop_circuits = True

            if CROSSFIRE:
                self.sig_drop_circuits = True

            if self.sig_drop_circuits:
                logger.info(f"Max link util, {iter_congestion}, below threshold, {self.congestion_threshold_lower_bound}. Reverting changes.")
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
            PREV_ITER_TM_DATA = tm_i_data
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
        return return_data

    def validate_simulation_inputs(self):
        """Validate the files implied by ``network_name``.
        If the topo file is not found, then the program halts.
        traffic and hosts files are generated if they are needed.
        Side effect: assigns object variables to verified filesystem paths:
        self.topo_file, self.hosts_file, and self.traffic_file.
        Also modifies self.base_graph.
        """
        name = self.network_name
        # Verify topology file and build base graph

        def verify_topo():
            logger.debug("verifying topology file.")
            gml_handle = os.path.join(
                SCRIPT_HOME, "data", "graphs", "gml", name + ".gml"
            )
            json_handle = os.path.join(
                SCRIPT_HOME, "data", "graphs", "json", name + ".json"
            )
            if (
                self.shakeroute
                and self.topology_programming_method != "baseline"
            ):
                base_topo_file = os.path.join(
                    SCRIPT_HOME,
                    "data",
                    "graphs",
                    "fiber_cut",
                    self.shakeroute,
                    self.topology_programming_method + ".gml",
                )

            elif isfile(gml_handle):
                base_topo_file = gml_handle

            elif isfile(json_handle):
                base_topo_file = json_handle
            else:
                logger.error(
                    f"Error topology file not found: {gml_handle} or {json_handle}",
                    exc_info=1,
                )
                exit(-1)
            logger.info(f"topology file check passed with base topology file: {base_topo_file}.")
            self.topo_file = base_topo_file

            # DON'T CALL nx.read_gml. Instead use FiberGraph.import_gml_graph
            # self.base_graph = FiberGraph(self.name, read_gml(base_topo_file))

            # self.import_gml_graph(base_topo_file)

            # try:
            #     self.base_graph.import_dot_graph(base_topo_file)
            # except KeyError:
            #     topo_file = os.path.join(SCRIPT_HOME, "data", "graphs", "dot", name+"-location.dot")
            #     self.base_graph.import_dot_graph(base_topo_file)
            # self.num_hosts = len([n for n in self.base_graph.G.nodes if n.startswith('h')])
            # self.num_hosts = len(self.base_graph.G.nodes)
            return

        # Verify hosts file
        def verify_hosts():
            logger.debug("verifying hosts file.")
            if self.shakeroute:
                hosts_file = os.path.join(
                    SCRIPT_HOME, "data", "hosts", self.shakeroute + ".hosts"
                )
            else:
                hosts_file = os.path.join(
                    SCRIPT_HOME, "data", "hosts", name + ".hosts"
                )
            hosts_folder = os.path.join(SCRIPT_HOME, "data", "hosts")
            try:  # sets self.num_hosts and checks file exists.
                assert isfile(
                    hosts_file
                ), "Error topology file not found: {}".format(hosts_file)
                logger.debug("Host file successfully located.")
                # g = read_dot(self.topo_file)
                # self.num_hosts = len([n for n in g.nodes if n.startswith('h')])
                lines = count_lines(hosts_file)
                assert (
                    lines == self.num_hosts
                ), "Host file has wrong number of lines. expected: {} got: {}".format(
                    self.num_hosts, lines
                )

            except AssertionError:  # Create the hosts file

                def create_host_file():
                    logger.debug("Creating host file.")
                    # g = read_dot(self.topo_file)
                    # self.num_hosts = len([n for n in g.nodes if n.startswith('h')])
                    # create host dir if needed.
                    makedirs(dirname(hosts_file), exist_ok=True)

                    with open(hosts_file, "w") as host_fob:
                        for i in range(1, self.num_hosts + 1):
                            host_fob.write("h" + str(i) + "\n")

                    logger.debug("Host file successfully created.")

                create_host_file()
            logger.debug("Host file check passed.")
            self.hosts_file = hosts_file
            return 

        # Verify traffic file
        def verify_traffic():
            logger.debug("verifying traffic file.")
            if self.traffic_file == "":
                logger.debug(
                    "no file stated. Generating common traffic file string"
                )
                traffic_file = os.path.join(
                    SCRIPT_HOME, "data", "traffic", name + ".txt"
                )
            else:
                logger.debug(
                    "Attempting to use traffic file provided by user."
                )
                logger.debug("file: {}".format(self.traffic_file))
                traffic_file = self.traffic_file

            try:
                if self.start_clean:  # start_clean
                    self._system("rm {}".format(traffic_file))
                assert isfile(
                    traffic_file
                ), "Error traffic file not found: {}".format(traffic_file)
                logger.debug("traffic file found.")
                line_count = count_lines(traffic_file)
                if line_count < self.iterations:
                    logger.error(
                        "traffic file found, but has too few lines. expected: {} got: {}".format(
                            self.iterations, line_count
                        )
                    )
                    assert False
                logger.debug("traffic file line-count passed.")
                # check total entries on each line is correct.
                with open(traffic_file, 'r') as tm_fob:
                    lines = tm_fob.readlines()
                    last_line = lines.pop()
                    # if (last_line.strip() != ""):
                    #     logger.error(f"File: {traffic_file} needs to end with an empty/blank line.")
                    #     assert False
                    num_expected_entries = self.num_hosts ** 2
                    for l in lines:
                        num_entries = len(l.strip().split())
                        if num_entries != num_expected_entries:                           
                            logger.error(f"Traffic matrix (TM): {traffic_file}. Network hosts (n): {self.num_hosts}. Line in TM should have n^2 entries ({num_expected_entries}). Got {num_entries}.")
                            assert False
                    # M = loadtxt(traffic_file)
                    # if sqrt(len(M)) != self.num_hosts:
                    #     logger.error(                            
                    #     )
                    #     assert False

            except AssertionError:
                if self.start_clean:
                    pass
                else:
                    logger.error(
                        f"Error verifying traffic file: {traffic_file}\n  Create one now? [y/n]"
                    )

                    create = input()
                    if create.lower().startswith("y"):
                        pass
                    else:
                        exit()

                rand_gravity_matrix(
                    self.num_hosts,
                    self.iterations,
                    self.magnitude,
                    traffic_file,
                )
                lines = count_lines(traffic_file)
                assert (
                    lines >= self.iterations
                ), "traffic file created, but has too few lines. expected: {} got: {}".format(
                    self.iterations, lines
                )

            logger.debug("Host file check passed.")
            self.traffic_file = traffic_file

        verify_topo()
        verify_hosts()
        verify_traffic()
        return









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
        top_k: Optional[int] = None,
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
            txp_count=self.wolf.get_txp_count(),
            candidate_set=self.candidate_link_choice_method,
            scale_down_factor=self.scale_down_factor,
            congestion_threshold_upper_bound=self.congestion_threshold_upper_bound,
            top_k=top_k,
            optimizer_time_limit=self.optimizer_time_limit_minutes * 60.0,
            use_cache=True,
            parallel_execution=True,
            solver=solver,
        )

        from onset.method_registry import _METHOD_REGISTRY
        method_config = _METHOD_REGISTRY.get(solver)
        if method_config is None or method_config.solve_fn is None:
            raise ValueError(f"No solver for method: {solver}")
        solve_fn = method_config.solve_fn
        result = solve_fn(problem)
        self.optimization_result = result
        self.opt_time = result.wall_time
        return result


# Wire handler callables into the method registry (avoids circular imports)
_METHOD_REGISTRY["doppler"] = MethodConfig(
    name="doppler", handler=_run_milp_method, is_milp=True,
    objective_mode="changes_plus_mlu", solver_method="doppler",
    solve_fn=_METHOD_REGISTRY["doppler"].solve_fn,
    uses_ecmp_multisol=False,
    description="Doppler reconnaissance defense (TNSM 2024)",
)
_METHOD_REGISTRY["onset_v3"] = MethodConfig(
    name="onset_v3", handler=_run_milp_method, is_milp=True,
    objective_mode="mlu", solver_method="onset_v3",
    solve_fn=_METHOD_REGISTRY["onset_v3"].solve_fn,
    uses_ecmp_multisol=True,
    description="ONSET DDoS defense — post major revision (TDSC 2025)",
)
_METHOD_REGISTRY["onset_v2"] = MethodConfig(
    name="onset_v2", handler=_run_milp_method, is_milp=True,
    objective_mode="mlu", solver_method="onset_v2",
    solve_fn=_METHOD_REGISTRY["onset_v2"].solve_fn,
    uses_ecmp_multisol=False,
    description="ONSET DDoS defense — path-based formulation (TDSC 2025)",
)
_METHOD_REGISTRY["onset"] = MethodConfig(
    name="onset", handler=_run_milp_method, is_milp=True,
    objective_mode="mlu", solver_method="onset",
    solve_fn=_METHOD_REGISTRY["onset"].solve_fn,
    uses_ecmp_multisol=False,
    description="Original topology programming formulation (OptSys 2021)",
)
_METHOD_REGISTRY["OTP"] = MethodConfig(
    name="OTP", handler=_run_otp, is_milp=False,
    objective_mode=None, solver_method=None, solve_fn=None,
    uses_ecmp_multisol=False,
    description="Offline Traffic Provisioning — shortcut-link heuristic",
)
_METHOD_REGISTRY["greylambda"] = MethodConfig(
    name="greylambda", handler=_run_greylambda, is_milp=False,
    objective_mode=None, solver_method=None, solve_fn=None,
    uses_ecmp_multisol=False,
    description="Greylambda — add circuits on fully-congested edges",
)
_METHOD_REGISTRY["cache"] = MethodConfig(
    name="cache", handler=_run_cache, is_milp=False,
    objective_mode=None, solver_method=None, solve_fn=None,
    uses_ecmp_multisol=False,
    description="Cache-based defense (Defender module)",
)
_METHOD_REGISTRY["BVT"] = MethodConfig(
    name="BVT", handler=_run_bvt, is_milp=False,
    objective_mode=None, solver_method=None, solve_fn=None,
    uses_ecmp_multisol=False,
    description="Bandwidth-variable transceiver emulation",
)
_METHOD_REGISTRY["TBE"] = MethodConfig(
    name="TBE", handler=_run_tbe, is_milp=False,
    objective_mode=None, solver_method=None, solve_fn=None,
    uses_ecmp_multisol=False,
    description="Temporary bandwidth expansion during flashcrowd",
)
_METHOD_REGISTRY["cli"] = MethodConfig(
    name="cli", handler=_run_cli, is_milp=False,
    objective_mode=None, solver_method=None, solve_fn=None,
    uses_ecmp_multisol=False,
    description="Interactive CLI mode",
)
