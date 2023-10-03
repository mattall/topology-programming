import sys
from numpy import loadtxt, sqrt
from itertools import combinations
from collections import Counter
from onset.alpwolf import AlpWolf
from onset.constants import SCRIPT_HOME
from onset.defender import Defender
from onset.utilities.config import CROSSFIRE
from onset.optimization_two import Link_optimization

# from mcf_net_difference import Link_optimization
from onset.constants import USER_HOME
from onset.utilities.diff_compare import diff_compare
from onset.utilities.gml_to_dot import Gml_to_dot
# from onset.utilities.logger import NewLogger
# logger = NewLogger().get_logger()
from onset.utilities.logger import logger
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
from onset.utilities.sysUtils import count_lines
from onset.utilities.tmg import rand_gravity_matrix
from onset.utilities.graph_utils import write_gml
from networkx.relabel import relabel_nodes
import traceback
from copy import deepcopy
from hashlib import sha1
from logging import FileHandler
from os import makedirs, path, system
from os.path import dirname, isfile


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
        candidate_link_choice_method="all",
        congestion_threshold_upper_bound=0.8,
        congestion_threshold_lower_bound=0.3,
        attack_proportion="",
        shakeroute=False,
        net_dir=False,
        fallow_tx_allocation_strategy="static",
        fallow_tx_allocation_file="",
        line_code="fixed",
        salt="",
    ):
        """Simulation initializer

        Args:
            network_name (str):
            num_hosts (int): number of hosts in the network

            test_name (str): Unique string, identifier for the simulation results

            iterations (int, optional): Number of simulated epochs. Defaults to 0.

            te_method (str, optional): Traffic engineering method - Must be supported
                by Yates or self-implemented. Defaults to '-ecmp'.

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
        # Set Experiment ID
        if self.use_heuristic.isdigit():
            self.EXPERIMENT_ID = "_".join(
                [
                    network_name,
                    test_name,
                    str(fallow_transponders),
                    self.attack_proportion,
                    self.te_method,
                ]
            ).replace("heuristic", "heuristic_{}".format(self.use_heuristic))
        else:
            self.EXPERIMENT_ID = "_".join(
                [
                    network_name,
                    test_name,
                    str(fallow_transponders),
                    self.attack_proportion,
                    self.te_method,
                ]
            )

        # Set Experiment absolute path
        if self.net_dir:
            self.EXPERIMENT_ABSOLUTE_PATH = path.join(
                SCRIPT_HOME,
                "data",
                "results",
                self.network_name,
                self.EXPERIMENT_ID,
            )

        else:
            self.EXPERIMENT_ABSOLUTE_PATH = path.join(
                SCRIPT_HOME, "data", "results", self.EXPERIMENT_ID
            )
        # The following three commands must be ordered as follows.
        self._init_logging()

        # self.base_graph = FiberGraph(self.name)
        self.validate_yates_params()  # Depends on base_graph.
        # Depends on validated yates params.
        self.wolf = AlpWolf(
            self.topo_file,
            self.fallow_transponders,
            fallow_tx_allocation_strategy=self.fallow_tx_allocation_strategy,
            fallow_tx_allocation_file=self.fallow_tx_allocation_file,
        )
        if self.topology_programming_method == "TBE": 
            self.wolf.restrict_bandwidth(0.8)

    def _system(self, command: str):
        logger.info("Calling system command: {}".format(command))
        return system(command)

    def _init_logging(self):
        logger.debug(
            "Experiment Absolute Path: {}".format(
                self.EXPERIMENT_ABSOLUTE_PATH
            )
        )
        logger.info("Experiment ID:            {}".format(self.EXPERIMENT_ID))
        # makedirs(self.EXPERIMENT_ABSOLUTE_PATH, exist_ok=True)
        # log_file = path.join(
        #     self.EXPERIMENT_ABSOLUTE_PATH, "{}.log".format(self.EXPERIMENT_ID)
        # )
        # if self.start_clean:
        # if True:
        #     if isfile(log_file):
        #         self._system("rm {}".format(log_file))
        # try:
        #     file_log_handler = FileHandler(log_file)
        # except FileNotFoundError:
        #     makedirs(path.dirname(log_file), exist_ok=True)
        #     file_log_handler = FileHandler(log_file)
        # file_log_handler.setFormatter(formatter)
        # logger.addHandler(file_log_handler)

    def _yates(self, topo_file, result_path, traffic_file=""):
        if traffic_file == "":
            traffic_file = self.traffic_file
        if self.shakeroute:
            result_path = path.join(self.network_name, result_path)
        command_args = [
            "yates",
            topo_file,
            traffic_file,
            traffic_file,
            self.hosts_file,
            self.te_method,
            "-num-tms",
            "1",
            "-out",
            result_path,
            "-budget",
            "3",
            ">>",
            f"{self.nonce}_yates.out",
        ]
        gurobi_status = self._system("gurobi_cl")
        if gurobi_status == 0:
            logger.info("gurobi_cl is in path.")
        else:
                raise(f"Error: gurobi_cl not in path {sys.path}")
        self._system(" ".join(command_args))
        max_congestion = read_result_val(
            path.join(
                SCRIPT_HOME,
                "data",
                "results",
                result_path,
                "MaxExpCongestionVsIterations.dat",
            )
        )
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
        GRAPHS_PATH = path.join(self.EXPERIMENT_ABSOLUTE_PATH, "graphs")
        makedirs(GRAPHS_PATH, exist_ok=True)

        # prep initial file
        initial_graph = self.wolf.logical_graph
        INITIAL_GRAPH_PATH = path.join(
            GRAPHS_PATH, self.network_name + "_0.dot"
        )
        INITIAL_RESULTS_REL_PATH = path.join(self.EXPERIMENT_ID, "__0")

        # Write Graphs to files
        self.export_logical_topo_to_gml(
            INITIAL_GRAPH_PATH.replace(".dot", ".gml"), G=initial_graph
        )
        Gml_to_dot(initial_graph, INITIAL_GRAPH_PATH)

        if (
            self._yates(INITIAL_GRAPH_PATH, INITIAL_RESULTS_REL_PATH)
            == "SIG_EXIT"
        ):
            return

        PATH_DIFF_FOLDER = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "path_diff"
        )
        CONGESTION_DIFF_FOLDER = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "congestion_diff"
        )
        makedirs(PATH_DIFF_FOLDER, exist_ok=True)
        makedirs(CONGESTION_DIFF_FOLDER, exist_ok=True)

        INITIAL_PATHS = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH,
            "__0",
            "paths",
            self.te_method.strip("-") + "_0",
        )
        INITIAL_CONGESTION = path.join(
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
            TEST_RESULTS_REL_PATH = path.join(
                self.EXPERIMENT_ID, "{}_{}".format(u, v)
            )
            test_alpwolf = deepcopy(self.wolf)
            for _ in range(circuits_to_add):
                test_alpwolf.add_circuit(u, v)

            TEST_GRAPH_PATH = path.join(
                GRAPHS_PATH, self.network_name + "_{}_{}.dot".format(u, v)
            )

            # Write Graphs to files
            Gml_to_dot(test_alpwolf.logical_graph, TEST_GRAPH_PATH)
            self.export_logical_topo_to_gml(
                TEST_GRAPH_PATH.replace(".dot", ".gml"),
                G=test_alpwolf.logical_graph,
            )

            if (
                self._yates(TEST_GRAPH_PATH, TEST_RESULTS_REL_PATH)
                == "SIG_EXIT"
            ):
                return

            TEST_PATHS = path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "{}_{}".format(u, v),
                "paths",
                self.te_method.strip("-") + "_0",
            )
            TEST_CONGESTION = path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "{}_{}".format(u, v),
                "MaxExpCongestionVsIterations.dat",
            )

            PATH_DIFF = path.join(
                self.EXPERIMENT_ABSOLUTE_PATH,
                "path_diff",
                "{}_{}.txt".format(u, v),
            )
            CONGESTION_DIFF = path.join(
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

            draw_graph(
                test_alpwolf.logical_graph,
                path.join(
                    GRAPHS_PATH, self.network_name + "_{}_{}".format(u, v)
                ),
            )

        PLOT_DIR = path.join(self.EXPERIMENT_ABSOLUTE_PATH, "plot_dir")
        CONGESTION_VS_PATHCHURN = path.join(
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
        CONGESTION_CDF = path.join(PLOT_DIR, "congestion_cdf")
        PATHCHURN_CDF = path.join(PLOT_DIR, "pathChurn_cdf")
        cdf_average_congestion(congestion_change, CONGESTION_CDF)
        cdf_churn(path_churn, PATHCHURN_CDF)

        return self.EXPERIMENT_ABSOLUTE_PATH

    def export_logical_topo_to_gml(self, name, G=None):
        if G == None:
            G = self.wolf.logical_graph

        # Relabels to be consistent with naming in Ripple.
        if 0:  # to make consistent with ripple examples.
            yates_to_ripple_map = {
                node: ("sw" + str(int(node) - 1)) for (node) in G
            }
        if 1:
            yates_to_ripple_map = {node: ("s{}".format(node)) for (node) in G}
        gml_view = relabel_nodes(G, yates_to_ripple_map, copy=True)
        write_gml(gml_view, name)
        del gml_view

    def perform_sim(
        self,
        circuits=1,
        start_iter=0,
        end_iter=0,
        repeat=False,
        unit="Gbps",
        demand_factor=1,
        dry=False
    ):
        sim_param_tag = f"{circuits}_{start_iter}_{end_iter}_{int(repeat)}_{unit}_{demand_factor:.1f}"

        if end_iter == 0:
            end_iter = self.iterations

        return_data = {
            "Iteration": [],
            "Experiment": [],
            "Strategy": [],
            "Routing": [],
            "Defense": [],
            "ReconfigTime": [],
            "Total Links Added": [],
            "Links Added": [],
            "Total Flux Links": [],
            "Total Links Dropped": [],
            "Links Dropped": [],
            "Congestion": [],
            "Loss": [],
            "Throughput": [],
            "Link Bandwidth Coefficient": [],
            "Demand Factor": [],
            "Optimization Time": [],
        }
        circuits_added = False

        if CROSSFIRE == True:
            sig_add_circuits = True
        else:
            sig_add_circuits = False

        sig_drop_circuits = False
        logger.debug("Yates.perform_sim")
        name = self.network_name
        iterations = self.iterations
        topo = self.topo_file
        traffic = self.traffic_file
        hosts = self.hosts_file
        te_method = self.te_method
        EXPERIMENT_ID = self.EXPERIMENT_ID
        EXPERIMENT_ABSOLUTE_PATH = self.EXPERIMENT_ABSOLUTE_PATH
        flux_circuits = (
            []
        )  # Stateful list of active circuits triggered from sig_add_circuits
        if repeat:
            # j is 1 the first time we see a traffic matrix, and 2 the seond.
            # if we are not repeating, then j should never be referenced.
            j = 1
        else:
            j = float("NaN")
        # Open traffic file and pass a new line from the file for every iteration.
        with open(traffic, "rb") as fob:
            PREV_ITER_ABS_PATH = ""
            for i in range(1, iterations + 2):
                if self.shakeroute:
                    ITERATION_ID = self.topology_programming_method

                elif repeat:
                    if j == 1:
                        ITERATION_ID = f"{name}_{i}-{j}-{iterations}_{sim_param_tag}"
                    elif j == 2:
                        ITERATION_ID = f"{name}_{i - 1}-{j}-{iterations}_{sim_param_tag}"

                else:
                    ITERATION_ID = f"{name}_{i}-{iterations}_{sim_param_tag}"

                ITERATION_REL_PATH = path.join(EXPERIMENT_ID, ITERATION_ID)
                ITERATION_ABS_PATH = path.join(
                    EXPERIMENT_ABSOLUTE_PATH, ITERATION_ID
                )
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

                # CONGESTION_PATH = path.join(
                #     EXPERIMENT_ABSOLUTE_PATH,
                #     ITERATION_ID,
                #     "EdgeCongestionVsIterations.dat",
                # )
                # MAX_CONGESTION_PATH = path.join(
                #     EXPERIMENT_ABSOLUTE_PATH,
                #     ITERATION_ID,
                #     "MaxCongestionVsIterations.dat",
                # )

                # Grab line from tm and throw it into a temp file to run yates.
                if repeat and i == (end_iter + 1) and j == 2:
                    pass
                    # don't read a new line if we are on the repetition step
                else:
                    tm_i_data_from_file = fob.readline()
                    tm_i_data = [
                        str(float(i) * demand_factor)
                        for i in tm_i_data_from_file.split()
                    ]
                    tm_i_data_to_temp_file = " ".join(tm_i_data)

                if i < start_iter:
                    continue
                if i > end_iter:
                    if repeat and i == (end_iter + 1) and j == 2:
                        pass
                    else:
                        continue
                if repeat:
                    j += 1
                if dry:
                    return_data = ITERATION_ABS_PATH
                    continue
                temp_tm_i = self.nonce
                temp_fob = open(self.nonce, "w")
                temp_fob.write(tm_i_data_to_temp_file)
                temp_fob.close()
                reconfig_time = 0
                new_circuit = []
                drop_circuits = []
                makedirs(ITERATION_ABS_PATH, exist_ok=True)

                # if i == 360:
                #     print("BREAK")
                # if 300 <= i: #<= 335 :
                #     pass
                # else:
                #     continue

                return_data["Experiment"].append(EXPERIMENT_ID)
                return_data["Iteration"].append(i)
                max_load = 1
                try:
                    # create dot graph and put it into the appropriate file for this run.
                    iteration_topo = ITERATION_ABS_PATH
                    if self.topology_programming_method == "cli":
                        client = self.wolf.cli()

                    elif (  # Heuristic method from very early testing.
                        self.topology_programming_method == "cache"
                        and sig_add_circuits
                        and not circuits_added
                    ):
                        defender = Defender(
                            self.network_name,
                            circuits,
                            self.candidate_link_choice_method,
                            self.use_heuristic,
                            PREV_ITER_ABS_PATH,
                            self.attack_proportion,
                        )
                        # TODO: Pass get_strategic_circuit the paths file from the previous iteration.
                        new_circuit = defender.get_strategic_circuit()
                        if (
                            type(new_circuit) == tuple
                            and len(new_circuit) == 2
                        ):
                            logger.debug(
                                "Adding {} ({}, {}) circuits.".format(
                                    circuits
                                ),
                                *new_circuit,
                            )
                            for _ in range(circuits):
                                u, v = new_circuit
                                self.wolf.add_circuit(u, v)

                            circuits_added = True

                        # elif new_circuit == []:
                        #     print("Could Not Add New Circuit")
                        #     self.exit_early = True

                    # elif self.strategy == 'onset' and sig_add_circuits and not circuits_added:

                    elif (  # Method from TDSC-23 - Link-flood DDoS Defense
                        self.topology_programming_method == "onset"
                        and sig_add_circuits
                    ):
                        optimizer = Link_optimization(
                            G=self.wolf.logical_graph,
                            BUDGET=4,
                            demand_matrix=self.nonce,
                            network=self.network_name,
                        )
                        # optimizer.run_model()
                        if self.te_method == "-ecmp":
                            max_load = 0.5
                            if self.shakeroute:
                                max_load = 10000.0
                        else:
                            max_load = 0.8

                        # optimizer.run_model_minimize_cost_v1(max_load)
                        # optimizer.run_model_minimize_cost()
                        new_circuit = []
                        optimizer.LINK_CAPACITY *= max_load
                        optimizer.run_model_mixed_objective()
                        new_circuit = optimizer.get_links_to_add()
                        # while len(new_circuit) == 0 and max_load < 5:
                        #     optimizer.LINK_CAPACITY *= max_load
                        #     optimizer.run_model_mixed_objective()
                        #     new_circuit = optimizer.get_links_to_add()
                        #     max_load += 0.25
                        if len(new_circuit) == 0:
                            # optimizer.BUDGET = 20
                            optimizer.LINK_CAPACITY += optimizer.LINK_CAPACITY
                            optimizer.run_model_mixed_objective()
                            new_circuit = optimizer.get_links_to_add()

                        circuit_tag = ""
                        if type(new_circuit) == list and len(new_circuit) > 0:
                            for nc in new_circuit:
                                u, v = nc
                                if circuit_tag == "":
                                    circuit_tag += f"circuit-{u}-{v}"
                                else:
                                    circuit_tag += f".{u}-{v}"

                                for _ in range(circuits):
                                    self.wolf.add_circuit(u, v)

                            circuits_added = True
                            flux_circuits.extend(new_circuit)

                        # elif new_circuit == []:
                        #     print("Could Not Add New Circuit")
                        #     self.exit_early = True

                        sig_add_circuits = False

                    elif (  # Method from TDSC-23 - Link-flood DDoS Defense
                        self.topology_programming_method == "onset_v2"
                        and sig_add_circuits
                    ):
                        txp_count_dict = self.wolf.get_txp_count() # Maps node NAMES to their total trandponders.
                        optimizer = Link_optimization(
                            G                   = self.wolf.logical_graph,
                            demand_matrix_file  = self.nonce,
                            network             = self.network_name,
                            core_G              = self.wolf.base_graph.copy(as_view=True),
                            txp_count           = txp_count_dict
                        )
                        # optimizer.run_model()
                        # if self.te_method == "-ecmp":
                        #     max_load = 0.5
                        #     if self.shakeroute:
                        #         max_load = 10000.0
                        # else:
                        #     max_load = 0.8

                        # optimizer.run_model_minimize_cost_v1(max_load)
                        # optimizer.run_model_minimize_cost()
                        new_circuit = []
                        result_topo = []
                        add_links = []
                        drop_links = []
                        # optimizer.LINK_CAPACITY *= max_load
                        result = optimizer.onset_optimizer()
                        try:
                            ((add_edges, drop_edges), opt_time) = result

                        except TypeError as e:
                            logger.error(f"{e} Optimization failed to yield a solution.", exc_info=True, stack_info=True)
                            ((add_edges, drop_edges), opt_time) = ([],[]), float("NaN")

                        circuit_tag = ""
                        new_circuit = add_links[:]
                        if drop_links:
                            for drop_circuit in drop_links:
                                u, v = drop_circuit
                                self.wolf.drop_circuit(u, v)

                        if add_links:
                            for nc in new_circuit:
                                u, v = nc
                                if circuit_tag == "":
                                    circuit_tag += f"circuit-{u}-{v}"
                                else:
                                    circuit_tag += f".{u}-{v}"

                                self.wolf.add_circuit(u, v)

                            circuits_added = True
                            # flux_circuits.extend(new_circuit)

                        # elif new_circuit == []:
                        #     print("Could Not Add New Circuit")
                        #     self.exit_early = True

                        sig_add_circuits = False

                    elif (  # Method from PDP+OTP HotNets-23
                        self.topology_programming_method == "OTP"
                        and sig_add_circuits
                    ):
                        edge_congestion_file = path.join(
                            PREV_ITER_ABS_PATH,
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
                                and c not in self.wolf.logical_graph.edges()
                            ]
                            logger.info(
                                f"Found the following shortcut: {shortcuts}"
                            )
                            return shortcuts

                        shortcuts = find_shortcut_link(congested_edges)
                        for edge in shortcuts:
                            u, v = edge
                            for _ in range(circuits):
                                self.wolf.add_circuit(u, v, 100)
                                flux_circuits.append((u, v))
                        # flux_circuits.extend(congested_edges)
                        sig_add_circuits = False

                    elif (  # Method from TNSM-23
                        self.topology_programming_method == "greylambda"
                        and sig_add_circuits
                    ):
                        edge_congestion_file = path.join(
                            PREV_ITER_ABS_PATH,
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
                                logger.error(f"Error, unable to unpack edge: {edge} of type: {type(edge)}")
                                raise
                            for _ in range(circuits):
                                added = self.wolf.add_circuit(u, v)
                                if added == 0:
                                    circuits_added = True

                        flux_circuits.extend(congested_edges)
                        sig_add_circuits = False

                    elif (  # Bandwidth Variable Transceivers - Emulate RADWAN
                        self.topology_programming_method == "BVT"
                        and sig_add_circuits
                    ):
                        edge_congestion_file = path.join(
                            PREV_ITER_ABS_PATH,
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
                        #         self.wolf.add_circuit(u, v)
                        # flux_circuits.extend(congested_edges)
                        sig_add_circuits = False

                    elif (  # Temporary Bandwidth Expansion - i.e., Spiffy
                        self.topology_programming_method == "TBE"
                        # and sig_add_circuits
                    ):
                        if "flashcrowd" in self.traffic_file \
                            and demand_factor > 0.9:

                            self.wolf.relax_restricted_bandwidth()
                        # sig_add_circuits = False

                    # Net Recon Defense Method
                    elif self.topology_programming_method == "skinwalker":
                        optimizer = Link_optimization(
                            G=self.wolf.logical_graph,
                            # BUDGET=0,
                            demand_matrix_file=self.nonce,
                            network=self.network_name,
                            core_G=self.wolf.base_graph.copy(as_view=True),
                        )
                        # optimizer.run_model()
                        if self.te_method == "-ecmp":
                            max_load = 0.5
                            if self.shakeroute:
                                max_load = 10000.0
                        else:
                            max_load = 0.8

                        # optimizer.run_model_minimize_cost_v1(max_load)
                        # optimizer.run_model_minimize_cost()
                        new_circuit = []
                        result_topo = []
                        add_links = []
                        drop_links = []
                        # optimizer.LINK_CAPACITY *= max_load
                        add_links, drop_links = optimizer.skinwalker()
                        # result_topo, add_links, drop_links = optimizer.optimize()
                        # result_topo, add_links, drop_links = optimizer.run_model_max_diff_ignore_demand()

                        circuit_tag = ""
                        new_circuit = add_links[:]
                        if drop_links:
                            for drop_circuit in drop_links:
                                u, v = drop_circuit
                                self.wolf.drop_circuit(u, v)

                        if add_links:
                            for nc in new_circuit:
                                u, v = nc
                                if circuit_tag == "":
                                    circuit_tag += f"circuit-{u}-{v}"
                                else:
                                    circuit_tag += f".{u}-{v}"

                                self.wolf.add_circuit(u, v)

                            circuits_added = True
                            # flux_circuits.extend(new_circuit)

                        # elif new_circuit == []:
                        #     print("Could Not Add New Circuit")
                        #     self.exit_early = True

                        sig_add_circuits = False

                    # Template for new TP methods
                    else:
                        # find new circuits
                        # call self.wolf.add_circuit(a,b)
                        # update flux_circuits with circuits added
                        # set sig_add_circuits false.
                        pass

                    # self.base_graph.G = Graph.copy(self.wolf.logical_graph)

                    # Save this iteration graph to GML and Dot
                    if circuits_added:
                        pass  # circuit_tag was updated above.
                    #     circuit_tag = "circuit-{}-{}".format(u,v)
                    #     circuit_tag = "circuit-{}".format(".".joint())
                    else:
                        circuit_tag = ""
                    # updated_topology_file = iteration_topo + circuit_tag
                    
                    if self.line_code == "BVT": 
                        self.wolf.logical_graph.edges[('63', '133')]["capacity"] *= 0.75
                    
                    updated_topology_file = iteration_topo
                    self.export_logical_topo_to_gml(
                        updated_topology_file + ".gml"
                    )
                    Gml_to_dot(
                        self.wolf.logical_graph,
                        iteration_topo + ".dot",
                        unit=unit,
                    )

                    # Draw the link graph for the instanced topology.
                    # self.base_graph._init_link_graph()
                        
                    iter_congestion = self._yates(
                        iteration_topo + ".dot",
                        ITERATION_REL_PATH,
                        traffic_file=temp_tm_i,
                    )
                    if len(new_circuit) > 0:
                        if self.topology_programming_method == "skinwalker":
                            reconfig_time = 1
                        else:
                            reconfig_time = get_reconfig_time(
                                updated_topology_file + ".gml", new_circuit
                            )
                    else:
                        reconfig_time = 0

                    return_data["ReconfigTime"].append(reconfig_time)
                    return_data["Strategy"].append(
                        "{} {}".format(
                            self.te_method, self.topology_programming_method
                        )
                    )
                    return_data["Routing"].append(
                        "{}".format(self.te_method).strip("-").upper()
                    )
                    return_data["Defense"].append(
                        "{}".format(self.topology_programming_method)
                    )

                    return_data["Congestion"].append(
                        read_result_val(
                            path.join(
                                ITERATION_ABS_PATH,
                                "MaxExpCongestionVsIterations.dat",
                            )
                        )
                    )
                    return_data["Loss"].append(
                        read_result_val(
                            path.join(
                                ITERATION_ABS_PATH,
                                "CongestionLossVsIterations.dat",
                            )
                        )
                    )
                    return_data["Throughput"].append(
                        read_result_val(
                            path.join(
                                ITERATION_ABS_PATH,
                                "TotalThroughputVsIterations.dat",
                            )
                        )
                    )
                    return_data["Total Links Added"].append(len(new_circuit))
                    return_data["Links Added"].append(new_circuit)
                    return_data["Total Flux Links"].append(len(flux_circuits))
                    return_data["Total Links Dropped"].append(
                        len(drop_circuits)
                    )
                    return_data["Links Dropped"].append(drop_circuits)
                    return_data["Link Bandwidth Coefficient"].append(max_load)
                    return_data["Demand Factor"].append(demand_factor)

                    if iter_congestion == "SIG_EXIT":
                        return

                    if (
                        iter_congestion
                        >= self.congestion_threshold_upper_bound
                    ):
                        sig_add_circuits = True

                    elif (
                        iter_congestion
                        <= self.congestion_threshold_lower_bound
                    ):
                        if len(flux_circuits) > 0:
                            sig_drop_circuits = True

                    if CROSSFIRE:
                        sig_drop_circuits = True

                    if sig_drop_circuits:
                        drop_circuits = flux_circuits[:]
                        if len(drop_circuits) > 0:
                            for dc in drop_circuits:
                                u, v = dc
                                for _ in range(circuits):
                                    self.wolf.drop_circuit(u, v)
                            flux_circuits = []
                            circuits_added = False
                            sig_drop_circuits = False

                except BaseException as e:
                    logger.error("Unknown Error", exc_info=True, stack_info=True)
                    # self._system("rm %s" % temp_tm_i)
                    if self.topology_programming_method == "greylambda":
                        return -1
                    return return_data

                # finally:
                # # Remove the temp file.
                # self._system("rm %s" % temp_tm_i)

                PREV_ITER_ABS_PATH = ITERATION_ABS_PATH
                # command_args = ["yates", iteration_topo, temp_tm_i, temp_tm_i, hosts,
                #                 te_method, "-num-tms", "1", "-out", ITERATION_REL_PATH]
                # logger.debug("Executing command: {}".format(" ".join(command_args)))
                # self._system(" ".join(command_args))

                # self.base_graph.set_weights(CONGESTION_PATH)
                # self.base_graph.draw_graphs(ITERATION_ABS_PATH)
                # edge_congestion = get_edge_congestion(CONGESTION_PATH)
                # PLOT_DIR = path.join(ITERATION_ABS_PATH, "plot_dir")
                # makedirs(PLOT_DIR, exist_ok=True)
                # try:
                #     congestion_heatmap(edge_congestion, path.join(
                #         PLOT_DIR, "edge_congestion_heatmap"))
                # except:
                #     logger.warning("Did not make heatmap for this iteration.")
                #     pass
        return return_data

    def validate_yates_params(self):
        """validates yates by ensuring that the necessary files implied by `self.name` exist.
        If the topo file is not found, then the program halts.
        traffic and hosts files are generated if they are needed.
        Side effect: Assigns the following object variables to strings describing a verified
        paths in the file system: self.topo_file, self.hosts_file, self.traffic_file
        Also modifies self.base_graph.
        """
        name = self.network_name
        # Verify topology file and build base graph

        def verify_topo():
            logger.debug("verifying topology file.")
            gml_handle = path.join(
                SCRIPT_HOME, "data", "graphs", "gml", name + ".gml"
            )
            json_handle = path.join(
                SCRIPT_HOME, "data", "graphs", "json", name + ".json"
            )
            if (
                self.shakeroute
                and self.topology_programming_method != "baseline"
            ):
                base_topo_file = path.join(
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
            #     topo_file = path.join(SCRIPT_HOME, "data", "graphs", "dot", name+"-location.dot")
            #     self.base_graph.import_dot_graph(base_topo_file)
            # self.num_hosts = len([n for n in self.base_graph.G.nodes if n.startswith('h')])
            # self.num_hosts = len(self.base_graph.G.nodes)

        # Verify hosts file
        def verify_hosts():
            logger.debug("verifying hosts file.")
            if self.shakeroute:
                hosts_file = path.join(
                    SCRIPT_HOME, "data", "hosts", self.shakeroute + ".hosts"
                )
            else:
                hosts_file = path.join(
                    SCRIPT_HOME, "data", "hosts", name + ".hosts"
                )
            hosts_folder = path.join(SCRIPT_HOME, "data", "hosts")
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

        # Verify traffic file
        def verify_traffic():
            logger.debug("verifying traffic file.")
            if self.traffic_file == "":
                logger.debug(
                    "no file stated. Generating common traffic file string"
                )
                traffic_file = path.join(
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
