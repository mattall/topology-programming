# main.py
# Written by MNH
# Takes network graph (.dot format) and TM Series.
# Produces two network simulations with traffic engineering sim (YATES).
#   1) Run YATES over the plain network graph.
#   2) Run YATES over the adaptive graph.
from unittest import result
from networkx.relabel import relabel_nodes
from alpwolf import AlpWolf
import argparse
import traceback

from copy import deepcopy
from logging import FileHandler
from os import makedirs
from os import path
from os import system

from os.path import isfile, dirname
# from onset.ONSET_MILP import Link_optimization
from onset.optimization import Link_optimization
from onset.utilities import SCRIPT_HOME, USER_HOME
from onset.utilities.plot_reconfig_time import get_reconfig_time
from onset.utilities.tmg import rand_gravity_matrix
from onset.utilities.logger import logger, formatter
from onset.utilities.sysUtils import count_lines
from onset.utilities.gml_to_dot import Gml_to_dot
from onset.utilities.write_gml import write_gml
from onset.utilities.diff_compare import diff_compare
from onset.utilities.plotters import cdf_average_congestion, cdf_churn, draw_graph, plot_points
from onset.utilities.post_process import post_process, post_proc_timeseries, read_result_val, read_link_congestion_to_dict
from onset.defender import Defender
from hashlib import sha1

CROSSFIRE = False

class Attack_Sim():
    def __init__(self, name: str,
                 num_hosts: int,
                 test: str,
                 iterations=0,
                 te_method='-ecmp',
                 start_clean=False,
                 magnitude=100*10**10,
                 traffic_file="",
                 strategy="",
                 fallow_transponders=5,
                 use_heuristic="",
                 method="all",
                 congestion_threshold_upper_bound=0.8,
                 congestion_threshold_lower_bound=0.3,
                 proportion="",
                 shakeroute=False,
                 net_dir=False,
                 fallow_tx_allocation_strategy="static",
                 fallow_tx_allocation_file="",                 
                 salt=""):
        self.nonce = "./temp/" + sha1("".join(
            [str(x) for x in
             [name, 
              test, 
              iterations, 
              te_method,
              start_clean, 
              magnitude, 
              traffic_file,
              strategy, 
              fallow_transponders, 
              use_heuristic, 
              method, 
              congestion_threshold_upper_bound, 
              congestion_threshold_lower_bound, 
              fallow_tx_allocation_strategy, 
              fallow_tx_allocation_file, 
              proportion, 
              salt]
             ]
        ).encode()).hexdigest()
        print("Nonce: ", self.nonce)
        logger.info("Initializing simulator: {} {} {}".format(
            name, test, iterations))
        self.name = name
        self.num_hosts = int(num_hosts)
        self.test = test
        self.iterations = iterations
        self.te_method = te_method
        self.traffic_file = traffic_file
        self.start_clean = start_clean
        self.magnitude = magnitude
        self.strategy = strategy
        self.topo_file = ""
        self.hosts_file = ""
        self.fallow_transponders = fallow_transponders
        self.use_heuristic = use_heuristic
        self.method = method
        self.congestion_threshold_upper_bound = congestion_threshold_upper_bound
        self.congestion_threshold_lower_bound = congestion_threshold_lower_bound
        self.exit_early = False
        self.attack_proportion = proportion
        self.shakeroute = shakeroute
        self.net_dir = net_dir
        self.fallow_tx_allocation_strategy = fallow_tx_allocation_strategy
        self.fallow_tx_allocation_file = fallow_tx_allocation_file
        # Set Experiment ID
        if self.use_heuristic.isdigit():
            self.EXPERIMENT_ID = "_".join([name, test, str(fallow_transponders), self.attack_proportion, self.te_method]).replace(
                "heuristic", "heuristic_{}".format(self.use_heuristic))
        else:
            self.EXPERIMENT_ID = "_".join(
                [name, test, str(fallow_transponders), self.attack_proportion, self.te_method])

        # Set Experiment absolute path
        if self.net_dir:
            self.EXPERIMENT_ABSOLUTE_PATH = path.join(
                SCRIPT_HOME, "data", "results", self.name, self.EXPERIMENT_ID)

        else:
            self.EXPERIMENT_ABSOLUTE_PATH = path.join(
                SCRIPT_HOME, "data", "results", self.EXPERIMENT_ID)
        # The following three commands must be ordered as follows.
        self._init_logging()

        # self.base_graph = FiberGraph(self.name)
        self.validate_yates_params()  # Depends on base_graph.
        # Depends on validated yates params.
        self.wolf = AlpWolf(self.topo_file, 
                            self.fallow_transponders, 
                            fallow_tx_allocation_strategy=self.fallow_tx_allocation_strategy, 
                            fallow_tx_allocation_file=self.fallow_tx_allocation_file)

    def _system(self, command: str):
        logger.info('Calling system command: {}'.format(command))
        system(command)

    def _init_logging(self):
        print("Experiment Absolute Path: {}".format(
            self.EXPERIMENT_ABSOLUTE_PATH))
        print("Experiment ID:            {}".format(self.EXPERIMENT_ID))
        makedirs(self.EXPERIMENT_ABSOLUTE_PATH, exist_ok=True)
        log_file = path.join(self.EXPERIMENT_ABSOLUTE_PATH,
                             '{}.log'.format(self.EXPERIMENT_ID))
        # if self.start_clean:
        if True:
            if isfile(log_file):
                self._system("rm {}".format(log_file))
        try:
            file_log_handler = FileHandler(log_file)
        except FileNotFoundError:
            makedirs(path.dirname(log_file), exist_ok=True)
            file_log_handler = FileHandler(log_file)
        file_log_handler.setFormatter(formatter)
        logger.addHandler(file_log_handler)

    def _yates(self, topo_file, result_path, traffic_file=""):
        if traffic_file == "":
            traffic_file = self.traffic_file
        if self.shakeroute:
            result_path = path.join(self.name, result_path)
        command_args = [USER_HOME + "/.opam/4.06.0/bin/yates", topo_file, traffic_file, traffic_file,
                        self.hosts_file, self.te_method, "-num-tms", "1", "-out", result_path, "-budget", "3"
                        ]

        self._system(" ".join(command_args))
        max_congestion = read_result_val(path.join(
            SCRIPT_HOME, "data", "results", result_path, "MaxExpCongestionVsIterations.dat"))
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
        INITIAL_GRAPH_PATH = path.join(GRAPHS_PATH, self.name + "_0.dot")
        INITIAL_RESULTS_REL_PATH = path.join(self.EXPERIMENT_ID, "__0")

        # Write Graphs to files
        self.export_logical_topo_to_gml(
            INITIAL_GRAPH_PATH.replace('.dot', '.gml'), G=initial_graph)
        Gml_to_dot(initial_graph, INITIAL_GRAPH_PATH)

        if self._yates(INITIAL_GRAPH_PATH, INITIAL_RESULTS_REL_PATH) == "SIG_EXIT":
            return

        PATH_DIFF_FOLDER = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "path_diff")
        CONGESTION_DIFF_FOLDER = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "congestion_diff")
        makedirs(PATH_DIFF_FOLDER, exist_ok=True)
        makedirs(CONGESTION_DIFF_FOLDER, exist_ok=True)

        INITIAL_PATHS = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "__0", "paths", self.te_method.strip('-')+"_0")
        INITIAL_CONGESTION = path.join(
            self.EXPERIMENT_ABSOLUTE_PATH, "__0", "MaxExpCongestionVsIterations.dat")
        # Run experiment
        if self.use_heuristic != "":
            # FIXME CHANGE BACK TO `candid_set='ranked'`
            # candidate_circuits = self.wolf.get_candidate_circuits(candid_set='ranked', k=5, l=5)
            candidate_circuits = self.wolf.get_candidate_circuits(
                candid_set='all')
        else:
            candidate_circuits = self.wolf.get_candidate_circuits(
                candid_set='all')

        for u, v in candidate_circuits:
            TEST_RESULTS_REL_PATH = path.join(
                self.EXPERIMENT_ID, "{}_{}".format(u, v))
            test_alpwolf = deepcopy(self.wolf)
            for _ in range(circuits_to_add):
                test_alpwolf.add_circuit(u, v)

            TEST_GRAPH_PATH = path.join(
                GRAPHS_PATH, self.name + "_{}_{}.dot".format(u, v))

            # Write Graphs to files
            Gml_to_dot(test_alpwolf.logical_graph, TEST_GRAPH_PATH)
            self.export_logical_topo_to_gml(TEST_GRAPH_PATH.replace(
                '.dot', '.gml'), G=test_alpwolf.logical_graph)

            if self._yates(TEST_GRAPH_PATH, TEST_RESULTS_REL_PATH) == "SIG_EXIT":
                return

            TEST_PATHS = path.join(self.EXPERIMENT_ABSOLUTE_PATH, "{}_{}".format(
                u, v), "paths", self.te_method.strip('-')+"_0")
            TEST_CONGESTION = path.join(self.EXPERIMENT_ABSOLUTE_PATH, "{}_{}".format(
                u, v), "MaxExpCongestionVsIterations.dat")

            PATH_DIFF = path.join(
                self.EXPERIMENT_ABSOLUTE_PATH, "path_diff", "{}_{}.txt".format(u, v))
            CONGESTION_DIFF = path.join(
                self.EXPERIMENT_ABSOLUTE_PATH, "congestion_diff", "{}_{}.txt".format(u, v))

            self._system("diff {} {} > {}".format(
                INITIAL_PATHS, TEST_PATHS, PATH_DIFF))
            self._system("diff {} {} > {}".format(
                INITIAL_CONGESTION, TEST_CONGESTION, CONGESTION_DIFF))
            path_churn.append(diff_compare(PATH_DIFF, 'path'))
            congestion_change.append(diff_compare(CONGESTION_DIFF))

            draw_graph(test_alpwolf.logical_graph, path.join(
                GRAPHS_PATH, self.name + "_{}_{}".format(u, v)))

        PLOT_DIR = path.join(self.EXPERIMENT_ABSOLUTE_PATH, "plot_dir")
        CONGESTION_VS_PATHCHURN = path.join(
            PLOT_DIR, "congestion_vs_pathChurn")
        makedirs(PLOT_DIR, exist_ok=True)
        plot_points(path_churn, congestion_change, "Path Churn",
                    "Congestion Change", CONGESTION_VS_PATHCHURN)
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
                node: ("sw" + str(int(node) - 1)) for (node) in G}
        if 1:
            yates_to_ripple_map = {node: ("s{}".format(node)) for (node) in G}
        gml_view = relabel_nodes(G, yates_to_ripple_map, copy=True)
        write_gml(gml_view, name)
        del gml_view

    def perform_sim(self, circuits=1, start_iter=0, end_iter=0, repeat=False):
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
        }
        circuits_added = False

        if CROSSFIRE == True:
            sig_add_circuits = True
        else:
            sig_add_circuits = False

        sig_drop_circuits = False
        logger.debug("Yates.perform_sim")
        name = self.name
        iterations = self.iterations
        topo = self.topo_file
        traffic = self.traffic_file
        hosts = self.hosts_file
        te_method = self.te_method
        EXPERIMENT_ID = self.EXPERIMENT_ID
        EXPERIMENT_ABSOLUTE_PATH = self.EXPERIMENT_ABSOLUTE_PATH
        flux_circuits = []  # Stateful list of active circuits triggered from sig_add_circuits
        j = 1
        # Open traffic file and pass a new line from the file for every iteration.
        with open(traffic, 'rb') as fob:
            sig_add_circuits = True
            PREV_ITER_ABS_PATH = ""
            for i in range(1, iterations+2):
                if self.shakeroute:
                    ITERATION_ID = "{}".format(self.strategy)

                elif repeat:
                    if j == 1:
                        ITERATION_ID = name + \
                            "_{}-{}-{}".format(i, j, iterations)
                    elif j == 2:
                        ITERATION_ID = name + \
                            "_{}-{}-{}".format(i-1, j, iterations)

                else:
                    ITERATION_ID = name+"_{}-{}".format(i, iterations)

                ITERATION_REL_PATH = path.join(EXPERIMENT_ID, ITERATION_ID)
                ITERATION_ABS_PATH = path.join(
                    EXPERIMENT_ABSOLUTE_PATH, ITERATION_ID)

                CONGESTION_PATH = path.join(
                    EXPERIMENT_ABSOLUTE_PATH, ITERATION_ID, "EdgeCongestionVsIterations.dat")
                MAX_CONGESTION_PATH = path.join(
                    EXPERIMENT_ABSOLUTE_PATH, ITERATION_ID, "MaxCongestionVsIterations.dat")

                # Grab line from tm and throw it into a temp file to run yates.
                if repeat and i == (end_iter + 1) and j == 2:
                    pass
                    # don't read a new line if we are on the repetition step
                else:
                    tm_i_data = fob.readline()

                if i < start_iter:
                    continue
                if i > end_iter:
                    if repeat and i == (end_iter + 1) and j == 2:
                        pass
                    else:
                        continue
                j += 1
                temp_tm_i = self.nonce
                temp_fob = open(self.nonce, 'wb')
                temp_fob.write(tm_i_data)
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
                    if self.strategy == 'cli':
                        client = self.wolf.cli()

                    elif self.strategy == 'cache' and sig_add_circuits and not circuits_added:
                        defender = Defender(
                            self.name, circuits, self.method, self.use_heuristic, PREV_ITER_ABS_PATH, self.attack_proportion)
                        # TODO: Pass get_strategic_circuit the paths file from the previous iteration.
                        new_circuit = defender.get_strategic_circuit()
                        if type(new_circuit) == tuple and len(new_circuit) == 2:
                            print("Adding {} ({}, {}) circuits.".format(
                                circuits), *new_circuit)
                            for _ in range(circuits):
                                u, v = new_circuit
                                self.wolf.add_circuit(u, v)

                            circuits_added = True

                        # elif new_circuit == []:
                        #     print("Could Not Add New Circuit")
                        #     self.exit_early = True

                    # elif self.strategy == 'onset' and sig_add_circuits and not circuits_added:
                    elif self.strategy == 'onset' and sig_add_circuits:
                        optimizer = Link_optimization(
                            G=self.wolf.logical_graph, BUDGET=3, demand_matrix=self.nonce, network=self.name)
                        # optimizer.run_model()
                        if self.te_method == "-ecmp":
                            max_load = 0.5
                            if self.shakeroute:
                                max_load = 10000.0
                        else:
                            max_load = 1.0

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

                        circuit_tag = ''
                        if type(new_circuit) == list and len(new_circuit) > 0:
                            for nc in new_circuit:
                                u, v = nc
                                if circuit_tag == "":
                                    circuit_tag += "circuit-" + u + "-" + v
                                else:
                                    circuit_tag += "." + u + "-" + v

                                for _ in range(circuits):
                                    self.wolf.add_circuit(u, v)

                            circuits_added = True
                            flux_circuits.extend(new_circuit)

                        # elif new_circuit == []:
                        #     print("Could Not Add New Circuit")
                        #     self.exit_early = True

                        sig_add_circuits = False

                    elif self.strategy == "greylambda" and sig_add_circuits:
                        edge_congestion_file = path.join(
                            PREV_ITER_ABS_PATH,
                            "EdgeCongestionVsIterations.dat")
                        edge_congestion_d = read_link_congestion_to_dict(
                            edge_congestion_file)
                        congested_edges = [
                            k for k in edge_congestion_d if edge_congestion_d[k] == 1]
                        for edge in congested_edges:
                            u, v = edge.strip("()").replace("s","").split(",")
                            u=int(u)
                            v=int(v)
                            for _ in range(circuits):
                                self.wolf.add_circuit(u, v)
                        flux_circuits.extend(congested_edges)
                        sig_add_circuits = False

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
                    updated_topology_file = iteration_topo
                    self.export_logical_topo_to_gml(
                        updated_topology_file + '.gml')
                    Gml_to_dot(self.wolf.logical_graph,
                               iteration_topo + '.dot')

                    # Draw the link graph for the instanced topology.
                    # self.base_graph._init_link_graph()

                    iter_congestion = self._yates(iteration_topo + '.dot',
                                                  ITERATION_REL_PATH,
                                                  traffic_file=temp_tm_i)
                    if len(new_circuit) > 0:
                        reconfig_time = get_reconfig_time(
                            updated_topology_file + '.gml',
                            new_circuit)
                    else:
                        reconfig_time = 0
                    return_data["ReconfigTime"].append(reconfig_time)
                    return_data["Strategy"].append(
                        "{} {}".format(self.te_method, self.strategy))
                    return_data["Routing"].append(
                        "{}".format(self.te_method).strip("-").upper())
                    return_data["Defense"].append("{}".format(self.strategy))

                    return_data["Congestion"].append(
                        read_result_val(
                            path.join(
                                ITERATION_ABS_PATH,
                                "MaxExpCongestionVsIterations.dat")
                        )
                    )
                    return_data["Loss"].append(
                        read_result_val(
                            path.join(
                                ITERATION_ABS_PATH,
                                "CongestionLossVsIterations.dat")
                        )
                    )
                    return_data["Throughput"].append(
                        read_result_val(
                            path.join(
                                ITERATION_ABS_PATH,
                                "TotalThroughputVsIterations.dat")
                        )
                    )
                    return_data["Total Links Added"].append(len(new_circuit))
                    return_data["Links Added"].append(new_circuit)
                    return_data["Total Flux Links"].append(len(flux_circuits))
                    return_data["Total Links Dropped"].append(
                        len(drop_circuits))
                    return_data["Links Dropped"].append(drop_circuits)
                    return_data["Link Bandwidth Coefficient"].append(max_load)

                    if iter_congestion == "SIG_EXIT":
                        return

                    if iter_congestion >= self.congestion_threshold_upper_bound:
                        sig_add_circuits = True

                    elif iter_congestion <= self.congestion_threshold_lower_bound:
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
                    print("Unknown Error")
                    print(repr(e))
                    traceback.print_exc()
                    self._system("rm %s" % temp_tm_i)
                    return return_data

                finally:
                    # Remove the temp file.
                    self._system("rm %s" % temp_tm_i)

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
        """ validates yates by ensuring that the necessary files implied by `self.name` exist.
            If the topo file is not found, then the program halts.
            traffic and hosts files are generated if they are needed.

            Side effect: Assigns the following object variables to strings describing a verified
            paths in the file system: self.topo_file, self.hosts_file, self.traffic_file

            Also modifies self.base_graph. 
        """
        name = self.name
        # Verify topology file and build base graph

        def verify_topo():
            logger.debug("verifying topology file.")
            if self.shakeroute and self.strategy != 'baseline':
                base_topo_file = path.join(
                    SCRIPT_HOME, "data", "graphs", "fiber_cut", self.shakeroute, self.strategy+".gml")

            else:
                base_topo_file = path.join(
                    SCRIPT_HOME, "data", "graphs", "gml", name+".gml")

            assert isfile(base_topo_file), "Error topology file not found: {}".format(
                base_topo_file)
            logger.debug("topology file check passed.")
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
                    SCRIPT_HOME, "data", "hosts", self.shakeroute+".hosts")
            else:
                hosts_file = path.join(
                    SCRIPT_HOME, "data", "hosts", name+".hosts")
            hosts_folder = path.join(SCRIPT_HOME, "data", "hosts")
            try:  # sets self.num_hosts and checks file exists.
                assert isfile(hosts_file), "Error topology file not found: {}".format(
                    hosts_file)
                logger.debug("Host file successfully located.")
                # g = read_dot(self.topo_file)
                # self.num_hosts = len([n for n in g.nodes if n.startswith('h')])
                lines = count_lines(hosts_file)
                assert lines == self.num_hosts, "Host file has wrong number of lines. expected: {} got: {}".format(
                    self.num_hosts, lines)

            except AssertionError:  # Create the hosts file
                def create_host_file():
                    logger.debug("Creating host file.")
                    # g = read_dot(self.topo_file)
                    # self.num_hosts = len([n for n in g.nodes if n.startswith('h')])
                    # create host dir if needed.
                    makedirs(dirname(hosts_file), exist_ok=True)
                    
                    with open(hosts_file, 'w') as host_fob:
                        for i in range(1, self.num_hosts+1):
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
                    "no file stated. Generating common traffic file string")
                traffic_file = path.join(
                    SCRIPT_HOME, "data", "traffic", name+".txt")
            else:
                logger.debug(
                    "Attempting to use traffic file provided by user.")
                logger.debug(
                    "file: {}".format(self.traffic_file))
                traffic_file = self.traffic_file

            try:
                if self.start_clean:  # start_clean
                    self._system("rm {}".format(traffic_file))
                assert isfile(traffic_file), "Error traffic file not found: {}".format(
                    traffic_file)
                logger.debug("traffic file found.")
                lines = count_lines(traffic_file)
                assert lines >= self.iterations, "traffic file found, but has too few lines. expected: {} got: {}".format(
                    self.iterations, lines)
                logger.debug("traffic file line-count passed.")

            except AssertionError:
                if self.start_clean:
                    pass
                else:
                    logger.error(
                        "Error verifying traffic file Create one now? [y/n]")

                    create = input()
                    if create.lower().startswith('y'):
                        pass
                    else:
                        exit()

                rand_gravity_matrix(
                    self.num_hosts, self.iterations, self.magnitude, traffic_file)
                lines = count_lines(traffic_file)
                assert lines >= self.iterations, "traffic file created, but has too few lines. expected: {} got: {}".format(
                    self.iterations, lines)

            logger.debug("Host file check passed.")
            self.traffic_file = traffic_file

        verify_topo()
        verify_hosts()
        verify_traffic()


def sanitize_magnitude(mag_arg: str) -> int:
    # WARNING. This function has been moved to .src.utilities.tmg.
    # Use that version instead.
    """Converts input magnitude arg into an integer
        Unit identifier is the 4th from list character in the string, mag_arg[-4].
        e.g., 1231904Gbps G is 4th from last.
        this returns 1231904 * 10**9.
    Args:
        mag_arg (str): number joined with either T, G, M, or K.

    Returns:
        int: Value corresponding to the input.
    """

    mag = mag_arg[-4].strip()
    coefficient = int(mag_arg[0:-4])
    logger.debug(coefficient)
    logger.debug(mag)
    exponent = 0
    if mag == 'T':
        exponent = 12
    elif mag == 'G':
        exponent = 9
    elif mag == 'M':
        exponent == 6
    elif mag == 'k':
        exponent == 3
    else:
        raise("ERROR: ill formed magnitude argument. Expected -m <n><T|G|M|k>bps, e.g., 33Gbps")
    result = coefficient * 10 ** exponent
    logger.debug("Result: {}".format(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", type=str,
                        help="str: Topology name. Should be accessible in ./data/graphs/gml/")
    parser.add_argument("num_hosts", type=int,
                        help="str: Number of nodes in the topology")
    parser.add_argument("test", type=str,
                        help="str: Type of test to run. Either 'add_circuit' to do preprocessing or anything else to run another experiment.")
    parser.add_argument("-i", "--iterations", type=int,
                        default=1, help="int: How many iterations to run for.")
    parser.add_argument("-te", "--trafficEngineering", type=str,
                        default="ecmp", help="Which TE method to use. e.g., -ecmp, -mcf")
    parser.add_argument("-c", "--clean", type=str, default="",
                        help="Remove old traffic matrices and other related data, to start new. Give magnitude for traffic matrix if starting clean. Can write <n><T|G|M|K>bps, e.g., 400Gbps, 3Tbps, 1000Mbps, 400kbps. This argument required if --clean is set and ignored otherwise.")
    parser.add_argument("-C", "--circuits", type=int, default=5,
                        help="Number of circuits to add")
    parser.add_argument("-s", "--strategy", type=str, default="",
                        help="defense strategy. type 'cli' for command line defense, or 'cache' to use preprocessed data to choose the best circuits to add in the presence of the attack.")
    parser.add_argument("-t", "--traffic_file", type=str, default="",
                        help="custom traffic file. Use this if you are simulating attacker behavior and the cache strategy.")
    parser.add_argument("-p", "--postProcess", action='store_true',
                        help="Run post processing (without experiment).")
    # parser.add_argument("-H", "--heuristic", action='store_true',
    #                     help="Use heuristic for selecting candidate links.")
    parser.add_argument("-H", "--heuristic", type=str, default="",
                        help="Use heuristic for selecting candidate links.\n" +
                        "\t(1) Link that reduces the congestion on most congested edge\n" +
                        "\t(2) Link that reduces max congestion"
                        "\t(3) Link that introduces that greatest number of new paths" +
                        "\t(4) Link that removes the greatest number of paths"
                        )

    parser.add_argument("-ts", "--time_series", action='store_true',
                        help="Post process time series data.")
    parser.add_argument("-tsFiles", "--time_series_files", nargs="+",
                        help="Required if -ts set. Time series data files.")
    parser.add_argument("-tsLabels", "--time_series_labels", nargs="+",
                        help="Required if -ts set. Time series data labels.")
    parser.add_argument("-tsIterations", "--time_series_iterations", type=int, default=0,
                        help="Required if -ts set. Time series iterations (int).")
    parser.add_argument("-r", "--result_ids", nargs="+",
                        help="result_ids for comparative analysis")
    parser.add_argument("-ftxas", "--fallow_tx_allocation_strategy", type=str, default="static", help="static, dynamic, or file.")
    parser.add_argument("-ftxaf", "--fallow_tx_allocation_file", type=str, default="", help="file containing fallow transponder allocation.")


    # parser.add_argument("-m", "--magnitude", help="Average magnitude for traffic matrix if starting clean. " +\
    # "Can write <n><T|G|M|K>bps, e.g., 400Gbps, 3Tbps, 1000Mbps, 400Kbps. " +\
    # "This argument required if --clean is set and ignored otherwise.")

    args = parser.parse_args()
    if args.clean:
        start_clean = True
        magnitude = sanitize_magnitude(args.clean)
    else:
        start_clean = False
        magnitude = 0

    te_method = "-" + args.trafficEngineering
    iterations = args.iterations
    topology = args.topology
    num_hosts = args.num_hosts
    test = args.test
    circuits = args.circuits
    strategy = args.strategy
    traffic_file = args.traffic_file
    postProcess = args.postProcess
    heuristic = args.heuristic
    result_ids = args.result_ids
    time_series = args.time_series
    time_series_files = args.time_series_files
    time_series_labels = args.time_series_labels
    time_series_iterations = args.time_series_iterations
    fallow_tx_allocation_strategy = args.fallow_tx_allocation_strategy
    fallow_tx_allocation_file = args.fallow_tx_allocation_file
    if heuristic:
        test += "_heuristic"  # use heuristic to choose new links
        method = "heuristic"
    else:
        test += "_circuits"  # use all possible new links
        method = "circuits"

    if postProcess:
        if time_series:
            post_proc_timeseries(time_series_files, topology,
                                 time_series_iterations, time_series_labels)
            exit()
        else:
            post_process(test, result_ids)
            exit()

    logger.info("Beginning simulation.")
    attack_sim = Attack_Sim(topology, num_hosts, test, iterations=iterations,
                            te_method=te_method,
                            start_clean=start_clean,
                            magnitude=magnitude,
                            traffic_file=traffic_file,
                            strategy=strategy,
                            fallow_transponders=circuits,
                            use_heuristic=heuristic,
                            method=method,
                            fallow_tx_allocation_strategy=fallow_tx_allocation_strategy,
                            fallow_tx_allocation_file=fallow_tx_allocation_file)

    if "add_circuit" in test:
        attack_sim.evaluate_performance_from_adding_link(circuits)
    else:
        attack_sim.perform_sim(circuits)

    # elif test == "multi_circuit":
    #     result_dirs = []
    #     for i in range(circuits):
    #         result_dirs.append(myYates.evaluate_performance_from_adding_link(circuits))
    #     congestion_data = {}
    #     for i, d in enumerate(result_dirs):
    #         congestion_data[str(i+1)] = readcsv()
