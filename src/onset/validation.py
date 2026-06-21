from __future__ import annotations

import os
from os import makedirs, system
from os.path import dirname, isfile
from copy import deepcopy

from onset.constants import SCRIPT_HOME
from onset.utilities.logger import logger
from onset.utilities.sysUtils import count_lines
from onset.utilities.tmg import rand_gravity_matrix
from onset.te import evaluate
from onset.utilities.gml_to_dot import Gml_to_dot
from onset.utilities.diff_compare import diff_compare
from onset.utilities.plotters import (
    cdf_average_congestion,
    cdf_churn,
    draw_graph,
    plot_points,
)
from onset.utilities.graph_utils import write_gml
from networkx.relabel import relabel_nodes

DRAW = False


def _system(command: str):
    logger.info("Calling system command: {}".format(command))
    return system(command)


def _evaluate_te(
    topo_file,
    result_path,
    traffic_file,
    shakeroute,
    hosts_file,
    te_method,
    exit_early,
    network_name,
):
    if shakeroute:
        result_path = os.path.join(network_name, result_path)
    result = evaluate(
        topo_file=topo_file,
        traffic_file=traffic_file,
        hosts_file=hosts_file,
        te_method=te_method,
        result_path=result_path,
        budget=3,
    )
    max_congestion = result.max_congestion
    logger.info("Max congestion: {}".format(max_congestion))
    if exit_early and float(max_congestion) == 1.0:
        logger.info("Max Congestion has reached 1. Ending simulation.")
        return "SIG_EXIT"
    return max_congestion


def export_logical_topo_to_gml(name, G):
    # Relabels to be consistent with naming in Ripple.
    if 0:  # to make consistent with ripple examples.
        te_to_ripple_map = {
            node: ("sw" + str(int(node) - 1)) for (node) in G
        }

    te_to_ripple_map = {node: ("s{}".format(node)) for (node) in G}
    gml_view = relabel_nodes(G, te_to_ripple_map, copy=True)
    write_gml(gml_view, name)
    del gml_view


def evaluate_performance_from_adding_link(
    wolf,
    network_name,
    experiment_absolute_path,
    experiment_id,
    use_heuristic,
    te_method,
    traffic_file,
    shakeroute,
    hosts_file,
    exit_early,
    circuits_to_add=1,
):
    path_churn = []
    congestion_change = []

    # prep directory
    GRAPHS_PATH = os.path.join(experiment_absolute_path, "graphs")
    makedirs(GRAPHS_PATH, exist_ok=True)

    # prep initial file
    initial_graph = wolf.logical_graph
    INITIAL_GRAPH_PATH = os.path.join(
        GRAPHS_PATH, network_name + "_0.dot"
    )
    INITIAL_RESULTS_REL_PATH = os.path.join(experiment_id, "__0")

    # Write Graphs to files
    export_logical_topo_to_gml(
        INITIAL_GRAPH_PATH.replace(".dot", ".gml"), G=initial_graph
    )
    Gml_to_dot(initial_graph, INITIAL_GRAPH_PATH)

    if (
        _evaluate_te(
            INITIAL_GRAPH_PATH, INITIAL_RESULTS_REL_PATH,
            traffic_file, shakeroute, hosts_file, te_method, exit_early, network_name,
        )
        == "SIG_EXIT"
    ):
        return

    PATH_DIFF_FOLDER = os.path.join(
        experiment_absolute_path, "path_diff"
    )
    CONGESTION_DIFF_FOLDER = os.path.join(
        experiment_absolute_path, "congestion_diff"
    )
    makedirs(PATH_DIFF_FOLDER, exist_ok=True)
    makedirs(CONGESTION_DIFF_FOLDER, exist_ok=True)

    INITIAL_PATHS = os.path.join(
        experiment_absolute_path,
        "__0",
        "paths",
        te_method.strip("-") + "_0",
    )
    INITIAL_CONGESTION = os.path.join(
        experiment_absolute_path,
        "__0",
        "MaxExpCongestionVsIterations.dat",
    )
    # Run experiment
    if use_heuristic != "":
        # FIXME CHANGE BACK TO `candid_set='ranked'`
        # candidate_circuits = wolf.get_candidate_circuits(candid_set='ranked', k=5, l=5)
        candidate_circuits = wolf.get_candidate_circuits(
            candid_set="all"
        )
    else:
        candidate_circuits = wolf.get_candidate_circuits(
            candid_set="all"
        )

    for u, v in candidate_circuits:
        TEST_RESULTS_REL_PATH = os.path.join(
            experiment_id, "{}_{}".format(u, v)
        )
        test_alpwolf = deepcopy(wolf)
        for _ in range(circuits_to_add):
            test_alpwolf.add_circuit(u, v)

        TEST_GRAPH_PATH = os.path.join(
            GRAPHS_PATH, network_name + "_{}_{}.dot".format(u, v)
        )

        # Write Graphs to files
        Gml_to_dot(test_alpwolf.logical_graph, TEST_GRAPH_PATH)
        export_logical_topo_to_gml(
            TEST_GRAPH_PATH.replace(".dot", ".gml"),
            G=test_alpwolf.logical_graph,
        )

        if (
            _evaluate_te(
                TEST_GRAPH_PATH, TEST_RESULTS_REL_PATH,
                traffic_file, shakeroute, hosts_file, te_method, exit_early, network_name,
            )
            == "SIG_EXIT"
        ):
            return

        TEST_PATHS = os.path.join(
            experiment_absolute_path,
            "{}_{}".format(u, v),
            "paths",
            te_method.strip("-") + "_0",
        )
        TEST_CONGESTION = os.path.join(
            experiment_absolute_path,
            "{}_{}".format(u, v),
            "MaxExpCongestionVsIterations.dat",
        )

        PATH_DIFF = os.path.join(
            experiment_absolute_path,
            "path_diff",
            "{}_{}.txt".format(u, v),
        )
        CONGESTION_DIFF = os.path.join(
            experiment_absolute_path,
            "congestion_diff",
            "{}_{}.txt".format(u, v),
        )

        _system(
            "diff {} {} > {}".format(INITIAL_PATHS, TEST_PATHS, PATH_DIFF)
        )
        _system(
            "diff {} {} > {}".format(
                INITIAL_CONGESTION, TEST_CONGESTION, CONGESTION_DIFF
            )
        )
        path_churn.append(diff_compare(PATH_DIFF, "path"))
        congestion_change.append(diff_compare(CONGESTION_DIFF))
        if DRAW: 
            draw_graph(test_alpwolf.logical_graph, os.path.join(
                    GRAPHS_PATH, network_name + "_{}_{}".format(u, v)))

    PLOT_DIR = os.path.join(experiment_absolute_path, "plot_dir")
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

    return experiment_absolute_path


def verify_topo(network_name, shakeroute, topology_programming_method):
    logger.debug("verifying topology file.")
    gml_handle = os.path.join(
        SCRIPT_HOME, "data", "graphs", "gml", network_name + ".gml"
    )
    json_handle = os.path.join(
        SCRIPT_HOME, "data", "graphs", "json", network_name + ".json"
    )
    if (
        shakeroute
        and topology_programming_method != "baseline"
    ):
        base_topo_file = os.path.join(
            SCRIPT_HOME,
            "data",
            "graphs",
            "fiber_cut",
            shakeroute,
            topology_programming_method + ".gml",
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
    return base_topo_file


def create_host_file(hosts_file, num_hosts):
    logger.debug("Creating host file.")
    makedirs(dirname(hosts_file), exist_ok=True)
    with open(hosts_file, "w") as host_fob:
        for i in range(1, num_hosts + 1):
            host_fob.write("h" + str(i) + "\n")
    logger.debug("Host file successfully created.")


def verify_hosts(network_name, shakeroute, num_hosts):
    logger.debug("verifying hosts file.")
    if shakeroute:
        hosts_file = os.path.join(
            SCRIPT_HOME, "data", "hosts", shakeroute + ".hosts"
        )
    else:
        hosts_file = os.path.join(
            SCRIPT_HOME, "data", "hosts", network_name + ".hosts"
        )
    hosts_folder = os.path.join(SCRIPT_HOME, "data", "hosts")
    try:  # sets num_hosts and checks file exists.
        assert isfile(
            hosts_file
        ), "Error topology file not found: {}".format(hosts_file)
        logger.debug("Host file successfully located.")
        lines = count_lines(hosts_file)
        assert (
            lines == num_hosts
        ), "Host file has wrong number of lines. expected: {} got: {}".format(
            num_hosts, lines
        )

    except AssertionError:  # Create the hosts file

        create_host_file(hosts_file, num_hosts)
    logger.debug("Host file check passed.")
    return hosts_file


def verify_traffic(network_name, traffic_file, start_clean, num_hosts, iterations, magnitude):
    logger.debug("verifying traffic file.")
    if traffic_file == "":
        logger.debug(
            "no file stated. Generating common traffic file string"
        )
        traffic_file = os.path.join(
            SCRIPT_HOME, "data", "traffic", network_name + ".txt"
        )
    else:
        logger.debug(
            "Attempting to use traffic file provided by user."
        )
        logger.debug("file: {}".format(traffic_file))
        # traffic_file already set from parameter

    try:
        if start_clean:  # start_clean
            _system("rm {}".format(traffic_file))
        assert isfile(
            traffic_file
        ), "Error traffic file not found: {}".format(traffic_file)
        logger.debug("traffic file found.")
        line_count = count_lines(traffic_file)
        if line_count < iterations:
            logger.error(
                "traffic file found, but has too few lines. expected: {} got: {}".format(
                    iterations, line_count
                )
            )
            assert False
        logger.debug("traffic file line-count passed.")
        # check total entries on each line is correct.
        with open(traffic_file, 'r') as tm_fob:
            lines = tm_fob.readlines()
            last_line = lines.pop()
            num_expected_entries = num_hosts ** 2
            for l in lines:
                num_entries = len(l.strip().split())
                if num_entries != num_expected_entries:                           
                    logger.error(f"Traffic matrix (TM): {traffic_file}. Network hosts (n): {num_hosts}. Line in TM should have n^2 entries ({num_expected_entries}). Got {num_entries}.")
                    assert False

    except AssertionError:
        if start_clean:
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
            num_hosts,
            iterations,
            magnitude,
            traffic_file,
        )
        lines = count_lines(traffic_file)
        assert (
            lines >= iterations
        ), "traffic file created, but has too few lines. expected: {} got: {}".format(
            iterations, lines
        )

    logger.debug("Host file check passed.")
    return traffic_file


def validate_simulation_inputs(
    network_name,
    shakeroute,
    topology_programming_method,
    traffic_file,
    start_clean,
    num_hosts,
    iterations,
    magnitude,
):
    """Validate the files implied by ``network_name``.
    If the topo file is not found, then the program halts.
    traffic and hosts files are generated if they are needed.
    Returns (topo_file, hosts_file, traffic_file).
    """
    topo_file = verify_topo(network_name, shakeroute, topology_programming_method)
    hosts_file = verify_hosts(network_name, shakeroute, num_hosts)
    traffic_file = verify_traffic(
        network_name, traffic_file, start_clean, num_hosts, iterations, magnitude
    )
    return topo_file, hosts_file, traffic_file
