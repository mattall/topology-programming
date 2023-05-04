# main.py
# Written by MNH
# Takes network graph (.dot format) and TM Series.
# Produces two network simulations with traffic engineering sim (YATES).
#   1) Run YATES over the plain network graph.
#   2) Run YATES over the adaptive graph.

from onset.utilities.sanitize_magnitude import sanitize_magnitude
from onset.simulator import Simulation
import argparse
from onset.utilities.logger import logger
from onset.utilities.post_process import post_process, post_proc_timeseries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "topology",
        type=str,
        help="str: Topology name. Should be accessible in ./data/graphs/gml/",
    )
    parser.add_argument(
        "num_hosts", type=int, help="str: Number of nodes in the topology"
    )
    parser.add_argument(
        "test",
        type=str,
        help="str: Type of test to run. Either 'add_circuit' to do preprocessing or anything else to run another experiment.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1,
        help="int: How many iterations to run for.",
    )
    parser.add_argument(
        "-te",
        "--trafficEngineering",
        type=str,
        default="ecmp",
        help="Which TE method to use. e.g., -ecmp, -mcf",
    )
    parser.add_argument(
        "-c",
        "--clean",
        type=str,
        default="",
        help="Remove old traffic matrices and other related data, to start new. Give magnitude for traffic matrix if starting clean. Can write <n><T|G|M|K>bps, e.g., 400Gbps, 3Tbps, 1000Mbps, 400kbps. This argument required if --clean is set and ignored otherwise.",
    )
    parser.add_argument(
        "-C",
        "--circuits",
        type=int,
        default=5,
        help="Number of circuits to add",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="",
        help="defense strategy. type 'cli' for command line defense, or 'cache' to use preprocessed data to choose the best circuits to add in the presence of the attack.",
    )
    parser.add_argument(
        "-t",
        "--traffic_file",
        type=str,
        default="",
        help="custom traffic file. Use this if you are simulating attacker behavior and the cache strategy.",
    )
    parser.add_argument(
        "-p",
        "--postProcess",
        action="store_true",
        help="Run post processing (without experiment).",
    )
    # parser.add_argument("-H", "--heuristic", action='store_true',
    #                     help="Use heuristic for selecting candidate links.")
    parser.add_argument(
        "-H",
        "--heuristic",
        type=str,
        default="",
        help="Use heuristic for selecting candidate links.\n"
        + "\t(1) Link that reduces the congestion on most congested edge\n"
        + "\t(2) Link that reduces max congestion"
        + "\t(3) Link that introduces that greatest number of new paths"
        + "\t(4) Link that removes the greatest number of paths",
    )

    parser.add_argument(
        "-ts",
        "--time_series",
        action="store_true",
        help="Post process time series data.",
    )
    parser.add_argument(
        "-tsFiles",
        "--time_series_files",
        nargs="+",
        help="Required if -ts set. Time series data files.",
    )
    parser.add_argument(
        "-tsLabels",
        "--time_series_labels",
        nargs="+",
        help="Required if -ts set. Time series data labels.",
    )
    parser.add_argument(
        "-tsIterations",
        "--time_series_iterations",
        type=int,
        default=0,
        help="Required if -ts set. Time series iterations (int).",
    )
    parser.add_argument(
        "-r",
        "--result_ids",
        nargs="+",
        help="result_ids for comparative analysis",
    )
    parser.add_argument(
        "-ftxas",
        "--fallow_tx_allocation_strategy",
        type=str,
        default="static",
        help="static, dynamic, or file.",
    )
    parser.add_argument(
        "-ftxaf",
        "--fallow_tx_allocation_file",
        type=str,
        default="",
        help="file containing fallow transponder allocation.",
    )

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
    test_name = args.test
    circuits = args.circuits
    topology_programming_method = args.strategy
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
        test_name += "_heuristic"  # use heuristic to choose new links
        candidate_link_choice_method = "heuristic"
    else:
        test_name += "_circuits"  # use all possible new links
        candidate_link_choice_method = "circuits"

    if postProcess:
        if time_series:
            post_proc_timeseries(
                time_series_files,
                topology,
                time_series_iterations,
                time_series_labels,
            )
            exit()
        else:
            post_process(test_name, result_ids)
            exit()

    logger.info("Beginning simulation.")
    sim = Simulation(
        topology,
        num_hosts,
        test_name,
        iterations=iterations,
        te_method=te_method,
        start_clean=start_clean,
        magnitude=magnitude,
        traffic_file=traffic_file,
        topology_programming_method=topology_programming_method,
        fallow_transponders=circuits,
        use_heuristic=heuristic,
        candidate_link_choice_method=candidate_link_choice_method,
        fallow_tx_allocation_strategy=fallow_tx_allocation_strategy,
        fallow_tx_allocation_file=fallow_tx_allocation_file,
    )

    if "add_circuit" in test_name:
        sim.evaluate_performance_from_adding_link(circuits)
    else:
        sim.perform_sim(circuits)
