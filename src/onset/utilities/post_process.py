# Create Two CDF Plots from the results data for each topology.
#   Plot 1. Path Churn.
#   Plot 2. Congestion
#       2a. Average
#       2b. 90-th percentile
#       2c. Max

import re
import os
from os import path, listdir, makedirs
from pickle import loads, dumps, load, dump
from unittest import result
from matplotlib.pyplot import plot, xlabel
from numpy import exp, result_type
import sys
from onset.utilities import SCRIPT_HOME, ZERO_INDEXED
from onset.utilities.plotters import congestion_multi_cdf, plot_points, plot_timeseries, plot_reconfig_time, congestion_multi_cdf_v2
# from __init__ import SCRIPT_HOME, ZERO_INDEXED
# from plotters import congestion_multi_cdf, plot_points, plot_timeseries


def post_process(test:str, result_ids:list, networks:list):
    """Post-process experiment data
    Args:
        test (str): file descriptor for finding test data. 
        iterations (list, optional): test iterations to include in plot CDFs. Defaults to [].
    """    
    data_root = path.join(SCRIPT_HOME, "data", "results")
    # experiment_tag = "_add_circuit_circuits_"
    # experiment_tag = "_" + test + "_" 
    experiment_tag = test  + "_"
    # get topology
    experiment_dirs = listdir(data_root)
    experiment_dirs = [ed for ed in experiment_dirs if experiment_tag in ed]
    experiment_dirs.sort()
    # networks = set([net.split(experiment_tag)[0] for net in experiment_dirs])
    # iterations = 5
    for net in networks:
        # plot_dir = path.join(SCRIPT_HOME, "temp_results", net + '_add_circuit')
        # plot_dir = path.join(SCRIPT_HOME, "data", "results", net + '_' + test)
        plot_dir = path.join(SCRIPT_HOME, "data", "results", test)
        makedirs(plot_dir, exist_ok=True)
        # if net != 'bellCanada': continue
        # if net == 'sprint': continue
        congestion_avg = {}
        congestion_difference_avg = {}
        congestion_90pct = {}
        congestion_difference_90pct = {}
        congestion_max = {}
        congestion_difference_max = {}
        num_paths = {}
        num_paths_difference = {}
        # for i in range(1, iterations+1):
        for res_id in result_ids:            
            i = int(res_id)
            # iteration_tag = experiment_tag + str(i)
            iteration_tag = test
            congestion_avg[i] = []
            congestion_difference_avg[i] = []
            congestion_90pct[i] = []
            congestion_difference_90pct[i] = []
            congestion_max[i] = []
            congestion_difference_max[i] = []
            num_paths[i] = []
            num_paths_difference[i] = []

            if res_id == result_ids[0]:  # only need to do this path prep business once. It is the same on each iteration.
                # data/results/ANS_add_circuit_circuits_3/
                base_tag = "__0/"
                metric_tags = ["MeanCongestionVsIterations.dat", "k90ExpCongestionVsIterations.dat",
                               "MaxCongestionVsIterations.dat", "NumPathsVsIterations.dat"]

                base_metric_paths = [
                    path.join(data_root, iteration_tag, base_tag, m_tag) for m_tag in metric_tags]

                base_metric_values = [read_result_val(
                    bmp) for bmp in base_metric_paths]

                circuit_regex = re.compile('^[0-9]+_[0-9]+$')
                circuit_tags = [tag for tag in listdir(
                    path.join(data_root, iteration_tag)) if circuit_regex.match(tag)]

                congestion_avg[0] = [base_metric_values[0]]
                congestion_difference_avg[0] = [base_metric_values[0]]
                congestion_90pct[0] = [base_metric_values[1]]
                congestion_difference_90pct[0] = [base_metric_values[1]]
                congestion_max[0] = [base_metric_values[2]]
                congestion_difference_max[0] = [base_metric_values[2]]
                num_paths[0] = [base_metric_values[3]]
                num_paths_difference[0] = [base_metric_values[3]]

            for circuit_tag in circuit_tags:
                circuit_metric_paths = [path.join(
                    data_root, iteration_tag, circuit_tag, m_tag) for m_tag in metric_tags]

                circuit_metric_values = [read_result_val(
                    cmp) for cmp in circuit_metric_paths]

                circuit_metric_percent_difference = [calc_percent_diff(
                    base_val, circuit_val) for base_val, circuit_val in zip(base_metric_values, circuit_metric_values)]

                congestion_difference_avg[i].append(
                    circuit_metric_percent_difference[0])

                congestion_difference_90pct[i].append(
                    circuit_metric_percent_difference[1])

                congestion_difference_max[i].append(
                    circuit_metric_percent_difference[2])

                num_paths_difference[i].append(
                    circuit_metric_percent_difference[3])

                congestion_avg[i].append(circuit_metric_values[0])
                congestion_90pct[i].append(circuit_metric_values[1])
                congestion_max[i].append(circuit_metric_values[2])
                num_paths[i].append(circuit_metric_values[3])

            # plot scatter plot of congestion_avg vs paths,
            # plot_points(num_paths[i], congestion_avg[i],
            #             "Total Paths", "Avg. Congestion",
            #             path.join(plot_dir, '{}_avg_congestion_vs_paths'.format(i)))
            plot_points(num_paths_difference[i], congestion_difference_avg[i],
                        "Total Paths\n(% difference)", "Avg. Congestion\n(% difference)",
                        path.join(plot_dir, '{}_avg_congestion_difference_vs_paths_difference'.format(i)))
            # plot_points(num_paths[i], congestion_difference_avg[i],
            #             "Total Paths", "Avg. Congestion\n(% difference)",
            #             path.join(plot_dir, '{}_avg_congestion_difference_vs_paths'.format(i)))

            # plot scatter plot of congestion_max vs path difference,
            # plot_points(num_paths[i], congestion_max[i], "Total Paths", "Max. Congestion",
            #             path.join(plot_dir, '{}_max_congestion_vs_paths'.format(i)))
            plot_points(num_paths_difference[i], congestion_difference_max[i],
                        "Total Paths\n(% difference)", "Max. Congestion \n(% difference)",
                        path.join(plot_dir, '{}_max_congestion_difference_vs_paths_difference'.format(i)))
            # plot_points(num_paths[i], congestion_difference_max[i],
            #             "Total Paths", "Max. Congestion \n(% difference)",
            #             path.join(plot_dir, '{}_max_congestion_difference_vs_paths'.format(i)))

            # plot scatter plot of 90th pctl congestion vs path difference,
            # plot_points(num_paths[i], congestion_90pct[i],
            #             "Total Paths", "90th Pctl. Congestion",
            #             path.join(plot_dir, '{}_90pctl_congestion_vs_paths'.format(i)))
            plot_points(num_paths_difference[i], congestion_difference_90pct[i],
                        "Total Paths\n(% difference)", "90th Pctl. Congestion \n(% difference)",
                        path.join(plot_dir, '{}_90pctl_congestion_difference_vs_paths_difference'.format(i)))
            # plot_points(num_paths[i], congestion_difference_90pct[i],
            #             "Total Paths", "90th Pctl. Congestion \n(% difference)",
            #             path.join(plot_dir, '{}_90pctl_congestion_difference_vs_paths'.format(i)))

            # plot scatter plot of congestion_avg vs congestion max,
            # plot_points(congestion_avg[i], congestion_max[i],
            #             "Avg. Congestion", "Max. Congestion",
            #             path.join(plot_dir, '{}_avg_vs_max_congestion'.format(i)))
            # plot_points(congestion_difference_avg[i], congestion_difference_max[i],
            #             "Avg. Congestion\n(% difference)", "Max. Congestion\n(% difference)",
            #             path.join(plot_dir, '{}_avg_vs_max_congestion_difference'.format(i)))

            # # congestion 90th pctl vs congestion max,
            # plot_points(congestion_90pct[i], congestion_max[i],
            #             "90th Pctl. Congestion", "Max. Congestion",
            #             path.join(plot_dir, '{}_90pct_vs_max_congestion'.format(i)))
            # plot_points(congestion_difference_90pct[i], congestion_difference_max[i],
            #             "90th Pctl. Congestion\n(% difference)", "Max. Congestion\n(% difference)",
            #             path.join(plot_dir, '{}_90pct_vs_max_congestion_difference'.format(i)))

            # # congestion avg vs 90th pctl and also difference for each congestion metric
            # plot_points(congestion_90pct[i], congestion_avg[i],
            #             "90th Pctl. Congestion", "Avg. Congestion",
            #             path.join(plot_dir, '{}_90pct_vs_avg_congestion'.format(i)))
            # plot_points(congestion_difference_90pct[i], congestion_difference_avg[i],
            #             "90th Pctl. Congestion\n(% difference)", "Avg. Congestion\n(% difference)",
            #             path.join(plot_dir, '{}_90pct_vs_avg_congestion_difference'.format(i)))

        # Finished collecting percent diff information for circuit added.
        # plot data to Path_dir/plot_dir

        congestion_multi_cdf(congestion_difference_avg, "$\Delta$ Avg. Congestion",
                             net, path.join(plot_dir, '{}_congestion_difference_average'.format(net)))
        congestion_multi_cdf(congestion_avg, "Avg. Congestion",
                             net, path.join(plot_dir, 'congestion_average'))
        congestion_multi_cdf(congestion_difference_90pct, "$\Delta$ 90th Pctl. Congestion",
                             net,  path.join(plot_dir, '{}_congestion_difference_90pct'.format(net)))
        congestion_multi_cdf(congestion_90pct, "90th Pctl. Congestion", 
                             net,  path.join(plot_dir, 'congestion_90pct'))
        congestion_multi_cdf(congestion_difference_max, "$\Delta$ Max. Congestion", 
                             net, path.join(plot_dir, '{}_congestion_difference_max'.format(net)))
        congestion_multi_cdf(congestion_max, "Max. Congestion",
                             net, path.join(plot_dir, 'congestion_max'))
        
        if len(congestion_max) == 2 and len(congestion_90pct) == 2 and len(congestion_avg) == 2:
            new_d = {"Max" : congestion_max[10], 
                     "90th Percentile": congestion_90pct[10], 
                     "Mean": congestion_avg[10]
                    }
            org_d = {"Max" : congestion_max[0], 
                     "90th Percentile": congestion_90pct[0], 
                     "Mean": congestion_avg[0]
                    }
            congestion_multi_cdf_v2(new_d, 
                                    org_d, 
                                    ["Mean", "90th Percentile", "Max"], "Max_90_mean",
                                    net, 
                                    path.join(plot_dir, "{}_Max_90_mean".format(net)))

def post_proc_timeseries(experiments, topology, iterations, series_labels=[], volume="", proportion=""):
    iter_folder = ["{}_{}-{}".format(topology, i, iterations)
                   for i in range(1, iterations+1)]
    measurement = ["CongestionLossVsIterations", 
                    # "k10CongestionVsIterations", 
                    # "k10ExpCongestionVsIterations",
                    # "k20CongestionVsIterations", 
                    # "k20ExpCongestionVsIterations", 
                    # "k30CongestionVsIterations",
                    # "k30ExpCongestionVsIterations", 
                    # "k40CongestionVsIterations", 
                    # "k40ExpCongestionVsIterations",
                    # "k50CongestionVsIterations", 
                    "k50ExpCongestionVsIterations", 
                    # "k60CongestionVsIterations",
                    # "k60ExpCongestionVsIterations", 
                    # "k70CongestionVsIterations", 
                    # "k70ExpCongestionVsIterations",
                    # "k80CongestionVsIterations", 
                    # "k80ExpCongestionVsIterations", 
                    # "k90CongestionVsIterations",
                    # "k90ExpCongestionVsIterations", 
                    # "k95CongestionVsIterations", 
                    # "k95ExpCongestionVsIterations",
                    "MaxCongestionVsIterations", 
                    "MaxExpCongestionVsIterations", 
                    # "MeanCongestionVsIterations",
                    # "MeanExpCongestionVsIterations", 
                    "NumPathsVsIterations", 
                    "TimeVsIterations",
                    "TotalSinkThroughputVsIterations"]
    # attack = [i*10 for i in range(1, iterations+1)]
    attack = [i*5 for i in range(iterations)]
    for m in measurement:
        exp_data = [[] for _ in range(len(experiments))]

        for itf in iter_folder:
            print(itf)
            experiment_file = [path.join(exp, itf, m+".dat")for exp in experiments]
            for i in range(len(experiments)):
                if path.exists(experiment_file[i]):
                    exp_data[i].append(read_result_val(experiment_file[i])) 
                # post_l = len(experiment1_data)
                # assert post_l - prev_l == 1
        num_experiments = len(exp_data)
        # plot both measurements as time series and
        # split camel case. source adapted from: https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
        ylabel = " ".join(
            re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', m)).split()[:-2])

        plots_dir = path.join(SCRIPT_HOME, "data", "results",
                              "{}_compare_{}_{}_{}".format(topology, iterations, volume, proportion))
        print("Plots Dir: ", plots_dir)
        makedirs(plots_dir, exist_ok=True)
        name = path.join(plots_dir, m)
        # save result as SCRIPT_HOME/results/<topology>_compare_<iterations>/measurement{.png & .pdf & csv}
        # plot_timeseries(attack, exp_data,
        #                 num_lines=len(exp_data), xlabel="Attack Size (Gbps)", ylabel=ylabel, name=name, series_labels=series_labels)
        
        # COMPARISON DATA FOR EXPERIMENT
        plot_timeseries(attack, exp_data,
                        num_lines=num_experiments, xlabel="Time (minutes)", ylabel=ylabel, name=name, series_labels=series_labels, save_legend=(m=="CongestionLossVsIterations"))

        # RECONFIG TIME PLOT
        plot_reconfig_time(attack, exp_data,
                        num_lines=num_experiments, xlabel="Time (minutes)", ylabel=ylabel, name=name, series_labels=series_labels, save_legend=(m=="CongestionLossVsIterations"))                        

        # plot_timeseries(attack, [experiment1_data, experiment2_data],
        #                 num_lines=2, xlabel="Attack Size (Gbps)", ylabel=ylabel, name=name, series_labels=series_labels)

        # and  experiment1/<topology>/measurement{.png & .pdf & csv}
        
        for data, exp in zip(exp_data, experiments):
            plot_timeseries(attack[:len(data)], data, 
                    num_lines=1, xlabel="Time (minutes)", ylabel=ylabel, name=path.join(exp, m))

        # plot_timeseries(attack, experiment1_data, num_lines=1,
        #                 xlabel="Attack Size (Gbps)", ylabel=ylabel, name=path.join(experiment1, m))
        # plot_timeseries(attack, experiment2_data, num_lines=1,
        #                 xlabel="Attack Size (Gbps)", ylabel=ylabel, name=path.join(experiment2, m))


def read_result_val(result_file: str) -> float:
    with open(result_file, 'r') as fob:
        fob.readline()  # skip first line.
        line_two = fob.readline()
        result_val = line_two.split()[2]
        if result_val == '-nan':
            result_val = 0 
        try:
            return float(result_val)
        except:
            print(f'File: {result_file} contained illegal value, {result_val}')
            return result_val


def calc_percent_diff(old: float, new: float) -> float:
    return ((new - old) / old)


def get_candidate_links(network, circuits,method, attack_proportion="") -> list:
    """Gives a list of candidate links to add to the base network. The links returned are verified to not increase the max link congestion in the network when the given number of circuits are given to the link. 

    NOTE: The circuits returned are indexed from 0 - they DO NOT map to congestion and other metrics collected
          by Yates. To map them to hosts in the Yates view, add 1.

    Args:
        network (str): name of the network.
        circuits (int): number of circuits to provide on the candidate links
        method (str): "all" or "heuristic" determines the set of data from which to choose the candidates
    Returns:
        list: [(percent_diff, node, node), ...] 3-tuples showing max congestion percent change for each candidate link (node, node). 
    """
    if circuits <= 0:
        return []

    candidate_links = []  # list of 3-tiples, (percent_diff, node, node)
    data_root = path.join(SCRIPT_HOME, "data", "results")
    if attack_proportion: 
        iteration_tag = network + "_add_circuit_" + method + "_" + str(circuits) + "_" + attack_proportion
    else:
        iteration_tag = network + "_add_circuit_" + method + "_" + str(circuits)
    circuit_regex = re.compile('^[0-9]+_[0-9]+$')
    circuit_tags = [tag for tag in listdir(
        path.join(data_root, iteration_tag)) if circuit_regex.match(tag)]
    base_tag = "__0/"
    metric_tag = "MaxExpCongestionVsIterations.dat"
    base_metric_path = path.join(
        data_root, iteration_tag, base_tag, metric_tag)
    base_metric = read_result_val(base_metric_path)
    for circuit_tag in circuit_tags:
        metric_path = path.join(data_root, iteration_tag,
                                circuit_tag, metric_tag)
        circuit_metric = read_result_val(metric_path)
        if circuit_metric < base_metric:
            node_a, node_b = circuit_tag.split("_")
            # if ZERO_INDEXED: # zero indexed nodes
            #     node_a = str(int(node_a) - 1)
            #     node_b = str(int(node_b) - 1)
            # one indexed nodes
            node_a = node_a
            node_b = node_b

            percent_diff = calc_percent_diff(base_metric, circuit_metric)
            candidate_links.append((percent_diff, node_a, node_b))
    candidate_links.sort()
    return candidate_links

def read_link_congestion_to_dict(f:str):
    from collections import defaultdict
    '''
    f: file descriptor (str). Absolute path.
    
    returns defaultdict of congestion on each edge.
    '''
    d = defaultdict(float)
    with open(f, 'r') as fob:
        # skip first two lines (header text).
        fob.readline()
        fob.readline()
        for line in fob.readlines():
            # only count congestion on core links (between sN and sM).
            if len(re.findall('s', line)) < 2:
                continue
            
            # expects line to look like r"\t\t(sX,sY) : Z\n" were X, Y are int and Z is float. 
            s1, s2 = line.strip().replace(" ", "").split(":")
            s2 = float(s2)
            d[s1] = s2
            
    return d 

if __name__ == "__main__":
    if 0:
        post_process()

    # cls = get_candidate_links("ANS", 2)
    # pass
    if 0:
        networks = ["ANS", "CRL", "sprint", "bellCanada", "surfNet"]
        for network in networks:
            # post_proc_timeseries("/home/matt/network_stability_sim/data/results/{}_attack".format(network),
            #                      "/home/matt/network_stability_sim/data/results/{}_attack-no-topology-adaptation".format(network), network, 20)
            post_proc_timeseries(["/home/matt/network_stability_sim/data/results/{}_attack-no-topology-adaptation".format(network),
                                  "/home/matt/network_stability_sim/data/results/{}_attack".format(network),
                                  "/home/matt/network_stability_sim/data/results/{}_attack_15".format(network)
                                 ], 
                                 network, 20, 
                                 series_labels=["No Defense", "Alp-Wolf: 10", "Alp-Wolf: 15"])
            # time_series_files = ["/home/matt/network_stability_sim/data/results/{}_attack_heuristic_0_{}".format(net, proportion), 
            #                 "/home/matt/network_stability_sim/data/results/{}_attack_heuristic_7_10_{}".format(net, proportion)]


    if 1: 
        time_series_files = [
                            "/home/matt/network_stability_sim/data/results-2022-02-11/sprint_optimal_3_link_attack_10_0Gbps-150Gbps",
                            
                            "/home/matt/network_stability_sim/data/results/sprint_optimal_3_link_attack_10_0Gbps-150Gbps_-mcf"
                            #  "/home/matt/network_stability_sim/data/results/sprint_attack_heuristic_7_10_0Gbps-150Gbps"
                            ]
        net = "sprint"
        iterations = 3
        post_proc_timeseries(time_series_files, net, iterations, series_labels=["ECMP", "MCF"])
