import matplotlib.pyplot as plt
import networkx as nx
import math
import os
import sys

from numpy import Infinity

from onset.utilities.plotters import plot_timeseries
from onset.utilities.post_process import read_result_val

plotted = False


def calc_haversine(lat1, lon1, lat2, lon2):
    # source: https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = pow(math.sin(dLat / 2), 2) + pow(math.sin(dLon / 2), 2) * math.cos(
        lat1
    ) * math.cos(lat2)
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


def calc_link_config_time(distance: float) -> float:
    # source OptSys 2021 Nance Hall et al.
    amps_on_path = math.ceil(distance / 80)  # one amp per 80 km
    reconfig_time = amps_on_path / 10  # 1 second per ten amps
    return reconfig_time


def main(network, iterations, agg_vol, benign_vol, atk_vol):
    joint_path = "/home/matt/network_stability_sim/data/results/{}_compare_{}_{}_{}_{}-{}".format(
        network, iterations, agg_vol, agg_vol, benign_vol, atk_vol
    )
    joint_id = joint_path.split("/")[-1]
    print("\nExperiment Joint ID: {}".format(joint_id))

    experiment_paths = [
        "/home/matt/network_stability_sim/data/results/{}_attack_heuristic_0_{}_{}-{}".format(
            network, agg_vol, benign_vol, atk_vol
        ),
        "/home/matt/network_stability_sim/data/results-2022-01-06/{}_attack_heuristic_7_10_{}_{}-{}".format(
            network, agg_vol, benign_vol, atk_vol
        ),
        "/home/matt/network_stability_sim/data/results/{}_attack_heuristic_7_10_{}_{}-{}".format(
            network, agg_vol, benign_vol, atk_vol
        ),
    ]
    labels = ["Baseline", "ONSET - Heuristic", "ONSET - Brute Force"]
    delay_file = "./data/link_delay.csv"
    if os.path.exists(delay_file):
        delay_fob = open(delay_file, "a")
    else:
        delay_fob = open(delay_file, "w")
        delay_fob.write("Network,Distance(km),Method,Time\n")

    ##########################################
    # GET LINK RECONFIG TIME FOR NEW CIRCUIT #
    ##########################################
    for experiment_path, method in zip(experiment_paths, labels):
        print("\tExperiment path: {}".format(experiment_path))

        experiment_id = experiment_path.split("/")[-1]
        print("\tExperiment ID: {}".format(experiment_id))

        experiment_files = os.listdir(experiment_path)
        TE_interval = 1 * 30  # 1 minute(s)
        epsilon = 0.01
        gml_file = [
            f
            for f in os.listdir(experiment_path)
            if f.endswith(".gml") and "circuit" in f
        ]
        if gml_file:
            gml_file = gml_file[0]  # there should only be one!
            # expects something like: '/ANS_3-3circuit-3-13.gml'
            new_link = gml_file.split("-")[-2:]  # becomes ['3', '13.gml']
            new_link[0] = "s" + new_link[0]  # becomes ['s3', '13.gml']
            new_link[1] = "s" + new_link[1][:-4]  # becomes ['s3', 's13']

            gml_file = os.path.join(experiment_path, gml_file)

            G = nx.read_gml(gml_file)

            lat1 = G.nodes[new_link[0]]["Latitude"]
            lon1 = G.nodes[new_link[0]]["Longitude"]
            lat2 = G.nodes[new_link[1]]["Latitude"]
            lon2 = G.nodes[new_link[1]]["Longitude"]

            distance = calc_haversine(lat1, lon1, lat2, lon2)
            reconfig_time = calc_link_config_time(distance)
            delay_fob.write(
                "{},{},{},{}\n".format(
                    network, distance, method, reconfig_time
                )
            )
        else:
            reconfig_time = 0
    delay_fob.close()
    metrics = [
        "CongestionLossVsIterations",
        "k10CongestionVsIterations",
        "k10ExpCongestionVsIterations",
        "k20CongestionVsIterations",
        "k20ExpCongestionVsIterations",
        "k30CongestionVsIterations",
        "k30ExpCongestionVsIterations",
        "k40CongestionVsIterations",
        "k40ExpCongestionVsIterations",
        "k50CongestionVsIterations",
        "k50ExpCongestionVsIterations",
        "k60CongestionVsIterations",
        "k60ExpCongestionVsIterations",
        "k70CongestionVsIterations",
        "k70ExpCongestionVsIterations",
        "k80CongestionVsIterations",
        "k80ExpCongestionVsIterations",
        "k90CongestionVsIterations",
        "k90ExpCongestionVsIterations",
        "k95CongestionVsIterations",
        "k95ExpCongestionVsIterations",
        "MaxCongestionVsIterations",
        "MaxExpCongestionVsIterations",
        "MeanCongestionVsIterations",
        "MeanExpCongestionVsIterations",
        "NumPathsVsIterations",
        "TimeVsIterations",
        "TotalSinkThroughputVsIterations",
    ]
    for metric in metrics:
        X = []
        Y = []

        print_metric = metric.split("VsIterations")[0]
        for experiment_path, method in zip(experiment_paths, labels):
            print("\tExperiment path: {}".format(experiment_path))

            experiment_id = experiment_path.split("/")[-1]
            print("\tExperiment ID: {}".format(experiment_id))

            # if "surfNet" not in experiment_id:
            #     plot_data = False
            #     continue

            experiment_files = os.listdir(experiment_path)
            TE_interval = 1 * 30  # 1 minute(s)

            # ##########################################
            # # GET LINK RECONFIG TIME FOR NEW CIRCUIT #
            # ##########################################

            # gml_file = [f for f in os.listdir(experiment_path) if f.endswith('.gml') and 'circuit' in f]
            # if gml_file:
            #     gml_file = gml_file[0] # there should only be one!
            #     # expects something like: '/ANS_3-3circuit-3-13.gml'
            #     new_link = gml_file.split('-')[-2:]     # becomes ['3', '13.gml']
            #     new_link[0] = 's' + new_link[0]         # becomes ['s3', '13.gml']
            #     new_link[1] = 's' + new_link[1][:-4]    # becomes ['s3', 's13']

            #     gml_file = os.path.join(experiment_path, gml_file)

            #     G = nx.read_gml(gml_file)

            #     lat1 = G.nodes[new_link[0]]['Latitude']
            #     lon1 = G.nodes[new_link[0]]['Longitude']
            #     lat2 = G.nodes[new_link[1]]['Latitude']
            #     lon2 = G.nodes[new_link[1]]['Longitude']

            #     distance = calc_haversine(lat1, lon1, lat2, lon2)
            #     reconfig_time = calc_link_config_time(distance)
            #     delay_fob.write("{},{},{}\n".format(network, distance, method, reconfig_time))
            # else:
            #     reconfig_time = 0

            ############################################
            # GET DISCRETE POINT VALUES FOR CONGESTION #
            ############################################

            exp_subfolders = []
            for exp_file in experiment_files:
                if os.path.isdir(os.path.join(experiment_path, exp_file)):
                    exp_subfolders.append(exp_file)

            exp_subfolders.sort()

            result_values = []
            time_values = []
            for i, exp_subfolder in enumerate(exp_subfolders):
                # value_path = os.path.join(experiment_path, exp_subfolder, 'CongestionLossVsIterations.dat')
                value_path = os.path.join(
                    experiment_path, exp_subfolder, metric + ".dat"
                )
                result_value = 100 * read_result_val(value_path)

                print("\tValue found in: {}".format(value_path))
                print("\tValue: {}".format(result_value))

                if i == 0:
                    result_values.append(result_value)
                    time_values.append(0)

                if i == 1:
                    result_values.append(result_values[-1])
                    time_values.append(TE_interval - epsilon)

                    result_values.append(result_value)
                    time_values.append(TE_interval)

                if i == 2:
                    result_values.append(result_values[-1])
                    time_values.append(TE_interval + reconfig_time - epsilon)

                    result_values.append(result_value)
                    time_values.append(TE_interval + reconfig_time)

                    result_values.append(result_value)
                    time_values.append(2 * TE_interval)

            X.append(time_values)
            Y.append(result_values)
            plot_file = os.path.join(
                experiment_path,
                experiment_id + "Link_reconfig_time_" + print_metric,
            )

            plot_timeseries(
                time_values,
                result_values,
                1,
                xlabel="Time (s)",
                ylabel=print_metric,
                name=plot_file,
            )
            print("Saving file to: {}.png".format(plot_file))
            plot_data = True

        if plot_data:
            os.makedirs(
                os.path.join(
                    ".", "data", "Link_reconfig_time_" + print_metric
                ),
                exist_ok=True,
            )
            plot_file = os.path.join(
                ".",
                "data",
                "Link_reconfig_time_" + print_metric,
                joint_id + "_" + print_metric,
            )
            # plot_file = os.path.join(joint_path, joint_id + "Link_reconfig_time_CongestionLoss")

            max_y = -Infinity
            min_y = Infinity
            for series in Y:
                for entry in series:
                    if entry > max_y:
                        max_y = entry
                    if entry < min_y:
                        min_y = entry

            # Generic ylim
            if max_y == min_y and max_y > 0:
                ylim = (0, max_y * 2)
            elif max_y == min_y and max_y < 0:
                ylim = (min_y * 2, 0)
            elif max_y == min_y and max_y > 0:
                ylim = (-10, 10)

            if "ExpCongestion" in metric and max_y > 100:
                ylim = (0, max_y)
            elif "Congestion" in metric:
                ylim = (0, 100)
            else:
                ylim = ylim

            global plotted
            plot_timeseries(
                X,
                Y,
                3,
                xlabel="Time (s)",
                ylabel=print_metric,
                name=plot_file,
                series_labels=labels,
                pass_X_direct=True,
                save_legend=(not plotted),
                log_scale=("Congestion" in metric),
                ylim=ylim,
            )
            plotted = True
            exit()
            print("Saving file to: {}.png".format(plot_file))


def get_reconfig_time(gml_file, circuits):
    reconfig_time = 0
    if gml_file:
        # circuits = gml_file.split("circuit-")[-1][:-3].split(".")
        for circuit in circuits:
            new_link = circuit[:]
            if not circuit:
                continue  # handle empty string case

            # expects something like: '3-13'
            # new_link = circuit.split('-')          # becomes ['3', '13']
            new_link[0] = "s" + new_link[0]  # becomes ['s3', '13']
            new_link[1] = "s" + new_link[1]  # becomes ['s3', 's13']

            G = nx.read_gml(gml_file)

            lat1 = G.nodes[new_link[0]]["Latitude"]
            lon1 = G.nodes[new_link[0]]["Longitude"]
            lat2 = G.nodes[new_link[1]]["Latitude"]
            lon2 = G.nodes[new_link[1]]["Longitude"]

            distance = calc_haversine(lat1, lon1, lat2, lon2)
            t = calc_link_config_time(distance)
            reconfig_time = max(t, reconfig_time)

    return reconfig_time


def multi_attack_timeseries(series1, series2, label1, label2, experiment_name):
    TE_interval = 1 * 30  # 1 minute(s)
    epsilon = 0.01

    delay_file = "./data/{}.csv".format(experiment_name)
    if os.path.exists(delay_file):
        delay_fob = open(delay_file, "a")
    else:
        delay_fob = open(delay_file, "w")
        delay_fob.write("Network,Time\n")

    for exp_path1, exp_path2 in zip(series1, series2):
        ###########################################
        # GET LINK RECONFIG TIME FOR NEW CIRCUITS #
        ###########################################

        print("\tExperiment 1 path: {}".format(exp_path1))
        print("\tExperiment 2 path: {}".format(exp_path2))

        experiment_id1 = exp_path1.split("/")[-1]
        experiment_id2 = exp_path2.split("/")[-1]

        print("\tExperiment 1 ID: {}".format(exp_path1))
        print("\tExperiment 2 ID: {}".format(exp_path2))

        experiment_files1 = os.listdir(exp_path1)
        experiment_files2 = os.listdir(exp_path2)

        gml_file1 = [
            f
            for f in os.listdir(exp_path1)
            if f.endswith(".gml") and "circuit" in f
        ]
        gml_file2 = [
            f
            for f in os.listdir(exp_path2)
            if f.endswith(".gml") and "circuit" in f
        ]

        gml_file1 = (
            os.path.join(exp_path1, gml_file1[0]) if gml_file1 else gml_file1
        )
        gml_file2 = (
            os.path.join(exp_path2, gml_file2[0]) if gml_file2 else gml_file2
        )

        reconfig_time1 = get_reconfig_time(gml_file1)
        reconfig_time2 = get_reconfig_time(gml_file2)

        delay_fob.write("{},{}\n".format(gml_file1, reconfig_time1))
        delay_fob.write("{},{}\n".format(gml_file2, reconfig_time2))

        continue
        delay_fob.close()
        metrics = [
            "CongestionLossVsIterations",
            "k10CongestionVsIterations",
            "k10ExpCongestionVsIterations",
            "k20CongestionVsIterations",
            "k20ExpCongestionVsIterations",
            "k30CongestionVsIterations",
            "k30ExpCongestionVsIterations",
            "k40CongestionVsIterations",
            "k40ExpCongestionVsIterations",
            "k50CongestionVsIterations",
            "k50ExpCongestionVsIterations",
            "k60CongestionVsIterations",
            "k60ExpCongestionVsIterations",
            "k70CongestionVsIterations",
            "k70ExpCongestionVsIterations",
            "k80CongestionVsIterations",
            "k80ExpCongestionVsIterations",
            "k90CongestionVsIterations",
            "k90ExpCongestionVsIterations",
            "k95CongestionVsIterations",
            "k95ExpCongestionVsIterations",
            "MaxCongestionVsIterations",
            "MaxExpCongestionVsIterations",
            "MeanCongestionVsIterations",
            "MeanExpCongestionVsIterations",
            "NumPathsVsIterations",
            "TimeVsIterations",
            "TotalSinkThroughputVsIterations",
        ]
        for metric in metrics:
            X = []
            Y = []

            print_metric = metric.split("VsIterations")[0]
            for experiment_path, method in zip(series1, labels):
                print("\tExperiment path: {}".format(experiment_path))

                experiment_id = experiment_path.split("/")[-1]
                print("\tExperiment ID: {}".format(experiment_id))

                experiment_files = os.listdir(experiment_path)
                TE_interval = 1 * 30  # 1 minute(s)

                ############################################
                # GET DISCRETE POINT VALUES FOR CONGESTION #
                ############################################

                exp_subfolders = []
                for exp_file in experiment_files:
                    if os.path.isdir(os.path.join(experiment_path, exp_file)):
                        exp_subfolders.append(exp_file)

                exp_subfolders.sort()

                result_values = []
                time_values = []
                for i, exp_subfolder in enumerate(exp_subfolders):
                    # value_path = os.path.join(experiment_path, exp_subfolder, 'CongestionLossVsIterations.dat')
                    value_path = os.path.join(
                        experiment_path, exp_subfolder, metric + ".dat"
                    )
                    result_value = 100 * read_result_val(value_path)

                    print("\tValue found in: {}".format(value_path))
                    print("\tValue: {}".format(result_value))

                    if i == 0:
                        result_values.append(result_value)
                        time_values.append(0)

                    if i == 1:
                        result_values.append(result_values[-1])
                        time_values.append(TE_interval - epsilon)

                        result_values.append(result_value)
                        time_values.append(TE_interval)

                    if i == 2:
                        result_values.append(result_values[-1])
                        time_values.append(
                            TE_interval + reconfig_time - epsilon
                        )

                        result_values.append(result_value)
                        time_values.append(TE_interval + reconfig_time)

                        result_values.append(result_value)
                        time_values.append(2 * TE_interval)

                X.append(time_values)
                Y.append(result_values)
                plot_file = os.path.join(
                    experiment_path,
                    experiment_id + "Link_reconfig_time_" + print_metric,
                )

                plot_timeseries(
                    time_values,
                    result_values,
                    1,
                    xlabel="Time (s)",
                    ylabel=print_metric,
                    name=plot_file,
                )
                print("Saving file to: {}.png".format(plot_file))
                plot_data = True

        if plot_data:
            os.makedirs(
                os.path.join(
                    ".", "data", "Link_reconfig_time_" + print_metric
                ),
                exist_ok=True,
            )
            plot_file = os.path.join(
                ".",
                "data",
                "Link_reconfig_time_" + print_metric,
                joint_id + "_" + print_metric,
            )
            # plot_file = os.path.join(joint_path, joint_id + "Link_reconfig_time_CongestionLoss")

            max_y = -Infinity
            min_y = Infinity
            for series in Y:
                for entry in series:
                    if entry > max_y:
                        max_y = entry
                    if entry < min_y:
                        min_y = entry

            # Generic ylim
            if max_y == min_y and max_y > 0:
                ylim = (0, max_y * 2)
            elif max_y == min_y and max_y < 0:
                ylim = (min_y * 2, 0)
            elif max_y == min_y and max_y > 0:
                ylim = (-10, 10)

            if "ExpCongestion" in metric and max_y > 100:
                ylim = (0, max_y)
            elif "Congestion" in metric:
                ylim = (0, 100)
            else:
                ylim = ylim

            global plotted
            plot_timeseries(
                X,
                Y,
                3,
                xlabel="Time (s)",
                ylabel=print_metric,
                name=plot_file,
                series_labels=labels,
                pass_X_direct=True,
                save_legend=(not plotted),
                log_scale=("Congestion" in metric),
                ylim=ylim,
            )
            plotted = True
            exit()
            print("Saving file to: {}.png".format(plot_file))


if __name__ == "__main__":
    if 0:
        networks = ["CRL", "sprint", "ANS", "surfNet", "bellCanada"]
        # traffic_vol = [100, 200, 300, 400, 5000]
        traffic_vol = [300]
        # traffic_split = [(80,20), (50,50), (20,80)]
        traffic_split = [(50, 50), (20, 80)]
        # networks = ['sprint', 'bellCanada']
        # traffic_vol = [300]
        # traffic_split = [(20, 80)]
        for network in networks:
            for agg_vol in traffic_vol:
                for ts in traffic_split:
                    benign_vol = str(int(ts[0] * agg_vol / 100)) + "Gbps"
                    atk_vol = str(int(ts[1] * agg_vol / 100)) + "Gbps"
                    str_agg_vol = str(agg_vol) + "Gbps"
                    main(network, 3, str_agg_vol, benign_vol, atk_vol)

    if 1:
        attack_folders_ecmp = [
            "./data/results/sprint_optimal_1_link_attack_100_0Gbps-100Gbps_-ecmp/",
            "./data/results/sprint_optimal_1_link_attack_100_0Gbps-150Gbps_-ecmp/",
            "./data/results/sprint_optimal_1_link_attack_100_0Gbps-200Gbps_-ecmp/",
            "./data/results/sprint_optimal_2_link_attack_100_0Gbps-100Gbps_-ecmp/",
            "./data/results/sprint_optimal_2_link_attack_100_0Gbps-150Gbps_-ecmp/",
            "./data/results/sprint_optimal_2_link_attack_100_0Gbps-200Gbps_-ecmp/",
            "./data/results/sprint_optimal_3_link_attack_100_0Gbps-100Gbps_-ecmp/",
            "./data/results/sprint_optimal_3_link_attack_100_0Gbps-150Gbps_-ecmp/",
            "./data/results/sprint_optimal_3_link_attack_100_0Gbps-200Gbps_-ecmp/",
        ]

        attack_folders_mcf = [
            "./data/results/sprint_optimal_1_link_attack_100_0Gbps-100Gbps_-mcf",
            "./data/results/sprint_optimal_1_link_attack_100_0Gbps-150Gbps_-mcf",
            "./data/results/sprint_optimal_1_link_attack_100_0Gbps-200Gbps_-mcf",
            "./data/results/sprint_optimal_2_link_attack_100_0Gbps-100Gbps_-mcf",
            "./data/results/sprint_optimal_2_link_attack_100_0Gbps-150Gbps_-mcf",
            "./data/results/sprint_optimal_2_link_attack_100_0Gbps-200Gbps_-mcf",
            "./data/results/sprint_optimal_3_link_attack_100_0Gbps-100Gbps_-mcf",
            "./data/results/sprint_optimal_3_link_attack_100_0Gbps-150Gbps_-mcf",
            "./data/results/sprint_optimal_3_link_attack_100_0Gbps-200Gbps_-mcf",
        ]

        multi_attack_timeseries(
            attack_folders_ecmp,
            attack_folders_mcf,
            "ECMP",
            "MCF",
            "Multi-target_Multi-volume_Attack",
        )
