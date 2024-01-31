import sys
sys.path.insert(0, "src/")
sys.path.insert(0, "scripts/")
import os
import json
from sys import argv
from multiprocessing import Pool
from glob import glob
from itertools import product
from TDSC.legacy.link_x_cap_worker import experiment

# TEST = False
TEST = True

def main(argv):
    DRY=False
    finished = 0
    unfinished = 0
    # Give a path to a dir where the simulation parameters can be loaded from.
    #   Each file in that dir must be formatted .json as follows
    #   # Note: be mindful of quotation marks `"`
    #   {
    #     "network_name": "<str>,
    #     "gml_file": "<path>",
    #     "links_targeted": <int>,
    #     "benign_volume": "<int>Gbps",
    #     "malicious_volume": "<int>Gbps"
    #    }
    if len(argv) == 1:
        EXPERIMENT_DIR = "scripts/TDSC/args/link_x_vol_barcharts/"
    elif len(argv) == 2:
        EXPERIMENT_DIR = argv[1]
        if EXPERIMENT_DIR == "dry":
            EXPERIMENT_DIR = "scripts/TDSC/args/link_x_vol_barcharts/"
            DRY = True
        print(DRY)
    else:
        print(
            f"""   
        USAGE: python {argv[0].split("/")[-1]} PATH\n\tExpected path to simulation params.
        PATH must contain .json files formatted as below. 
        Note: be mindful of quotation marks (")
        {{
          "network_name": "<str>",
          "gml_file": "<path>",
          "links_targeted": <int>,
          "benign_volume": "<int>Gbps",
          "malicious_volume": "<int>Gbps"
         }}"""
        )
        exit()

    
    if TEST:
        # surfNet_optimal_1_link_attack_conservative_10_0Gbps-200Gbps_-ecmp
        PARALLEL = False
        # NETWORKS = ["ANS"]
        # NETWORKS = ["CRL", "ANS", "sprint", "bellCanada", "surfNet"]
        NETWORKS = ["CRL"]
        # NETWORKS = ["sprint"]
        # NETWORKS = ["surfNet"]
        # VOLUMES = ["100Gbps"]
        VOLUMES = ["200Gbps"]
        NUM_TARGETED = [5]
        # NUM_TARGETED = [5]
        TE_METHODS = ["-mcf"]
        # TE_METHODS = ["-ecmp", "-mcf"]
        # CANDIDATE_SET = ["max"]

        # NETWORKS = ["bellCanada", "surfNet"]
        # VOLUMES = ["100Gbps", "150Gbps", "200Gbps"]
        # NUM_TARGETED = [1, 2, 3, 4, 5]
        # TE_METHODS = ["-ecmp", "-mcf"]
        CANDIDATE_SET = ["max"]

    else:
        PARALLEL = True
        NETWORKS = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
        VOLUMES = ["100Gbps", "150Gbps", "200Gbps"]
        NUM_TARGETED = [1, 2, 3, 4, 5]
        TE_METHODS = ["-ecmp", "-mcf"]
        CANDIDATE_SET = ["conservative"]
        # NETWORKS = ["ANS"]
        # NETWORKS = ["surfNet", "bellCanada"]
        # VOLUMES = ["200Gbps"]
        # NUM_TARGETED = [5]
        # TE_METHODS = ["-mcf"]
        # CANDIDATE_SET = ["conservative", "liberal", "max"]
        pool = Pool()


    data = {}
    exp_args = []
    for te_method, exp, candidate_set in product(TE_METHODS, glob(f"{EXPERIMENT_DIR}*.json"), CANDIDATE_SET):
        config = json.load(open(exp, "r"))
        network = config["network_name"]
        gml_file = config["gml_file"]
        atk_vol = config["malicious_volume"]
        benign_vol = config["benign_volume"]
        n_targets = config["links_targeted"]

        if network not in NETWORKS:
            continue
        if atk_vol not in VOLUMES:
            continue
        if n_targets not in NUM_TARGETED:
            continue
        exp_result = os.path.join(f"data/archive/results/results-10-30-2023/{network}_optimal_{n_targets}_link_attack_{candidate_set}_10_0Gbps-{atk_vol}_{te_method}/{network}_3-3_1_0_0_0_Gbps_1.0/MaxExpCongestionVsIterations.dat")
        if os.path.exists(exp_result) and not TEST:
            # print(exp_result)
            continue
        print(te_method, exp, candidate_set)
        print(exp_result)
        unfinished += 1
        data["Attack"] = f"{n_targets}x{atk_vol.strip('Gbps')}"
        # print(f"{te_method}: {network}")
        # print(f"  gml_file:         {gml_file}")
        # print(f"  malicious_volume: {atk_vol}")
        # print(f"  benign_volume:    {benign_vol}")
        # print(f"  links_targeted:   {n_targets}")

        iterations = 2
        tag = "oneShot"
        traffic_file = f"data/archive/traffic/traffic-05-16-2022/{network}_benign_{benign_vol}_{n_targets}x{atk_vol}_{iterations}_{tag}.txt"
        assert os.path.exists(traffic_file), f"traffic file: {traffic_file} not found"
        assert os.path.exists(gml_file), f"gml file: {gml_file} not found"

        proportion = "{}-{}".format(benign_vol, atk_vol)
        # print("Traffic File: {}".format(traffic_file))
        exp_args.append((network, 
                    n_targets, 
                    iterations, 
                    te_method, 
                    traffic_file, 
                    proportion,
                    candidate_set))
    print(unfinished)
    if DRY: 
        exit()
    if PARALLEL:
        pool = Pool()
        pool.map_async(experiment, exp_args)
        pool.close()
        pool.join()

    else:
        for ea in exp_args:
            experiment(ea)

    # with open("data/results/time.csv", "a") as fob:
    #     fob.write(",".join([exp,
    #                         network_name,
    #                         gml_file,
    #                         aggregate_volume,
    #                         benign_to_malicious_ratio,
    #                         attack_accuracy,
    #                         str(precompute_initialization_time),
    #                         str(precompute_time),
    #                         str(baseline_initialization_time),
    #                         str(baseline_time),
    #                         str(onset_init_time),
    #                         str(onset_time) + "\n"])
    #     )

    # labels = ["Baseline", "ONSET."]

    # time_series_files = ["/home/matt/network_stability_sim/data/results/{}_attack_heuristic_0_{}".format(net, proportion),
    #                 "/home/matt/network_stability_sim/data/results/{}_attack_heuristic_7_10_{}".format(net, proportion)]
    # post_proc_timeseries(time_series_files, net, iterations, labels, volume, proportion)

    # create a csv file  test.csv and store
    # it in a variable as outfile
    


if __name__ == "__main__":
    main(argv)
