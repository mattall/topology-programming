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
# PARALLEL = False
TEST = False
def main(argv):
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
        PARALLEL = False
        NETWORKS = ["sprint"]
        VOLUMES = ["200Gbps"]
        NUM_TARGETED = [5]
        TE_METHODS = ["-mcf"]
    else:
        PARALLEL = True
        NETWORKS = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
        VOLUMES = ["100Gbps", "150Gbps", "200Gbps"]
        NUM_TARGETED = [1, 2, 3, 4, 5]
        TE_METHODS = ["-ecmp", "-mcf"]
        pool = Pool()


    data = {}
    exp_args = []
    for te_method, exp in product(TE_METHODS, glob(f"{EXPERIMENT_DIR}*.json")):
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

        data["Attack"] = f"{n_targets}x{atk_vol.strip('Gbps')}"
        print(f"{te_method}: {network}")
        print(f"  gml_file:         {gml_file}")
        print(f"  malicious_volume: {atk_vol}")
        print(f"  benign_volume:    {benign_vol}")
        print(f"  links_targeted:   {n_targets}")

        iterations = 3
        n_targets
        tag = "oneShot"
        traffic_file = f"data/archive/traffic/traffic-05-16-2022/{network}_benign_{benign_vol}_{n_targets}x{atk_vol}_{iterations}_{tag}.txt"
        assert os.path.exists(traffic_file), f"traffic file: {traffic_file} not found"
        assert os.path.exists(gml_file), f"gml file: {gml_file} not found"

        proportion = "{}-{}".format(benign_vol, atk_vol)
        print("Traffic File: {}".format(traffic_file))
        exp_args.append((network, 
                    n_targets, 
                    iterations, 
                    te_method, 
                    traffic_file, 
                    proportion))
    if PARALLEL:
        pool.map_async(experiment, exp_args)
        pool.close()
        pool.join()

    else:
        experiment(exp_args)


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
