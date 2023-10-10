from typing import DefaultDict

from numpy import product
from src.utilities.post_process import post_proc_timeseries
from src.utilities.tmg import sanitize_magnitude, make_human_readable
from net_sim import Attack_Sim
from time import time
import os
import csv
import json
from multiprocessing import Pool
from itertools import product

# ALL NETWORKS
# networks = ['ANS']
# networks = ['sprint']
# networks = ['sprint', 'ANS', 'CRL']


# networks = ['CRL', 'bellCanada', 'surfNet']
# EXPERIMENT_DIR = "./cvi_config/"

# # networks = ['dumbbell_3_2']
# EXPERIMENT_DIR = "/home/mhall/network_stability_sim/sim_config/cvi_config/"
# experiments = os.listdir(EXPERIMENT_DIR)
# te_method = "-semimcfecmp"
# with open("data/results/time.csv", "w") as fob:
#     fob.write("Experiment,Network,GML File,Aggregate Volume,Benign to Malicious Ratio,Attack Accuracy,Precompute Initialization Time,Precompute Time,Baseline Initialization Time,Baseline Time,ONSET Initialization Time,ONSET Time\n")
#         #exp, network_name, gml_file, aggregate_volume, benign_to_malicious_ratio, attack_accuracy])

data = {}
te_methods = ["-ecmp", "-mcf"]
# te_methods = ["-ecmp"]
strategies = ["optimal", "baseline"]
# strategies = ["optimal"]
net = "sprint"
# traffic = "/home/mhall/network_stability_sim/data/traffic/rolling_attack/sprint_rolling_attack_3_round.txt"
traffic = "./data/traffic/rolling_attack/sprint-rolling-mixed-type-attack.txt"
for te_method, strategy in product(te_methods, strategies):
    t0_onset_init = time()
    attack_sim = Attack_Sim(net, 
                            "rolling_attack_{}".format(strategy),
                            iterations=720, 
                            te_method=te_method,
                            method="optimal",
                            traffic_file=traffic,
                            strategy=strategy, 
                            use_heuristic="",
                            fallow_transponders=100, 
                            congestion_threshold_upper_bound=0.8,
                            congestion_threshold_lower_bound=0.1,
                            )
    result = attack_sim.perform_sim(circuits=10)           
    for key in result:
        if key in data:
            data[key].extend(result[key])
        else:
            data[key] = result[key]                        


with open("./data/results/{}_rolling_mixed_type_attack.csv".format(net), 'w') as fob: 
    writer = csv.writer(fob)
    writer.writerow(data.keys())
    writer.writerows(zip(*data.values()))
            
