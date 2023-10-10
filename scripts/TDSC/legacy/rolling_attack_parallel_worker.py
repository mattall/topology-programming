from typing import DefaultDict
import sys 
from sys import argv
import os 
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./src/utilities'))
from src.utilities import SCRIPT_HOME
from src.utilities.post_process import post_proc_timeseries
from src.utilities.tmg import sanitize_magnitude, make_human_readable
from net_sim import Attack_Sim
from time import time

import csv


def main(argv):
    if len(argv) == 1:
        network         = "ANS"
        te_method       = "-ecmp"
        strategy        = "optimal"
    else:
        network         = argv[1]
        te_method       = argv[2]
        strategy        = argv[3]
        print("network: {}\nte_method: {}\nstrategy: {}".format(network, te_method, strategy))
    traffic_file    = SCRIPT_HOME + "/data/traffic/rolling_attack/" + network + "-rolling-mixed-type-attack.txt"

    data = DefaultDict(list)
    attack_sim = Attack_Sim(network, 
                            "rolling_attack_{}".format(strategy),
                            iterations=720, 
                            te_method=te_method,
                            method="optimal",
                            traffic_file=traffic_file,
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

    with open("./data/results/{}_{}_{}_rolling_mixed_type_attack.csv".format(network, te_method, strategy ), 'w') as fob: 
        writer = csv.writer(fob)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
            
if __name__ == "__main__":
    main(argv)