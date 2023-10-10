from typing import DefaultDict
import sys
import os
import csv
sys.path.append(0,"src")
from onset.simulator import Simulation
from time import time
from itertools import product
from sys import argv

def main(argv):
    if len(argv) == 1:
        network         = 'sprint'
        te_methods      = ["-mcf"]
        strategies      = ["baseline", "optimal"]
        attack_string   = "200E9"
    else:
        network = argv[1]
        te_methods = [argv[2]]
        strategies      = [argv[3]]
        attack_string   = argv[4] 
        # print("network: {}\nte_method: {}\nstrategy: {}\nattack: {}".format(network, te_methods, strategies, attack_string))
   
    gml_file        = f"data/graphs/gml/{network}.gml"
    traffic_file    = f"data/traffic/{network}_crossfire_every_node_{attack_string}.txt"
    iterations      = int(os.popen('wc -l ' + traffic_file).read().split()[0])

    data = DefaultDict(list)
    for te_method, strategy in product(te_methods, strategies):
        attack_sim = Simulation(network, 
                                f"crossfire_attack_{strategy}_{attack_string}",
                                iterations=iterations, 
                                te_method=te_method,
                                method="optimal",
                                traffic_file=traffic_file,
                                strategy=strategy, 
                                use_heuristic="",
                                fallow_transponders=100, 
                                congestion_threshold_upper_bound=0,
                                congestion_threshold_lower_bound=float("inf"),
                                )

        result = attack_sim.perform_sim(circuits=10)
        for key in result:
            data[key].extend(result[key])

    result_file = f"data/results/{network}_{te_method}_{strategy}_{attack_string}_crossfire_every_link.csv"

    print(f"writing results to: {result_file}")

    with open(result_file, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

if __name__ == "__main__":
    main(argv)