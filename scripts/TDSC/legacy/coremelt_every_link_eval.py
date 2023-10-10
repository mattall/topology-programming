from typing import DefaultDict
import sys
import os
import csv
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./src/utilities'))
from src.utilities import SCRIPT_HOME
from src.utilities.post_process import post_proc_timeseries
from net_sim import Attack_Sim
from time import time
from sys import argv

def main(argv):
    if len(argv) == 1:
        network         = 'CRL'
        te_methods      = ["-mcf"]
        attack          = '3.00e+11'
    else:
        network = argv[1]
        te_methods = [argv[2]]
        # print("network: {}\nte_method: {}".format(network, te_methods))

    gml_file        = SCRIPT_HOME + "/data/graphs/gml/" + network + ".gml"
    traffic_file    = SCRIPT_HOME + "/data/traffic/" + network + "_coremelt_every_link_" + attack + ".txt"
    iterations = int(os.popen('wc -l ' + traffic_file).read().split()[0])
    data = DefaultDict(list)
    for te_method in te_methods: 
        if 1: ########################## Baseline ########################
            t0_baseline_init = time()
            attack_sim = Attack_Sim(network, 
                                    "baseline_"+ attack, 
                                    iterations=iterations, 
                                    te_method=te_method,
                                    method="none",    
                                    traffic_file=traffic_file,
                                    strategy="baseline", 
                                    use_heuristic='no',
                                    fallow_transponders=0,
                                    )

            result = attack_sim.perform_sim(circuits=0)
            for key in result:
                data[key].extend(result[key])

        if 0: ########################## ONSET ########################
            t0_baseline_init = time()
            attack_sim = Attack_Sim(network, 
                                    "coremelt_attack{}_{}".format("optimal", "1.00e+11"),
                                    iterations=iterations, 
                                    te_method=te_method,
                                    method="optimal",    
                                    traffic_file=traffic_file,
                                    strategy="optimal", 
                                    use_heuristic='',
                                    fallow_transponders=100,
                                    congestion_threshold_upper_bound=0,
                                    congestion_threshold_lower_bound=float("inf"),
                                    )

            result = attack_sim.perform_sim(circuits=10)
            for key in result:
                data[key].extend(result[key])                
        
        results_file = SCRIPT_HOME + "/data/results/{}_coremelt_every_link_".format(network) + attack + ".csv"
        print("writing results to: " + results_file)
        with open(results_file, "w") as outfile:    
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

if __name__ == "__main__":
    main(argv)