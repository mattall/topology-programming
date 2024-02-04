from typing import DefaultDict
import sys
import csv
sys.path.insert(0,"src")
sys.path.insert(0,"src/utilities")
from onset.simulator import Simulation
from sys import argv

def main(argv):
    if len(argv) == 1:
       network, te_method, attack_string, iter_i, total_iters = "ANS", "-mcf", "100E9", "4", "12"

    else:
        network, te_method, attack_string, iter_i, total_iters = argv[1:]
    
    iter_i = int(iter_i)
    total_iters = int(total_iters)
    hosts = {
        "sprint" : 11, 
        "ANS": 18, 
        "CRL": 33, 
        "bellCanada" : 48, 
        "surfNet" : 50
    }

    traffic_file    = f"data/archive/traffic/traffic-05-16-2022/{network}_crossfire_every_node_{attack_string}.txt"
    attack_sim = Simulation(network, 
                            hosts[network],
                            f"crossfire_attack_{te_method}_{attack_string}",
                            iterations=total_iters, 
                            te_method=te_method,
                            traffic_file=traffic_file,                                                     
                            topology_programming_method="onset_v3",                            
                            fallow_transponders=2, 
                            # congestion_threshold_upper_bound=0,
                            # congestion_threshold_lower_bound=float("inf"),                            
                            )

    result = attack_sim.perform_sim(start_iter=iter_i, end_iter=iter_i, repeat=True)
    data = DefaultDict(list)
    for key in result:
        data[key].extend(result[key])

    result_file = f"data/reports/{network}_{te_method}_{attack_string}_{iter_i}_{total_iters}_crossfire_every_link.csv"
    print(f"writing results to: {result_file}")
    with open(result_file, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

if __name__ == "__main__":
    main(argv)