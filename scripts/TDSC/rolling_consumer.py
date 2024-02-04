
from typing import DefaultDict
import sys
import csv
sys.path.insert(0,"src")
sys.path.insert(0,"src/utilities")
from onset.simulator import Simulation
from sys import argv

def main(argv):
    if len(argv) == 1: 
        network, te_method, tp_method = 'CRL', '-ecmp', 'onset_v3'
    else: 
        try: 
            network, te_method, tp_method = argv[1:]
        except Exception as E: 
            print(E)            
            print(f"\targs: {argv[1:]}")
    
    hosts = {
        "sprint" : 11, 
        "ANS": 18, 
        "CRL": 33, 
        "bellCanada" : 48, 
        "surfNet" : 50}

    exp_start_iter = {
        # ('ANS', '-ecmp', 'baseline') : 720, 
        # ('ANS', '-ecmp', 'onset_v3') : 720,
        # ('ANS', '-mcf', 'baseline') : 720, 
        # ('ANS', '-mcf', 'onset_v3') : 720,
        # ('CRL', '-ecmp', 'baseline') : 586, 
        # ('CRL', '-ecmp', 'onset_v3') : 719,
        # ('CRL', '-mcf', 'baseline') : 49, 
        # ('CRL', '-mcf', 'onset_v3') : 49,
        # ('sprint', '-ecmp', 'baseline') : 0, 
        # ('sprint', '-ecmp', 'onset_v3') : 542,
        # ('sprint', '-mcf', 'baseline') : 0, 
        # ('sprint', '-mcf', 'onset_v3') : 0,
        # ('bellCanada', '-ecmp', 'baseline') : 0, 
        # ('bellCanada', '-ecmp', 'onset_v3') : 0,
        # ('bellCanada', '-mcf', 'baseline') : 0, 
        # ('bellCanada', '-mcf', 'onset_v3') : 0,
        # ('surfNet', '-ecmp', 'baseline') : 0, 
        ('surfNet', '-ecmp', 'onset_v3') :  543
        # ('surfNet', '-mcf', 'baseline') : 0, 
        # ('surfNet', '-mcf', 'onset_v3') : 543,
    }
    traffic_file    = f"data/archive/traffic/traffic-05-16-2022/rolling_attack/{network}-rolling-mixed-type-attack.txt"    
    attack_sim = Simulation(network, 
                            hosts[network],
                            f"{network}_{te_method}_{tp_method}_rolling_mixed_type_attack.csv",                             
                            te_method=te_method,
                            traffic_file=traffic_file,                        
                            topology_programming_method=tp_method,                            
                            fallow_transponders=2, 
                            congestion_threshold_upper_bound=0.8,
                            congestion_threshold_lower_bound=0.2,
                            iterations=720
                            )

    result = attack_sim.perform_sim(start_iter=exp_start_iter[network, te_method, tp_method])
    data = DefaultDict(list)
    for key in result:
        data[key].extend(result[key])

    result_file = f"data/results/{network}_{te_method}_{tp_method}_rolling_mixed_type_attack.csv"
    print(f"writing results to: {result_file}")
    with open(result_file, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))

if __name__ == "__main__":
    main(argv)

