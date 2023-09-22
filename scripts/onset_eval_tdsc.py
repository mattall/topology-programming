from typing import DefaultDict
from pprint import pprint
import sys
import os
import csv
# sys.path.append(os.path.abspath('.'))
# sys.path.append(os.path.abspath('./src'))
# sys.path.append(os.path.abspath('./src/utilities'))

sys.path.insert(0, "./src/")
sys.path.insert(0, "./scripts/")

from onset.simulator import Simulation
from onset.utilities.post_process import post_proc_timeseries
from time import time
from sys import argv


def main(argv):
    if len(argv) == 1:
        network               = "surfNet"
        hosts                 = "50"
        te_method             = "-mcf"
        traffic_type          = "coremelt"
        iter_i                = '89'
        experiment_tag        = "onset_coremelt"
        network_instance      = "surfNet"
        repeat                = "repeat" 
        traffic_file          = "data/traffic/surfNet_coremelt_every_link_2.00e+11.txt"
        n_fallow_transponders = '1'
        optical_strategy      = "onset_v2"
        fallow_tx_allocation  = "file"
        ftx_file              = "data/txp/surfNet_0.txt"

        # network                 = 'Comcast'
        # hosts                   = 149
        # te_method               = "-semimcfraeke"
        # traffic_type            = "FlashCrowd"
        # iter_i                  = 1
        # experiment_tag          = "NoFailures"
        # network_instance        = network
        # repeat                  = "False"
        # traffic_file            = "/home/mhall/OLTE/data/traffic/{}_x100_10000_pareto-matrix.txt".format(network)
        # n_fallow_transponders   = "20"

    else:
        _                      ,\
        network                ,\
        hosts                  ,\
        te_method              ,\
        traffic_type           ,\
        iter_i                 ,\
        experiment_tag         ,\
        network_instance       ,\
        repeat                 ,\
        traffic_file           ,\
        n_fallow_transponders  ,\
        optical_strategy       ,\
        fallow_tx_allocation   ,\
        ftx_file               = argv

    if (os.path.exists(traffic_file)):
        pass
    else: 
        raise FileNotFoundError(traffic_file)
    
    iter_i = int(iter_i)
    hosts = int(hosts)
    iterations = int(os.popen('wc -l ' + traffic_file).read().split()[0])
    data = DefaultDict(list)
    
    t0_baseline_init = time()
    attack_sim = Simulation(network_instance, 
                            hosts,
                            "_".join([traffic_type,experiment_tag]), 
                            iterations=iterations, 
                            te_method=te_method,
                            topology_programming_method=optical_strategy,
                            traffic_file=traffic_file,                            
                            fallow_transponders=n_fallow_transponders,
                            fallow_tx_allocation_strategy=fallow_tx_allocation,
                            fallow_tx_allocation_file=ftx_file,
                            salt=str(iter_i)
                            )
    if repeat == "repeat":
        result = attack_sim.perform_sim(circuits=1, start_iter=iter_i, end_iter=iter_i, repeat = True)
    else:
        result = attack_sim.perform_sim(circuits=1, start_iter=iter_i, end_iter=iter_i)

    for key in result:
        data[key].extend(result[key])

    pprint(data)
    # results_file = SCRIPT_HOME + "/data/results/{}_coremelt_every_link_{}_{}_{}".format(
    #     network, attack, net_iter.split('/')[1], iter_i) + ".csv"
    # print("writing results to: " + results_file)
    # with open(results_file, "w") as outfile:    
    #     writer = csv.writer(outfile)
    #     writer.writerow(data.keys())
    #     writer.writerows(zip(*data.values()))

if __name__ == "__main__":
    main(argv)
