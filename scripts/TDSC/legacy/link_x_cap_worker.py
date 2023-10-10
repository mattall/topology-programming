import os
import csv

def experiment(args):
    net, n_targets, iterations, te_method, traffic_file, proportion = args
    from onset.utilities.post_process import post_proc_timeseries
    from onset.simulator import Simulation
    hosts = {
        "sprint" : 11, 
        "ANS": 18, 
        "CRL": 33, 
        "bellCanada" : 48, 
        "surfNet" : 50
    }


    attack_sim = Simulation(
        net,
        hosts[net],
        f"optimal_{n_targets}_link_attack",
        iterations=iterations,
        te_method=te_method,
        traffic_file=traffic_file,
        fallow_transponders=10,
        congestion_threshold_upper_bound=0.8,
        congestion_threshold_lower_bound=0.2,
        topology_programming_method="onset_v1_1",
        attack_proportion=proportion,
    )

    result = attack_sim.perform_sim()
    result_file = f"data/reports/{net}_multi-attack{te_method}-{os.getpid()}.csv"
    with open(result_file, "w") as outfile:
        # pass the csv file to csv.writer function.
        # writer = csv.DictWriter(outfile, fieldnames=result.keys())
        writer = csv.writer(outfile)

        # writer.writeheader()
        # pass the dictionary keys to writerow
        # function to frame the columns of the csv file
        writer.writerow(result.keys())

        # make use of writerows function to append
        # the remaining values to the corresponding
        # columns using zip function.
        writer.writerows(zip(*result.values()))
        print("wrote result to: " + result_file)
    return 0 
