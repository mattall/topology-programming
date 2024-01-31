import csv
from sys import argv
from sys import path as sys_path
from os import path as os_path
sys_path.insert(0, "src/")
sys_path.insert(0, "scripts/")

def experiment(*args):
    network, n_targets, te_method, volume_per_target, candidate_set = args
    iterations = 2
    traffic_file = f"data/archive/traffic/traffic-05-16-2022/{network}_benign_0Gbps_{n_targets}x{volume_per_target}Gbps_{iterations}_oneShot.txt"
    assert os_path.exists(traffic_file), f"Error traffic file not found: {traffic_file}.\n\tCheck network: {network}, n_targets: {n_targets}, and volume_per_target: {volume_per_target}"
    exp_id = "_".join(str(a) for a in args)
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
        network,
        hosts[network],
        f"optimal_{n_targets}_link_attack_{candidate_set}",
        iterations=iterations,
        te_method=te_method,
        traffic_file=traffic_file,
        fallow_transponders=2,
        congestion_threshold_upper_bound=0.8,
        congestion_threshold_lower_bound=0.2,
        topology_programming_method="onset_v3",
        candidate_link_choice_method=candidate_set
    )

    result = attack_sim.perform_sim()
    result_file = f"data/reports/" + exp_id  + ".csv"
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

def main(argv):    
    if len(argv) == 1:
        network, n_targets, te_method, volume_per_target, candidate_set = "sprint", 5, "-mcf", 200, "max"
    else: 
        try: 
            network, n_targets, te_method, volume_per_target, candidate_set = argv[1:]
        except:    
            print(f"Usage: python {argv[0]} network n_targets te_method volume_per_target candidate_set")
            exit -1

    print(" ".join(argv[1:]))
    experiment(network, n_targets, te_method, volume_per_target, candidate_set)
    
if __name__ == "__main__": 
    main(argv)