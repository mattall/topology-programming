from src.utilities.post_process import post_proc_timeseries
from net_sim import Attack_Sim

network = "linear_10" #
networks = ["linear_10"]
iterations = 3 #
volume = "300Gbps" #
benign_vol     =   "240" # 
atk_vol     =   "60"
targets = "1"
tag     = "oneShot"
traffic_file = '/home/matt/network_stability_sim/data/traffic/{}_{}Gbps_{}Gbps_{}_{}.txt'.format(network, benign_vol, atk_vol, iterations, tag)
congestion_threshold = 1
proportion="40-20"
# networks = ["sprint", "surfNet", "sprint", "bellCanada"]

if 0: # Preprocess ( Same as exp1.py )
    print("PREPROCESSING")
    for net in networks:
        circuits=10
        attack_sim = Attack_Sim(net, 
                                    "add_circuit_heuristic", 
                                    te_method="-ecmp",
                                    method="heuristic",
                                    # traffic_file= "/home/matt/network_stability_sim/data/traffic/" + net + ".txt", 
                                    traffic_file=traffic_file,
                                    strategy="cache", 
                                    use_heuristic="yes",
                                    fallow_transponders=circuits,
                                    proportion=proportion)
        attack_sim.evaluate_performance_from_adding_link(circuits)   

if 0: # Baseline
    print("BASELINE ANALYSIS")
    for net in networks:    
        h = "yes"
        attack_sim = Attack_Sim(net, 
                                "attack_heuristic", 
                                iterations=iterations, 
                                te_method="-ecmp",
                                method="heuristic",    
                                # traffic_file="/home/matt/network_stability_sim/data/traffic/{}_{}Gbps_{}Gbps_{}_{}.txt".format(net, benign_vol, atk_vol, targets, tag),
                                traffic_file=traffic_file,
                                strategy="cache", 
                                use_heuristic=h,
                                fallow_transponders=0,
                                proportion=proportion)
        attack_sim.perform_sim(circuits=0)

if 0: # Test all Heuristics
    for net in networks:
        # for i in [1, 2, 3, 4, 5, 6, 7]:
        # # for i in [7]:
        #     h = str(i)
        #     attack_sim = Attack_Sim(net, 
        #                             "attack_heuristic", 
        #                             iterations=iterations, 
        #                             te_method="-ecmp",
        #                             method="heuristic",
        #                             # traffic_file="/home/matt/network_stability_sim/data/traffic/sprint_10Gbps_20.txt", 
        #                             traffic_file=traffic_file,
        #                             strategy="cache", 
        #                             use_heuristic=h,
        #                             fallow_transponders=10,
        #                             congestion_threshold=1)
        #     attack_sim.perform_sim(circuits=10)
        
        labels = ["Baseline", 
                "Dynamic Congestion", 
                "Static Add Paths From Central Link",
                "Static Remove Paths From Central Link",
                "Static Add Many Paths",
                "Static Remove Many Paths",
                "Dynamic Add Many Paths + add paths to congested link.",
                "Dynamic Add Many Paths + remove paths from congested link."] 

        time_series_files = ["/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_0", 
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_1_10", 
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_2_10", 
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_3_10", 
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_4_10",
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_5_10",
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_6_10",
                            "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_7_10"]
        post_proc_timeseries(time_series_files, net, iterations, labels)

if 0: # Test Heuristics 7
    print("TESTING HEURISTIC 7")
    # h_nets = ["linear_10"]
    # h_nets = ["ANS", "bellCanada"]
    # for net in h_nets:
    for net in networks:
        for i in [7]:
            h = str(i)
            attack_sim = Attack_Sim(net, 
                                    "attack_heuristic", 
                                    iterations=iterations, 
                                    te_method="-ecmp",
                                    method="heuristic",
                                    # traffic_file="/home/matt/network_stability_sim/data/traffic/{}_{}Gbps_{}Gbps_{}_{}.txt".format(net, benign_vol, atk_vol, targets, tag),
                                    traffic_file=traffic_file,
                                    strategy="cache", 
                                    use_heuristic=h,
                                    fallow_transponders=10, 
                                    congestion_threshold=congestion_threshold,
                                    proportion=proportion)
            attack_sim.perform_sim(circuits=10)

if 1: # Plot attack results
    for net in networks:
        labels = ["Baseline", 
                  "Dynamic Add Many Paths + remove paths from congested link."] 

        time_series_files = ["/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_0_{}".format(proportion), 
                             "/home/matt/network_stability_sim/data/results/" + net + "_attack_heuristic_7_10_{}".format(proportion)]
        post_proc_timeseries(time_series_files, net, iterations, labels, volume, proportion)