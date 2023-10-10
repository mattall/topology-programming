import os
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), "network_stability_sim"))
sys.path.append(os.path.join(os.path.expanduser('~'), "network_stability_sim", "src"))
from net_sim import Attack_Sim
from src.utilities.post_process import post_process

networks = ["CRL"]#, "CRL", "surfNet", "sprint", "bellCanada"]
# networks = ["sprint", "CRL", "surfNet", "sprint", "bellCanada"]
# networks = ["sprint"]
# networks = ["grid_3", "linear_10", "whisker_3_2"]
for net in networks:
    circuits=10
    attack_sim = Attack_Sim(net, 
                            "add_circuit_heuristic", 
                            te_method="-ecmp",
                            method="heuristic",
                            traffic_file= "/home/mhall/network_stability_sim/data/traffic/" + net + ".txt", 
                            strategy="cache", 
                            use_heuristic="yes",
                            fallow_transponders=circuits,
                            iterations=1)
    attack_sim.evaluate_performance_from_adding_link(circuits)

# post_process("ANS_add_circuit_heuristic_10__-ecmp", ["10"], ["ANS"])