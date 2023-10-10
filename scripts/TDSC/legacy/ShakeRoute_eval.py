from collections import defaultdict
import os
import sys

sys.path.append(os.path.join(os.path.expanduser('~'), "network_stability_sim"))
sys.path.append(os.path.join(os.path.expanduser('~'), "network_stability_sim", "src"))
from itertools import combinations
import networkx as nx
import csv
from net_sim import Attack_Sim
PROJECT_HOME = os.path.join(os.path.expanduser('~'), "network_stability_sim")


def get_performance_after_cuts(network:str, cut_dir:str, result_file:str):
    cut_files = [os.path.join(cut_dir, cut_scenario) for cut_scenario in os.listdir(cut_dir) ]
    cut_scenarios = [ cut_scenario.strip(".gml") for cut_scenario in os.listdir(cut_dir) ]
    n_scenarios = len(cut_files)
    

    results = defaultdict(list)
    # baseline
    sim = Attack_Sim(network, "cut_resilience", 1, '-ecmp', 
                    traffic_file="", 
                    strategy="optimal",
                    fallow_transponders=100,
                    congestion_threshold_upper_bound=0.8,
                    congestion_threshold_lower_bound=0.1
    )
    res = sim.perform_sim(dry=True)
    for key in res: 
        results[key].extend(res[key])

    for scenario, file in zip(cut_scenarios, cut_files):
        sim = Attack_Sim(scenario, "cut_resilience", 1, '-ecmp', 
                        traffic_file= os.path.join(PROJECT_HOME, "data", "traffic", network+".txt"),
                        strategy="optimal",
                        fallow_transponders=100,
                        congestion_threshold_upper_bound=0.8,
                        congestion_threshold_lower_bound=0.1,
                        shakeroute=network
        )
        res = sim.perform_sim(dry=True)
        for key in res: 
            results[key].extend(res[key])

    with open("./data/results/{}_shakeroute.csv".format(network), 'w') as fob: 
        writer = csv.writer(fob)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))
            

def enumerate_cut_scenarios(G:nx.Graph, k:int, out_dir:str, net_name:str):
    """ Enumerates k-link fiber cut scenarios. 
        Writes graph induced by the cut to a file. 

    Args:
        network (nx.Graph): Base network topology
        k (int): Max number of links to consider for cutting.
        out_dir (str): path to directory in which to write output files. 
    """    
    
    edges = [e for e in G.edges()]
    cut_scenarios = []
    for i in range(1, k+1):
        for c in combinations(edges, i):
            cut_scenarios.append(c)
    
    for i, cs in enumerate(cut_scenarios):
        temp_G = G.copy()
        temp_G.remove_edges_from(cs)
        out_file = os.path.join(out_dir, "{}_{}_{}_edges.gml".format(net_name, i, len(cs)))
        nx.write_gml(temp_G, out_file)

def enumerate_alt_paths_from_scenarios(G:nx.Graph, net_name:str):
    """ Enumerates k-link fiber cut scenarios. 
        Writes graph induced by the cut to a file. 

    Args:
        network (nx.Graph): Base network topology
        k (int): Max number of links to consider for cutting.
        out_dir (str): path to directory in which to write output files. 
    """    
    import pickle as pkl

    path_lengths = []
    paths_exist = 0
    paths_dne = 0

    edges = [e for e in G.edges()]
    for e in edges:
        # print("Removing edge, ", e)
        temp_G = G.copy()
        temp_G.remove_edge(e[0], e[1])
        try:
            alt_path_p = nx.shortest_path(temp_G, e[0], e[1])
            paths_exist += 1
            path_lengths.append(len(alt_path_p) - 1)
        except nx.exception.NetworkXNoPath:
            paths_dne += 1
    
    path_info = {"path_lengths":path_lengths, "paths_exist": paths_exist, "paths_dne":paths_dne}
    with open("./data/shakeroute/{}_alt_path_data.pkl".format(net_name), 'wb') as fob:
        pkl.dump(path_info, fob)
        

def main():
    graphs_dir = os.path.join(os.path.expanduser('~'), "network_stability_sim", "data", "graphs", "gml")
    networks = ["ANS", "sprint", "darkstrand"]

    network_files = [os.path.join(graphs_dir, net + ".gml") for net in networks]
    for net, nf in zip(networks, network_files):
        # print(net)
        # out_dir = os.path.join(os.path.expanduser('~'), "network_stability_sim", "data", 
        #                         "graphs", "shakeroute", net)
        # os.makedirs(out_dir, exist_ok=True)

        # # G = nx.Graph(read_dot(nf))
        G = nx.read_gml(nf)
        # # enumerate_cut_scenarios(G, 2, out_dir, net)
        # get_performance_after_cuts(net, out_dir, "")
        enumerate_alt_paths_from_scenarios(G, net)

if __name__ == "__main__":
    main()