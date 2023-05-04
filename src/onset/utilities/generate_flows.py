'''
    Generates flows for the given topology.
    by Abduarraheem Elfandi
    and Matt Nance-Hall
'''
import networkx as nx
import random
from sys import argv, exit
from os import path
from onset.utilities.recon_utils import read_json_graph, write_flows
from itertools import product
from numpy import load
from os import path

def generate_flows(file_path, min_tf, max_tf):
    """
    file_path (str): path to topology file.
    min_tf (int|float): minimum amount of tracing flows.
    max_tf (int|float): maximum amount of tracing flows.
    Output: flow_li (List[tuple]): [(src, dest, tracing flows), ...]

    NOTE could be improved to be better.
    """
    flows = []
    print("Generating flows for undirected graph.")

    if file_path.endswith(".gml"):
        G = nx.read_gml(file_path, label="id")
    elif file_path.endswith(".json"):
        G = read_json_graph(file_path)
    else:
        raise BaseException("Expected file to be .gml or .json")
    
    nodes = G.nodes()
    for i, j in product(nodes, repeat=2):
        if i == j: continue
        n_flows = random.randint(min_tf, max_tf)
        flows.append((f"client_{i}", f"client_{j}", n_flows))

    return flows


def tm_to_flows(tm_path):
    """
    file_path (str): path to topology file.        
    Output: flow_li (List[tuple]): [(src, dest, tracing flows), ...]    
    """
    dir_name = path.dirname(tm_path)
    base_name_no_ext = path.splitext(tm_path)[0]
    flows_file = path.join(dir_name, base_name_no_ext + "_flows.txt")

    flows = []

    tm = load(tm_path)
    for i, j in tm: 
        if tm[i][j] > 0:
            flows.append((f"client_{i}", f"client_{j}", tm[i][j]))
    
    return flows

def main(argv):
    # for testing purposes
    try:
        topology_file = argv[1]
        min_flow = float(argv[2])
        max_flow = float(argv[3])

    except:
        print("usage: python3 network.py topology_file.gml min_flow max_flow [output_file]")
        exit()

    try:        
        output_file = argv[4]

    except:
        output_file = topology_file + 'flow.txt'
        print(f'Output file not provided {output_file} will be generated.')

    flows = generate_flows(topology_file, min_flow, max_flow)
    write_flows(flows, output_file=output_file)

if __name__ == "__main__":
    main(argv)