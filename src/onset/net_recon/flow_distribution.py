import networkx as nx
import matplotlib.pyplot as plt
import pickle
from sys import argv, exit
from numpy import array

from onset.network_model import Network
from onset.utilities.recon_utils import  fdN_plt, cdf_plt
from onset.utilities.sysUtils import postfix_str
from onset.utilities.graph import write_json_graph
from onset.utilities.flows import read_flows
from onset.utilities.logger import logger

from onset.constants import CLEAN_START

def calc_flow_distribution(G, flows):
    """
    G: Physical topology of the network.
    Flows(List[tuple]): Adversary's tracing flow [(src, dest, tracing flows), ...].
    Generates the logical topology which reflects
    the adversary's view of the network.
    Based on EqualNet's algorithm 1 of logical topology generation.
    """
    H = nx.Graph()  # adversary's view of the network.
    for src, dest, tracing_flows in flows:
        shortest_paths = list(nx.all_shortest_paths(G, src, dest))
        num_shortest_paths = len(shortest_paths)
        logger.debug(f"{num_shortest_paths} paths between {src} and {dest}")
        for path in shortest_paths:
            logger.debug(path)
            path_len = len(path)            
            # if there multiple shortest paths.
            path_tf = tracing_flows // num_shortest_paths
            logger.debug(f"path flow: {path_tf}")
            # path_tf = tracing_flows
            # first node will always be the source node.
            # incoming_node = src # first node is the source

            for i in range(1, path_len):
                # incoming_node_ip = # ip address of the node connecting
                current_node = n_p_i = path[i]
                if i == 1:
                    prev_interface_addr = path[0]
                    interface_addr = G.nodes[current_node]["client_interface"]
                elif i == len(path) - 1: 
                    prev_node = path[i - 1]
                    prev_interface_addr = interface_addr
                    interface_addr = path[i]
                else:
                    prev_node = path[i - 1]
                    prev_interface_addr = interface_addr
                    interface_addr = G.nodes[current_node]["interface_map"][prev_node]                                        
                # print(G[current_node])
                # node_interface = G.nodes[current_node]['client_interface']
                # node_interface = G[path[i-1]]['interface_map'][prev_ip]
                
                link = (prev_interface_addr, interface_addr)
                if "client" in link[0] or "client" in link[1]: continue
                # print(H.nodes.data())
                # print(link)
                if not H.has_node(interface_addr):
                    node_attr = {
                        "fdN": path_tf,
                        # "client_interface": node_interface,
                        # "interface_map": {incoming_node_ip: dest}
                    }
                    H.add_node(interface_addr, **node_attr)
                else:
                    try:  # a node can already exist but no fdN value for it has been set.
                        H.nodes[interface_addr]['fdN'] += path_tf
                    except:
                        H.nodes[interface_addr]['fdN'] = path_tf
                if not H.has_edge(prev_interface_addr, interface_addr):
                    link_attr = {"fdL": {}}
                    link_attr['fdL'][str(link)] = path_tf
                    H.add_edge(prev_interface_addr, interface_addr, **link_attr)
                else:
                    try:
                        H[prev_interface_addr][interface_addr]['fdL'][str(link)] += path_tf
                    except KeyError as e:
                        # Since in networkx H[u][v] = H[v][u],
                        # in the case there isn't no tracing flow for (v,u).
                        H[prev_interface_addr][interface_addr]['fdL'][str(link)] = path_tf
    fdL = nx.get_edge_attributes(H,"fdL")
    fdN = nx.get_node_attributes(H,"fdN")
    return H, fdN, fdL

def main(argv):
    try:
        topology_file = argv[1]
        flows_file = argv[2]
    except:
        logger.error(f"usage: python {argv[0]} TopologyFile FlowsFile")
        exit()
        
    try:
        descriptor = argv[3]
    except:
        descriptor = ""

    
    # output_file = f"{topology_file.split('.')[:-1][0]}_attacker_view"
    
    ground_truth_file = postfix_str(topology_file, "modeled")
    output_file = postfix_str(topology_file, "outsider_view")
    
    if topology_file.endswith(".pkl"):
        with open(topology_file, 'rb') as fob:
            network = pickle.load(fob)
    else:
        network = Network(topology_file, ground_truth_file)

    G = network.graph
    write_json_graph(G, ground_truth_file)
    flows = read_flows(flows_file)
    H, fdN, fdL = calc_flow_distribution(G, flows)
    
    # nx.write_gml(H, output_file)
    write_json_graph(H, output_file)
    nx.draw(H, with_labels=True, font_weight='bold')
    plt.savefig(f'{output_file}_plot.jpg')
    plt.clf()

    fdN_plt(H, output_file=f"{output_file}_attacker_fdN")
    
    cdf_plt(array(list(fdN.values()))/1000, "Flow Density (K)", postfix_str(output_file, "attacker_view_ccdf"), complement=True, label=descriptor)

if __name__ == "__main__":
    main(argv)