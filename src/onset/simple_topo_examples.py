'''
Creates 3 generic simple example topologies, traffic matrices, and paths
for testing with Ripple, and Ripple + ONSET

Linear graph.

series of 3 or more links.
A <-> B <-> C

Mesh.
4 by 4 grid topology.
A <-> B <-> C <-> D
^     ^     ^     ^
|     |     |     |
v     v     v     v
E <-> F <-> G <-> H
...

Dumbbell.
Linear topology with whiskers.
A             F
 \           /
  \         /
B---D <-> E---G
  /         \
 /           \
C             H
'''

# library for creating graphs. 
import networkx as nx
from networkx.algorithms.centrality.reaching import local_reaching_centrality

# creating traffic matrices
from utilities.tmg import rand_gravity_matrix
from attacker import read_tm_to_tc, Attacker

# json writing utilities
from utilities.matrix_to_json_flows import write_flows_to_json
from utilities.paths_to_json import convert_paths_onset_to_json
from utilities.write_gml import write_gml

def get_linear_topo(n_links:int) -> nx.Graph:
    # creates and returns linear topology with n_links
    # series of 3 or more links.
    # e.g., A <-> B <-> C

    G = nx.Graph()
    my_nodes = ['s{}'.format(n+1) for n in range(n_links+1)]
    i = 0
    for u, v in zip(my_nodes, my_nodes[1:]):
        G.add_edge(u, v, capacity=100)
        G.nodes[u]["Longitude"] = i
        G.nodes[u]["Latitude"] = 0
        i += 1

    G.nodes[my_nodes[-1]]["Longitude"] = i
    G.nodes[my_nodes[-1]]["Latitude"] = 0
    
    return G

def get_whisker_topo(n_span_hops:int, n_whiskers) -> nx.Graph:
    # creates and returns whisker topology with n_span_hops links in the middle and n_whiskers on each end.
    # Linear topology with whiskers
    # e.g., 
    # get_whisker_topo(n_span_hops=1, n_whiskers=3).
    # A             F
    #  \           /
    #   \         /
    # B---D <-> E---G
    #   /         \
    #  /           \
    # C             H

    G = nx.Graph()
    
    nodes = ['s{}'.format(n + 1) for n in range( 2*(n_whiskers+1) + (n_span_hops - 1) )]
    
    left_whisk_nodes = nodes[:n_whiskers]
    left_span_node = nodes[n_whiskers]
    span_nodes = nodes[n_whiskers:-n_whiskers]
    right_span_node = nodes[-n_whiskers-1]
    right_whisk_nodes = nodes[-n_whiskers:]
    
    mid_latitude = n_whiskers / 2 
    longitude = 0

    for i, lwn in enumerate(left_whisk_nodes):
        G.add_edge(lwn, left_span_node, capacity=100)
        G.nodes[lwn]['Latitude'] = i
        G.nodes[lwn]['Longitude'] = longitude
    
    # Left whiskers placed, move longitude.
    longitude += 1 
    G.nodes[left_span_node]['Longitude'] = longitude
    G.nodes[left_span_node]['Latitude'] = mid_latitude 
    

    # first span node placed, move longitude
    longitude += 1 

    for u, v in zip(span_nodes, span_nodes[1:]):
        G.add_edge(u, v, capacity=100)
        G.nodes[v]['Longitude'] = longitude
        G.nodes[v]['Latitude'] = mid_latitude
        
        
        # after each span hop is placed, move longitude
        longitude += 1

    for i, rwn in enumerate(right_whisk_nodes):
        G.add_edge(right_span_node, rwn, capacity=100)
        G.nodes[rwn]['Longitude'] = longitude
        G.nodes[rwn]['Latitude'] = i
        
    return G

def get_whisker_topo(n_span_hops:int, n_whiskers) -> nx.Graph:
    # creates and returns whisker topology with n_span_hops links in the middle and n_whiskers on each end.
    # Linear topology with whiskers
    # e.g., 
    # get_whisker_topo(n_span_hops=1, n_whiskers=3).
    # A             F
    #  \           /
    #   \         /
    # B---D <-> E---G
    #   /         \
    #  /           \
    # C             H

    G = nx.Graph()
    
    nodes = ['s{}'.format(n + 1) for n in range( 2*(n_whiskers+1) + (n_span_hops - 1) )]
    
    left_whisk_nodes = nodes[:n_whiskers]
    left_span_node = nodes[n_whiskers]
    span_nodes = nodes[n_whiskers:-n_whiskers]
    right_span_node = nodes[-n_whiskers-1]
    right_whisk_nodes = nodes[-n_whiskers:]
    
    mid_latitude = n_whiskers / 2 
    longitude = 0

    for i, lwn in enumerate(left_whisk_nodes):
        G.add_edge(lwn, left_span_node, capacity=100)
        G.nodes[lwn]['Latitude'] = i
        G.nodes[lwn]['Longitude'] = longitude
    
    # Left whiskers placed, move longitude.
    longitude += 1 
    G.nodes[left_span_node]['Longitude'] = longitude
    G.nodes[left_span_node]['Latitude'] = mid_latitude 
    

    # first span node placed, move longitude
    longitude += 1 

    for u, v in zip(span_nodes, span_nodes[1:]):
        G.add_edge(u, v, capacity=100)
        G.nodes[v]['Longitude'] = longitude
        G.nodes[v]['Latitude'] = mid_latitude
        
        
        # after each span hop is placed, move longitude
        longitude += 1

    for i, rwn in enumerate(right_whisk_nodes):
        G.add_edge(right_span_node, rwn, capacity=100)
        G.nodes[rwn]['Longitude'] = longitude
        G.nodes[rwn]['Latitude'] = i
        
    return G    

def get_grid_topo(grid_dimension) -> nx.Graph:
    G = nx.generators.grid_graph(dim=(grid_dimension, grid_dimension))
    nx.set_edge_attributes(G, 100, 'capacity')
    for (x, y) in G.nodes():
        G.nodes[(x,y)]["Longitude"] = x
        G.nodes[(x,y)]["Latitude"] = y
        
    labels = {node : 's{}'.format(i+1) for i, node in enumerate(G.nodes())}
    nx.relabel_nodes(G, labels,copy=False)
    return G


def main():
    # Step 1: create graphs.
    GRAPH_FOLDER = "/home/matt/network_stability_sim/data/graphs/gml/"
    linear_length = 10
    linear_G = get_linear_topo(linear_length)

    whiskers, span_hops = 3, 2
    dumbbell_G = get_dumbbell_topo(n_whiskers=whiskers, n_span_hops=span_hops)
    
    grid_dim = 3
    grid_G = get_grid_topo(grid_dim)
    linear_topo_file = GRAPH_FOLDER + "linear_"  + str(linear_length) + ".gml"
    dumbbell_topo_file = GRAPH_FOLDER + "dumbbell_" + str(whiskers) + "_" + str(span_hops) + ".gml"
    grid_topo_file = GRAPH_FOLDER + "grid_"    + str(grid_dim) + ".gml"
    write_gml(linear_G, linear_topo_file)
    print('wrote graph to: ' + linear_topo_file)
    
    write_gml(dumbbell_G, dumbbell_topo_file)
    print('wrote graph to: ' + dumbbell_topo_file)

    write_gml(grid_G, grid_topo_file)
    print('wrote graph to: ' + grid_topo_file)
    
    # Step 2: Create traffic flows for benign and attack traffic. 

    # Step 3: Create traffic flows for benign and attack traffic. 

if __name__ == "__main__":
    main()