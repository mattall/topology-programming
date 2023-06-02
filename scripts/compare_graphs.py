from sys import argv

import networkx as nx

from onset.utilities.graph_utils import read_json_graph
from onset.net_sim import Simulation
'''
A and B are Isomorphic (~=) Graphs.
Would like to find an Isomorphism of the two that is some how "Similfied"
or "Ruduced".
Perhapse, if the columns were oredered from least to greatest/greatest to least

>>> A = \
Matrix(array([[0, 0, 1, 0, 0],
       [0, 0, 1, 1, 1],
       [1, 1, 0, 1, 0],
       [0, 1, 1, 0, 0],
       [0, 1, 0, 0, 0]]))
>>> B = \
Matrix(array([[0, 1, 1, 1, 0],
       [1, 0, 0, 1, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1],
       [0, 0, 0, 1, 0]]))

A ~= B becasue there is a permutation matrix, P, such that
A = (P)(B)(P^T)

>>> P = \
Matrix(array([[0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0]]))

P*B*P.transpose()
Matrix([
[0, 0, 1, 0, 0],
[0, 0, 1, 1, 1],
[1, 1, 0, 1, 0],
[0, 1, 1, 0, 0],
[0, 1, 0, 0, 0]])
>>> P*B*P.transpose()==A
True

'''
def main(argv):
    try: 
        G_path = argv[1]
        H_path = argv[2]

    except:
        print(f"USAGE: {argv[0]} FILE_A FILE_B")
        # exit()
        G_path = "data/graphs/json/campus/campus_ground_truth.json"
        H_path = "data/graphs/json/campus/campus_reconstruction.json"

    G = read_json_graph(G_path)
    H = read_json_graph(H_path)
         
    '''This takes a long time, even for very small grpahs (14 nodes)'''
    # sim = nx.similarity.optimize_graph_edit_distance(G, H)
    # for v in nx.optimize_graph_edit_distance(G, H):
    #     sim = v
    # print(f"Graph edit distance: {sim}")

    Gamma = G.copy()

    graph_permutation_set = set() # set of candidate optical-layer mutations to apply to the graph G.

    union_size = len(set(G.edges()).union(set(H.edges())))
    intersection_size = len(set(G.edges()).intersection(set(H.edges())))
    
    
if __name__=="__main__":
    main(argv)



