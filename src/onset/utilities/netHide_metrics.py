import numpy as np
import networkx as nx
from nltk import edit_distance
import gurobipy as gp

def accuracy(s, d, P, V):
    """ From Section 4.2 of Meier et al., 2018.
    The accuracy metric is a function that maps two paths for a given
    flow to a value v \in [0,1]. The value captures the similarity 
    between the paths in the virtual and physical topology.
    
    Args:
        s (node): source
        d (node): destination
        P (graph): physical topology (nodes, links, forwarding trees)
        V (graph): virtual topology (nodes, links, forwarding trees)
        
    Returns: 
        float [0,1]: similarity of a path from s to d in the physical and
                virtual topology 
    """           
    path_P = nx.shortest_path(P, s, d)
    path_V = nx.shortest_path(V, s, d)
    Levenshtein_distance = edit_distance(path_P, path_V)
    pathLen_P = len(path_P) - 1
    pathLen_V = len(path_V) - 1

    return 1 - (Levenshtein_distance / (pathLen_P + pathLen_V))
    
def utility(s, d, P, V):
    """ From Section 4.3 of Meier et al., 2018.
    The utility metric is measures the representation of physical
    events such as link failures. It incorporates the likelihood that
    a failure in the physical topology P is visible in the virtual 
    topology V and that a failure in V actually exists in P. 
    
    Args:
        s (node): source
        d (node): destination
        P (graph): physical topology (nodes, links, forwarding trees)
        V (graph): virtual topology (nodes, links, forwarding trees)
        
    Returns: 
        float [0,1]: utility of an (s, d) path in the virtual topology
    """
    path_P_nodes = nx.shortest_path(P, s, d)
    path_V_nodes = nx.shortest_path(V, s, d)
    path_P_links = [(u, v) for (u, v) in zip(path_P_nodes, path_P_nodes[1:])]
    path_V_links = [(u, v) for (u, v) in zip(path_V_nodes, path_P_nodes[1:])]
    pathLen_P = len(path_P_links)
    pathLen_V = len(path_V_links)
    U = [0 for _ in range(pathLen_V)]
    for n in range(1, pathLen_V + 1) :
        subPath_V = path_V_links[:n]
        C = set(path_P_links) & set(subPath_V)
        U[n-1] = 0.5 * ((len(C) / pathLen_P) + (len(C) / len(subPath_V)))

    u = np.sum(U) / pathLen_V

    return u

