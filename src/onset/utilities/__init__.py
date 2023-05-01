from collections import defaultdict
from os.path import realpath, curdir
import os
SCRIPT_HOME = os.path.join(os.path.expanduser('~'), "network_stability_sim")
USER_HOME = os.path.join(os.path.expanduser('~'))
ZERO_INDEXED = False

def parse_edges(path):
    path = path.strip().strip('[]')
    edges = path.split(', ')[1:-1]
    edge_list = []
    for e in edges:
        nodes = e.strip('()')
        a, b = nodes.split(',')
        # if ZERO_INDEXED: # zero indexed nodes
        #     a = str(int(a.strip('s')) - 1)
        #     b = str(int(b.strip('s')) - 1)

        # else: # one indexed nodes
        a = a.strip('s')
        b = b.strip('s')
            
        edge_list.append((a, b))
    
    return edge_list

def find_target_link(path_file):
    """Finds a target link from a path file. Returns the number of times a link appears in a path, and the flows that target 
        each link

    Args:
        path_file (str): absolute path to a the path file

    Returns:
        tuple: (0. edge_use:dict (edge:pair -> count:int), 1. edge_flows:dict (edge:pair -> flows:list:pair), 2. most_used_edge:int)
    """    
    with open(path_file, 'r') as fob:
        edge_use = defaultdict(int)
        edge_flows = defaultdict(list)
        most_used_edge = 0
        a, b = None, None
        for line in fob:
            if line.startswith('h'):
                host_line = line.split(' -> ')
                a, b = [h.strip(' \n:') for h in host_line]
                if ZERO_INDEXED: # zero indexed nodes
                    a = str(int(a.strip('h')) - 1)
                    b = str(int(b.strip('h')) - 1)
                else: # one indexed nodes
                    a = a.strip('h')
                    b = b.strip('h')
                        
            if line.startswith('['): # line contains a path
                path, percent = line.strip().split("@")
                path_edges = parse_edges(path)
                for edge in path_edges:
                    edge_use[edge] += 1
                    edge_flows[edge].append((a, b))
                    if edge_use[edge] == max(edge_use.values()):
                        most_used_edge = edge

    return edge_use, edge_flows, most_used_edge
