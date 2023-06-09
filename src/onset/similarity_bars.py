from itertools import product
import json

from collections import defaultdict
from sys import argv
from os import devnull
from typing import DefaultDict

import networkx as nx
import pandas as pd 

from onset.similarity import my_accuracy_method
from onset.similarity import suggest_mutation
from onset.utilities.graph_utils import read_json_graph
from onset.utilities.graph_utils import import_gml_graph
from onset.utilities.plotters import cdf_plt
from onset.utilities.graph_utils import get_edge_flows


def main(argv):
    try:
        G_path = argv[1]
    except:
        G_path = "data/graphs/json/regional/ground_truth_regional.json"
        H_path = "data/graphs/json/regional/reconstruction_regional.json"

    G = read_json_graph(G_path, stringify=True)
    H = read_json_graph(H_path, stringify=True)

    accuracy = my_accuracy_method(G, H)    
    J = suggest_mutation(G)
    if J is None:
        exit()
