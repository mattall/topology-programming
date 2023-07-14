import json

from collections import defaultdict
from sys import argv
from typing import DefaultDict

import networkx as nx

from onset.utilities.graph_utils import read_json_graph
from onset.utilities.plotters import plot_cdf
# from onset.utilities.graph_utils import get_edge_flows


def main(argv):
    try:
        G_path = argv[1]
    except:
        G_path = "data/graphs/json/campus/campus_ground_truth.json"
        H_path = "data/graphs/json/campus/campus_reconstruction.json"

    G = read_json_graph(G_path, stringify=True)
    H = read_json_graph(H_path, stringify=True)

    G_btwness = nx.edge_betweenness_centrality(G, normalized=False)
    H_btwness = nx.edge_betweenness_centrality(H, normalized=False)
    sorted_btwness = lambda btwness: sorted(
        btwness, key=btwness.get, reverse=True
    )

    sorted_btwness_G = sorted_btwness(G_btwness)
    sorted_btwness_H = sorted_btwness(H_btwness)

    print("e_G\te_H\tbetweenness(e_G))\tbetweenness(e_H)\tdifference")
    for e_G, e_H in zip(sorted_btwness_G, sorted_btwness_H):
        max_betweenness = G_btwness[e_G]
        min_betweenness = H_btwness[e_H]
        print(f"{e_G}\t{e_H}\t{max_betweenness}\t{min_betweenness}", end="\t")
        if e_G == e_H:
            print(f"{G_btwness[e_G] - H_btwness[e_H]}")
        else:
            print("inf")

    # n_G_paths = count_paths(G)
    # n_H_paths = count_paths(H)

    # print(sum([G_btwness[b] for b in G_btwness]), n_G_paths)
    # print(sum([H_btwness[b] for b in H_btwness]), n_H_paths)
    # p = nx.shortest_path(G)
    # print(p)

    my_G_btwness = my_betweenness_method(G, normalize=True)
    my_H_btwness = my_betweenness_method(H, normalize=True)

    my_sorted_btwness_G = sorted_btwness(my_G_btwness)
    my_sorted_btwness_H = sorted_btwness(my_H_btwness)

    print("e_G\te_H\tbtwness(e_G))\tbtwness(e_H)\tdifference")
    for e_G, e_H in zip(my_sorted_btwness_G, my_sorted_btwness_H):
        print(
            f"{e_G}\t{e_H}\t{my_G_btwness[e_G]:.3f}\t{my_H_btwness[e_H]:.3f}",
            end="\t",
        )
        if e_G == e_H:
            print(f"{my_G_btwness[e_G] - my_H_btwness[e_H]:.3f}")
        else:
            print("inf")

    # print(x, y, p[x][y])

    # print(sum(len(path for nx.shortest_path(G))))
    # print(len(nx.shortest_path(H)))

    accuracy = 0

    # union = 0
    # intersection = 0
    # for e_G in my_G_btwness:
    #     b_G = my_G_btwness[e_G]
    #     if e_G in my_H_btwness:
    #         b_H = my_H_btwness[e_H]
    #         intersection += abs(my_H_btwness)

    my_accuracy_method(G, H)
    


def count_paths(G):
    count = 0
    for x in G.nodes():
        for y in G.nodes():
            if x == y:
                continue
            paths = nx.all_shortest_paths(G, x, y)
            for p in paths:
                count += 1
                print(p)

    return count


def my_betweenness_method(G, normalize=False):
    d = defaultdict(int)
    count = 0
    for x in G.nodes():
        for y in G.nodes():
            if x == y:
                continue
            paths = nx.all_shortest_paths(G, x, y)
            for p in paths:
                for u, v in zip(p, p[1:]):
                    key = tuple(sorted((u, v)))
                    d[key] += 1
                count += 1

    if normalize:
        for key in d:
            d[key] = d[key] / count

    return d


def key_to_json(data):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, (tuple, frozenset)):
        return str(data)
    raise TypeError


def to_json(data):
    if data is None or isinstance(
        data, (bool, int, float, tuple, range, str, list)
    ):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}
    raise TypeError

def import_gml_graph(path):  # , label=None, destringizer=None):
    # G = read_gml(path, label, destringizer)
    G = nx.read_gml(path, label="id")
    # Note: Because of something weird in read_gml,
    # must remake node IDs into strings manually.
    if min(G.nodes) == 0:
        node_to_str_map = {node: str(node + 1) for (node) in G.nodes}
        # node_to_str_map = {node: ("sw" + str(node)) for (node) in G.nodes}
        nx.relabel_nodes(G, node_to_str_map, copy=False)

    # nx.relabel_nodes(G, )
    # nx.relabel_nodes(G, lambda x: _sanitize(x))
    position = {}
    for node in G.nodes():
        position[node] = (
            G.nodes()[node]["Longitude"],
            G.nodes()[node]["Latitude"],
        )

    nx.set_node_attributes(G, position, "position")

    color = {}
    for node in G.nodes():
        color[node] = "blue"
    nx.set_node_attributes(G, color, "color")
    return G


def get_paths(source: nx.Graph | str, target_json_file=""):
    if isinstance(source, str) and source.endswith(".gml"):
        G = import_gml_graph(source)
    elif isinstance(source, nx.Graph):
        G = source
    else:
        raise Exception("BAD SOURCE. Expected.gml file or nx.Graph instance")

    assert nx.is_connected(G)
    path_id = 0
    source_nodes = G.nodes()
    target_nodes = G.nodes()
    path_dict = DefaultDict(dict)
    for s in source_nodes:
        for t in target_nodes:
            if s != t:
                s_t_paths = nx.all_shortest_paths(G, s, t)
                for s_t_path in s_t_paths:
                    path_dict["path{}".format(path_id)]["src"] = f"{s}"
                    path_dict["path{}".format(path_id)]["dst"] = f"{t}"
                    path_dict["path{}".format(path_id)]["nhop"] = len(s_t_path)
                    path_dict["path{}".format(path_id)]["hops"] = [
                        f"{node}" for node in s_t_path
                    ]
                    path_id += 1

    if isinstance(target_json_file, str) and target_json_file.endswith(".json"):
        with open(target_json_file, "w") as fob:
            json.dump(
                {"paths": path_dict, "npath": len(path_dict)}, fob, indent=4
            )
            print(f"Saved JSON Paths to: {target_json_file}")

    return path_dict


def get_edge_flows(G, paths=None):
    """
    returns dictionary mapping each edge to the set of source and destination
    pairs whose shortest-path traverse the edge.

    e.g.,
        G:  1---2---3
                |
            4---5---6

     >>> get_edge_flows(G)
    {
        (1,2): {(1,2),(1,3),(1,4),(1,5),(1,6)},
        (2,3): {(1,3),(2,3),(3,4),(3,5),(3,6)},
        (2,5): {(1,4),(1,5),(1,6),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6)},
        (4,5): {(1,4),(2,4),(3,4),(4,5),(4,6)},
        (5,6): {(1,6),(2,6),(3,6),(4,6),(5,6)},
    }
    """
    if isinstance(paths, str):
        paths = read_paths(paths)
    else:
        paths = get_paths(G)
    edge_flows = DefaultDict(set)
    
    for net_path in paths:
        src = paths[net_path]["src"]
        dst = paths[net_path]["dst"]
        hops = paths[net_path]["hops"]
        for u, v in zip(hops[:], hops[1:]):
            edge = tuple(sorted((u, v)))
            edge_flows[edge].add(tuple(sorted((src, dst))))
    return edge_flows

def read_paths(path_file: str) -> dict:
    if path_file.endswith(".json"):
        return read_json_paths(path_file)

def read_json_paths(input_file):
    with open(input_file, "r") as fob:
        jobj = json.load(fob)
    if "paths" in jobj:          
        return jobj["paths"]
    else:
        return jobj

def my_accuracy_method(G, H):
    accuracy_record = {}
    all_edges = set(G.edges()).union(set(H.edges()))
    G_edge_flows = get_edge_flows(G)
    H_edge_flows = get_edge_flows(H)
    my_G_btwness = my_betweenness_method(G, normalize=True)
    my_H_btwness = my_betweenness_method(H, normalize=True)

    for an_edge in all_edges:
        e = tuple(sorted(an_edge))
        if e in G.edges():
            G_e_flows = G_edge_flows[e]
        else:
            G_e_flows = set()
        if e in H.edges():
            H_e_flows = H_edge_flows[e]
        else:
            H_e_flows = set()
        e_union = G_e_flows.union(H_e_flows)
        e_intersection = G_e_flows.intersection(H_e_flows)
        get_e_flow_weights = lambda G, flows: {
            f: 1 / len(list(nx.all_shortest_paths(G, *f))) for f in flows
        }
        G_e_flow_weights = get_e_flow_weights(G, G_e_flows)
        H_e_flow_weights = get_e_flow_weights(H, H_e_flows)
        sum_G_flow_weights = sum([v for v in G_e_flow_weights.values()])
        sum_H_flow_weights = sum([v for v in H_e_flow_weights.values()])

        accuracy_record[e] = {
            "G_flows": G_e_flows,
            "num_G_flows": len(G_e_flows),
            "H_flows": H_e_flows,
            "num_H_flows": len(H_e_flows),
            "union": e_union,
            "size_union": len(e_union),
            "intersection": e_intersection,
            "size_intersection": len(e_intersection),
            "G_betweenness": my_G_btwness[e],
            "H_betweenness": my_H_btwness[e],
            "Jaccard simularity": jaccard_similarity(G_e_flows, H_e_flows),
            "overlap coefficient": overlap_coefficient(G_e_flows, H_e_flows),
            "G_flow_weights": G_e_flow_weights,
            "H_flow_weights": H_e_flow_weights,
            "sum_G_flow_weights": sum_G_flow_weights,
            "sum_H_flow_weights": sum_H_flow_weights,
            "Jaccard_weighted": jaccard_similarity(
                G_e_flows, H_e_flows, G_e_flow_weights, H_e_flow_weights
            ),
            "overlap_weighted": overlap_coefficient(
                G_e_flows, H_e_flows, G_e_flow_weights, H_e_flow_weights
            ),
        }
    
    # with open("accuracy.json", "w") as fob:
    #     json.dump(to_json(accuracy_record), fob)
    csv_keys = [
        "edge",
        "G_flows",
        "num_G_flows",
        "H_flows",
        "num_H_flows",
        "union",
        "size_union",
        "intersection",
        "size_intersection",
        "G_betweenness",
        "H_betweenness",
        "Jaccard simularity",
        "overlap coefficient",
        "G_flow_weights",
        "H_flow_weights",
        "sum_G_flow_weights",
        "sum_H_flow_weights",
        "Jaccard_weighted",
        "overlap_weighted",
    ]
    
    with open("accuracy.csv", "w") as fob:                
        for k in csv_keys:
            fob.write(f"{k};")
        fob.write("\n")
        for e in accuracy_record:
            fob.write(f"{e};")
            for k in csv_keys[1:]:
                fob.write(f"{accuracy_record[e][k]};")
            fob.write("\n")
    
    return 


def jaccard_similarity(
    A: set, B: set, A_weights: dict = 1, B_weights: dict = 1
):
    if A_weights == 1:
        try:
            return len(A.intersection(B)) / len(A.union(B))
        except ZeroDivisionError as e:
            return "nan"
    else:
        U_size = weighted_union(A, B, A_weights, B_weights)
        I_size = weighted_intersection(A, B, A_weights, B_weights)
        try:
            return I_size / U_size
        except ZeroDivisionError:
            return "nan"


def overlap_coefficient(
    A: set, B: set, A_weights: dict = 1, B_weights: dict = 1
):
    if A_weights == 1:
        try:
            return len(A.intersection(B)) / min(len(A), len(B))
        except ZeroDivisionError:
            return "nan"
    else:
        A_size = sum([v for v in A_weights.values()])
        B_size = sum([v for v in B_weights.values()])
        I_size = weighted_intersection(A, B, A_weights, B_weights)
        try:
            return I_size / min(A_size, B_size)
        except ZeroDivisionError:
            return "nan"


def weighted_union(A, B, A_weights, B_weights):
    weighted_u = 0
    U = list(A.union(B))
    for i in range(len(U)):
        edge = U[i]
        if edge in A_weights:
            A_i = A_weights[edge]
        else:
            A_i = 0
        if edge in B_weights:
            B_i = B_weights[edge]
        else:
            B_i = 0

        weighted_u += max(A_i, B_i)
    return weighted_u


def weighted_intersection(A, B, A_weights, B_weights):
    weighted_i = 0
    I = list(A.intersection(B))
    for i in range(len(I)):
        edge = I[i]
        A_i = A_weights[edge]
        B_i = B_weights[edge]
        weighted_i += min(A_i, B_i)
    return weighted_i


if __name__ == "__main__":
    main(argv)
