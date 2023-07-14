import json

from collections import defaultdict
from sys import argv

import networkx as nx

from onset.utilities.graph_utils import read_json_graph, get_edge_flows


def main(argv):
    try:
        G_path = argv[1]
    except:
        G_path = "/home/matt/src/topology-programming/data/graphs/json/campus/campus_ground_truth.json"
        H_path = "/home/matt/src/topology-programming/data/graphs/json/campus/campus_reconstruction.json"

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
                    key = str(sorted([u, v]))
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


def my_accuracy_method(G, H):
    accuracy_record = {}
    all_edges = set(G.edges()).union(set(H.edges()))
    G_edge_flows = get_edge_flows(G)
    H_edge_flows = get_edge_flows(H)
    for an_edge in all_edges:
        e = tuple(sorted(an_edge))
        if e in G.edges():
            G_e_flows = G_edge_flows[e]
        else:
            G_e_flows = set()
        if e in H.edges():
            H_e_flows = H_edge_flows[e]
        elif e not in G.edges():
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
            "Jaccard": jaccard_similarity(G_e_flows, H_e_flows),
            "overlap": overlap_coefficient(G_e_flows, H_e_flows),
            "G_flows": G_e_flows,
            "num_G_flows": len(G_e_flows),
            "H_flows": H_e_flows,
            "num_H_flows": len(H_e_flows),
            "union": e_union,
            "size_union": len(e_union),
            "intersection": e_intersection,
            "size_intersection": len(e_intersection),
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
    with open("accuracy.json", "w") as fob:
        json.dump(to_json(accuracy_record), fob)


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
