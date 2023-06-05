from itertools import product
import json

from collections import defaultdict
from sys import argv
from typing import DefaultDict

import networkx as nx

from onset.utilities.graph_utils import read_json_graph
from onset.utilities.graph_utils import import_gml_graph
from onset.utilities.plotters import cdf_plt
from onset.utilities.graph_utils import get_edge_flows


def main(argv):
    try:
        G_path = argv[1]
    except:
        G_path = "data/graphs/json/campus/campus_ground_truth.json"
        H_path = "data/graphs/json/campus/campus_reconstruction.json"

    G = read_json_graph(G_path, stringify=True)
    H = read_json_graph(H_path, stringify=True)

    accuracy = my_accuracy_method(G, H)

    x_label = "Similarity Distribution"
    jac_sim_dist = [accuracy[e]["Jaccard similarity"] for e in accuracy]
    jac_norm_sim_dist = [accuracy[e]["Jaccard weighted"] for e in accuracy]
    fig, ax = cdf_plt(
        jac_sim_dist, xlabel=x_label, label="Jaccard", clear_fig=False
    )
    cdf_plt(
        jac_norm_sim_dist,
        xlabel=x_label,
        label="Jaccard Weighted",
        fig=fig,
        ax=ax,
        output_file="Jaccard_Similarity",
    )

    x_label = "Overlap Distribution"
    dist_1 = [accuracy[e]["overlap coefficient"] for e in accuracy]
    dist_2 = [accuracy[e]["overlap weighted"] for e in accuracy]
    fig, ax = cdf_plt(dist_1, xlabel=x_label, label="Overlap", clear_fig=False)
    cdf_plt(
        dist_2,
        xlabel=x_label,
        label="Overlap Weighted",
        fig=fig,
        ax=ax,
        output_file="Overlap_similarity",
    )

    G_btwness = [accuracy[e]["G betweenness"] for e in accuracy]
    H_btwness = [accuracy[e]["H betweenness"] for e in accuracy]
    x_label = "Betweenness Centrality Distribution"
    fig, ax = cdf_plt(
        G_btwness, xlabel=x_label, label="Ground Truth", clear_fig=False
    )
    cdf_plt(
        H_btwness,
        xlabel=x_label,
        label="Ricci Reconstruction",
        fig=fig,
        ax=ax,
        output_file="Betweenness_centrality",
    )
    GB = nx.edge_betweenness_centrality(G)

    most_central = max(GB, key=GB.get)
    least_central = min(GB, key=GB.get)

    J = G.copy()
    print(f"most central edge:{most_central}")
    J.remove_edge(*most_central)

    if (least_central[0], most_central[0]) not in J.edges():
        print(f"adding edges{(least_central[0], most_central[0])}")
        J.add_edge(least_central[0], most_central[0])

    mutated_accuracy = my_accuracy_method(G, J)
    x_label = "Similarity Distribution for Mutated Graph"
    jac_sim_dist = [
        mutated_accuracy[e]["Jaccard similarity"] for e in mutated_accuracy
    ]
    jac_norm_sim_dist = [
        mutated_accuracy[e]["Jaccard weighted"] for e in mutated_accuracy
    ]
    fig, ax = cdf_plt(
        jac_sim_dist, xlabel=x_label, label="Jaccard", clear_fig=False
    )
    cdf_plt(
        jac_norm_sim_dist,
        xlabel=x_label,
        label="Jaccard Weighted",
        fig=fig,
        ax=ax,
        output_file="Jaccard_Similarity_mute",
    )

    x_label = "Overlap Distribution for Mutated Graph"
    dist_1 = [
        mutated_accuracy[e]["overlap coefficient"] for e in mutated_accuracy
    ]
    dist_2 = [
        mutated_accuracy[e]["overlap weighted"] for e in mutated_accuracy
    ]
    fig, ax = cdf_plt(dist_1, xlabel=x_label, label="Overlap", clear_fig=False)
    cdf_plt(
        dist_2,
        xlabel=x_label,
        label="Overlap Weighted",
        fig=fig,
        ax=ax,
        output_file="Overlap_similarity_mute",
    )

    G_btwness = [
        mutated_accuracy[e]["G betweenness"] for e in mutated_accuracy
    ]
    H_btwness = [
        mutated_accuracy[e]["H betweenness"] for e in mutated_accuracy
    ]
    x_label = "Betweenness Centrality Distribution"
    fig, ax = cdf_plt(
        G_btwness, xlabel=x_label, label="Ground Truth", clear_fig=False
    )
    cdf_plt(
        H_btwness,
        xlabel=x_label,
        label="Mutated Graph",
        fig=fig,
        ax=ax,
        output_file="Betweenness_centrality_mute",
    )
    
    mutated_vs_ricci_accuracy = my_accuracy_method(H, J)
    
    # Similarity GT v. Ricci, Ricci v. Mute, Mute v. GT
    
    x_label = "Overlap Distribution"
    dist_1 = [
        accuracy[e]["overlap weighted"] for e in accuracy
    ]
    dist_2 = [
        mutated_accuracy[e]["overlap weighted"] for e in mutated_accuracy
    ]
    dist_3 = [
        mutated_vs_ricci_accuracy[e]["overlap weighted"] for e in mutated_vs_ricci_accuracy
    ]
    
    fig, ax = cdf_plt(dist_1, xlabel=x_label, label="$G(t_0)$ vs. Ricci($t_0$)", clear_fig=False)
    fig, ax = cdf_plt(dist_2, xlabel=x_label, label="$G(t_0)$ vs. $G(t_1)$", clear_fig=False, fig=fig, ax=ax)
    cdf_plt(
        dist_3,
        xlabel=x_label,
        label="$G(t_1)$ vs. Ricci($t_0$)",
        fig=fig,
        ax=ax,
        output_file="Overlap_similarity_3-way_weighted",
    )
    
    # Similarity GT v. Ricci, Ricci v. Mute, Mute v. GT
    
    x_label = "Jaccard Similarity Distribution"
    metric = "Jaccard weighted"
    dist_1 = [
        accuracy[e][metric] for e in accuracy
    ]
    dist_2 = [
        mutated_accuracy[e][metric] for e in mutated_accuracy
    ]
    dist_3 = [
        mutated_vs_ricci_accuracy[e][metric] for e in mutated_vs_ricci_accuracy
    ]
    
    fig, ax = cdf_plt(dist_1, xlabel=x_label, label="$G(t_0)$ vs. Ricci($t_0$)", clear_fig=False)
    fig, ax = cdf_plt(dist_2, xlabel=x_label, label="$G(t_0)$ vs. $G(t_1)$", clear_fig=False, fig=fig, ax=ax)
    cdf_plt(
        dist_3,
        xlabel=x_label,
        label="$G(t_1)$ vs. Ricci($t_0$)",
        fig=fig,
        ax=ax,
        output_file="Jaccard_similarity_3-way_weighted",
    )
    
    # Similarity GT v. Ricci, Ricci v. Mute, Mute v. GT
    
    x_label = "Edge Betweenness Distribution ($\sigma(*)$)"
    metric = "G betweenness"
    dist_1 = [
        accuracy[e][metric] for e in accuracy
    ]
    metric = "H betweenness"
    dist_2 = [
        accuracy[e][metric] for e in accuracy
    ]
    dist_3 = [
        mutated_vs_ricci_accuracy[e][metric] for e in mutated_vs_ricci_accuracy
    ]
    
    fig, ax = cdf_plt(dist_1, xlabel=x_label, label="$\sigma(G(t_0))$", clear_fig=False)
    fig, ax = cdf_plt(dist_2, xlabel=x_label, label="$\sigma$(Ricci($t_0$))", clear_fig=False, fig=fig, ax=ax)
    cdf_plt(
        dist_3,
        xlabel=x_label,
        label="$\sigma(G(t_1))$",
        fig=fig,
        ax=ax,
        output_file="betweenness_3-way_weighted",
    )
    
    
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


def path_normalized_betweenness_method(G, normalize=False) -> dict:
    """
    Betweenness for edges, scaled by total number of paths in a network
        rather than by the number of source/destination pairs
    Args:
        G (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.

    Returns:
        dict: _description_
    """
    d = defaultdict(int)
    count = 0
    for x, y in product(G.nodes(), repeat=2):
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


def re_key(d):
    new_d = defaultdict(int)
    for key in d.keys():
        new_key = tuple(sorted(key))
        new_d[new_key] = d[key]
    return new_d


def my_accuracy_method(G, H):
    accuracy_record = {}
    all_edges = set(G.edges()).union(set(H.edges()))
    G_edge_flows = get_edge_flows(G)
    H_edge_flows = get_edge_flows(H)
    # my_G_btwness = path_normalized_betweenness_method(G, normalize=True)
    # my_H_btwness = path_normalized_betweenness_method(H, normalize=True)

    my_G_btwness = re_key(nx.edge_betweenness_centrality(G))
    my_H_btwness = re_key(nx.edge_betweenness_centrality(H))
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
            "G flows": G_e_flows,
            "num G flows": len(G_e_flows),
            "H flows": H_e_flows,
            "num H flows": len(H_e_flows),
            "union": e_union,
            "size union": len(e_union),
            "intersection": e_intersection,
            "size intersection": len(e_intersection),
            "G betweenness": my_G_btwness[e],
            "H betweenness": my_H_btwness[e],
            "Jaccard similarity": jaccard_similarity(G_e_flows, H_e_flows),
            "overlap coefficient": overlap_coefficient(G_e_flows, H_e_flows),
            "G flow_weights": G_e_flow_weights,
            "H flow weights": H_e_flow_weights,
            "sum G flow_weights": sum_G_flow_weights,
            "sum H flow_weights": sum_H_flow_weights,
            "Jaccard weighted": jaccard_similarity(
                G_e_flows, H_e_flows, G_e_flow_weights, H_e_flow_weights
            ),
            "overlap weighted": overlap_coefficient(
                G_e_flows, H_e_flows, G_e_flow_weights, H_e_flow_weights
            ),
        }

    # with open("accuracy.json", "w") as fob:
    #     json.dump(to_json(accuracy_record), fob)
    csv_keys = [
        "G flows",
        "num G flows",
        "H flows",
        "num H flows",
        "union",
        "size union",
        "intersection",
        "size intersection",
        "G betweenness",
        "H betweenness",
        "Jaccard similarity",
        "overlap coefficient",
        "G flow_weights",
        "H flow weights",
        "sum G flow_weights",
        "sum H flow_weights",
        "Jaccard weighted",
        "overlap weighted",
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

    return accuracy_record


def jaccard_similarity(
    A: set, B: set, A_weights: dict = 1, B_weights: dict = 1
):
    if A_weights == 1:
        try:
            return len(A.intersection(B)) / len(A.union(B))
        except ZeroDivisionError as e:
            return float("NaN")
    else:
        U_size = weighted_union(A, B, A_weights, B_weights)
        I_size = weighted_intersection(A, B, A_weights, B_weights)
        try:
            return I_size / U_size
        except ZeroDivisionError:
            return float("NaN")


def overlap_coefficient(
    A: set, B: set, A_weights: dict = 1, B_weights: dict = 1
):
    if A_weights == 1:
        try:
            return len(A.intersection(B)) / min(len(A), len(B))
        except ZeroDivisionError:
            return float("NaN")
    else:
        A_size = sum([v for v in A_weights.values()])
        B_size = sum([v for v in B_weights.values()])
        I_size = weighted_intersection(A, B, A_weights, B_weights)
        try:
            return I_size / min(A_size, B_size)
        except ZeroDivisionError:
            return float("NaN")


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
