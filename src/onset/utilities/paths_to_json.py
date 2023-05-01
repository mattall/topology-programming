# convert paths to JSON file. 

'''
{
    "paths":{
        "path{}".foramt(id): {
            "dst": node_id,
            "src": node_id,
            "nhop": int, 
            "hops": [
                node_id,
                ...
            ]
            }
        },
        ...
}
'''
import json
from os import makedirs, path
from typing import DefaultDict
import networkx as nx


def import_gml_graph(path):#, label=None, destringizer=None):
    # G = read_gml(path, label, destringizer)
    G = nx.read_gml(path, label="id")
    # Note: Because of something weird in read_gml, must remake node IDs into strings manually.
    if min(G.nodes)==0:
        node_to_str_map = {node: str(node+1) for (node) in G.nodes}
        # node_to_str_map = {node: ("sw" + str(node)) for (node) in G.nodes}
        nx.relabel_nodes(G, node_to_str_map, copy=False)

    # nx.relabel_nodes(G, )
    # nx.relabel_nodes(G, lambda x: _sanitize(x))
    position = {}
    for node in G.nodes():
        position[node] = (G.nodes()[node]['Longitude'], G.nodes()[node]['Latitude'])
    
    nx.set_node_attributes(G, position, 'position')

    color = {}
    for node in G.nodes():
        color[node] = 'blue'
    nx.set_node_attributes(G, color, 'color')
    return G

def get_paths(source_gml_file, target_json_file):
    G = import_gml_graph(source_gml_file)
    new_graph_file = None
    # G = nx.read_gml(source_gml_file)
    source_dir = path.dirname(source_gml_file)
    source_base_file = path.basename(source_gml_file)

    target_dir = path.dirname(target_json_file)
    target_base_file = path.basename(target_json_file)

    if not nx.is_connected(G):
        LCC = max(nx.strongly_connected_components(G.to_directed()), key=len)
        G = G.subgraph(LCC)
        new_graph_file = path.join(source_dir, "connected_" + source_base_file)
        node_label_map = {i : str(j+1) for (i, j) in zip(G.nodes(), range(len(G.nodes()))) }
        G = nx.relabel_nodes(G, node_label_map)
        write_gml(G, new_graph_file)
        target_json_file = path.join(target_dir, "connected_" + target_base_file)


    pid = 0
    source_nodes = G.nodes()
    target_nodes = G.nodes()
    path_dict = DefaultDict(dict)
    for source in source_nodes:
        for target in target_nodes:
            if source != target:
                s_t_paths = nx.all_shortest_paths(G, source, target)
                for s_t_path in s_t_paths:
                    # path_dict["path{}".format(pid)]["src"] = G.nodes[source]["label"]
                    # path_dict["path{}".format(pid)]["dst"] = G.nodes[target]["label"]
                    path_dict["path{}".format(pid)]["src"] = "s{}".format(source)
                    path_dict["path{}".format(pid)]["dst"] = "s{}".format(target)
                    path_dict["path{}".format(pid)]["nhop"] = len(s_t_path)
                    # path_dict["path{}".format(pid)]["hops"] = [G.nodes[node]["label"] for node in s_t_path]
                    path_dict["path{}".format(pid)]["hops"] = ["s{}".format(node) for node in s_t_path]
                    pid += 1 
    with open(target_json_file, 'w') as fob:
        json.dump( {"paths":path_dict, "npath":len(path_dict)}, fob, indent=4 )
        print("Saved JSON Paths to: {}".format(target_json_file))
    
    return target_json_file, len(G.nodes)
    
def convert_paths_onset_to_json(source_file, target_file):
    paths = {}
    with open(source_file, 'r') as fob:
        path_tag = 0
        for line in fob:
            if line.startswith('h'):
                src, dst = line.split("->")
                src = src.strip()
                dst = dst.strip().strip(":").strip()

            if line.startswith("["):
                path_id = "path{}".format(path_tag)
                paths[path_id] = {}
                paths[path_id]["src"] = src.replace('h','s')
                paths[path_id]["dst"] = dst.replace('h','s')
                # print(paths[path_id]["src"], paths[path_id]["dst"])

                # print(line.strip())
                hops = line.split('@')[0] # ignore everything after '@'
                hops = hops.split('),')[:-1] # only look at core hops, not edge.
                hop_nodes = [hop.split(',')[-1] for hop in hops]
                paths[path_id]["nhop"] = len(hop_nodes)
                paths[path_id]["hops"] = hop_nodes
                path_tag += 1

    with open(target_file, 'w') as fob:
        json.dump({"paths":paths, "npath":len(paths)}, fob, indent=4)


if __name__ == "__main__":
    if 0:  # Base Paths
        convert_paths_onset_to_json("/home/matt/network_stability_sim/data/results/sprint_add_circuit_heuristic_10/__0/paths/ecmp_0", 
                                    "/home/matt/ripple/simulator/topologies/sprint/path.json")

    if 1: # Reroute Paths
        convert_paths_onset_to_json("/home/matt/network_stability_sim/data/results.old/surfNet_add_circuit_heuristic_10/9_20/paths/ecmp_0",
                                    "/home/matt/ripple/simulator/topologies/surfNet/9_20_paths.json")

    if 0: 
        with open("/home/matt/network_stability_sim/notes.json", 'r') as fob:
            config = json.load(fob)

        for network in config:
            makedirs(path.join("/home/matt/ripple/simulator/topologies", network), exist_ok=True)
            # txt_path = config[network]["BasePathsONSET"]
            # json_path = config[network]["BasePathsJSON"]
            # convert_paths_onset_to_json(txt_path, json_path)
            txt_path = config[network]["AdaptedPathsONSET"]
            json_path = config[network]["AdaptedPathsJSON"]
            print(json_path)
            convert_paths_onset_to_json(txt_path, json_path)
            