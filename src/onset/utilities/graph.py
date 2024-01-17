from ipaddress import ip_address
from os import path
from sys import exit
from typing import DefaultDict
import networkx as nx


import json

from networkx import read_gml
from networkx.classes.function import get_edge_attributes


def write_json_graph(G, output_file):
    if not output_file.endswith(".json"):
        output_file += ".json"

    jobj = nx.adjacency_data(G)
    with open(output_file, "w") as fob:
        json.dump(jobj, fob, indent=4)
    return None

def find_paths_with_flow(source, target, links):
    graph = {}

    # Build the graph from the given links
    for link in links:
        if link[0] not in graph:
            graph[link[0]] = []
        graph[link[0]].append((link[1], link[2]))

    # DFS function to find paths
    def dfs(current, path, total_flow):
        if current == target:
            result.append((path, total_flow))
            return
        if current not in graph:
            return
        for neighbor, flow in graph[current]:
            dfs(neighbor, path + [(current, neighbor)], min(total_flow, flow))

    result = []
    dfs(source, [], float('inf'))

    return result

def write_paths(paths, out_file): 
    source = paths[0][0][0][0] 
    target = paths[0][0][-1][-1]
    
    for path, val in paths: 
        assert path[ 0][ 0] == source
        assert path[-1][-1] == target

    with open(out_file, 'a') as fob: 
        fob.write(f"{source} -> {target} :\n")
        for path, val in paths: 
            fob.write(f"{path} @ {val}\n")
        fob.write('\n')


def print_paths(paths):
    # paths are list of pairs (path, val) 
    # 
    # e.g., [([(3, 1), (1, 2), (2, 4)], 32), ([(3, 5), (5, 2), (2, 4)], 8)] 
    # 
    # path[0] = ([(3, 1), (1, 2), (2, 4)], 32)
    # path[0][ 0] = [(3, 1), (1, 2), (2, 4)]
    # path[0][ 0][ 0] = (3, 1)    
    # path[0][ 0][ 0][ 0] = 3 
    # path[0][ 0][-1] = (2, 4)    
    # path[0][ 0][-1][-1] = 4 
    #
    # prints 
    # 3 -> 4
    # [(3, 1), (1, 2), (2, 4)] @ 32
    # [(3, 5), (5, 2), (2, 4)] @ 8
    source = paths[0][0][0][0] 
    target = paths[0][0][-1][-1]
    
    for path, val in paths: 
        assert path[ 0][ 0] == source
        assert path[-1][-1] == target
    
    print(f"{source} -> {target} :")
    for path, val in paths: 
        print(path, "@", val)
    print()


def find_shortest_paths(source, dest, edges, path_limit):
    G = nx.Graph()

    for edge in edges:
        G.add_edge(*edge)
    
    paths = []
    shortest_path_len = float("inf")
    for p in nx.shortest_simple_paths(G, source=source, target=dest):
        shortest_path_len = min(shortest_path_len, len(p))
        if len(p) > shortest_path_len: 
            break
        if len(paths) >= path_limit:
            break
        
        paths.append(p)
    
    return paths


class Gml_to_dot:
    def __init__(self, gml, outFile):
        # Get providers
        print("[Gml_to_dot] inFile: {}, outFile: {}".format(gml, outFile))
        self.write_gml_to_dot(gml, outFile)

    # def __call__(self, inFile, outfile):

    def reduce_range(self, nodes, links, link_capacity):
        return nodes, links, link_capacity

        # links and nodes are sets.

        nodes = sorted(nodes)
        links = sorted(links)

        node_map = {}  # map input node number to a 1 to len(nodes) range
        for n in range(len(nodes)):
            node_map[nodes[n]] = n + 1

        g_nodes = [node_map[n] for n in nodes]
        g_links = [(node_map[a], node_map[b]) for a, b in links]

        g_link_capacity = {
            (node_map[a], node_map[b]): link_capacity[(a, b)] for a, b in links
        }

        return g_nodes, g_links, g_link_capacity

    def mac_range(self, max):
        # https://stackoverflow.com/questions/8484877/mac-address-generator-in-python
        mac = "00:00:00:"
        mac_list = []
        for number in range(max):
            hex_num = hex(number)[2:].zfill(6)
            mac_list.append("{}{}{}:{}{}:{}{}".format(mac, *hex_num))
        return mac_list

    def write_dot_graph(self, nodes, links, link_capacity, name):
        nodes = list(nodes)
        links = list(links)
        switch_ips = [str(ip_address(a)) for a in range(len(nodes))]
        host_ips = [str(ip_address(a)) for a in range(2**16 + len(nodes))]
        mac_addrs = self.mac_range(len(nodes) * 2)
        # with open("./" + name + '.dot', 'w') as fob:
        with open(name, "w") as fob:
            fob.write("digraph topology {\n\n")
            for x in sorted(nodes):
                mac = mac_addrs.pop()
                ip = switch_ips.pop()
                fob.write(
                    's{0}\t[type=switch,id={0},mac="{1}",ip="{2}"];\n'.format(
                        x, mac, ip
                    )
                )

            fob.write("\n")
            for x in sorted(nodes):
                mac = mac_addrs.pop()
                ip = host_ips.pop()
                fob.write(
                    'h{0}\t[type=host,mac="{1}",ip="{2}"];\n'.format(
                        x, mac, ip
                    )
                )

            fob.write("\n")
            for x in sorted(nodes):
                capacity = max(
                    link_capacity.values()
                )  # ensure congestion happens "in network," not at hosts.
                fob.write(
                    'h{0} -> s{0}\t[src_port=0, dst_port=0, cost=1, capacity="{1}Gbps"];\n'.format(
                        x, capacity
                    )
                )
                fob.write(
                    's{0} -> h{0}\t[src_port=0, dst_port=0, cost=1, capacity="{1}Gbps"];\n'.format(
                        x, capacity
                    )
                )

            fob.write("\n")
            for a, b in sorted(links):
                try:
                    capacity = link_capacity[(a, b)]
                except:
                    for l in link_capacity:
                        print(l, link_capacity[l])
                    exit()

                fob.write(
                    's{0} -> s{1}\t[src_port={1}, dst_port={0}, cost=1, capacity="{2}Gbps"];\n'.format(
                        a, b, capacity
                    )
                )
                fob.write(
                    's{0} -> s{1}\t[src_port={1}, dst_port={0}, cost=1, capacity="{2}Gbps"];\n'.format(
                        b, a, capacity
                    )
                )

            fob.write("}")

    def write_gml_to_dot(self, gml, out_file):
        if type(gml) is str and path.isfile(gml):
            G = read_gml(gml)
        else:
            G = gml

        links = set(G.edges)
        nodes = set(G.nodes)

        try:
            link_capacity = get_edge_attributes(G, "capacity")
            assert link_capacity
        except AssertionError:
            link_capacity = {}
            for link in links:
                link_capacity[link] = 10

        vertices, edges, edge_capacity = self.reduce_range(
            nodes, links, link_capacity
        )

        self.write_dot_graph(vertices, edges, edge_capacity, out_file)


def _link_on_path(path, link):
    # function adapted from:
    # https://www.geeksforgeeks.org/python-check-for-sublist-in-list/
    # Check for Sublist in List
    # Using any() + list slicing + generator expression
    res = any(
        path[idx : idx + len(link)] == link
        for idx in range(len(path) - len(link) + 1)
    )
    return res


def link_on_path(path, link):
    l2 = [link[1], link[0]]
    return _link_on_path(path, link) or _link_on_path(path, l2)


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
        node_label_map = {
            i: str(j + 1) for (i, j) in zip(G.nodes(), range(len(G.nodes())))
        }
        G = nx.relabel_nodes(G, node_label_map)
        write_gml(G, new_graph_file)
        target_json_file = path.join(
            target_dir, "connected_" + target_base_file
        )

    pid = 0
    source_nodes = G.nodes()
    target_nodes = G.nodes()
    path_dict = DefaultDict(dict)
    for source in source_nodes:
        for target in target_nodes:
            if source != target:
                s_t_paths = nx.all_shortest_paths(G, source, target)
                for s_t_path in s_t_paths:
                    path_dict["path{}".format(pid)]["src"] = "s{}".format(
                        source
                    )
                    path_dict["path{}".format(pid)]["dst"] = "s{}".format(
                        target
                    )
                    path_dict["path{}".format(pid)]["nhop"] = len(s_t_path)
                    path_dict["path{}".format(pid)]["hops"] = [
                        "s{}".format(node) for node in s_t_path
                    ]
                    pid += 1
    with open(target_json_file, "w") as fob:
        json.dump({"paths": path_dict, "npath": len(path_dict)}, fob, indent=4)
        print("Saved JSON Paths to: {}".format(target_json_file))

    return target_json_file, len(G.nodes)


def convert_paths_onset_to_json(source_file, target_file):
    paths = {}
    with open(source_file, "r") as fob:
        path_tag = 0
        for line in fob:
            if line.startswith("h"):
                src, dst = line.split("->")
                src = src.strip()
                dst = dst.strip().strip(":").strip()

            if line.startswith("["):
                path_id = "path{}".format(path_tag)
                paths[path_id] = {}
                paths[path_id]["src"] = src.replace("h", "s")
                paths[path_id]["dst"] = dst.replace("h", "s")
                # print(paths[path_id]["src"], paths[path_id]["dst"])

                # print(line.strip())
                hops = line.split("@")[0]  # ignore everything after '@'
                hops = hops.split("),")[
                    :-1
                ]  # only look at core hops, not edge.
                hop_nodes = [hop.split(",")[-1] for hop in hops]
                paths[path_id]["nhop"] = len(hop_nodes)
                paths[path_id]["hops"] = hop_nodes
                path_tag += 1

    with open(target_file, "w") as fob:
        json.dump({"paths": paths, "npath": len(paths)}, fob, indent=4)


def read_json_graph(input_file):
    with open(input_file, "r") as fob:
        jobj = json.load(fob)

    G = nx.adjacency_graph(jobj, multigraph=False)
    return G


def parse_edges(path):
    path = path.strip().strip("[]")
    edges = path.split(", ")[1:-1]
    edge_list = []
    for e in edges:
        nodes = e.strip("()")
        a, b = nodes.split(",")
        # if ZERO_INDEXED: # zero indexed nodes
        #     a = str(int(a.strip('s')) - 1)
        #     b = str(int(b.strip('s')) - 1)

        # else: # one indexed nodes
        a = a.strip("s")
        b = b.strip("s")

        edge_list.append((a, b))

    return edge_list


def write_gml(G, name):
    with open(name, "w") as fob:
        fob.write("graph [\n")
        for node in sorted(G.nodes()):
            id = node.strip("s")
            fob.write("\tnode [\n")
            fob.write("\t\tid {}\n".format(id))
            fob.write('\t\tlabel "{}"\n'.format(node))
            for key, value in G.nodes[node].items():
                fob.write("\t\t{} {}\n".format(key, value))
            fob.write("\t]\n")
        for s, t in G.edges():
            src = s.strip("s")
            dst = t.strip("s")
            fob.write("\tedge [\n")
            fob.write("\t\tsource {}\n".format(src))
            fob.write("\t\ttarget {}\n".format(dst))
            for key, value in G.edges[s, t].items():
                fob.write("\t\t{} {}\n".format(key, value))
            fob.write("\t]\n")
        fob.write("]")
        return


def is_subpath(a, b, input_path, distance=1):
    # Returns True if 'a' and 'b' are a subpath of 'path' separated by a fixed 'distance' whose default is 1.
    if a in input_path and b in input_path:
        first = input_path.index(a)
        second = input_path.index(b)
        if first + distance == second:
            return True
    return False
