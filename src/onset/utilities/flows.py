import json
from numpy import load, loadtxt, sqrt
from os import path
from onset.utilities.graph import read_json_graph

import networkx as nx

import random
from itertools import product

from onset.utilities.logger import logger


def generate_flows(file_path, min_tf, max_tf):
    """
    file_path (str): path to topology file.
    min_tf (int|float): minimum amount of tracing flows.
    max_tf (int|float): maximum amount of tracing flows.
    Output: flow_li (List[tuple]): [(src, dest, tracing flows), ...]
    NOTE could be improved to be better.
    """
    flows = []
    print("Generating flows for undirected graph.")

    if file_path.endswith(".gml"):
        G = nx.read_gml(file_path, label="id")
    elif file_path.endswith(".json"):
        G = read_json_graph(file_path)
    else:
        raise BaseException("Expected file to be .gml or .json")

    nodes = G.nodes()
    for i, j in product(nodes, repeat=2):
        if i == j:
            continue
        n_flows = random.randint(min_tf, max_tf)
        flows.append((f"client_{i}", f"client_{j}", n_flows))

    return flows


def tm_to_flows(tm_path):
    """
    file_path (str): path to topology file.
    Output: flow_li (List[tuple]): [(src, dest, tracing flows), ...]
    """
    dir_name = path.dirname(tm_path)
    base_name_no_ext = path.splitext(tm_path)[0]
    flows_file = path.join(dir_name, base_name_no_ext + "_flows.txt")

    flows = []

    tm = load(tm_path)
    for i, j in tm:
        if tm[i][j] > 0:
            flows.append((f"client_{i}", f"client_{j}", tm[i][j]))

    return flows


# Read data from traffic matrix
def read_tm_to_tc(
    tc: dict, tm: list, paths: dict, malicious: bool, rolling: int
):
    """
    Args:
        tc (dict): traffic classes object, can be empty.
        tm (list): traffic matrix assumed to be passed as a 1-D list
        paths (dict): paths object
        malicious (bool): traffic being read is malicious or not
        rolling (int): -1 if traffic is not malicious. Otherwise, which phase
            of attack is the traffic from.
    """
    tc_index = len(tc)
    dimension = int(sqrt(len(tm)))

    for entry in range(len(tm)):
        rate = int(tm[entry] // 1000)
        i = (entry // dimension) + 1
        j = (entry % dimension) + 1
        if i == j:
            continue

        dst = "s" + str(j)
        src = "s" + str(i)
        flow_paths = {}
        for path in paths:
            if paths[path]["src"] == src and paths[path]["dst"] == dst:
                flow_paths[path] = paths[path]

        n_paths = len(flow_paths)
        for flow_path in flow_paths:
            path_rate = rate // n_paths
            if path_rate == 0:
                continue

            tc["tc{}".format(tc_index)] = {
                "dst": dst,
                "malicious": malicious,
                "nhops": paths[flow_path]["nhop"],
                "rolling": rolling,
                "hops": paths[flow_path]["hops"],
                "src": src,
                "flow_rate": int(rate // n_paths),
            }
            tc_index += 1


def write_flows_to_json(
    base_traffic_matrix,
    attack_traffic_matrix,
    json_paths,
    out_file,
    targets,
    congestion_factor,
):
    # For each source, destination in the traffic matrix,
    #   map source, destinations to a (set of) path(s).

    # Load traffic matrix and path data.
    base_tm = loadtxt(base_traffic_matrix, dtype=int)
    attack_tm = loadtxt(attack_traffic_matrix, dtype=int)
    path_obj = json.load(open(json_paths, "r"))
    paths = path_obj["paths"]
    # Prepare flow object meta-data
    flow_obj = {}
    target_links = [str(t) for t in targets]
    flow_obj["effective cong"] = {
        target_link: congestion_factor for target_link in target_links
    }
    flow_obj["target_link"] = {
        "link{}".format(i): {
            "dst": "s{}".format(targets[i][1]),
            "src": "s{}".format(targets[i][0]),
        }
        for i in range(len(targets))
    }
    flow_obj["nlink"] = len(targets)
    tc = {}
    read_tm_to_tc(tc, base_tm, paths, False, -1)
    read_tm_to_tc(tc, attack_tm, paths, True, 1)
    flow_obj["traffic_class"] = tc
    with open(out_file, "w") as fob:
        json.dump(flow_obj, fob, indent=4)


def write_flows(flows, output_file="output_tracing_flows.csv"):
    """
    flows (List[tuple]): List of flow tuples.
    output_file (str, optional): output file. Default "output_flows.csv".
    """
    with open(output_file, "w") as fp:
        for flow in flows:
            fp.write(",".join(str(x) for x in flow))
            fp.write("\n")
    print(f"file written to: {output_file}")


def read_flows(flows_file):
    """
    flow_file (str): Path to flows_file.
    Output: List of flow tuples. [(src, dest, tracing_flows), ...]
    """
    flows = []
    with open(flows_file, "r") as fp:
        for line in fp:
            flow_str = line.strip().split(",")
            # flow = tuple(eval(val) for val in flow_str)
            flow = (flow_str[0], flow_str[1], eval(flow_str[2]))
            flows.append(flow)
    return flows


def sanitize_magnitude(mag_arg: str) -> int:
    # WARNING. This function has been moved to .src.utilities.tmg.
    # Use that version instead.
    """Converts input magnitude arg into an integer
        Unit identifier is the 4th from list character in the string, mag_arg[-4].
        e.g., 1231904Gbps G is 4th from last.
        this returns 1231904 * 10**9.
    Args:
        mag_arg (str): number joined with either T, G, M, or K.
    Returns:
        int: Value corresponding to the input.
    """

    mag = mag_arg[-4].strip()
    coefficient = int(mag_arg[0:-4])
    logger.debug(coefficient)
    logger.debug(mag)
    exponent = 0
    if mag == "T":
        exponent = 12
    elif mag == "G":
        exponent = 9
    elif mag == "M":
        exponent == 6
    elif mag == "k":
        exponent == 3
    else:
        raise (
            "ERROR: ill formed magnitude argument. Expected -m <n><T|G|M|k>bps, e.g., 33Gbps"
        )
    result = coefficient * 10**exponent
    logger.debug("Result: {}".format(result))
    return result
