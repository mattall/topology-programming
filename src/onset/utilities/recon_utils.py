import json
import random
import ipaddress

import networkx as nx

from os import path

from onset.utilities.sysUtils import save_raw_data
from onset.utilities.plotters import new_figure
from onset.constants import PLT_BASE
from onset.constants import PLT_HIGHT
from onset.constants import IPV4
from onset.constants import PLOT_DIR


def random_ipv4():
    """
    Generates random IPV4 address.
    """
    ip_addr = ipaddress.IPv4Address(random.randint(0, IPV4))
    return str(ip_addr)


def random_ipv4_address_space(prefix=24):
    """
    Generates random IPV4 address space.
    """
    ip_addr = ipaddress.IPv4Address(random.randint(0, IPV4))
    interface = ipaddress.IPv4Network(
        f"{ip_addr}/{prefix}", strict=False
    )  # setting strict=False will mask
    return str(interface)


def random_ipv4_interface(ip_address_space):
    """
    Input:  ip_address_space: IPV4 ip_address_space. (e.g '10.0.1.0/24')
    Output: Random IPV4 address that is in the same subnet as ip address space.
    """
    network = ipaddress.ip_network(ip_address_space)
    # print(network)
    # print(list(network.hosts()))
    host = str(random.choice(list(network.hosts())))
    return host

def fdN_plt(G, output_file="fdN_plt"):
    """
    G: Graph to get node flow density of.
    output_file: plot output file name.
    Outputs plot for flow density for each node.
    """
    
    output_file = path.join(PLOT_DIR, output_file)
    fdNs = nx.get_node_attributes(G, "fdN")
    fdN_sorted = {
        node: node_fdN
        for node, node_fdN in sorted(
            fdNs.items(), key=lambda fdN: fdN[1], reverse=True
        )
        if "client" not in node
    }
    labels = nodes = list(fdN_sorted.keys())
    nodes_fdN = list(fdN_sorted.values())
    fig, ax = new_figure(scale=6)
    ax.bar(range(len(nodes)), nodes_fdN)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Logical Nodes")
    ax.set_ylabel("Flow Density")
    ax.set_title(f"Leakage: {max(nodes_fdN) - min(nodes_fdN)}")
    fig.savefig(output_file + ".jpg")
    fig.savefig(output_file + ".pdf")
    fig.clf()
    save_raw_data(labels, nodes_fdN, output_file,"Interface", "Flow Density")
    json.dump(fdN_sorted, open(f"{output_file}.json", "w"), indent=4)

    return None

def calculate_fdL(G, flows):
    """
    flows (List[tuple]): [(src, dest, tracing_flows), ...]
    Output: Links flow density.
    """

    links = G.edges

    for src, dest, tracing_flows in flows:
        shortest_paths = list(nx.all_shortest_paths(G, src, dest))
        num_shortest_paths = len(shortest_paths)
        path_tf = tracing_flows / num_shortest_paths
        for path in shortest_paths:
            path_len = len(path)
            for i in range(path_len - 1):
                link = (path[i], path[i + 1])
                try:
                    # Maybe a check needed to see if L3/virtual link or not.
                    links[link]["fdL"][
                        str(link)
                    ] += path_tf  # networkx doesn't like tuples as keys when exporting so turn it

                except (
                    KeyError
                ) as e:  # set tracing flow for link when there isn't one.
                    links[link]["fdL"][
                        str(link)
                    ] = path_tf  # networkx doesn't like tuples as keys when exporting so turn it into string.

    return nx.get_edge_attributes(G, "fdL")

