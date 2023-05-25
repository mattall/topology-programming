import json
import random
import ipaddress

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from os import path

from onset.utilities.sysUtils import save_raw_data
from onset.constants import PLT_BASE
from onset.constants import PLT_HIGHT
from onset.constants import IPV4
from onset.constants import PLOT_DIR

plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "font.size": 26,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 2,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.minor.size": 6,
        "ytick.minor.size": 6,
        "xtick.minor.width": 2,
        "ytick.minor.width": 2,
        "lines.linewidth": 2,
    }
)

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


def new_figure(scale=1):
    return plt.subplots(figsize=(scale * PLT_BASE, scale * PLT_HIGHT))


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


def cdf_plt(
    distribution,
    xlabel="X",
    output_file="cdf",
    complement=False,
    clear_fig=True,
    fig=False,
    ax=False,
    label="",
):
    output_file = path.join(PLOT_DIR, output_file)
    s = float(distribution.sum())
    cdf = distribution.cumsum(0) / s
    # sort the data:
    data_sorted = np.sort(distribution)

    # calculate the proportional values of samples
    p = 1.0 * np.arange(len(distribution)) / (len(distribution) - 1)

    if fig and ax:
        pass
    else:
        fig, ax = new_figure(scale=3)

    if complement:
        ccdf = 1 - p
        ylabel = xlabel
        xlabel = "CCDF"
        X = ccdf
        Y = data_sorted
        ax.set_xlim([0, 1])
        # ax.set_ybound(lower=0)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        # ax.set_yticks([0, 10, 20, 30])
        save_raw_data(X, Y, output_file, ylabel, xlabel)
    else:
        ylabel = "CDF"
        Y = cdf
        X = data_sorted
        ax.set_ylim([0, 1])
        save_raw_data(X, Y, output_file, xlabel, ylabel)

    ax.plot(X, Y, label=label)
    ax.grid()
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # ax.set_yticks([0, 10, 20, 30])
    fig.savefig(output_file + ".jpg")
    fig.savefig(output_file + ".pdf")
    if clear_fig:
        plt.clf()
        return None
    else:
        return fig, ax
    
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
