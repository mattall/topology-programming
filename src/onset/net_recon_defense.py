"""
Network obfuscation algorithm.
Author: Matt Nance-Hall

Context:
    -A network's base topology, G, is a collections Nodes and Links.
    -The routing topology is the set of paths for network flows on G.
    -The rank of a link, r_l, is the proportion of network flows that traverse it.
    -We can transform the base topology from G to G' with optical circuit switching.
    -The operation also changes the  r_l for all links.

Threat:
    -An adversary that maps the network's routing topology can target critical links 
    with high volume flows and disrupt connectivity for benign users.

Goal:
    -Increase difficulty of mapping the routing topology for an adversary.

Method:
    -At random points in time change the base topology with random addition or removal of 
    links such that utility of the network is not affect for benign users.

Detail:
    -Initialize the Reconfigurable Link Set as an empty set which will later contain node pairs.
    -Calculate link rank, r_l, for all links in the base topology.
    -Choose a random cutoff and add all links with r_l greater than the cutoff to the Reconfigurable Link Set.
    -Add pairs of nodes who are adjacent to links in the Reconfigurable Link Set to the set too.
    -At a random point in time, generate a "proposed update" by choosing a random subset form the Reconfigurable Link Set.
    -Simulate current traffic with the "proposed update" and if traffic performance is acceptable, push the change by 
    adding/removing links from the chosen subset---The updated graph is G'.
    -If performance drops in the simulation choose a new subset for the "proposed update".

Evaluation:
    -Show that graphs G and G' are not homomorphic.
    -Show that flow density of nodes and links in G and G' are significantly different

    Possible tests for similarity of the distributions
        - Two-sample Kolmogorov–Smirnov test
        - Wilcoxon Rank Sum Test
        - Cramér–von Mises test

   Show the time required to find G'. 
    """

# builtins
from sys import argv
from itertools import combinations

# 3rd party
import networkx as nx
import numpy as np

# customs
from onset.utilities.flow_distribution import calc_flow_distribution
from onset.utilities.flows import tm_to_flows
from onset.simulator import Simulation


def adaptive_recon(G: nx.Graph, flows: list):
    """
    Returns a network topology that is a variation on the original topology.

    G : the network topology
    flows: traffic demand flows. Formatted as a list: [(client_i, client_j, demand), ...]

    returns: H, a new network topology.
    """

    sim = Simulation()
    reconfigurable_link_set = set()

    H, node_flow_density, link_flow_density = calc_flow_distribution(G, flows)

    for u, v in combinations(G.nodes(), 2):
        if u == v:
            continue


def main(argv):
    pass


if __name__ == "__main__":
    main(argv)
