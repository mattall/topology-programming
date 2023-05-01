# Created by MNH
# API for Link-flood Protection with Optical Functions (AlpWolf)
#
# Class definition of the API
#
# .

from cmath import log
from collections import defaultdict
from copy import copy
from itertools import combinations
from math import ceil

from networkx import Graph, read_gml, set_node_attributes, relabel_nodes
from networkx.algorithms.centrality.betweenness import edge_betweenness_centrality
from networkx.algorithms.centrality.betweenness import betweenness_centrality
from onset.utilities.logger import logger

# from pprint import pprint


class AlpWolf():
    """API for Link Flood Protection with Optical Layer Functions (ALPWOLF) 
    https://www.merriam-webster.com/dictionary/fallow fallow adjective (2) 
    Definition of fallow (Entry 4 of 4) 
        1: left untilled or unsown after plowing 
        2: DORMANT, INACTIVE —used especially in the phrase to lie fallow 
        at this very moment there are probably important inventions lying 
        fallow — Harper's Ferry
    """

    def __init__(self, base_graph_file:str, fallow_transponders=5, fallow_tx_allocation_strategy='static', fallow_tx_allocation_file="") -> None:
        """Base fiber topology used for simulation. Tracks lambda allocations between node pairs with from transponders.    
        Args:
            base_graph_file (str): path to file containing base network topology
            fallow_transponders (int, optional): Number of fallow transponders at each node. Defaults to 5.
            fallow_tx_allocation_strategy (str, optional): Describes how to allocate transponders. Defaults to 'static'.
                can also be "dynamic" or "file". If it is "dynamic" then the top 90th percentile nodes get 'fallow_transponders' and others get 'fallow_transponders/2'.
            fallow_tx_allocation_file (str, optional): _description_. Defaults to "".
                Used when fallow_tx_allocation_strategy = "file", contains a path to a file that explicitly states the number of fallow transponders per node. 
            
        """        
        self.circuit_bandwidth = 100  # Gb/s
        self.transponders_per_degree = 1
        self.base_graph_file = base_graph_file
        self.base_graph = self.import_gml_graph(base_graph_file)
        self.logical_graph = Graph()
        self.n_nodes = len(self.base_graph.nodes())
        # node-pair (u,v) -> num_circuits (int)
        self.circuits = defaultdict(int)
        self.fallow_transponders = fallow_transponders
        self.fallow_tx_allocation_strategy = fallow_tx_allocation_strategy
        self.fallow_tx_allocation_file = fallow_tx_allocation_file
        self.n_super_nodes = ceil(self.n_nodes * 0.1)
        self.commands = {
            'list nodes': 'shows all nodes',
            'list links': 'shows all point-to-point fiber links',
            'list circuits': 'lists the circuits and their bandwidth',
            'list transponders u': 'shows the transponders at node u, their status (service or idle)',
            'get bandwidth u v': 'shows bandwidth on the (u, v) link',
            'add circuit u v': 'creates a new circuit from u to v.',
            'drop circuit u v': 'removes a circuit from u to v.',
            'help': 'show this message.',
            'CTRL+C': 'quit',
        }
        self._init_transponders()
        self._init_position()
        
    
    def import_gml_graph(self, path):#, label=None, destringizer=None):
        # MOVING THIS TO UTILITIES
        # self.G = read_gml(path, label, destringizer)
        self.G = read_gml(path, label="id")
        # Note: Because of something weird in read_gml, must remake node IDs into strings manually.
        if min(self.G.nodes)==0:
            node_to_str_map = {node: str(node+1) for (node) in self.G.nodes}
            # node_to_str_map = {node: ("sw" + str(node)) for (node) in self.G.nodes}
            relabel_nodes(self.G, node_to_str_map, copy=False)

        # nx.relabel_nodes(self.G, )
        # nx.relabel_nodes(self.G, lambda x: _sanitize(x))
        position = {}
        for node in self.G.nodes():
            try:
                position[node] = (self.G.nodes()[node]['Longitude'], self.G.nodes()[node]['Latitude'])
            except KeyError:
                from numpy.random import random
                logger.error("Key Error for position of node. Generating Random position label")
                long, lat = [int(x) for x in random(2)*100]
                logger.info("Setting (Longitude, Latitude) of Node, {}, to ({}, {})".format(node, long, lat))
                (self.G.nodes()[node]['Longitude'], self.G.nodes()[node]['Latitude']) = long, lat
                position[node] = (self.G.nodes()[node]['Longitude'], self.G.nodes()[node]['Latitude'])
            
        set_node_attributes(self.G, position, 'position')
        color = {}
        for node in self.G.nodes():
            color[node] = 'blue'
            
        set_node_attributes(self.G, color, 'color')
        return self.G
    
    def _init_transponders(self):
        """Initializes transponders in the input graph.

        Transponder count is parameterized based on the degree of the node plus a fixed
        number of fallow transponders.

        Each transponder ID for a node is initially set to -1, unassigned.

        Each for getAvailableTransponder and addCircuit to handle mapping of transponders to 
        pair nodes on each circuit. 

        Raises:
            Exception: If it cannot add a circuit due to transponder constraints then it fails. 
        """
        # The transponder node-attribute is a dictionary (transponder ID -> node). node -1 implies 
        # transponder is unassigned.
        # If the transponder is not in use the ID maps to the node that houses that transponder.

        # first give each node_n the prescribed number of transponders.
        if self.fallow_tx_allocation_strategy == "static":
            for node_n in self.base_graph.nodes:
                self.base_graph.nodes[node_n]['transponder'] = {}
                transponder_count = self.transponders_per_degree * \
                    self.base_graph.degree(node_n) + self.fallow_transponders
                for i in range(transponder_count):
                    self.base_graph.nodes[node_n]['transponder'][i] = -1

        elif self.fallow_tx_allocation_strategy == "dynamic":
            btwness = betweenness_centrality(self.base_graph)
            sorted_btwness = sorted(btwness, key=btwness.get, reverse=True)
            super_nodes = sorted_btwness[:self.n_super_nodes]
            for node_n in self.base_graph.nodes:                
                self.base_graph.nodes[node_n]['transponder'] = {}
                if node_n in super_nodes:
                    logger.info("Node {} is a super node.".format(node_n))
                    transponder_count = int(self.transponders_per_degree * self.base_graph.degree(node_n) \
                                        + self.fallow_transponders)
                else: 
                    transponder_count = int(self.transponders_per_degree * self.base_graph.degree(node_n) \
                                        + self.fallow_transponders / 2)
                for i in range(transponder_count):
                    self.base_graph.nodes[node_n]['transponder'][i] = -1
        
        elif self.fallow_tx_allocation_strategy == "file":
            ftx_alloc_dict = defaultdict(int)
            
            # read number of fallow fallow transponders for each node from the file
            with open(self.fallow_tx_allocation_file, 'r') as fob:
                for line in fob: 
                    node_n, num_ftx = line.strip().split(',')
                    ftx_alloc_dict[node_n] = int(num_ftx)
                    
            for node_n in self.base_graph.nodes:
                # node_n gets as many transponders needed to connect to adjacent nodes 
                #   : self.transponders_per_degree * (self.base_graph.degree(node_n)
                # plust as many as needed to make connections based on ftx_alloc_dict
                #   : self.transponders_per_degree * ftx_alloc_dict[node_n].
                self.base_graph.nodes[node_n]['transponder'] = {}
                transponder_count = self.transponders_per_degree * \
                    (self.base_graph.degree(node_n) + ftx_alloc_dict[node_n])
                for i in range(transponder_count):
                    self.base_graph.nodes[node_n]['transponder'][i] = -1
        
        else:
            raise("Undefined")
        
        for u, v in self.base_graph.edges:
            for i in range(self.transponders_per_degree):
                trans_u = self.get_available_transponder(
                    self.base_graph.nodes[u]['transponder'])
                trans_v = self.get_available_transponder(
                    self.base_graph.nodes[v]['transponder'])
                if trans_u >= 0 and trans_v >= 0:
                    self.add_circuit(u, v, trans_u, trans_v)
                else:
                    raise Exception(
                        "Error, insufficient transponders at nodes for circuit.")

    def _init_position(self):
        """Sets Longitude and Latitude for nodes based on the input graph.
        """
        for n in self.base_graph.nodes():
            node_n = self.base_graph.nodes[n]
            self.logical_graph.add_node(
                n, Longitude=node_n["Longitude"], Latitude=node_n["Latitude"])

    def get_available_transponder(self, t_dict: dict) -> int:
        """Returns an index to an available transponder in the transponder dictionary (t_dict)
        If no transponder is available, returns -1. 

        Args:
            t_dict (dict): Transponder attribute of a particular node.

        Returns:
            int: ID of transponder on node that is unassigned. 
        """
        for transponder in t_dict:
            assignment = t_dict[transponder]
            if assignment == -1:  # transponder is unassigned.
                return transponder
        return -1

    def get_peer_transponder(self, t_dict: dict, node_n: int) -> int:
        """Returns an index to a transponder in t_dict paired to the node_n
        If no transponder is paired to node_n, returns -1. 

        Args:
            t_dict (dict): Transponder attribute of a particular node.
            node_n (int): Node for which we look for a connection in t_dict

        Returns:
            int: ID of transponder on paired to node_n or -1 if none are paired to it. 
        """
        for transponder in t_dict:
            assignment = t_dict[transponder]
            if assignment == node_n:  # transponder is unassigned.
                return transponder
        return -1

    def can_add_circuit(self, u: str, v: str, transponder_u: int = None, transponder_v: int = None) -> bool:
        logger.info("Testing if we can add circuit {} {}.".format(u, v))
        if transponder_u == None or transponder_v == None:
            transponder_u = self.get_available_transponder(
                self.base_graph.nodes[u]['transponder'])

            transponder_v = self.get_available_transponder(
                self.base_graph.nodes[v]['transponder'])

        if transponder_u == -1 or transponder_v == -1:
            logger.info("Cannot add circuit. Transponder pair unavailable")
            return False
        return True

    def add_circuit(self, u: str, v: str, transponder_u: int = None, transponder_v: int = None) -> int:
        """Adds a circuit between nodes u and v.

        Throws an assertion error if the transponder indices given are anything but -1 (i.e., they
        are already assigned).

        Updated transponders at both nodes to map to eachother. 

        Adds 1 to circuit count for each direction of the circuit.

        Add an edge connecting the circuits to the logical graph if it doesn't exist, and increases
        the bandwidth by circuit_bandwidth.

        Args:
            u (str): node descriptor
            v (str): node descriptor
            transponder_u (int, optional): Transponder index in u that is unassigned. Defaults to None.
            transponder_v (int, optional): Transponder index in v that is unassigned. Defaults to None.

        Returns: 
            0 on success.
            -1 of failure.
        """
        # logger.info("Adding circuit {} {}.".format(u, v))
        if transponder_u == None or transponder_v == None:
            transponder_u = self.get_available_transponder(
                self.base_graph.nodes[u]['transponder'])

            transponder_v = self.get_available_transponder(
                self.base_graph.nodes[v]['transponder'])

        if transponder_u == -1 or transponder_v == -1:
            logger.error(f"Couldn't add circuit ({u}, {v}). Transponder pair unavailable")
            return -1

        # Make sure transponders are currently unassigned.
        assert self.base_graph.nodes[u]['transponder'][transponder_u] == -1
        assert self.base_graph.nodes[v]['transponder'][transponder_v] == -1

        # Assign them to each other
        self.base_graph.nodes[u]['transponder'][transponder_u] = v
        self.base_graph.nodes[v]['transponder'][transponder_v] = u

        self.circuits[(u, v)] += 1
        # ensure opposite pairing has same number of circuits.
        self.circuits[(v, u)] = self.circuits[(u, v)]

        # update logical graph
        if (u, v) in self.logical_graph.edges:
            self.logical_graph[u][v]['capacity'] += self.circuit_bandwidth
        else:
            self.logical_graph.add_edge(u, v)
            self.logical_graph[u][v]['capacity'] = self.circuit_bandwidth

        logger.info("Successfully added circuit {} {}.".format(u, v))
        return 0

    def drop_circuit(self, u: str, v: str, transponder_u=None, transponder_v=None):
        """Remove the u, v circuit if it exists.

        Throws assertion error if the transponders for the nodes do not map correctly,
        or if the circuit bandwidth is 0 or less. 

        Sets transponder assignments for dropped circuit to -1 (unassigned) at each node. 
        Subtracts 1 from circuit count for the circuit in both directions. 

        Removes the logical edge connecting the two nodes if the circuit bandwidth is reduced
        to 0.

        Args:
            u (str): node descriptor
            v (str): node descriptor
            transponder_u (int, optional): transponder index in u that maps to v. Defaults to None.
            transponder_v (int, optional): transponder index in v that maps to u. Defaults to None.
        """
        logger.info("Dropping circuit {} {}.".format(u, v))
        if self.circuits[(u, v)] == 0:
            logger.info("Cannot drop Circuit {} {} - does not exist.".format(u, v))
            return 0

        if transponder_u == None or transponder_v == None:
            transponder_v = self.get_peer_transponder(
                self.base_graph.nodes[v]['transponder'], u)
            transponder_u = self.get_peer_transponder(
                self.base_graph.nodes[u]['transponder'], v)

        # Verify output. V's transponder should point to U...
        assert self.base_graph.nodes[v]['transponder'][transponder_v] == u, "While dropping circut, node {}'s transponder did not map to {}. Instead it mapped to {}".format(v, u, self.base_graph.nodes[v]['transponder'][transponder_v])
        assert self.base_graph.nodes[u]['transponder'][transponder_u] == v, "While dropping circut, node {}'s transponder did not map to {}. Instead it mapped to {}".format(u, v, self.base_graph.nodes[u]['transponder'][transponder_u])
        assert self.circuits[(u, v)] > 0, "No circuit to drop between {} and {}".format(u, v)

        # Drop assignments, reassigning transponder to -1.
        self.base_graph.nodes[v]['transponder'][transponder_v] = -1
        self.base_graph.nodes[u]['transponder'][transponder_u] = -1

        self.circuits[(u, v)] -= 1
        # ensure opposite pairing has same number of circuts.
        self.circuits[(v, u)] = self.circuits[(u, v)]

        # update logical graph
        self.logical_graph[u][v]['capacity'] -= self.circuit_bandwidth
        assert self.logical_graph[u][v]['capacity'] >= 0, "Error - drop_circuit - capacity cannot be negative!"
        if self.logical_graph[u][v]['capacity'] == 0:
            self.logical_graph.remove_edge(u, v)
            del self.circuits[(u, v)]
            del self.circuits[(v, u)]

        logger.info("Successfully dropped circuit {} {}.".format(u, v))
        return 0

    def cli_help(self):
        """Prints the available cli commands.
        """
        logger.info("Commands:\n" + "\n\t".join(self.commands))

    def list_nodes(self):
        """Lists all of the network nodes.
        """
        logger.info(", ".join([str(i) for i in list(self.base_graph.nodes())]))

    def get_nodes(self):
        """Return list of network nodes.
        """
        return list(self.base_graph.nodes())

    def list_links(self):
        """Lists all of the physical links in the base graph. 
        """
        for e in self.base_graph.edges():
            logger.info(e)

    def get_links(self):
        """Return the physical links in the base graph. 
        """
        return self.base_graph.edges()

    def get_candidate_circuits(self, candid_set="all", k=0, l=0) -> list:
        """Returns the node pairs that can currently establish a circuit

        Args:
            candid_set (str: 'all', 'ranked'): 
                all: Return all combinations of non-adjacent node pairs.
                ranked: Return 'k * l' node pairs based on centrality metrics.
                          First, the k-most central links in the graph are found.
                          Then, for each link, a pair of nodes marking the l-most
                          common subpath on the link are returned.
            k (int > 0), required if candidate_set='ranked': k-most shared links.
            l (int > 0), required if candidate_set='ranked': l-most shared subpaths 
                                                         on each link.

        Returns:
            list: list of candidate circuits
        """
        assert k < len(self.logical_graph.edges()), "Error, More candidate links than total links in network."
        if self.fallow_transponders == 0:
            return []
            
        if candid_set == "all":
            logger.info("getting list of candidate circutes.")
            candidates = []
            for u, v in combinations(self.get_nodes(), 2):
                if (u, v) not in self.circuits:
                    if self.can_add_circuit(u, v):
                        candidates.append((u, v))
            return candidates
        elif candid_set == "ranked":
            def choose_candidates(adjacent_links: list, target_link: tuple, l: int, candidate_link_set: set, current_link_set:set) -> list:
                """ Creates a list of candidate links from a target link and adjacent link. 
                Example: 
                    adjacent_links = [('Cheyenne', 'Boulder', 0.18181818181818182)
                                    ('Stockton', 'Washington, DC', 0.14393939393939395)
                                    ('Cheyenne', 'Kansas City', 0.13939393939393938)
                                    ('Stockton', 'Anaheim', 0.13333333333333333)
                                    ('Stockton', 'Seattle', 0.13030303030303028)
                                    ('Stockton', 'New York (Pennsauken)', 0.08181818181818182)
                                    ('Stockton', 'Chicago', 0.07121212121212121)]
                    target_link = ('Cheyenne', 'Stockton')
                    l = 5
                    Returns: [('Stockton', 'Boulder')
                            ('Cheyenne', 'Washington, DC'),
                            ('Stockton', 'Kansas City'),
                            ('Cheyenne', 'Anaheim'),
                            ('Cheyenne', 'Seattle')]
                Args:
                    adjacent_links (list): [(s:str, t:Str, betweenness:float)]. Assumes 
                                        either 's' or 't' is a node in target link.
                                        This link is therefore adjacent to the targetlink. 
                                        List is given sorted by betweenness. Betweenness 
                                        is ignored by this function.
                    target_link (tuple): (u:str, v:str). Target link is assumed adjacent to 
                                        each adjacent link.
                    l (int): Max number of candidate links to return.
                Returns:
                    list: [(candidate_a:str, candidate_b:str)]. list of tuples for candidate links. 
                        list tuples each have a node from adjacent links and a node
                        from target link. The target link in the pair is the one that was
                        NOT included with the original adjacent link. 
                """
                candidates = []
                for s, t, btwness in adjacent_links:
                    if s in target_link:
                        candidate_a = target_link[0] if s == target_link[1] else target_link[1]
                        candidate_b = t
                    elif t in target_link:
                        candidate_a = target_link[0] if t == target_link[1] else target_link[1]
                        candidate_b = s
                    else:
                        raise Exception("Error. Got a list of 'adjacent links' where ({}, {}) was not adjacent to {}.".format(
                            s, t, target_link))
                    if (candidate_a, candidate_b) not in (candidate_link_set | current_link_set): # ONLY ADD NEW LINKS 
                        if (candidate_b, candidate_a) not in (candidate_link_set | current_link_set):
                            candidates.append((candidate_a, candidate_b))
                    if len(candidates) == l:
                        return candidates
                return candidates

            candidates = []
            assert k > 0 and l > 0, "Error: candidate_set 'ranked' requires 'k' and 'l' > 0"
            btwness = edge_betweenness_centrality(self.base_graph)
            btwness_popable = copy(btwness)
            for _ in range(k):
                most_central = max(btwness_popable, key=btwness_popable.get)
                btwness_popable.pop(most_central)
                adjacent_links = []
                for node in most_central:
                    for neighbor in self.base_graph[node]:
                        if neighbor in most_central:
                            continue
                        if (node, neighbor) in btwness:
                            node_neighbor_btwness = btwness[node, neighbor]
                        else:
                            node_neighbor_btwness = btwness[neighbor, node]

                        adjacent_links.append(
                            (node, neighbor, node_neighbor_btwness))

                adjacent_links = sorted(
                    adjacent_links, key=lambda x: x[-1], reverse=True)
                candidates.extend(choose_candidates(
                    adjacent_links, most_central, l, set(candidates), set(self.logical_graph.edges)))
            return candidates

    def get_circuits(self):
        """Return all circuits and their bandwidth in Gb/s
        """
        return self.circuits

    def list_circuits(self):
        """Lists all circuits and their bandwidth in Gb/s
        """
        for c in sorted(self.circuits):
            logger.info("{}: {}  Gb/s".format(c, self.circuits[c]))

    def get_bandwidth(self, u, v) -> int:
        """Returns the bandwidth between a pair of hosts

        Args:
            u (str): maps to a node in the logical network graph. 
            v (str): maps to a node in the logical network graph. 

        Returns:
            int: Bandwidth between the nodes, Gb/s
        """
        return self.logical_graph[u][v]['capacity']

    def list_transponders(self, node_n):
        """Lists the transponders and their assignment for a node_n.
        Transponders are a dictionary object, where transponder IDs 
        map to other nodes. A mapping to -1 indicated that transponder
        is unassigned.

        Args:
            node_n (str): a node descriptor.
        """
        logger.info(self.base_graph.nodes[node_n]['transponder'])

    def cli(self):
        """Starts the AlpWolf command line interface for issuing topology change commands.
        """
        logger.info("Starting client session command. Hit CTRL+C to end.")
        self.cli_help()

        while True:
            try:
                command = input('> ')
                if command == 'list nodes':
                    self.list_nodes()

                elif command == 'list links':
                    self.list_links()

                elif command.startswith('list circuits'):
                    self.list_circuits()

                elif command.startswith('list transponders'):
                    comm_args = command.split()
                    if len(comm_args) == 3:
                        node_n = comm_args[2]
                        if node_n in self.base_graph.nodes():
                            self.list_transponders(node_n)
                        else:
                            logger.error("Node  {} not found.".format(node_n))
                    else:
                        logger.error("Invalid command.")

                elif command.startswith('get bandwidth'):
                    comm_args = command.split()
                    if len(comm_args) == 4:
                        u, v = comm_args[2], comm_args[3]
                        if (u, v) in self.logical_graph.edges():
                            logger.info(
                                "{} {} bandwidth: {} Gb/s".format(u, v, self.get_bandwidth(u, v)))
                        else:
                            logger.error("Edge {} not found.".format((u, v)))
                    else:
                        logger.error("Invalid command.")

                elif command.startswith('add circuit'):
                    # verify transponders available at both ends.
                    comm_args = command.split()
                    if len(comm_args) == 4:
                        u, v = comm_args[2], comm_args[3]
                        if u not in self.base_graph.nodes():
                            logger.error("Node  {} not found.".format(u))
                        elif v not in self.base_graph.nodes():
                            logger.error("Node  {} not found.".format(v))
                        else:
                            self.add_circuit(u, v)

                    else:
                        logger.error("Invalid command.")

                elif command.startswith('drop circuit'):
                    comm_args = command.split()
                    if len(comm_args) == 4:
                        u, v = comm_args[2], comm_args[3]
                        if (u, v) in self.circuits:
                            self.drop_circuit(u, v)
                        else:
                            logger.error(
                                "Circuit {} not found.".format((u, v)))
                    else:
                        logger.error("Invalid command.")

                elif command.startswith('help'):
                    self.cli_help()

                else:
                    logger.error("Invalid command.")

            except KeyboardInterrupt:
                logger.info("Ending session")
                break
