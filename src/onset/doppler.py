# builtins
import os
from itertools import combinations, permutations, product
from collections import defaultdict
from multiprocessing import Pool
from copy import deepcopy
import json
import pickle

# third-party
from gurobipy import Model, GRB, quicksum, min_, max_, Env, tupledict
import networkx as nx
import numpy as np
from tqdm import tqdm
from time import process_time
from math import ceil

# customs
from onset.utilities.plot_reconfig_time import calc_haversine
from onset.utilities.graph_utils import link_on_path
from onset.constants import SCRIPT_HOME
from onset.utilities.logger import logger

class Link_optimization:
    '''
    idea/intent: 
        G is an instance of a logical network represented as an
        *undirected* graph.

        core_G is an instance of a physical network, represented as an 
        *undirected* graph. 

        G may change with successive calls to methods of the class, but core_G
        should never be modified. 
    '''
    def __init__(
        self,
        G: nx.Graph,
        demand_matrix_file: str,
        network: str,
        core_G=None,
        use_cache=False,
        parallel_execution=False,
        txp_count=None,
        BUDGET=1,
        compute_paths=False,
        candidate_set="max"
    ):
        self.debug = False
        self.G = deepcopy(G)
        self.all_node_pairs = list(permutations(self.G.nodes, 2))
        if isinstance(demand_matrix_file, str):
            self.demand_matrix_file = demand_matrix_file
            self.demand_matrix = np.loadtxt(demand_matrix_file)
            self.initialize_demand()

        elif isinstance(demand_matrix_file, dict):
            self.demand_matrix_file = None
            self.demand_matrix = None
            self.demand_dict = demand_matrix_file

        else:
            logger.info("unknown demand type")
            self.demand_matrix_file = None
            self.demand_matrix = None
            self.demand_dict = None

        self.network = network
        if isinstance(core_G, nx.Graph):
            self.core_G = core_G.copy(as_view=True)
            self.txp_count = [
                2 * len(core_G.nodes[x]["transponder"]) for x in core_G.nodes
            ]
        else:
            self.core_G = self.G.copy(as_view=True)
            self.txp_count = txp_count

        # Only proceed if instance had not set self.txp_count.
        if self.txp_count is None:
            # set up a default self.txp_count if one isn't passed.
            if txp_count is None:
                self.txp_count = [
                    (len(self.core_G[node]) + 1) for node in self.core_G.nodes
                ]
            # set to the passed value
            else:
                self.txp_count = txp_count

        if isinstance(txp_count, list):
            self.txp_count = txp_count

        if isinstance(txp_count, dict):
            self.txp_count = txp_count

        self.nodes = self.core_G.nodes
        self.use_cache = use_cache
        self.PARALLEL = parallel_execution
        self.k = 1
        self.MAX_DISTANCE = 5000  # km
        # self.MAX_DISTANCE = float("inf")  # km
        self.LINK_CAPACITY = 100 * 10**9  # bps
        self.BUDGET = BUDGET
        self.super_graph = nx.Graph()
        self.model = None
        self.flow_vars = None
        self.flow_paths = tupledict()
        self.candidate_set = candidate_set
        self.candidate_links = []
        self.tunnel_list = []
        self.links_to_add = []
        self.links_to_drop = []
        self.new_G = None
        self.tunnel_dict = defaultdict(lambda : defaultdict(list)) # map [s][t] -> list of tunnels [[s, u_1, t], [s, u_2, t], ...]
        self.original_tunnel_dict = defaultdict(lambda : defaultdict(list))
        self.original_tunnel_list = []
        self.core_shortest_path_len = defaultdict(lambda: defaultdict(lambda: np.inf)) #  map [s][t] -> (int) shortest path len 
        if self.demand_dict == None:
            self.initialize_demand()
        self.initialize_candidate_links()
        self.add_candidate_links_to_super_graph()
        if compute_paths:
            self.get_shortest_paths()

    def populate_changes(self, edge_vars):
        self.new_G = new_G = nx.Graph()
        for (u, v) in edge_vars: 
            if edge_vars[(u, v)].x == 1:
                new_G.add_edge(u, v)
            
        links_to_add = [e for e in new_G.edges if e not in self.G.edges]
        links_to_drop = [e for e in self.G.edges if e not in new_G.edges]
        assert nx.is_strongly_connected(new_G.to_directed(as_view=True))
        self.links_to_add = links_to_add
        self.links_to_drop = links_to_drop

        return 0
        
    def write_iss(self, verbose=False):
        m = self.model
        m.computeIIS()
        if verbose:
            for c in self.model.getConstrs():
                if c.IISConstr:
                    logger.info(f"\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}")
                for v in m.getVars():
                    if v.IISLB:
                        logger.info(f"\t{v.varname} ≥ {v.LB}")
                    if v.IISUB:
                        logger.info(f"\t{v.varname} ≤ {v.UB}")
        iis_file = f"logs/iismodel/{self.network}_{self.candidate_set}_{os.getpid()}.ilp"
        m.write(iis_file)
        logger.error(f"Model is infeasible. Wrote model IIS to {iis_file}.")

    def get_shortest_paths_from_k_link_super_graphs(self, k):
        # Run after finding candidate links
        # Given K, find every combination of k candidate links
        # For each combination, in parallel,
        #   1. Create a supergraph on the base topology
        #   2. Find the shortest paths - as short as or shorter than -
        #       the original shortest path for every pair of nodes.
        #       2.1 If the path includes any candidate links, save this path.
        #   3. Return the set of saved paths.
        logger.info("Getting work ready.")
        work = []
        count = 0
        for i in range(1, k + 1):
            link_sets = list(combinations(self.candidate_links, i))
            for c, link_set in enumerate(link_sets):
                count += 1
                work.append([count, 1, link_set, self.G, self.network])
        logger.info("re-indexing work.")
        for w in work:
            w[1] = len(work)
        # work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])
        logger.info("Starting the work.")
        p = Pool(64)
        p.starmap(work_log, work, chunksize=8)
        p.terminate()
        p.join()

        # modifies self.tunnel_list
        self.tunnel_list = self.original_tunnel_list[:]
        for w in work:
            self.load_path_list(w[0])

        for i, tunnel in enumerate(self.tunnel_list):
            source, target = tunnel[0], tunnel[-1]
            self.tunnel_dict[(source, target)].append(i)

        self.save_paths()

    def load_path_list(self, identifier):
        if self.use_cache:
            temp_file = "./.temp/{}.json".format(identifier)
            with open(temp_file, "r") as fob:
                obj = json.load(fob)

            os.remove(temp_file)
            self.tunnel_list.extend(obj["list"])
            return

        else:
            return

    def initialize_demand(self):
        #####################################################
        #### Read Traffic Matrix into Source/Destinations ###
        #####################################################
        """
        demand_dict: {(source, target): demand_value}
        """
        self.demand_dict = {}
        dim = int(np.sqrt(len(self.demand_matrix)))
        for (i, j), (source, dest) in zip(
            permutations(range(len(self.G.nodes)), 2), self.all_node_pairs
        ):
            self.demand_dict[(source, dest)] = float(self.demand_matrix[dim * i + j])

    def update_shortest_path_len(self, this_path_len, source, target):
        prev_path_len = self.core_shortest_path_len[source][target]
        self.core_shortest_path_len[source][target] = self.core_shortest_path_len[target][source] = min(prev_path_len, this_path_len)

    def initialize_candidate_links(self, p=0.1):
        """ Create list of candidate links a.k.a. Shortcuts

        Args:
            candidate_set (str, optional): describes set from which candidate links are chosen. Defaults to "max".
                max:        Any pair of vertices separated by self.MAX_DISTANCE or less and 2-hops away.
                    e.g., 
                        for (u, v) below, 

                        (u_1)--\             /--(v_1)      
                                \           /       
                        (u_2)----(u) ---- (v)
                                /           \ 
                        (u_3)--/             \--(v_2)

                        The candidate set of links is
                        (u, v_1),   (u, v_2)
                        (u_1, v),   (u_2, v),   (u_3, v)                        
                        (u_1, v_1), (u_2, v_1), (u_3, v_1)
                        (u_1, v_2), (u_2, v_2), (u_3, v_2)


                liberal:    Links from K_{N(u), N(v)} for all (u, v) top `p` percent of links, ranked by edge centrality.
                    N(u) and N(v) are the neighborhoods of u and v respectively, or all adjacent vertices to u and v
                    K_{N(u), N(v)} is the complete bi-partite graph connecting the neighborhoods of u and v.
                    e.g., 
                        from the neighborhood of (u, v) above, the candidate set of links is
                        (u_1, v_1), (u_2, v_1), (u_3, v_1)
                        (u_1, v_2), (u_2, v_2), (u_3, v_2)

                conservative:   Links from K_{N(u), v} and K_{u, v} for all (u, v) top `p` percent of links, ranked by
                    edge centrality.
                    Eg., 
                        from the neighborhood of (u, v) above, the candidate set of links is
                        (u_1, v), (u_2, v), (u_3, v)
                        (u, v_1), (u, v_2)
        """
        core_G = self.core_G.copy(as_view=True)
        candidate_set = self.candidate_set
        if core_G == None:
            core_G = deepcopy(self.G)

        if candidate_set == "max":
            candidate_links = [sorted(l) for l in core_G.edges]

            for source, target in self.all_node_pairs:
                if (source, target) not in core_G.edges and sorted(
                    (source, target)
                ) not in candidate_links:
                    shortest_paths = list(
                        nx.all_shortest_paths(core_G, source, target))
                    shortest_path_len = len(shortest_paths[0])
                    shortest_path_hops = shortest_path_len - 1
                    self.update_shortest_path_len(shortest_path_len, source, target)
                    if shortest_path_hops == 2:
                        hop_dist_ok = True
                    else:
                        hop_dist_ok = False

                    geo_dist_ok = False
                    min_geo_dist = float("inf")
                    for p in shortest_paths:
                        distance = 0
                        for u, v in zip(p, p[1:]):
                            if "Latitude" not in core_G.nodes[u]:
                                distance = 1
                                break
                            distance += calc_haversine(
                                core_G.nodes[u]["Latitude"],
                                core_G.nodes[u]["Longitude"],
                                core_G.nodes[v]["Latitude"],
                                core_G.nodes[v]["Longitude"],
                            )

                        min_geo_dist = min(min_geo_dist, distance)

                        if min_geo_dist <= self.MAX_DISTANCE:
                            geo_dist_ok = True
                            break

                    if not geo_dist_ok:
                        logger.info(f"Minimum distance, {min_geo_dist} km, "\
                                    f"too far between {source} and {target}")
                        
                    if geo_dist_ok and hop_dist_ok:
                        candidate_links.append(sorted((source, target)))

        elif candidate_set == "liberal":
            btwness = nx.edge_betweenness_centrality(core_G, normalized=False)
            btwness_sorted_keys = sorted(btwness, key=btwness.get, reverse=True)
            nun_ranked_links = ceil(p * len(btwness_sorted_keys))
            logger.info(
                f"conservative candidate link selection from top %{p * 100} "\
                    + f"links: {nun_ranked_links} / {len(btwness_sorted_keys)}")
            ranked_links = btwness_sorted_keys[:nun_ranked_links]
            candidate_Graph = nx.Graph()
            for (u, v) in ranked_links:
                u_neighbors = list(nx.neighbors(core_G, u))
                v_neighbors = list(nx.neighbors(core_G, v))
                logger.debug(f"link {u, v}.\t"\
                            + f"Neighbors of {u}: {len(u_neighbors)}\t"\
                            + f"Neighbors of {v}: {len(v_neighbors)}")
                candidate_Graph.add_edges_from([
                    (n_u, n_v) for n_u, n_v in product(u_neighbors, v_neighbors) 
                    if n_u != n_v 
                        and (n_u, n_v) not in core_G.edges])
            candidate_links = list(candidate_Graph.edges)
        
        elif candidate_set == "conservative":
            btwness = nx.edge_betweenness_centrality(core_G, normalized=False)
            btwness_sorted_keys = sorted(btwness, key=btwness.get, reverse=True)
            nun_ranked_links = ceil(p * len(btwness_sorted_keys))
            logger.info(
                f"conservative candidate link selection from top %{p * 100} "\
                    + f"links: {nun_ranked_links} / {len(btwness_sorted_keys)}")
            ranked_links = btwness_sorted_keys[:nun_ranked_links]
            candidate_Graph = nx.Graph()
            for (u, v) in ranked_links:
                u_neighbors = list(nx.neighbors(core_G, u))
                v_neighbors = list(nx.neighbors(core_G, v))
                logger.debug(f"link {u, v}.\t"\
                            + f"Neighbors of {u}: {len(u_neighbors)}\t"\
                            + f"Neighbors of {v}: {len(v_neighbors)}")
                candidate_Graph.add_edges_from([
                    (n_u, v) for n_u in u_neighbors
                    if n_u != v 
                        and (n_u, v) not in core_G.edges])
                candidate_Graph.add_edges_from([
                    (u, n_v) for n_v in v_neighbors 
                    if u != n_v 
                        and (u, n_v) not in core_G.edges])
            candidate_links = list(candidate_Graph.edges)

        logger.info(f"Total candidate links: {len(candidate_links)}")
        self.candidate_links = candidate_links

    def add_candidate_links_to_super_graph(self):
        self.super_graph.add_edges_from(self.candidate_links)
        self.super_graph.add_edges_from(self.core_G.edges)

    def get_shortest_paths(self):
        ########################################
        #### Derive paths that use shortcuts ###
        ########################################
        self.get_shortest_original_paths()
        try:
            assert self.use_cache
            # Loads both tunnel list and tunnel dict.
            self.load_paths()
            logger.info("Loaded paths from disc.")

            # Don't feel link making keys strings just to save dict to json. Maybe later.
            # self.tunnel_dict = defaultdict(list)
            # for i, tunnel in enumerate(self.tunnel_list):
            #     source, target = tunnel[0], tunnel[-1]
            #     self.tunnel_dict[(source, target)].append(i)

        except:
            if self.PARALLEL:
                self.get_shortest_paths_from_k_link_super_graphs(self.k)
            else:
                super_graph = self.super_graph
                '''
                if source not in self.core_shortest_path_len or target not in self.core_shortest_path_len[source]:
                    shortest_s_t_path_len \
                        = self.core_shortest_path_len[source][target] \
                        = self.core_shortest_path_len[target][source] \
                        = nx.shortest_path_length(G, source, target)
                    s_t_paths = nx.shortest_simple_paths(G, source, target)
                    for s_t_path in tqdm(s_t_paths, desc="Path", leave=False):
                        if len(s_t_path) - 1 > shortest_s_t_path_len:
                            break
                        self.original_tunnel_list.append(s_t_path)
                        self.original_tunnel_dict[source][target].append(s_t_path) 
                        reversed_path = list(reversed(s_t_path))
                        self.original_tunnel_list.append(reversed_path)
                        self.original_tunnel_dict[target][source].append(reversed_path) 

                '''
                
                for s, t in tqdm(self.all_node_pairs, desc="Pre-computing paths."):
                    if s not in self.tunnel_dict or t not in self.tunnel_dict[s]:
                        s_t_paths = nx.shortest_simple_paths(super_graph, s, t)
                        shortest_s_t_path_len = self.core_shortest_path_len[s][t]
                        for s_t_path in tqdm(
                            s_t_paths,
                            desc="Calculating ({}, {}) paths shorter than {} hops.".format(
                                s, t, shortest_s_t_path_len
                            ),
                        ):
                            if len(s_t_path) - 1 > shortest_s_t_path_len:
                                break
                            reversed_path = list(reversed(s_t_path))
                            self.tunnel_list.append(s_t_path)
                            self.tunnel_list.append(reversed_path)
                            self.tunnel_dict[s][t].append(s_t_path)
                            self.tunnel_dict[t][s].append(reversed_path)

                self.save_paths()
            logger.info("Computed paths and saved to disc.")

    def get_shortest_original_paths(self):
        ########################################
        #### Derive paths that use shortcuts ###
        ########################################
        try:
            assert self.use_cache
            self.load_original_paths()
            logger.info("Loaded original paths from disc.")
            
            # for i, tunnel in enumerate(self.original_tunnel_list):
            #     source, target = tunnel[0], tunnel[-1]
            #     self.original_tunnel_dict[(source, target)].append(i)
            #     self.core_shortest_path_len[(source, target)] = len(tunnel)
            for s, t in self.all_node_pairs:
                self.core_shortest_path_len[s][t]  = len(self.original_tunnel_dict[s][t][0]) - 1

        except:
            G = self.G.copy(as_view=True)
            for source, target in tqdm(self.all_node_pairs, desc="Pre-computing paths."):
                if source not in self.core_shortest_path_len or target not in self.core_shortest_path_len[source]:
                    shortest_s_t_path_len \
                        = self.core_shortest_path_len[source][target] \
                        = self.core_shortest_path_len[target][source] \
                        = nx.shortest_path_length(G, source, target)
                    s_t_paths = nx.shortest_simple_paths(G, source, target)
                    for s_t_path in tqdm(s_t_paths, desc="Path", leave=False):
                        if len(s_t_path) - 1 > shortest_s_t_path_len:
                            break
                        self.original_tunnel_list.append(s_t_path)
                        self.original_tunnel_dict[source][target].append(s_t_path) 
                        reversed_path = list(reversed(s_t_path))
                        self.original_tunnel_list.append(reversed_path)
                        self.original_tunnel_dict[target][source].append(reversed_path) 

            self.save_original_paths_list()
            self.save_original_paths_dict()
            logger.info("Computed original paths and saved to disc.")

    def load_paths(self):
        with open(
            f"./data/paths/optimization/{self.network}_{self.candidate_set}.json", "r"
        ) as fob:
            json_obj = json.load(fob)
            self.tunnel_list = json_obj["list"]
            self.tunnel_dict = json_obj["tunnels"]
        return

    def save_paths(self):
        with open(
            f"./data/paths/optimization/{self.network}_{self.candidate_set}.json", "w"
        ) as fob:
            return json.dump({"list": self.tunnel_list, 
                              "tunnels": self.tunnel_dict}, fob)
        

    def load_original_paths(self):
        with open(
            "./data/paths/optimization/{}_original_tunnel_list.json".format(self.network),
            "r",
        ) as fob:
            json_obj = json.load(fob)
            self.original_tunnel_list = json_obj["list"]

        with open(
            "./data/paths/optimization/{}_original_tunnel_dict.json".format(self.network),
            "r",
        ) as fob:
            json_obj = json.load(fob)
            self.original_tunnel_dict = json_obj["tunnels"]
        return

    def save_original_paths_list(self):
        os.makedirs("./data/paths/optimization/", exist_ok=True)
        with open(
            "./data/paths/optimization/{}_original_tunnel_list.json".format(self.network),
            "w",
        ) as fob:
            return json.dump({"list": self.original_tunnel_list}, fob)
    
    def save_original_paths_dict(self):
        os.makedirs("./data/paths/optimization/", exist_ok=True)
        with open(
            "./data/paths/optimization/{}_original_tunnel_dict.json".format(self.network),
            "w",
        ) as fob:
            return json.dump({"tunnels": self.original_tunnel_dict}, fob)


    def doppler(self):
        LINK_CAPACITY = self.LINK_CAPACITY

        m = self.model = Model("Doppler")

        # Convert the graph to a directed graph

        G_0 = self.G.copy(as_view=True).to_directed()
        directionless_edges = self.super_graph.edges
        G_prime = self.super_graph.to_directed()
        demand = self.demand_dict

        # Graphs should only differ in edges
        assert set(G_0.nodes) == set(G_prime.nodes)

        # Get transponder count
        txp_count = self.txp_count

        # Get the list of nodes and edges
        nodes = list(G_0.nodes)
        initial_edges = list(G_0.edges)
        prime_edges = self.prime_edges = list(G_prime.edges)

        # Add integer variables for node degree
        node_degree_vars = m.addVars(
            len(nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
        )

        # Edge vars that can be toggled on or off.
        edge_vars = m.addVars(prime_edges, vtype=GRB.BINARY, name="edge")

        # Initial edges, a constant vector accessible the same as edge_vars
        initial_edge_vars = tupledict(
            {(u, v): 1 if (u, v) in initial_edges else 0 for (u, v) in prime_edges}
        )

        # The overlapping set of edges for initial_edge_vars and edge_vars
        link_intersection = m.addVars(
            prime_edges,
            vtype=GRB.BINARY,
            name="link_intersection",
        )
        m.addConstrs(
            link_intersection[(u, v)] == min_(edge_vars[u, v], initial_edge_vars[u, v])
            for (u, v) in prime_edges
        )

        # Enforce max degree based on transponders at each node
        for v_idx, vertex in enumerate(nodes):
            m.addConstr(
                node_degree_vars[v_idx]
                == sum(
                    edge_vars[u, v]
                    for (u, v) in edge_vars
                    # if u == vertex or v == vertex
                    if u == vertex  # only interested in originating match
                )
            )
            m.addConstr(txp_count[vertex] >= node_degree_vars[v_idx])

        # Add flow variables for commodities and edges
        flow_vars = m.addVars(
            demand.keys(), prime_edges, vtype=GRB.CONTINUOUS, name="flow"
        )

        # Add conservation of flow constraints for nodes and commodities
        for node in nodes:
            for source, target in demand.keys():
                inflow = quicksum(
                    flow_vars[source, target, u, v] for (u, v) in G_prime.in_edges(node)
                )
                outflow = quicksum(
                    flow_vars[source, target, u, v]
                    for (u, v) in G_prime.out_edges(node)
                )
                if node == source:
                    m.addConstr(inflow - outflow == -demand[source, target])
                elif node == target:
                    m.addConstr(inflow - outflow == demand[source, target])
                else:
                    m.addConstr(inflow - outflow == 0)

        # Add capacity constraints for edges
        edge_capacity = tupledict()
        for u, v in prime_edges:
            edge_capacity[u, v] = m.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=LINK_CAPACITY, name=f"capacity_{u}_{v}"
            )
            m.addConstr(
                edge_capacity[u, v]
                >= quicksum(
                    flow_vars[source, target, u, v]
                    for (source, target) in demand.keys()
                )
            )

        # Enforce symetrical bi-directional capacity
        for u, v in directionless_edges:
            m.addConstr(edge_capacity[u, v] == edge_capacity[v, u])

        # Add binary constraint on edges
        for u, v in prime_edges:
            m.addConstr(edge_capacity[u, v] == edge_vars[(u, v)] * LINK_CAPACITY)

        # Set the objective to minimize the total flow
        # m.setObjective(quicksum(flow_vars[source, target, u, v] for (source, target) in demand.keys() for (u, v) in prime_edges), sense=GRB.MINIMIZE)
        m.setObjective(
            quicksum(link_intersection[u, v] for (u, v) in prime_edges),
            sense=GRB.MINIMIZE,
        )

        # Optimize the model
        start = process_time()
        m.optimize()
        end = process_time()
        opt_time = start - end
        self.flow_vars = flow_vars
        if m.status == GRB.Status.OPTIMAL:
            logger.info("Optimal solution found.")
            self.populate_changes(edge_vars)

        else:
            logger.error("No optimal solution found.")
            if self.debug: 
                self.write_iss()

        return opt_time
