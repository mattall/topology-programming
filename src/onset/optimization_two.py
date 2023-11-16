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

    def get_flow_allocations(self):
        m = self.model
        flow_vars = self.flow_vars
        demand = self.demand_dict
        prime_edges = self.prime_edges
        flow_paths = self.flow_paths
        if m.status == GRB.Status.OPTIMAL:
            logger.info("Optimal solution found.")
            for source, target in demand.keys():
                paths = []
                for u, v in prime_edges:
                    if flow_vars[source, target, u, v].x > 0:
                        paths.append(
                            {
                                "path": [u, v],
                                "weight": flow_vars[source, target, u, v].x,
                            }
                        )
                flow_paths[(source, target)] = paths

                logger.info(f"Flow from {source} to {target}:")
                for path in paths:
                    logger.info(f"Path: {path['path']} - Weight: {path['weight']}")

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

    def onset_v1(self):
        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        demand_dict = self.demand_dict
        # self.BUDGET = min(len(self.candidate_links), 10)
        self.BUDGET = len(self.candidate_links)
        with Env() as env, Model("ONSET", env=env) as m:
            self.model = m
            m.setParam("NodefileStart", 2)
            m.setParam("SoftMemLimit", 5)  # Process dies if uses more than 5 GB

            # ## Variables
            #
            # #### Inputs
            #
            # - $G(V,E)$ : network $G$, vertices $V$, edges, $E$
            # - $e' \in E$      : Original edges
            # - $\hat{e} \in E$ : Candidate edges
            # - *Note*: $\hat{e} \cap e' = \emptyset$ and $\hat{e} \cup e' = E$
            # - $d_{s,t} \in D$ : Demand $d$ between two nodes $s$ and $t$ in demand matrix $D$
            # - $p \in \mathcal{P}$ : A path, sequence of 2 or more nodes $\{v_0, \dots, v_n\}$ s.t., $ \forall i \in N^{[1, n]}, (v_{i-1}, v_i) \in E$
            # - $P_{s,t} \in \mathcal{P}$ : Set of paths $P$ from $s$ to $t$ among all paths $\mathcal{P}$
            # - $\hat{P_{s,t}} \in \mathcal{P}$ : Set of paths $P$ that contain at least one candidate link, $\hat{e}$
            # - ${P'_{s,t}} \in \mathcal{P}$ : Set of paths that contain only original links, ${e'}$
            # - *Note*: $\hat{P_{s,t}} \cap {P'_{s,t}} = \emptyset$ and $\hat{P_{s,t}} \cup {P'_{s,t}} = \mathcal{P}$
            # - ${P^e_{s,t}} \in \mathcal{P}$ : Set of paths that contain a particular link, $e$
            # - $B$: Budget. Number of candidate links that can be provisioned in the network
            # - $C$: Link capacity
            #
            # ### Decision Variables
            #
            # - $b_{e}$: Binary variable. $1$ if edge $e$ is active. $0$ if $e$ is inactive.
            # - $b_{p}$: Binary variable. $1$ if path $p$ is active. $0$ if $p$ is inactive.
            # - $U_{e}$: Utilization of edge $e$
            # - $flow_{p}$: $flow$ allocated onto path $p$
            #
            # ### Auxillary Variable/Constraint
            # - $M$: Objective function auxillary variable, $M = max_{e \in E} U_e$

            logger.info("Initializing Optimizer variables")
            logger.info("\tInitializing candidate_link_vars")
            candid_link_vars = m.addVars(
                range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
            )

            logger.info("\tInitializing path_vars")
            path_vars = m.addVars(
                range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
            )

            logger.info("\tInitializing link_util")
            link_util = m.addVars(
                range(len(super_graph.edges())),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=LINK_CAPACITY,
                name="link_util",
            )

            # # unbound
            # link_util = m.addVars(range(len(super_graph.edges())),
            #                     vtype=GRB.CONTINUOUS, lb=0, name="link_util")
            logger.info("\tInitializing norm_link_util")
            norm_link_util = m.addVars(
                range(len(super_graph.edges())),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=1,
                name="norm_link_util",
            )

            m.addConstrs(
                (
                    norm_link_util[i] == link_util[i] / LINK_CAPACITY
                    for i in range(len(super_graph.edges()))
                ),
                name="norm_link_util_constr",
            )

            logger.info("\tInitializing flow_p")
            flow_p = m.addVars(
                range(len(super_paths_list)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                name="flow_p",
            )

            logger.info("Setting Objective.")
            M = m.addVar(vtype=GRB.CONTINUOUS, name="M")
            m.addConstr(M == max_(norm_link_util), name="aux_objective_constr_M")

            # ### Objective
            # Minimize $M$

            N = m.addVar(vtype=GRB.CONTINUOUS, name="N", lb=1)
            m.addConstr(N == quicksum(candid_link_vars), name="aux_objective_constr_M")

            O = m.addVar(vtype=GRB.CONTINUOUS, name="O")
            m.addConstr(O == M + N, name="aux_objective_constr_O")

            m.setObjective(O, sense=GRB.MINIMIZE)

            logger.info("Initializing Constraints")
            # ### Budget Constraint
            #
            # $\sum_{\hat{e} \in E} b_\hat{e} \leq B$
            # logger.info("Adding Model Constraint, Budget: {}".format(self.BUDGET))
            logger.info("\tInitializing Budget")
            m.addConstr(quicksum(candid_link_vars) <= self.BUDGET, "budget")

            # ### Active Paths Constraints
            #
            # $\forall p \in \mathcal{P} : flow_{p} \leq C * b_p$
            #
            # $\forall p \in \mathcal{P}: b_{p} = \min_{e \in p} b_e$
            #
            ##############################
            # Add Active Path Constraint #
            ##############################
            logger.info("\tInitializing flow_binary and path_link constraints")

            path_candidate_links = [[] for _ in range(len(path_vars))]
            for p_i in tqdm(
                range(len(path_vars)),
                desc="Initialling candidate link & path binary vars 1",
                total=len(path_vars),
            ):
                path_i_links = list(
                    zip(super_paths_list[p_i], super_paths_list[p_i][1:])
                )
                for link_l in path_i_links:
                    l1 = list(link_l)
                    l2 = [l1[1], l1[0]]
                    if l1 in candidate_links or l2 in candidate_links:
                        candidate_index = (
                            candidate_links.index(l1)
                            if l1 in candidate_links
                            else candidate_links.index(l2)
                        )
                        path_candidate_links[p_i].append(candidate_index)

                # with open(path_candidate_link_file, 'wb') as pkl:
                #      pickle.dump(path_candidate_links, pkl)

            for p_i in tqdm(
                range(len(path_vars)),
                desc="Initialling candidate link & path binary vars 2",
                total=len(path_vars),
            ):
                m.addConstr(
                    flow_p[p_i] <= path_vars[p_i] * LINK_CAPACITY,
                    "flow_binary_{}".format(p_i),
                )
                if path_candidate_links[p_i] != []:
                    path_candidate_link_vars = [
                        candid_link_vars[var_i] for var_i in path_candidate_links[p_i]
                    ]
                    m.addConstr(
                        path_vars[p_i] == min_(path_candidate_link_vars),
                        name="path_link_constr_{}".format(p_i),
                    )

                else:
                    m.addConstr(
                        path_vars[p_i] == True,
                        name="path_link_constr_{}".format(p_i),
                    )

            # ### Total demand for a path constraint
            #
            # $\forall d_{s,t} \in D : d_{s,t} \leq \sum_{p \in P_{s,t}} flow_{p}$
            #
            # #####################################################
            # # Find demand per tunnel considering active tunnels #
            # #####################################################
            logger.info("\tInitializing Find demand per tunnel considering active tunnels")
            for source, target in demand_dict:
                P = self.tunnel_dict[(source, target)]

                m.addConstr(
                    demand_dict[(source, target)] <= quicksum(flow_p[p] for p in P),
                    "flow_{}_{}".format(source, target),
                )

            # ### Total link utilization from all active paths constraint
            #
            # $\forall e \in E: U_e = \sum_{p | e \in p } flow_{p} $
            #
            # $\forall e \in E: \sum_{p | e \in p } flow_{p} \leq C $
            #
            ###############################################################
            # Find Demand per link considering demand from active tunnels #
            ###############################################################
            logger.info(
                "\tInitializing Find Demand per link considering demand from active tunnels"
            )
            link_tunnels_file = os.path.join(
                SCRIPT_HOME,
                "data",
                "paths",
                "optimization",
                self.network + "_tunnels.pkl",
            )

            # FILE EXISTS
            if os.path.exists(link_tunnels_file):
                # NOT EMPTY
                logger.info("Loading Link Tunnels file: {}".format(link_tunnels_file))
                if os.path.getsize(link_tunnels_file) > 0:
                    with open(link_tunnels_file, "rb") as pkl:
                        network_tunnels = pickle.load(pkl)

            else:
                logger.info("Generating Link Tunnels: {}".format(link_tunnels_file))
                network_tunnels = []
                # for all links.
                if 0:
                    for link_i, (link_source, link_target) in enumerate(
                        list(super_graph.edges())
                    ):
                        link_tunnels = []
                        # for all tunnels
                        for tunnel_i, tunnel in enumerate(super_paths_list):
                            if link_on_path(tunnel, [link_source, link_target]):
                                link_tunnels.append(tunnel_i)

                        network_tunnels.append(link_tunnels)

                if 1:
                    network_tunnels = [[] for _ in range(len(super_graph.edges()))]
                    link_index = list(super_graph.edges())[:]
                    for tunnel_i, tunnel in tqdm(
                        enumerate(super_paths_list),
                        desc="Generating link tunnels list.",
                        total=len(super_paths_list),
                    ):
                        for u, v in zip(tunnel, tunnel[1:]):
                            try:
                                l_id = link_index.index((u, v))
                            except ValueError:
                                l_id = link_index.index((v, u))

                            network_tunnels[l_id].append(tunnel_i)

                # Write File for future
                with open(link_tunnels_file, "wb") as pkl:
                    pickle.dump(network_tunnels, pkl)

            assert len(network_tunnels) == len(super_graph.edges())

            for link_i, (link_source, link_target) in tqdm(
                enumerate(list(super_graph.edges())),
                desc="Initializing link utilization constraints.",
                total=len(super_graph.edges()),
            ):
                link_tunnels = network_tunnels[link_i]
                # if 0:
                #     # FILE EXISTS
                #     if os.path.exists(link_i_tunnels_file):
                #         # NOT EMPTY
                #         if os.path.getsize(link_i_tunnels_file) > 0:
                #             with open(link_i_tunnels_file, 'rb') as pkl:
                #                 link_tunnels = pickle.load(pkl)

                #         # IS EMPTY
                #         else:
                #             link_tunnels = []

                #     # FILE DOES NOT EXIST
                #     else:
                #         # Enumerate Tunnels
                #         for tunnel_i, tunnel in enumerate(super_paths_list):
                #             if link_on_path(tunnel, [link_source, link_target]):
                #                 link_tunnels.append(tunnel_i)

                #         # Write File for future
                #         with open(link_i_tunnels_file, 'wb') as pkl:
                #             pickle.dump(link_tunnels, pkl)

                # if 1:
                #     link_tunnels = []
                #     for tunnel_i, tunnel in enumerate(super_paths_list):
                #         if link_on_path(tunnel, [link_source, link_target]):
                #             link_tunnels.append(tunnel_i)

                m.addConstr(
                    link_util[link_i] == quicksum(flow_p[i] for i in link_tunnels),
                    "link_demand_{}".format(link_i),
                )

                m.addConstr(
                    self.LINK_CAPACITY >= quicksum(flow_p[i] for i in link_tunnels),
                    "link_utilization_{}".format(link_i),
                )

            m.update()
            self.model.optimize()
            if self.model.status == GRB.Status.OPTIMAL:
                links_to_add = []
                for clv in candid_link_vars:
                    if candid_link_vars[clv].x == 1:
                        links_to_add.append(candidate_links[clv])
                self.links_to_add = links_to_add

    def onset_v2(self):
        LINK_CAPACITY = self.LINK_CAPACITY

        m = self.model = Model("OnsetOptimization")

        # Convert the graph to a directed graph

        G_0 = self.G.copy(as_view=True).to_directed()
        directionless_edges = self.super_graph.edges
        G_prime = self.super_graph.to_directed()
        demand = self.demand_dict

        # Graphs should only differ in edges
        assert set(G_0.nodes) == set(G_prime.nodes)

        # Get transponder count
        txp_count = self.txp_count
        assert isinstance(
            txp_count, dict
        ), f"txp_count type error. expected <class 'dict'>, got: {type(txp_count)}"

        # Get the list of nodes and edges
        nodes = self.nodes

        # ensure all nodes are reflected in txp_count.
        assert set(txp_count.keys()) == set(
            nodes
        ), f"txp_count vs. nodes set mismatch error. The following sets should be equal. \n\tnodes: {set(nodes)}\n\ttxp_count.keys(): {set(txp_count.keys())}"

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
        # link_intersection = m.addVars(
        #     prime_edges,
        #     vtype=GRB.BINARY,
        #     name="link_intersection",
        # )
        # m.addConstrs(
        #     link_intersection[(u, v)] == min_(edge_vars[u, v], initial_edge_vars[u, v])
        #     for (u, v) in prime_edges
        # )

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

        # Enforce symmetrical bi-directional capacity
        for u, v in directionless_edges:
            m.addConstr(edge_capacity[u, v] == edge_capacity[v, u])

        # Add binary constraint on edges
        for u, v in prime_edges:
            m.addConstr(edge_capacity[u, v] == edge_vars[(u, v)] * LINK_CAPACITY)

        # Set the objective to minimize the total flow
        # m.setObjective(quicksum(flow_vars[source, target, u, v] for (source, target) in demand.keys() for (u, v) in prime_edges), sense=GRB.MINIMIZE)
        m.setObjective(
            max(flow_vars["*", "*", u, v] for (u, v) in prime_edges), sense=GRB.MINIMIZE
        )
        # m.setObjective(
        #     quicksum(link_intersection[u, v] for (u, v) in prime_edges),
        #     sense=GRB.MINIMIZE,
        # )
        m.update()

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

    def onset_v1_1(self):
        LINK_CAPACITY = self.LINK_CAPACITY
        # super_paths_list = self.tunnel_list
        tunnel_dict = self.tunnel_dict
        m = self.model = Model("Onset v1.1 Optimization")
        # Convert the graph to a directed graph
        core_G = self.core_G.copy(as_view=True).to_directed()
        G_0 = self.G.copy(as_view=True).to_directed()
        directionless_edges = self.super_graph.edges
        G_prime = self.super_graph.to_directed()
        demand = self.demand_dict

        # Graphs should only differ in edges
        assert set(G_0.nodes) == set(G_prime.nodes)

        # Get transponder count
        txp_count = self.txp_count
        assert isinstance(
            txp_count, dict
        ), f"txp_count type error. expected <class 'dict'>, got: {type(txp_count)}"

        # Get the list of nodes and edges
        nodes = self.nodes

        # ensure all nodes are reflected in txp_count.
        assert set(txp_count.keys()) == set(
            nodes
        ), f"txp_count vs. nodes set mismatch error. The following sets should be equal. \n\tnodes: {set(nodes)}\n\ttxp_count.keys(): {set(txp_count.keys())}"

        # list of directional edges that have potential to exist
        prime_edges = self.prime_edges = list(G_prime.edges)

        # Add integer variables for node degree
        node_degree_vars = m.addVars(
            len(nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
        )

        # Edge vars that can be toggled on or off.
        edge_vars = m.addVars(prime_edges, vtype=GRB.BINARY, name="edge")

        # Enforce max degree based on transponders at each node
        for v_idx, vertex in enumerate(nodes):
            m.addConstr(
                node_degree_vars[v_idx]
                == sum(
                    edge_vars[u, v]
                    for (u, v) in edge_vars
                    if u == vertex  # only interested in originating match
                )
            )
            m.addConstr(txp_count[vertex] >= node_degree_vars[v_idx])

        # Edges in the core (base aka original aka physical aka fiber) graph
        # are required to be present.
        for u, v in core_G.edges:
            m.addConstr(
                edge_vars[u, v] == 1,
                name=f"core_edge_{u}_{v}"
            )
        # Helper references
        links_in_flow = tupledict()
        flows_with_link = tupledict()
        for s, t in prime_edges:
            flows_with_link[s, t] = set()

        logger.info("\tInitializing path_vars and flow_vars")
        path_vars = tupledict()
        flow_vars = tupledict()
        for s, t in self.all_node_pairs:
            for i, path_idx in enumerate(tunnel_dict[s][t]):
                #   initializing vars
                path_vars[s, t, i] = m.addVar(
                    vtype=GRB.BINARY, name=f"path_{s}_{t}_{i}"
                )
                flow_vars[s, t, i] = m.addVar(
                    vtype=GRB.CONTINUOUS, name=f"flow_{s}_{t}_{i}"
                )
                #   setting up helper references
                links_in_flow[s, t, i] = set()
                path_links = zip(
                    self.tunnel_list[i],
                    self.tunnel_list[i][1:]
                )
                for u, v in path_links:
                    links_in_flow[s, t, i].add((u, v))
                    flows_with_link[u, v].add((s, t, i))

        for s, t, i in tqdm(
            path_vars,
            desc="Initialling candidate link & path binary vars",
            total=len(path_vars),
        ):
            m.addConstr(
                flow_vars[s, t, i] <= path_vars[s, t, i] * LINK_CAPACITY,
                f"flow_lte_path_{s}_{t}_{i}",
            )

            m.addConstr(
                path_vars[s, t, i] == min_([edge_vars[u, v]
                                           for (u, v) in links_in_flow[s, t, i]]),
                name=f"path_eq_min_edge_in_path__{s}_{t}_{i}"
            )

        logger.info("\tInitializing link_util")
        link_util = m.addVars(
            prime_edges,
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=LINK_CAPACITY,
            name="link_util",
        )

        # logger.info("\tInitializing norm_link_util")
        # norm_link_util = m.addVars(
        #     prime_edges,
        #     vtype=GRB.CONTINUOUS,
        #     lb=0,
        #     ub=1,
        #     name="norm_link_util",
        # )

        link_capacity = tupledict(prime_edges)
        logger.info("\tInitializing Find demand per tunnel considering active tunnels")
        for s, t in demand:
            m.addConstr(
                demand[s, t] <= flow_vars.sum(s, t, '*'),
                f"dem_{s}_{t}_lte_flow".format(s, t),
            )

        for u, v in prime_edges:
            link_capacity[u, v] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=LINK_CAPACITY,
                name=f"capacity_{u}_{v}"
            )
            m.addConstr(
                link_util[u, v]
                <= quicksum(
                    flow_vars[s, t, i]
                    for (s, t, i) in flows_with_link[u, v]
                )
            )

            m.addConstr(
                link_capacity[u, v]
                >= quicksum(
                    flow_vars[s, t, i]
                    for (s, t, i) in flows_with_link[u, v]
                )
            )
        # Enforce symmetrical bi-directional capacity
        for u, v in directionless_edges:
            m.addConstr(link_capacity[u, v] == link_capacity[v, u])
        # Add binary constraint on edges
        # TODO: in mode advanced version, link capacity should be related
        # to transponder allocation on u, v and the capacity potential for
        # each of those transponders. For now, we will assume fixed capacity
        # per transponder and one transponder pair per link.
        for u, v in prime_edges:
            m.addConstr(
                link_capacity[u, v] == edge_vars[(u, v)] * LINK_CAPACITY,
                name=f"link_cap_{u}_{v}"
            )
            # m.addConstr(
            #     norm_link_util[u, v] == link_util[u, v] / link_capacity[u, v],
            #     name=f"norm_link_cap_{u}_{v}"
            #     )

        logger.info("Setting Objective.")
        M = m.addVar(vtype=GRB.CONTINUOUS, name="M")
        N = m.addVar(vtype=GRB.CONTINUOUS, name="N", lb=1)
        O = m.addVar(vtype=GRB.CONTINUOUS, name="O")
        m.addConstr(M == max_(link_util), name="aux_objective_constr_M")
        m.addConstr(N == quicksum(edge_vars), name="aux_objective_constr_M")
        m.addConstr(O == M + N, name="aux_objective_constr_O")
        m.setObjective(O, sense=GRB.MINIMIZE)

        m.update()
        start = process_time()
        m.optimize()
        end = process_time()
        opt_time = end - start
        self.flow_vars = flow_vars
        if self.model.status == GRB.Status.OPTIMAL:
            self.populate_changes(edge_vars)
        else:
            logger.error("No optimal solution found.")
            if self.debug: 
                self.write_iss()
        return opt_time
    
    def onset_v1_1_no_cap(self):
        LINK_CAPACITY = self.LINK_CAPACITY
        # super_paths_list = self.tunnel_list
        tunnel_dict = self.tunnel_dict
        m = self.model = Model("Onset v1.1 Optimization - No capacity constraint")
        # Convert the graph to a directed graph
        core_G = self.core_G.copy(as_view=True).to_directed()
        G_0 = self.G.copy(as_view=True).to_directed()
        directionless_edges = self.super_graph.edges
        G_prime = self.super_graph.to_directed()
        demand = self.demand_dict

        # Graphs should only differ in edges
        assert set(G_0.nodes) == set(G_prime.nodes)

        # Get transponder count
        txp_count = self.txp_count
        assert isinstance(
            txp_count, dict
        ), f"txp_count type error. expected <class 'dict'>, got: {type(txp_count)}"

        # Get the list of nodes and edges
        nodes = self.nodes

        # ensure all nodes are reflected in txp_count.
        assert set(txp_count.keys()) == set(
            nodes
        ), f"txp_count vs. nodes set mismatch error. The following sets should be equal. \n\tnodes: {set(nodes)}\n\ttxp_count.keys(): {set(txp_count.keys())}"

        # list of directional edges that have potential to exist
        prime_edges = self.prime_edges = list(G_prime.edges)

        # Add integer variables for node degree
        node_degree_vars = m.addVars(
            len(nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
        )

        # Edge vars that can be toggled on or off.
        edge_vars = m.addVars(prime_edges, vtype=GRB.BINARY, name="edge")

        # Enforce max degree based on transponders at each node
        for v_idx, vertex in enumerate(nodes):
            m.addConstr(
                node_degree_vars[v_idx]
                == sum(
                    edge_vars[u, v]
                    for (u, v) in edge_vars
                    if u == vertex  # only interested in originating match
                )
            )
            m.addConstr(txp_count[vertex] >= node_degree_vars[v_idx])

        # Edges in the core (base aka original aka physical aka fiber) graph
        # are required to be present.
        for u, v in core_G.edges:
            m.addConstr(
                edge_vars[u, v] == 1,
                name=f"core_edge_{u}_{v}"
            )
        # Helper references
        links_in_flow = tupledict()
        flows_with_link = tupledict()
        for s, t in prime_edges:
            flows_with_link[s, t] = set()

        logger.info("\tInitializing path_vars and flow_vars")
        path_vars = tupledict()
        flow_vars = tupledict()
        for s, t in self.all_node_pairs:
            for i, path_idx in enumerate(tunnel_dict[s][t]):
                #   initializing vars
                path_vars[s, t, i] = m.addVar(
                    vtype=GRB.BINARY, name=f"path_{s}_{t}_{i}"
                )
                flow_vars[s, t, i] = m.addVar(
                    vtype=GRB.CONTINUOUS, name=f"flow_{s}_{t}_{i}"
                )
                #   setting up helper references
                links_in_flow[s, t, i] = set()
                path_links = zip(
                    self.tunnel_list[i],
                    self.tunnel_list[i][1:]
                )
                for u, v in path_links:
                    links_in_flow[s, t, i].add((u, v))
                    flows_with_link[u, v].add((s, t, i))

        for s, t, i in tqdm(
            path_vars,
            desc="Initialling candidate link & path binary vars",
            total=len(path_vars),
        ):
            # m.addConstr(
            #     flow_vars[s, t, i] <= path_vars[s, t, i] * LINK_CAPACITY,                
            #     f"flow_lte_path_{s}_{t}_{i}",
            # )
            m.addConstr(
                path_vars[s, t, i] == min_([edge_vars[u, v]
                                           for (u, v) in links_in_flow[s, t, i]]),
                name=f"path_eq_min_edge_in_path__{s}_{t}_{i}"
            )

        logger.info("\tInitializing link_util")
        link_util = m.addVars(
            prime_edges,
            vtype=GRB.CONTINUOUS,
            lb=0,
            # ub=LINK_CAPACITY,
            name="link_util",
        )

        # logger.info("\tInitializing norm_link_util")
        # norm_link_util = m.addVars(
        #     prime_edges,
        #     vtype=GRB.CONTINUOUS,
        #     lb=0,
        #     ub=1,
        #     name="norm_link_util",
        # )

        link_capacity = tupledict(prime_edges)
        logger.info("\tInitializing Find demand per tunnel considering active tunnels")
        for s, t in demand:
            m.addConstr(
                demand[s, t] <= flow_vars.sum(s, t, '*'),
                f"dem_{s}_{t}_lte_flow".format(s, t),
            )

        for u, v in prime_edges:
            # link_capacity[u, v] = m.addVar(
            #     vtype=GRB.CONTINUOUS,
            #     lb=0,
            #     ub=LINK_CAPACITY,
            #     name=f"capacity_{u}_{v}"
            # )
            m.addConstr(
                link_util[u, v]
                <= quicksum(
                    flow_vars[s, t, i]
                    for (s, t, i) in flows_with_link[u, v]
                )
            )
            # m.addConstr(
            #     link_capacity[u, v]
            #     >= quicksum(
            #         flow_vars[s, t, i]
            #         for (s, t, i) in flows_with_link[u, v]
            #     )
            # )
        # Enforce symmetrical bi-directional capacity
        for u, v in directionless_edges:
            # m.addConstr(link_capacity[u, v] == link_capacity[v, u])
            m.addConstr(edge_vars[u, v] == edge_vars[v, u])
        # Add binary constraint on edges
        # TODO: in mode advanced version, link capacity should be related
        # to transponder allocation on u, v and the capacity potential for
        # each of those transponders. For now, we will assume fixed capacity
        # per transponder and one transponder pair per link.
        # for u, v in prime_edges:
        #     m.addConstr(
        #         link_capacity[u, v] == edge_vars[(u, v)] * LINK_CAPACITY,
        #         name=f"link_cap_{u}_{v}"
        #     )
        #     # m.addConstr(
        #     #     norm_link_util[u, v] == link_util[u, v] / link_capacity[u, v],
        #     #     name=f"norm_link_cap_{u}_{v}"
        #     #     )

        logger.info("Setting Objective.")
        M = m.addVar(vtype=GRB.CONTINUOUS, name="M")
        N = m.addVar(vtype=GRB.CONTINUOUS, name="N", lb=1)
        O = m.addVar(vtype=GRB.CONTINUOUS, name="O")
        m.addConstr(M == max_(link_util), name="aux_objective_constr_M")
        m.addConstr(N == quicksum(edge_vars), name="aux_objective_constr_M")
        m.addConstr(O == M + N, name="aux_objective_constr_O")
        m.setObjective(O, sense=GRB.MINIMIZE)

        m.update()
        start = process_time()
        m.optimize()
        end = process_time()
        opt_time = end - start
        self.flow_vars = flow_vars
        if self.model.status == GRB.Status.OPTIMAL:
            self.populate_changes(edge_vars)
        else:
            logger.error("No optimal solution found.")
            if self.debug: 
                self.write_iss()
        return opt_time

        _, opt_time = clock(m.optimize)
        self.flow_vars = flow_vars
        if m.status == GRB.Status.OPTIMAL:
            logger.info("Optimal solution found.")
            resultant_edges = [
                (u, v) for (u, v) in edge_vars if edge_vars[(u, v)].x == 1
            ]
            add_edges = [e for e in resultant_edges if e not in self.G.edges]
            drop_edges = [e for e in self.G.edges if e not in resultant_edges]
            G_new = nx.DiGraph()
            G_new.add_edges_from(resultant_edges)
            assert nx.is_strongly_connected(G_new)
            add_edges = list(set([tuple(sorted((u, v))) for u, v in add_edges]))
            drop_edges = list(set([tuple(sorted((u, v))) for u, v in drop_edges]))
            return ((add_edges, drop_edges), opt_time)

        else:
            logger.error("No optimal solution found.")
            write_iss(m)

        return -1

        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        demand_dict = self.demand_dict
        # self.BUDGET = min(len(self.candidate_links), 10)
        self.BUDGET = len(self.candidate_links)
        with Env() as env, Model("ONSET_v1_1", env=env) as m:
            self.model = m
            m.setParam("NodefileStart", 2)
            m.setParam("SoftMemLimit", 5)  # Process dies if uses more than 5 GB

            # ## Variables
            #
            # #### Inputs
            #
            # - $G(V,E)$ : network $G$, vertices $V$, edges, $E$
            # - $e' \in E$      : Original edges
            # - $\hat{e} \in E$ : Candidate edges
            # - *Note*: $\hat{e} \cap e' = \emptyset$ and $\hat{e} \cup e' = E$
            # - $d_{s,t} \in D$ : Demand $d$ between two nodes $s$ and $t$ in demand matrix $D$
            # - $p \in \mathcal{P}$ : A path, sequence of 2 or more nodes $\{v_0, \dots, v_n\}$ s.t., $ \forall i \in N^{[1, n]}, (v_{i-1}, v_i) \in E$
            # - $P_{s,t} \in \mathcal{P}$ : Set of paths $P$ from $s$ to $t$ among all paths $\mathcal{P}$
            # - $\hat{P_{s,t}} \in \mathcal{P}$ : Set of paths $P$ that contain at least one candidate link, $\hat{e}$
            # - ${P'_{s,t}} \in \mathcal{P}$ : Set of paths that contain only original links, ${e'}$
            # - *Note*: $\hat{P_{s,t}} \cap {P'_{s,t}} = \emptyset$ and $\hat{P_{s,t}} \cup {P'_{s,t}} = \mathcal{P}$
            # - ${P^e_{s,t}} \in \mathcal{P}$ : Set of paths that contain a particular link, $e$
            # - $B$: Budget. Number of candidate links that can be provisioned in the network
            # - $C$: Link capacity
            #
            # ### Decision Variables
            #
            # - $b_{e}$: Binary variable. $1$ if edge $e$ is active. $0$ if $e$ is inactive.
            # - $b_{p}$: Binary variable. $1$ if path $p$ is active. $0$ if $p$ is inactive.
            # - $U_{e}$: Utilization of edge $e$
            # - $flow_{p}$: $flow$ allocated onto path $p$
            #
            # ### Auxillary Variable/Constraint
            # - $M$: Objective function auxillary variable, $M = max_{e \in E} U_e$

            logger.info("Initializing Optimizer variables")
            logger.info("\tInitializing candidate_link_vars")
            candid_link_vars = m.addVars(
                range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
            )

            logger.info("\tInitializing path_vars")
            path_vars = m.addVars(
                range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
            )

            logger.info("\tInitializing link_util")
            link_util = m.addVars(
                range(len(super_graph.edges())),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=LINK_CAPACITY,
                name="link_util",
            )

            # # unbound
            # link_util = m.addVars(range(len(super_graph.edges())),
            #                     vtype=GRB.CONTINUOUS, lb=0, name="link_util")
            logger.info("\tInitializing norm_link_util")
            norm_link_util = m.addVars(
                range(len(super_graph.edges())),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=1,
                name="norm_link_util",
            )

            m.addConstrs(
                (
                    norm_link_util[i] == link_util[i] / LINK_CAPACITY
                    for i in range(len(super_graph.edges()))
                ),
                name="norm_link_util_constr",
            )

            logger.info("\tInitializing flow_p")
            flow_p = m.addVars(
                range(len(super_paths_list)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                name="flow_p",
            )

            logger.info("Setting Objective.")
            M = m.addVar(vtype=GRB.CONTINUOUS, name="M")
            m.addConstr(M == max_(norm_link_util), name="aux_objective_constr_M")

            # ### Objective
            # Minimize $M$

            N = m.addVar(vtype=GRB.CONTINUOUS, name="N", lb=1)
            m.addConstr(N == quicksum(candid_link_vars), name="aux_objective_constr_M")

            O = m.addVar(vtype=GRB.CONTINUOUS, name="O")
            m.addConstr(O == M + N, name="aux_objective_constr_O")

            m.setObjective(O, sense=GRB.MINIMIZE)

            logger.info("Initializing Constraints")
            # ### Budget Constraint
            #
            # $\sum_{\hat{e} \in E} b_\hat{e} \leq B$
            # logger.info("Adding Model Constraint, Budget: {}".format(self.BUDGET))
            # logger.info("\tInitializing Budget")
            # m.addConstr(quicksum(candid_link_vars) <= self.BUDGET, "budget")

            # ### Active Paths Constraints
            #
            # $\forall p \in \mathcal{P} : flow_{p} \leq C * b_p$
            #
            # $\forall p \in \mathcal{P}: b_{p} = \min_{e \in p} b_e$
            #
            ##############################
            # Add Active Path Constraint #
            ##############################
            logger.info("\tInitializing flow_binary and path_link constraints")

            path_candidate_links = [[] for _ in range(len(path_vars))]
            path_candidate_link_file = os.path.join(
                SCRIPT_HOME,
                "data",
                "paths",
                "optimization",
                self.network + "_path_candidate_links.pkl",
            )

            # try to preload constraint
            # if os.path.exists(path_candidate_link_file):
            #     with open(path_candidate_link_file, 'rb') as pkl:
            #         path_candidate_links = pickle.load(pkl)

            # else:
            path_candidate_links = [[] for _ in range(len(path_vars))]
            for p_i in tqdm(
                range(len(path_vars)),
                desc="Initialling candidate link & path binary vars 1",
                total=len(path_vars),
            ):
                path_i_links = list(
                    zip(super_paths_list[p_i], super_paths_list[p_i][1:])
                )
                for link_l in path_i_links:
                    l1 = list(link_l)
                    l2 = [l1[1], l1[0]]
                    if l1 in candidate_links or l2 in candidate_links:
                        candidate_index = (
                            candidate_links.index(l1)
                            if l1 in candidate_links
                            else candidate_links.index(l2)
                        )
                        path_candidate_links[p_i].append(candidate_index)

                # with open(path_candidate_link_file, 'wb') as pkl:
                #      pickle.dump(path_candidate_links, pkl)

            for p_i in tqdm(
                range(len(path_vars)),
                desc="Initialling candidate link & path binary vars 2",
                total=len(path_vars),
            ):
                m.addConstr(
                    flow_p[p_i] <= path_vars[p_i] * LINK_CAPACITY,
                    "flow_binary_{}".format(p_i),
                )
                if path_candidate_links[p_i] != []:
                    path_candidate_link_vars = [
                        candid_link_vars[var_i] for var_i in path_candidate_links[p_i]
                    ]
                    m.addConstr(
                        path_vars[p_i] == min_(path_candidate_link_vars),
                        name="path_link_constr_{}".format(p_i),
                    )

                else:
                    m.addConstr(
                        path_vars[p_i] == True,
                        name="path_link_constr_{}".format(p_i),
                    )

            # ### Total demand for a path constraint
            #
            # $\forall d_{s,t} \in D : d_{s,t} \leq \sum_{p \in P_{s,t}} flow_{p}$
            #
            # #####################################################
            # # Find demand per tunnel considering active tunnels #
            # #####################################################
            logger.info("\tInitializing Find demand per tunnel considering active tunnels")
            for source, target in demand_dict:
                P = self.tunnel_dict[(source, target)]

                m.addConstr(
                    demand_dict[(source, target)] <= quicksum(flow_p[p] for p in P),
                    "flow_{}_{}".format(source, target),
                )

            # ### Total link utilization from all active paths constraint
            #
            # $\forall e \in E: U_e = \sum_{p | e \in p } flow_{p} $
            #
            # $\forall e \in E: \sum_{p | e \in p } flow_{p} \leq C $
            #
            ###############################################################
            # Find Demand per link considering demand from active tunnels #
            ###############################################################
            logger.info(
                "\tInitializing Find Demand per link considering demand from active tunnels"
            )
            link_tunnels_file = os.path.join(
                SCRIPT_HOME,
                "data",
                "paths",
                "optimization",
                self.network + "_tunnels.pkl",
            )

            # FILE EXISTS
            if os.path.exists(link_tunnels_file):
                # NOT EMPTY
                logger.info("Loading Link Tunnels file: {}".format(link_tunnels_file))
                if os.path.getsize(link_tunnels_file) > 0:
                    with open(link_tunnels_file, "rb") as pkl:
                        network_tunnels = pickle.load(pkl)

            else:
                logger.info("Generating Link Tunnels: {}".format(link_tunnels_file))
                network_tunnels = []
                # for all links.
                if 0:
                    for link_i, (link_source, link_target) in enumerate(
                        list(super_graph.edges())
                    ):
                        link_tunnels = []
                        # for all tunnels
                        for tunnel_i, tunnel in enumerate(super_paths_list):
                            if link_on_path(tunnel, [link_source, link_target]):
                                link_tunnels.append(tunnel_i)

                        network_tunnels.append(link_tunnels)

                if 1:
                    network_tunnels = [[] for _ in range(len(super_graph.edges()))]
                    link_index = list(super_graph.edges())[:]
                    for tunnel_i, tunnel in tqdm(
                        enumerate(super_paths_list),
                        desc="Generating link tunnels list.",
                        total=len(super_paths_list),
                    ):
                        for u, v in zip(tunnel, tunnel[1:]):
                            try:
                                l_id = link_index.index((u, v))
                            except ValueError:
                                l_id = link_index.index((v, u))

                            network_tunnels[l_id].append(tunnel_i)

                # Write File for future
                with open(link_tunnels_file, "wb") as pkl:
                    pickle.dump(network_tunnels, pkl)

            assert len(network_tunnels) == len(super_graph.edges())

            for link_i, (link_source, link_target) in tqdm(
                enumerate(list(super_graph.edges())),
                desc="Initializing link utilization constraints.",
                total=len(super_graph.edges()),
            ):
                link_tunnels = network_tunnels[link_i]
                # if 0:
                #     # FILE EXISTS
                #     if os.path.exists(link_i_tunnels_file):
                #         # NOT EMPTY
                #         if os.path.getsize(link_i_tunnels_file) > 0:
                #             with open(link_i_tunnels_file, 'rb') as pkl:
                #                 link_tunnels = pickle.load(pkl)

                #         # IS EMPTY
                #         else:
                #             link_tunnels = []

                #     # FILE DOES NOT EXIST
                #     else:
                #         # Enumerate Tunnels
                #         for tunnel_i, tunnel in enumerate(super_paths_list):
                #             if link_on_path(tunnel, [link_source, link_target]):
                #                 link_tunnels.append(tunnel_i)

                #         # Write File for future
                #         with open(link_i_tunnels_file, 'wb') as pkl:
                #             pickle.dump(link_tunnels, pkl)

                # if 1:
                #     link_tunnels = []
                #     for tunnel_i, tunnel in enumerate(super_paths_list):
                #         if link_on_path(tunnel, [link_source, link_target]):
                #             link_tunnels.append(tunnel_i)

                m.addConstr(
                    link_util[link_i] == quicksum(flow_p[i] for i in link_tunnels),
                    "link_demand_{}".format(link_i),
                )

                m.addConstr(
                    self.LINK_CAPACITY >= quicksum(flow_p[i] for i in link_tunnels),
                    "link_utilization_{}".format(link_i),
                )

            m.update()
            self.model.optimize()
            if self.model.status == GRB.Status.OPTIMAL:
                links_to_add = []
                for clv in candid_link_vars:
                    if candid_link_vars[clv].x == 1:
                        links_to_add.append(candidate_links[clv])
                self.links_to_add = links_to_add

    def run_model_max_diff(self):
        core_G = self.core_G.copy(as_view=True)
        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        demand_dict = self.demand_dict
        super_graph_edge_list = list(super_graph.edges)
        initial_graph_edges_list = list(self.G.edges)
        # self.BUDGET = min(len(self.candidate_links), 10)
        # self.BUDGET = len(self.candidate_links)
        with Env() as env, Model("ONSET", env=env) as m:
            self.model = m
            m.setParam("NodefileStart", 2)
            m.setParam("SoftMemLimit", 5)  # Process dies if uses more than 5 GB

            logger.info("Initializing Optimizer variables")
            logger.info("\tInitializing candidate_link_vars")

            candid_link_vars = m.addVars(
                len(super_graph.edges),
                vtype=GRB.BINARY,
                name="candidate_links",
            )
            vertices = list(core_G.nodes)
            initial_links = [1 if e in self.G.edges else 0 for e in super_graph.edges]

            link_intersection = m.addVars(
                len(super_graph.edges),
                vtype=GRB.BINARY,
                name="link_intersection",
            )
            for i in range(len(super_graph.edges)):
                m.addConstr(link_intersection[i] <= candid_link_vars[i])
                m.addConstr(link_intersection[i] <= initial_links[i])

            logger.info("\t initializing node degree constraint.")
            # degree of node must be <= to the total available transponders
            txp_count = [len(core_G.nodes[x]["transponder"]) for x in core_G.nodes]
            node_degree_vars = m.addVars(
                len(core_G.nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
            )
            for v_idx in range(len(vertices)):
                m.addConstr(
                    node_degree_vars[v_idx]
                    == sum(
                        candid_link_vars[eid]
                        for eid, (v1, v2) in enumerate(super_graph_edge_list)
                        if v1 == vertices[v_idx] or v2 == vertices[v_idx]
                    )
                )
                m.addConstr(node_degree_vars[v_idx] <= txp_count[v_idx])

            logger.info("\tInitializing path_vars")
            path_vars = m.addVars(
                range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
            )

            logger.info("\tInitializing link_util")
            link_util = m.addVars(
                range(len(super_graph.edges)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=LINK_CAPACITY,
                name="link_util",
            )

            logger.info("\tInitializing norm_link_util")
            norm_link_util = m.addVars(
                range(len(super_graph.edges)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=1,
                name="norm_link_util",
            )

            m.addConstrs(
                (
                    norm_link_util[i] == link_util[i] / LINK_CAPACITY
                    for i in range(len(super_graph.edges))
                ),
                name="norm_link_util_constr",
            )

            logger.info("\tInitializing flow_p")
            flow_p = m.addVars(
                range(len(super_paths_list)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                name="flow_p",
            )

            logger.info("Setting Objective.")
            M = m.addVar(vtype=GRB.INTEGER, name="M")
            m.addConstr(
                M == link_intersection.sum(),
                name="objective_constr_min_graph_intersection_M",
            )

            m.setObjective(M, sense=GRB.MINIMIZE)

            logger.info("Initializing Constraints")

            ##############################
            # Add Active Path Constraint #
            ##############################
            logger.info("\tInitializing flow_binary and path_link constraints")

            path_candidate_links = [[] for _ in range(len(path_vars))]
            path_candidate_link_file = os.path.join(
                SCRIPT_HOME,
                "data",
                "paths",
                "optimization",
                self.network + "_path_candidate_links.pkl",
            )

            # try to preload constraint
            # if os.path.exists(path_candidate_link_file):
            #     with open(path_candidate_link_file, 'rb') as pkl:
            #         path_candidate_links = pickle.load(pkl)

            # else:
            path_candidate_links = [[] for _ in range(len(path_vars))]
            for p_i in tqdm(
                range(len(path_vars)),
                desc="Initialling candidate link & path binary vars 1",
                total=len(path_vars),
            ):
                path_i_links = list(
                    zip(super_paths_list[p_i], super_paths_list[p_i][1:])
                )
                for link_l in path_i_links:
                    l1 = list(link_l)
                    l2 = [l1[1], l1[0]]
                    if l1 in candidate_links or l2 in candidate_links:
                        candidate_index = (
                            candidate_links.index(l1)
                            if l1 in candidate_links
                            else candidate_links.index(l2)
                        )
                        path_candidate_links[p_i].append(candidate_index)

                # with open(path_candidate_link_file, 'wb') as pkl:
                #      pickle.dump(path_candidate_links, pkl)

            for p_i in tqdm(
                range(len(path_vars)),
                desc="Initialling candidate link & path binary vars 2",
                total=len(path_vars),
            ):
                m.addConstr(
                    flow_p[p_i] <= path_vars[p_i] * LINK_CAPACITY,
                    "flow_binary_{}".format(p_i),
                )

                path_candidate_link_vars = [
                    candid_link_vars[var_i] for var_i in path_candidate_links[p_i]
                ]
                m.addConstr(
                    path_vars[p_i] == min_(path_candidate_link_vars),
                    name="path_link_constr_{}".format(p_i),
                )

            #####################################################
            # Find demand per tunnel considering active tunnels #
            #####################################################
            logger.info("\tInitializing Find demand per tunnel considering active tunnels")
            for source, target in demand_dict:
                P = self.tunnel_dict[(source, target)]

                m.addConstr(
                    demand_dict[(source, target)] <= quicksum(flow_p[p] for p in P),
                    "flow_{}_{}".format(source, target),
                )

            ###############################################################
            # Find Demand per link considering demand from active tunnels #
            ###############################################################
            logger.info(
                "\tInitializing Find Demand per link considering demand from active tunnels"
            )
            link_tunnels_file = os.path.join(
                SCRIPT_HOME,
                "data",
                "paths",
                "optimization",
                self.network + "_tunnels.pkl",
            )

            logger.info("Generating Link Tunnels: {}".format(link_tunnels_file))
            network_tunnels = [[] for _ in range(len(super_graph.edges))]
            link_index = list(super_graph.edges)[:]
            for tunnel_i, tunnel in tqdm(
                enumerate(super_paths_list),
                desc="Generating link tunnels list.",
                total=len(super_paths_list),
            ):
                for u, v in zip(tunnel, tunnel[1:]):
                    try:
                        l_id = link_index.index((u, v))
                    except ValueError:
                        l_id = link_index.index((v, u))

                    network_tunnels[l_id].append(tunnel_i)

            # # Write File for future
            # with open(link_tunnels_file, "wb") as pkl:
            #     pickle.dump(network_tunnels, pkl)

            assert len(network_tunnels) == len(super_graph.edges)

            for link_i, (link_source, link_target) in tqdm(
                enumerate(list(super_graph.edges)),
                desc="Initializing link utilization constraints.",
                total=len(super_graph.edges),
            ):
                link_tunnels = network_tunnels[link_i]

                m.addConstr(
                    link_util[link_i]
                    == candid_link_vars[link_i]
                    * quicksum(flow_p[i] for i in link_tunnels),
                    "link_demand_{}".format(link_i),
                )

                m.addConstr(
                    self.LINK_CAPACITY >= link_util[link_i],
                    "link_utilization_{}".format(link_i),
                )

            m.update()
            start = process_time()
            m.optimize()
            end = process_time()
            opt_time = start - end
            if m.status == GRB.Status.OPTIMAL:
                logger.info("Optimal solution found.")
                self.populate_changes(candid_link_vars)
            else:
                logger.error("No optimal solution found.")
                if self.debug: 
                    self.write_iss()
            return opt_time
        
    def run_model_max_diff_ignore_demand(self):
        core_G = self.core_G.copy(as_view=True)
        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        super_graph_edge_list = list(super_graph.edges)
        initial_graph_edges_list = list(self.G.edges)
        # self.BUDGET = min(len(self.candidate_links), 10)
        # self.BUDGET = len(self.candidate_links)
        with Env() as env, Model("ONSET", env=env) as m:
            self.model = m
            m.setParam("NodefileStart", 2)
            m.setParam("SoftMemLimit", 5)  # Process dies if uses more than 5 GB

            # ## Variables
            #
            # #### Inputs
            #
            # - $G(V,E)$ : network $G$, vertices $V$, edges, $E$
            # - $e' \in E$      : Original edges
            # - $\hat{e} \in E$ : Candidate edges
            # - *Note*: $\hat{e} \cap e' = \emptyset$ and $\hat{e} \cup e' = E$
            # - $d_{s,t} \in D$ : Demand $d$ between two nodes $s$ and $t$ in demand matrix $D$
            # - $p \in \mathcal{P}$ : A path, sequence of 2 or more nodes $\{v_0, \dots, v_n\}$ s.t., $ \forall i \in N^{[1, n]}, (v_{i-1}, v_i) \in E$
            # - $P_{s,t} \in \mathcal{P}$ : Set of paths $P$ from $s$ to $t$ among all paths $\mathcal{P}$
            # - $\hat{P_{s,t}} \in \mathcal{P}$ : Set of paths $P$ that contain at least one candidate link, $\hat{e}$
            # - ${P'_{s,t}} \in \mathcal{P}$ : Set of paths that contain only original links, ${e'}$
            # - *Note*: $\hat{P_{s,t}} \cap {P'_{s,t}} = \emptyset$ and $\hat{P_{s,t}} \cup {P'_{s,t}} = \mathcal{P}$
            # - ${P^e_{s,t}} \in \mathcal{P}$ : Set of paths that contain a particular link, $e$
            # - $B$: Budget. Number of candidate links that can be provisioned in the network
            # - $C$: Link capacity
            #
            # ### Decision Variables
            #
            # - $b_{e}$: Binary variable. $1$ if edge $e$ is active. $0$ if $e$ is inactive.
            # - $b_{p}$: Binary variable. $1$ if path $p$ is active. $0$ if $p$ is inactive.
            # - $U_{e}$: Utilization of edge $e$
            # - $flow_{p}$: $flow$ allocated onto path $p$
            #
            # ### Auxillary Variable/Constraint
            # - $M$: Objective function auxillary variable, $M = max_{e \in E} U_e$

            print("Initializing Optimizer variables")
            print("\tInitializing candidate_link_vars")
            # candid_link_vars = m.addVars(
            #     range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
            # )

            candid_link_vars = m.addVars(
                len(super_graph.edges),
                vtype=GRB.BINARY,
                name="candidate_links",
            )
            vertices = list(core_G.nodes)
            initial_links = [1 if e in self.G.edges else 0 for e in super_graph.edges]

            link_intersection = m.addVars(
                len(super_graph.edges),
                vtype=GRB.BINARY,
                name="link_intersection",
            )
            for i in range(len(super_graph.edges)):
                m.addConstr(link_intersection[i] <= candid_link_vars[i])
                m.addConstr(link_intersection[i] <= initial_links[i])

            print("\t initializing node degree constraint.")
            # degree of node must be <= to the total available transponders
            txp_count = [len(core_G.nodes[x]["transponder"]) for x in core_G.nodes]
            node_degree_vars = m.addVars(
                len(core_G.nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
            )
            for v_idx in range(len(vertices)):
                m.addConstr(
                    node_degree_vars[v_idx]
                    == sum(
                        candid_link_vars[eid]
                        for eid, (v1, v2) in enumerate(super_graph_edge_list)
                        if v1 == vertices[v_idx] or v2 == vertices[v_idx]
                    )
                )

                m.addConstr(node_degree_vars[v_idx] <= txp_count[v_idx])

            m.addConstr()

            print("Setting Objective.")
            M = m.addVar(vtype=GRB.INTEGER, name="M")
            m.addConstr(
                M == link_intersection.sum(),
                name="objective_constr_min_graph_intersection_M",
            )

            m.setObjective(M, sense=GRB.MINIMIZE)

            m.update()
            start = process_time()
            m.optimize()
            end = process_time()
            opt_time = start - end
            if m.status == GRB.Status.OPTIMAL:
                logger.info("Optimal solution found.")
                self.populate_changes(candid_link_vars)
            else:
                logger.error("No optimal solution found.")
                if self.debug: 
                    self.write_iss()
            return opt_time

    def get_links_to_add(self):
        return self.links_to_add
        
    def get_links_to_drop(self):
        return self.links_to_drop


def work_log(identifier, total, candidate_link_list, G, network):
    with open("./data/paths/optimization/{}_original.json".format(network), "r") as fob:
        json_obj = json.load(fob)
        original_tunnel_list = json_obj["list"]

    original_shortest_s_t_path = {}
    original_tunnel_dict = defaultdict(list)
    for i, tunnel in enumerate(original_tunnel_list):
        source, target = tunnel[0], tunnel[-1]
        original_tunnel_dict[(source, target)].append(i)
        original_shortest_s_t_path[(source, target)] = len(tunnel)

    # print(" Creating supergraph with {} link(s)".format(candidate_link_list))
    super_graph = G.copy()
    for new_link in candidate_link_list:
        super_graph.add_edge(*new_link)

    def all_candidate_on_path(candidates, path):
        candidates_found = 0
        for c in candidates:
            for link in zip(path, path[1:]):
                candidates_found += (c[0] == link[0] and c[1] == link[1]) or (
                    c[0] == link[1] and c[1] == link[0]
                )
        return candidates_found == len(candidates)

    print("{} / {} = {}".format(identifier, total, identifier / total))
    # print("Pre-computing paths on super-graph with {} link(s)".format(candidate_link_list))
    tunnel_list = []
    tunnel_dict = defaultdict(list)
    # for source in tqdm(super_graph.nodes, desc="Pre-computing paths."):
    for source in super_graph.nodes:
        for target in super_graph.nodes:
            if (
                source != target
                and "{}----{}".format(source, target) not in tunnel_dict
            ):
                s_t_paths = nx.shortest_simple_paths(super_graph, source, target)
                shortest_s_t_path_len = original_shortest_s_t_path[(source, target)]
                # for s_t_path in tqdm(s_t_paths,
                #                 desc="Calculating ({}, {}) paths shorter than {} hops.".format(
                #                     source, target, shortest_s_t_path_len)):
                for s_t_path in s_t_paths:
                    if len(s_t_path) > shortest_s_t_path_len:
                        break

                    if all_candidate_on_path(candidate_link_list, s_t_path):
                        reversed_path = list(reversed(s_t_path))
                        tunnel_list.append(s_t_path)
                        tunnel_list.append(reversed_path)
                        tunnel_dict["{}----{}".format(source, target)].append(s_t_path)
                        tunnel_dict["{}----{}".format(target, source)].append(
                            reversed_path
                        )

    with open("./.temp/{}.json".format(identifier), "w") as fob:
        # json.dump({"list": tunnel_list, "dict": tunnel_dict},fob)
        json.dump({"list": tunnel_list}, fob, indent=4)


def main():
    if __name__ == "__main__":

        topo_path = "data/graphs/gml/surfNet.gml"
        network = "surfNet"
        demand_matrix = "data/traffic/surfNet"

        G = nx.read_gml(topo_path)

        # optimizer = Link_optimization(G, demand_matrix, network)
        # Link_optimization(G, demand_matrix, network, candidate_set="liberal")
        # Link_optimization(G, demand_matrix, network, candidate_set="conservative")

        Link_optimization(G, demand_matrix, network, candidate_set="conservative", compute_paths=True)
        """
        Solution for multi objective problem. 
        demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_50Gbps_targets_5_iteration_2_strength_200"

        12 links added. 
        ['Atlanta', 'Cheyenne']
        ['Cheyenne', 'Chicago']
        ['Cheyenne', 'Washington, DC']
        ['Atlanta', 'Boulder']
        ['Atlanta', 'Chicago']
        ['Boulder', 'Seattle']
        ['Boulder', 'Stockton']
        ['Anaheim', 'Boulder']
        ['Boulder', 'Kansas City']
        ['Boulder', 'New York (Pennsauken)']
        ['Fort Worth', 'Stockton']
        ['Kansas City', 'Stockton']

        Solving for fixed budget of 11 links should be infeasible.


        #### UPDATE ####
        Solution was found for 11 links. Objective value was 1E11
        ['Atlanta', 'Cheyenne']
        ['Atlanta', 'Boulder']
        ['Atlanta', 'Stockton']
        ['Boulder', 'Seattle']
        ['Boulder', 'Stockton']
        ['Anaheim', 'Boulder']
        ['Boulder', 'Fort Worth']
        ['Boulder', 'Kansas City']
        ['Boulder', 'Chicago']
        ['Boulder', 'New York (Pennsauken)']
        ['Boulder', 'Washington, DC']

        Solving for fixed budget of 10 links should REALLY be infeasible.
        
        
        """
        # optimizer.run_model_minimize_cost()


main()
