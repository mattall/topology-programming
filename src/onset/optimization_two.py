# builtins
import os
from itertools import combinations, permutations
from collections import defaultdict
from multiprocessing import Pool
import json
import pickle

# third-party
from gurobipy import Model, GRB, quicksum, min_, max_, Env, tupledict
import networkx as nx
import numpy as np
from tqdm import tqdm
from onset.constants import SCRIPT_HOME

# customs
from onset.utilities.plot_reconfig_time import calc_haversine
from onset.utilities.graph_utils import link_on_path


class Link_optimization:
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
        compute_paths=False
    ):
        self.G = G
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
            print("unknow demand type")
            self.demand_matrix_file = None
            self.demand_matrix = None
            self.demand_dict = None

        self.network = network
        if isinstance(core_G, nx.Graph):
            self.core_G = core_G
            self.txp_count = [
                2*len(core_G.nodes[x]["transponder"]) for x in core_G.nodes
            ]
        else:
            self.core_G = self.G.copy(as_view=True)
        
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
        self.candidate_links = []
        self.tunnel_list = []
        self.links_to_add = []
        self.tunnel_dict = defaultdict(list)
        self.original_tunnel_list = []
        self.core_shortest_path_len = defaultdict(lambda: np.inf)
        if self.demand_dict == None:
            self.initialize_demand()
        self.initialize_candidate_links()
        self.add_candidate_links_to_super_graph()
        if compute_paths:
            self.get_shortest_paths()
        
    def get_shortest_paths_from_k_link_super_graphs(self, k):
        # Run after finding candidate links
        # Given K, find every combination of k candidate links
        # For each combination, in parallel,
        #   1. Create a supergraph on the base topology
        #   2. Find the shortest paths as short as or shorter than
        #       the original shortest path for every pair of nodes.
        #       2.1 If the path includes any candidate links, save this path.
        #   3. Return the set of saved paths.
        print("Getting work ready.")
        work = []
        count = 0
        for i in range(1, k + 1):
            link_sets = list(combinations(self.candidate_links, i))
            for c, link_set in enumerate(link_sets):
                count += 1
                work.append([count, 1, link_set, self.G, self.network])
        print("reindexing work.")
        for w in work:
            w[1] = len(work)
        # work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])
        print("Starting the work.")
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
            self.demand_dict[(source, dest)] = float(
                self.demand_matrix[dim * i + j]
            )

    def update_shortest_path_len(self, this_path_len, source, target):
        prev_path_len = self.core_shortest_path_len[(source, target)]
        self.core_shortest_path_len[
            (source, target)
        ] = self.core_shortest_path_len[(target, source)] = min(
            prev_path_len, this_path_len
        )

    def initialize_candidate_links(self):
        ########################################################
        #### Create list of candidate links a.k.a. Shortcuts ###
        ########################################################
        core_G = self.core_G
        candidate_links = self.candidate_links
        if core_G == None:
            core_G = self.G

        candidate_links = [sorted(l) for l in core_G.edges]

        for source, target in self.all_node_pairs:
            if (source, target) not in core_G.edges and sorted(
                (source, target)
            ) not in candidate_links:
                shortest_paths = list(
                    nx.all_shortest_paths(core_G, source, target)
                )
                shortest_path_len = len(shortest_paths[0])
                shortest_path_hops = shortest_path_len - 1
                self.update_shortest_path_len(
                    shortest_path_len, source, target
                )
                if shortest_path_hops == 2:
                    hop_dist_ok = True
                else:
                    hop_dist_ok = False

                geo_dist_ok = False
                min_geo_dist = float("inf")
                for p in shortest_paths:
                    distance = 0
                    for u, v in zip(p, p[1:]):
                        if "Latitide" not in core_G.nodes[u]:
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
                    print(
                        "Minimum distance, {} km, too far between {} and {}".format(
                            min_geo_dist, source, target
                        )
                    )

                if geo_dist_ok and hop_dist_ok:
                    candidate_links.append(sorted((source, target)))

        self.candidate_links = candidate_links

    def add_candidate_links_to_super_graph(self):
        self.super_graph.add_edges_from(self.candidate_links)

    def get_shortest_paths(self):
        ########################################
        #### Derive paths that use shortcuts ###
        ########################################
        self.get_shortest_original_paths()
        try:
            assert self.use_cache
            self.load_paths()
            print("Loaded paths from disc.")

            # Don't feel link making keys strings just to save dict to json. Maybe later.
            self.tunnel_dict = defaultdict(list)
            for i, tunnel in enumerate(self.tunnel_list):
                source, target = tunnel[0], tunnel[-1]
                self.tunnel_dict[(source, target)].append(i)

        except:
            if self.PARALLEL:
                self.get_shortest_paths_from_k_link_super_graphs(self.k)
            else:
                self.tunnel_list = []
                self.tunnel_dict = defaultdict(list)
                super_graph = self.super_graph
                for source in tqdm(
                    super_graph.nodes, desc="Pre-computing paths."
                ):
                    for target in super_graph.nodes:
                        if (
                            source != target
                            and (source, target) not in self.tunnel_dict
                        ):
                            s_t_paths = nx.shortest_simple_paths(
                                super_graph, source, target
                            )
                            shortest_s_t_path_len = (
                                self.core_shortest_path_len[(source, target)]
                            )
                            for s_t_path in tqdm(
                                s_t_paths,
                                desc="Calculating ({}, {}) paths shorter than {} hops.".format(
                                    source, target, shortest_s_t_path_len
                                ),
                            ):
                                if len(s_t_path) > shortest_s_t_path_len:
                                    break

                                reversed_path = list(reversed(s_t_path))
                                self.tunnel_list.append(s_t_path)
                                self.tunnel_list.append(reversed_path)
                                # self.tunnel_dict[(source, target)].append(s_t_path)
                                # self.tunnel_dict[(target, source)].append(reversed_path)

                for i, tunnel in enumerate(self.tunnel_list):
                    source, target = tunnel[0], tunnel[-1]
                    self.tunnel_dict[(source, target)].append(i)

                self.save_paths()
            print("Computed paths and saved to disc.")

    def get_shortest_original_paths(self):
        ########################################
        #### Derive paths that use shortcuts ###
        ########################################
        try:
            assert self.use_cache
            self.load_original_paths()
            print("Loaded original paths from disc.")
            self.original_tunnel_dict = defaultdict(list)
            for i, tunnel in enumerate(self.original_tunnel_list):
                source, target = tunnel[0], tunnel[-1]
                self.original_tunnel_dict[(source, target)].append(i)
                self.core_shortest_path_len[(source, target)] = len(tunnel)

        except:
            self.original_tunnel_list = []
            self.core_shortest_path_len = {}
            G = self.G
            for source in tqdm(G.nodes, desc="Pre-computing paths."):
                for target in G.nodes:
                    if (
                        source != target
                        and (source, target) not in self.core_shortest_path_len
                    ):
                        s_t_paths = nx.shortest_simple_paths(G, source, target)
                        shortest_s_t_path_len = np.inf
                        for s_t_path in tqdm(
                            s_t_paths, desc="Path", leave=False
                        ):
                            if shortest_s_t_path_len == np.inf:
                                shortest_s_t_path_len = len(s_t_path)
                                self.core_shortest_path_len[
                                    (source, target)
                                ] = shortest_s_t_path_len
                                self.core_shortest_path_len[
                                    (target, source)
                                ] = shortest_s_t_path_len
                            if len(s_t_path) > shortest_s_t_path_len:
                                break
                            self.original_tunnel_list.append(s_t_path)
                            reversed_path = list(reversed(s_t_path))
                            self.original_tunnel_list.append(reversed_path)

            self.save_original_paths()
            print("Computed original paths and saved to disc.")

    def load_paths(self):
        if self.use_cache:
            with open(
                "./data/paths/optimization/{}.json".format(self.network), "r"
            ) as fob:
                json_obj = json.load(fob)
                self.tunnel_list = json_obj["list"]
            return
        else:
            return

    def save_paths(self):
        with open(
            "./data/paths/optimization/{}.json".format(self.network), "w"
        ) as fob:
            return json.dump({"list": self.tunnel_list}, fob, indent=4)

    def load_original_paths(self):
        if self.use_cache:
            with open(
                "./data/paths/optimization/{}_original.json".format(
                    self.network
                ),
                "r",
            ) as fob:
                json_obj = json.load(fob)
                self.original_tunnel_list = json_obj["list"]
            return
        else:
            return

    def save_original_paths(self):
        os.makedirs("./data/paths/optimization/", exist_ok=True)
        with open(
            "./data/paths/optimization/{}_original.json".format(self.network),
            "w",
        ) as fob:
            return json.dump(
                {"list": self.original_tunnel_list}, fob, indent=4
            )
    
    def get_flow_allocations(self):
        m = self.model
        flow_vars = self.flow_vars
        demand = self.demand_dict
        prime_edges = self.prime_edges
        flow_paths = self.flow_paths
        if m.status == GRB.Status.OPTIMAL:
            print("Optimal solution found.")        
            for (source, target) in demand.keys():
                paths = []
                for (u, v) in prime_edges:
                    if flow_vars[source, target, u, v].x > 0:
                        paths.append({
                            "path": [u, v],
                            "weight": flow_vars[source, target, u, v].x
                        })
                flow_paths[(source, target)] = paths
                
                print(f"Flow from {source} to {target}:")
                for path in paths:
                    print(f"Path: {path['path']} - Weight: {path['weight']}")


    def mcf(self):
        LINK_CAPACITY = self.LINK_CAPACITY

        m = self.model = Model("MulticommodityFlow")
        
        # Convert the graph to a directed graph
        
        G_0 = self.G.to_directed()
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
        initial_edge_vars = tupledict({(u, v): 1 if (u,v) in initial_edges else 0 for (u,v) in prime_edges })

        # The overlapping set of edges for initial_edge_vars and edge_vars
        link_intersection = m.addVars(
            prime_edges,
            vtype=GRB.BINARY,
            name="link_intersection",
        )    
        m.addConstrs(
            link_intersection[(u, v)] == min_(edge_vars[u,v], initial_edge_vars[u,v]) for (u, v) in prime_edges
        )

        # Enforce max degree based on transponders at each node
        for v_idx, vertex in enumerate(nodes):
            m.addConstr(
                node_degree_vars[v_idx]
                == sum(
                    edge_vars[u, v]
                    for (u, v) in edge_vars
                    if u == vertex or v == vertex
                )
            )
            m.addConstr( txp_count[v_idx] >= node_degree_vars[v_idx])



        # Add flow variables for commodities and edges
        flow_vars = m.addVars(demand.keys(), prime_edges, vtype=GRB.CONTINUOUS, name="flow")
        
        # Add conservation of flow constraints for nodes and commodities
        for node in nodes:
            for (source, target) in demand.keys():
                inflow = quicksum(flow_vars[source, target, u, v] for (u, v) in G_prime.in_edges(node))
                outflow = quicksum(flow_vars[source, target, u, v] for (u, v) in G_prime.out_edges(node))                
                if node == source:
                    m.addConstr(inflow - outflow == -demand[source, target])
                elif node == target:
                    m.addConstr(inflow - outflow == demand[source, target])
                else:
                    m.addConstr(inflow - outflow == 0)
        
        # Add capacity constraints for edges
        edge_capacity = tupledict()
        for (u, v) in prime_edges:
            edge_capacity[u, v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=LINK_CAPACITY, name=f"capacity_{u}_{v}")
            m.addConstr(edge_capacity[u, v] >= quicksum(flow_vars[source, target, u, v] for (source, target) in demand.keys()))

        # Enforce symetrical bi-directional capacity
        for (u, v) in directionless_edges:
            m.addConstr(edge_capacity[u, v] == edge_capacity[v, u])


        # Add binary constraint on edges
        for (u, v) in prime_edges:        
            m.addConstr(edge_capacity[u,v] == edge_vars[(u,v)] * LINK_CAPACITY)
        
        # Set the objective to minimize the total flow        
        # m.setObjective(quicksum(flow_vars[source, target, u, v] for (source, target) in demand.keys() for (u, v) in prime_edges), sense=GRB.MINIMIZE)
        m.setObjective(quicksum(link_intersection[u,v] for (u,v) in prime_edges), sense=GRB.MINIMIZE)
        
        # Optimize the model
        m.optimize()
        self.flow_vars = flow_vars
        if m.status == GRB.Status.OPTIMAL:
            print("Optimal solution found.")
            resultant_edges = [
                (u,v) 
                for (u,v) in edge_vars
                if edge_vars[(u,v)].x == 1
            ] 
            add_edges = [
                e for e in resultant_edges if e not in self.G.edges
            ]
            drop_edges = [
                e for e in self.G.edges if e not in resultant_edges
            ]
            G_new = nx.DiGraph()
            G_new.add_edges_from(resultant_edges)
            assert(nx.is_strongly_connected(G_new))
            add_edges = list(set([tuple(sorted((u, v))) for u, v in add_edges]))
            drop_edges = list(set([tuple(sorted((u, v))) for u, v in drop_edges]))
            return (add_edges, drop_edges)


        else:
            print("No optimal solution found.")
        
        return -1

    def run_model_max_diff(self):
        core_G = self.core_G
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
            m.setParam(
                "SoftMemLimit", 5
            )  # Process dies if uses more than 5 GB

            print("Initializing Optimizer variables")
            print("\tInitializing candidate_link_vars")

            candid_link_vars = m.addVars(
                len(super_graph.edges),
                vtype=GRB.BINARY,
                name="candidate_links",
            )
            vertices = list(core_G.nodes)
            initial_links = [
                1 if e in self.G.edges else 0 for e in super_graph.edges
            ]

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
            txp_count = [
                len(core_G.nodes[x]["transponder"]) for x in core_G.nodes
            ]
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

            print("\tInitializing path_vars")
            path_vars = m.addVars(
                range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
            )

            print("\tInitializing link_util")
            link_util = m.addVars(
                range(len(super_graph.edges)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=LINK_CAPACITY,
                name="link_util",
            )

            print("\tInitializing norm_link_util")
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

            print("\tInitializing flow_p")
            flow_p = m.addVars(
                range(len(super_paths_list)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                name="flow_p",
            )

            print("Setting Objective.")
            M = m.addVar(vtype=GRB.INTEGER, name="M")
            m.addConstr(
                M == link_intersection.sum(),
                name="objective_constr_min_graph_intersection_M",
            )

            m.setObjective(M, sense=GRB.MINIMIZE)

            print("Initializing Constraints")

            ##############################
            # Add Active Path Constraint #
            ##############################
            print("\tInitializing flow_binary and path_link constraints")

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
                    candid_link_vars[var_i]
                    for var_i in path_candidate_links[p_i]
                ]
                m.addConstr(
                    path_vars[p_i] == min_(path_candidate_link_vars),
                    name="path_link_constr_{}".format(p_i),
                )

            #####################################################
            # Find demand per tunnel considering active tunnels #
            #####################################################
            print(
                "\tInitializing Find demand per tunnel considering active tunnels"
            )
            for source, target in demand_dict:
                P = self.tunnel_dict[(source, target)]

                m.addConstr(
                    demand_dict[(source, target)]
                    <= quicksum(flow_p[p] for p in P),
                    "flow_{}_{}".format(source, target),
                )

            ###############################################################
            # Find Demand per link considering demand from active tunnels #
            ###############################################################
            print(
                "\tInitializing Find Demand per link considering demand from active tunnels"
            )
            link_tunnels_file = os.path.join(
                SCRIPT_HOME,
                "data",
                "paths",
                "optimization",
                self.network + "_tunnels.pkl",
            )
            
            print("Generating Link Tunnels: {}".format(link_tunnels_file))
            network_tunnels = [
                [] for _ in range(len(super_graph.edges))
            ]
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
                    == candid_link_vars[link_i] * quicksum(flow_p[i] for i in link_tunnels),
                    "link_demand_{}".format(link_i),
                )

                m.addConstr(
                    self.LINK_CAPACITY
                    >= link_util[link_i],
                    "link_utilization_{}".format(link_i),
                )

            m.update()
            m.optimize()
            if m.status == GRB.Status.OPTIMAL:
                # links_to_add = []
                # for clv in candid_link_vars:
                #     if candid_link_vars[clv].x == 1:
                #         links_to_add.append(candidate_links[clv])
                # self.links_to_add = links_to_add
                resultant_edges = [
                    list(super_graph.edges)[i]
                    for i in range(len(candid_link_vars))
                    if candid_link_vars[i].x == 1
                ]
                add_edges = [
                    e for e in resultant_edges if e not in self.G.edges
                ]
                drop_edges = [
                    e for e in self.G.edges if e not in resultant_edges
                ]
                return (resultant_edges, add_edges, drop_edges)
            else: 
                m.computeIIS()
                # Print out the IIS constraints and variables
                print('\nThe following constraints and variables are in the IIS:')
                for c in m.getConstrs():
                    if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

                for v in m.getVars():
                    if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                    if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

                m.write('iismodel.ilp')



    # def run_model_max_diff_ignore_demand(self):
    #     core_G = self.core_G
    #     super_graph = self.super_graph
    #     candidate_links = self.candidate_links
    #     super_paths_list = self.tunnel_list
    #     LINK_CAPACITY = self.LINK_CAPACITY
    #     super_graph_edge_list = list(super_graph.edges)
    #     initial_graph_edges_list = list(self.G.edges)
    #     # self.BUDGET = min(len(self.candidate_links), 10)
    #     # self.BUDGET = len(self.candidate_links)
    #     with Env() as env, Model("ONSET", env=env) as m:
    #         self.model = m
    #         m.setParam("NodefileStart", 2)
    #         m.setParam(
    #             "SoftMemLimit", 5
    #         )  # Process dies if uses more than 5 GB

    #         # ## Variables
    #         #
    #         # #### Inputs
    #         #
    #         # - $G(V,E)$ : network $G$, vertices $V$, edges, $E$
    #         # - $e' \in E$      : Original edges
    #         # - $\hat{e} \in E$ : Candidate edges
    #         # - *Note*: $\hat{e} \cap e' = \emptyset$ and $\hat{e} \cup e' = E$
    #         # - $d_{s,t} \in D$ : Demand $d$ between two nodes $s$ and $t$ in demand matrix $D$
    #         # - $p \in \mathcal{P}$ : A path, sequence of 2 or more nodes $\{v_0, \dots, v_n\}$ s.t., $ \forall i \in N^{[1, n]}, (v_{i-1}, v_i) \in E$
    #         # - $P_{s,t} \in \mathcal{P}$ : Set of paths $P$ from $s$ to $t$ among all paths $\mathcal{P}$
    #         # - $\hat{P_{s,t}} \in \mathcal{P}$ : Set of paths $P$ that contain at least one candidate link, $\hat{e}$
    #         # - ${P'_{s,t}} \in \mathcal{P}$ : Set of paths that contain only original links, ${e'}$
    #         # - *Note*: $\hat{P_{s,t}} \cap {P'_{s,t}} = \emptyset$ and $\hat{P_{s,t}} \cup {P'_{s,t}} = \mathcal{P}$
    #         # - ${P^e_{s,t}} \in \mathcal{P}$ : Set of paths that contain a particular link, $e$
    #         # - $B$: Budget. Number of candidate links that can be provisioned in the network
    #         # - $C$: Link capacity
    #         #
    #         # ### Decision Variables
    #         #
    #         # - $b_{e}$: Binary variable. $1$ if edge $e$ is active. $0$ if $e$ is inactive.
    #         # - $b_{p}$: Binary variable. $1$ if path $p$ is active. $0$ if $p$ is inactive.
    #         # - $U_{e}$: Utilization of edge $e$
    #         # - $flow_{p}$: $flow$ allocated onto path $p$
    #         #
    #         # ### Auxillary Variable/Constraint
    #         # - $M$: Objective function auxillary variable, $M = max_{e \in E} U_e$

    #         print("Initializing Optimizer variables")
    #         print("\tInitializing candidate_link_vars")
    #         # candid_link_vars = m.addVars(
    #         #     range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
    #         # )

    #         candid_link_vars = m.addVars(
    #             len(super_graph.edges),
    #             vtype=GRB.BINARY,
    #             name="candidate_links",
    #         )
    #         vertices = list(core_G.nodes)
    #         initial_links = [
    #             1 if e in self.G.edges else 0 for e in super_graph.edges
    #         ]

    #         link_intersection = m.addVars(
    #             len(super_graph.edges),
    #             vtype=GRB.BINARY,
    #             name="link_intersection",
    #         )
    #         for i in range(len(super_graph.edges)):
    #             m.addConstr(link_intersection[i] <= candid_link_vars[i])
    #             m.addConstr(link_intersection[i] <= initial_links[i])

    #         print("\t initializing node degree constraint.")
    #         # degree of node must be <= to the total available transponders
    #         txp_count = [
    #             len(core_G.nodes[x]["transponder"]) for x in core_G.nodes
    #         ]
    #         node_degree_vars = m.addVars(
    #             len(core_G.nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
    #         )
    #         for v_idx in range(len(vertices)):
    #             m.addConstr(
    #                 node_degree_vars[v_idx]
    #                 == sum(
    #                     candid_link_vars[eid]
    #                     for eid, (v1, v2) in enumerate(super_graph_edge_list)
    #                     if v1 == vertices[v_idx] or v2 == vertices[v_idx]
    #                 )
    #             )

    #             m.addConstr(node_degree_vars[v_idx] <= txp_count[v_idx])

    #         m.addConstr()

    #         print("Setting Objective.")
    #         M = m.addVar(vtype=GRB.INTEGER, name="M")
    #         m.addConstr(
    #             M == link_intersection.sum(),
    #             name="objective_constr_min_graph_intersection_M",
    #         )

    #         m.setObjective(M, sense=GRB.MINIMIZE)

    #         m.update()
    #         m.optimize()
    #         if m.status == GRB.Status.OPTIMAL:
    #             # links_to_add = []
    #             # for clv in candid_link_vars:
    #             #     if candid_link_vars[clv].x == 1:
    #             #         links_to_add.append(candidate_links[clv])
    #             # self.links_to_add = links_to_add
    #             resultant_edges = [
    #                 list(super_graph.edges)[i]
    #                 for i in range(len(candid_link_vars))
    #                 if candid_link_vars[i].x == 1
    #             ]
    #             add_edges = [
    #                 e for e in resultant_edges if e not in self.G.edges
    #             ]
    #             drop_edges = [
    #                 e for e in self.G.edges if e not in resultant_edges
    #             ]
    #             return (resultant_edges, add_edges, drop_edges)

    # def get_links_to_add(self):
    #     if len(self.links_to_add) > 0:
    #         return self.links_to_add
    #     else:
    #         return []


def work_log(identifier, total, candidate_link_list, G, network):
    with open(
        "./data/paths/optimization/{}_original.json".format(network), "r"
    ) as fob:
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
                s_t_paths = nx.shortest_simple_paths(
                    super_graph, source, target
                )
                shortest_s_t_path_len = original_shortest_s_t_path[
                    (source, target)
                ]
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
                        tunnel_dict["{}----{}".format(source, target)].append(
                            s_t_path
                        )
                        tunnel_dict["{}----{}".format(target, source)].append(
                            reversed_path
                        )

    with open("./.temp/{}.json".format(identifier), "w") as fob:
        # json.dump({"list": tunnel_list, "dict": tunnel_dict},fob)
        json.dump({"list": tunnel_list}, fob, indent=4)


def main():
    if __name__ == "__main__":
        # G = nx.read_gml("./data/graphs/gml/sprint.gml")
        # G = nx.read_gml("./data/graphs/gml/linear_3.gml")
        # GML_File = (
        #     "/home/mhall/network_stability_sim/data/graphs/gml/sprint_test.gml"
        # )

        topo_path = "/home/m/src/topology-programming/data/results/ground_truth_uh_circuits_5__-mcf/ground_truth_1-1.gml"
        network = "campus"

        demand_matrix = "/home/mhall/network_stability_sim/data/traffic/sprint_benign_0Gbps_targets_5_iteration_2_strength_100_mix"

        G = nx.read_gml(topo_path)

        demand_matrix = (
            "/home/m/src/topology-programming/data/traffic/ground_truth.txt"
        )

        # demand_matrix = "./data/traffic-2022-01-29/sprint_240Gbps.txt"
        # demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_50Gbps_targets_5_iteration_2_strength_200"
        # demand_matrix = "/home/matt/network_stability_sim/temp/sprint_benign_50Gbps_5x200Gbps_3_oneShot.txt"
        # demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_0Gbps_targets_3_iteration_1_strength_100_atk"
        # demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_0Gbps_targets_3_iteration_1_strength_150_atk"

        # demand_matrix = "/home/mhall/network_stability_sim/data/traffic/linear_3_benign_0Gbps_targets_1_iteration_1_strength_150_atk"

        # demand_matrix = "/home/matt/network_stability_sim/26495ffb2405ed09de0cf24bc1d54c5d0eb56579"
        # network = "linear_3"

        optimizer = Link_optimization(G, demand_matrix, network)

        # optimizer.run_model()

        # optimizer.run_model_minimize_cost()
        # optimizer.run_model_mixed_objective()
        result, add, drop = optimizer.run_model_max_diff()

        # optimizer.run_model_v2()
        # optimizer.run_model_minimize_cost_v1()
        print(optimizer.get_links_to_add())
        # for i in range(len(optimizer.candidate_links)):
        #     print(optimizer.model.getVarByName("b_link[{}]".format(i)))

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
