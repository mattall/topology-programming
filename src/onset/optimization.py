# builtins
from distutils.command.build_scripts import first_line_re
from email.policy import default
import os
import sys
from itertools import combinations, permutations
from collections import defaultdict
from multiprocessing import Pool
import json
import pickle

# third-party
from gurobipy import Model, GRB, quicksum, min_, max_, Env
import networkx as nx
import numpy as np
from tqdm import tqdm
from onset.constants import SCRIPT_HOME

# customs
from onset.utilities.plot_reconfig_time import calc_haversine
from onset.utilities.graph_utils import link_on_path


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

    with open("./temp/{}.json".format(identifier), "w") as fob:
        # json.dump({"list": tunnel_list, "dict": tunnel_dict},fob)
        json.dump({"list": tunnel_list}, fob, indent=4)


class Link_optimization:
    def __init__(
        self, G: nx.Graph, BUDGET: int, demand_matrix: str, network: str
    ):
        self.PARALLEL = True
        self.k = 1
        self.MAX_DISTANCE = 5000  # km
        self.LINK_CAPACITY = 100 * 10**9  # bps
        self.BUDGET = BUDGET
        self.network = network
        # self.G = nx.read_gml("./data/graphs/gml/sprint.gml")
        self.G = G
        self.super_graph = G.copy()
        self.all_node_pairs = list(permutations(self.G.nodes, 2))
        # self.demand_matrix = np.loadtxt("./data/traffic/sprint_240Gbps.txt")
        self.demand_matrix_file = demand_matrix
        self.demand_matrix = np.loadtxt(demand_matrix)
        self.model = None
        self.demand_dict = {}
        self.candidate_links = []
        self.tunnel_list = []
        self.links_to_add = []
        self.tunnel_dict = defaultdict(list)
        self.original_tunnel_list = []
        self.original_shortest_s_t_path = defaultdict(lambda: np.inf)
        self.initialize_demand()
        self.initialize_candidate_links()
        self.add_candidate_links_to_super_graph()
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
        temp_file = "./temp/{}.json".format(identifier)
        with open(temp_file, "r") as fob:
            obj = json.load(fob)

        os.remove(temp_file)
        self.tunnel_list.extend(obj["list"])

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

    def initialize_candidate_links(self):
        ########################################################
        #### Create list of candidate links a.k.a. Shortcuts ###
        ########################################################
        self.candidate_links = []
        G = self.G
        for source, target in self.all_node_pairs:
            if (source, target) not in G.edges and sorted(
                (source, target)
            ) not in self.candidate_links:
                distance = calc_haversine(
                    G.nodes[source]["Latitude"],
                    G.nodes[source]["Longitude"],
                    G.nodes[target]["Latitude"],
                    G.nodes[target]["Longitude"],
                )
                if distance < self.MAX_DISTANCE:
                    self.candidate_links.append(sorted((source, target)))
                else:
                    print(
                        "Distance, {} km, too far between {} and {}".format(
                            distance, source, target
                        )
                    )

    def add_candidate_links_to_super_graph(self):
        self.super_graph.add_edges_from(self.candidate_links)

    def get_shortest_paths(self):
        ########################################
        #### Derive paths that use shortcuts ###
        ########################################
        self.get_shortest_original_paths()
        try:
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
                                self.original_shortest_s_t_path[
                                    (source, target)
                                ]
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
            self.load_original_paths()
            print("Loaded original paths from disc.")
            self.original_tunnel_dict = defaultdict(list)
            for i, tunnel in enumerate(self.original_tunnel_list):
                source, target = tunnel[0], tunnel[-1]
                self.original_tunnel_dict[(source, target)].append(i)
                self.original_shortest_s_t_path[(source, target)] = len(tunnel)

        except:
            self.original_tunnel_list = []
            self.original_shortest_s_t_path = {}
            G = self.G
            for source in tqdm(G.nodes, desc="Pre-computing paths."):
                for target in G.nodes:
                    if (
                        source != target
                        and (source, target)
                        not in self.original_shortest_s_t_path
                    ):
                        s_t_paths = nx.shortest_simple_paths(G, source, target)
                        shortest_s_t_path_len = np.inf
                        for s_t_path in tqdm(
                            s_t_paths, desc="Path", leave=False
                        ):
                            if shortest_s_t_path_len == np.inf:
                                shortest_s_t_path_len = len(s_t_path)
                                self.original_shortest_s_t_path[
                                    (source, target)
                                ] = shortest_s_t_path_len
                                self.original_shortest_s_t_path[
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
        with open(
            "./data/paths/optimization/{}.json".format(self.network), "r"
        ) as fob:
            json_obj = json.load(fob)
            self.tunnel_list = json_obj["list"]

    def save_paths(self):
        with open(
            "./data/paths/optimization/{}.json".format(self.network), "w"
        ) as fob:
            return json.dump({"list": self.tunnel_list}, fob, indent=4)

    def load_original_paths(self):
        with open(
            "./data/paths/optimization/{}_original.json".format(self.network),
            "r",
        ) as fob:
            json_obj = json.load(fob)
            self.original_tunnel_list = json_obj["list"]

    def save_original_paths(self):
        os.makedirs("./data/paths/optimization/", exist_ok=True)
        with open(
            "./data/paths/optimization/{}_original.json".format(self.network),
            "w",
        ) as fob:
            return json.dump(
                {"list": self.original_tunnel_list}, fob, indent=4
            )

    def run_model_mixed_objective(self):
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
            m.setParam(
                "SoftMemLimit", 5
            )  # Process dies if uses more than 5 GB

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
            candid_link_vars = m.addVars(
                range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
            )

            print("\tInitializing path_vars")
            path_vars = m.addVars(
                range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
            )

            print("\tInitializing link_util")
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
            print("\tInitializing norm_link_util")
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

            print("\tInitializing flow_p")
            flow_p = m.addVars(
                range(len(super_paths_list)),
                vtype=GRB.CONTINUOUS,
                lb=0,
                name="flow_p",
            )

            print("Setting Objective.")
            M = m.addVar(vtype=GRB.CONTINUOUS, name="M")
            m.addConstr(
                M == max_(norm_link_util), name="aux_objective_constr_M"
            )

            # ### Objective
            # Minimize $M$

            N = m.addVar(vtype=GRB.CONTINUOUS, name="N", lb=1)
            m.addConstr(
                N == quicksum(candid_link_vars), name="aux_objective_constr_M"
            )

            O = m.addVar(vtype=GRB.CONTINUOUS, name="O")
            m.addConstr(O == M + N, name="aux_objective_constr_O")

            m.setObjective(O, sense=GRB.MINIMIZE)

            print("Initializing Constraints")
            # ### Budget Constraint
            #
            # $\sum_{\hat{e} \in E} b_\hat{e} \leq B$
            # print("Adding Model Constraint, Budget: {}".format(self.BUDGET))
            print("\tInitializing Budget")
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
                if path_candidate_links[p_i] != []:
                    path_candidate_link_vars = [
                        candid_link_vars[var_i]
                        for var_i in path_candidate_links[p_i]
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

            # ### Total link utilization from all active paths constraint
            #
            # $\forall e \in E: U_e = \sum_{p | e \in p } flow_{p} $
            #
            # $\forall e \in E: \sum_{p | e \in p } flow_{p} \leq C $
            #
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

            # FILE EXISTS
            if os.path.exists(link_tunnels_file):
                # NOT EMPTY
                print(
                    "Loading Link Tunnels file: {}".format(link_tunnels_file)
                )
                if os.path.getsize(link_tunnels_file) > 0:
                    with open(link_tunnels_file, "rb") as pkl:
                        network_tunnels = pickle.load(pkl)

            else:
                print("Generating Link Tunnels: {}".format(link_tunnels_file))
                network_tunnels = []
                # for all links.
                if 0:
                    for link_i, (link_source, link_target) in enumerate(
                        list(super_graph.edges())
                    ):
                        link_tunnels = []
                        # for all tunnels
                        for tunnel_i, tunnel in enumerate(super_paths_list):
                            if link_on_path(
                                tunnel, [link_source, link_target]
                            ):
                                link_tunnels.append(tunnel_i)

                        network_tunnels.append(link_tunnels)

                if 1:
                    network_tunnels = [
                        [] for _ in range(len(super_graph.edges()))
                    ]
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
                    link_util[link_i]
                    == quicksum(flow_p[i] for i in link_tunnels),
                    "link_demand_{}".format(link_i),
                )

                m.addConstr(
                    self.LINK_CAPACITY
                    >= quicksum(flow_p[i] for i in link_tunnels),
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

    def run_model(self):
        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        demand_dict = self.demand_dict
        self.model = m = Model("ONSET")
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

        candid_link_vars = m.addVars(
            range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
        )

        path_vars = m.addVars(
            range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
        )

        link_util = m.addVars(
            range(len(super_graph.edges())),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=LINK_CAPACITY,
            name="link_util",
        )

        flow_p = m.addVars(
            range(len(super_paths_list)),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="flow_p",
        )

        M = m.addVar(vtype=GRB.CONTINUOUS, name="M")

        # ### Objective
        # Minimize $M$

        m.addConstr(M == max_(link_util), name="max_link_util")
        m.setObjective(M, sense=GRB.MINIMIZE)

        # ### Budget Constraint
        #
        # $\sum_{\hat{e} \in E} b_\hat{e} \leq B$
        print("Adding Model Constraint, Budget: {}".format(self.BUDGET))
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
        for p_i in range(len(path_vars)):
            m.addConstr(
                flow_p[p_i] <= path_vars[p_i] * LINK_CAPACITY,
                "flow_binary_{}".format(p_i),
            )

            path_i_links = list(
                zip(super_paths_list[p_i], super_paths_list[p_i][1:])
            )
            candidate_links_on_path = []
            for link_l in path_i_links:
                l1 = list(link_l)
                l2 = [l1[1], l1[0]]
                if l1 in candidate_links or l2 in candidate_links:
                    candidate_index = (
                        candidate_links.index(l1)
                        if l1 in candidate_links
                        else candidate_links.index(l2)
                    )
                    candidate_links_on_path.append(candidate_index)

            if candidate_links_on_path:
                path_candidate_link_vars = [
                    candid_link_vars[var_i]
                    for var_i in candidate_links_on_path
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
        for source, target in demand_dict:
            P = self.tunnel_dict[(source, target)]

            m.addConstr(
                demand_dict[(source, target)]
                <= quicksum(flow_p[p] for p in P),
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
        for link_i, (link_source, link_target) in tqdm(
            enumerate(list(super_graph.edges())),
            desc="Initializing link utilization constraints.",
            total=len(super_graph.edges()),
        ):
            link_tunnels = []
            for tunnel_i, tunnel in enumerate(super_paths_list):
                if link_on_path(tunnel, [link_source, link_target]):
                    link_tunnels.append(tunnel_i)

            m.addConstr(
                link_util[link_i] == quicksum(flow_p[i] for i in link_tunnels),
                "link_demand_{}".format(link_i),
            )

            m.addConstr(
                self.LINK_CAPACITY
                >= quicksum(flow_p[i] for i in link_tunnels),
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

    def run_model_v2(self):
        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        demand_dict = self.demand_dict
        self.model = m = Model("ONSET")
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

        candid_link_vars = m.addVars(
            range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
        )

        path_vars = m.addVars(
            range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
        )

        link_util = m.addVars(
            range(len(super_graph.edges())),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=1,
            name="link_util",
        )

        flow_p = m.addVars(
            range(len(super_paths_list)),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="flow_p",
        )

        M = m.addVar(vtype=GRB.CONTINUOUS, name="M")

        # ### Objective
        # Minimize $M$

        m.addConstr(M == max_(link_util), name="max_link_util")
        m.setObjective(M, sense=GRB.MINIMIZE)

        # ### Budget Constraint
        #
        # $\sum_{\hat{e} \in E} b_\hat{e} \leq B$
        print("Adding Model Constraint, Budget: {}".format(self.BUDGET))
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
        for p_i in range(len(path_vars)):
            m.addConstr(
                flow_p[p_i] <= (path_vars[p_i]) / LINK_CAPACITY,
                "flow_binary_{}".format(p_i),
            )

            path_i_links = list(
                zip(super_paths_list[p_i], super_paths_list[p_i][1:])
            )
            candidate_links_on_path = []
            for link_l in path_i_links:
                l1 = list(link_l)
                l2 = [l1[1], l1[0]]
                if l1 in candidate_links or l2 in candidate_links:
                    candidate_index = (
                        candidate_links.index(l1)
                        if l1 in candidate_links
                        else candidate_links.index(l2)
                    )
                    candidate_links_on_path.append(candidate_index)

            if candidate_links_on_path:
                path_candidate_link_vars = [
                    candid_link_vars[var_i]
                    for var_i in candidate_links_on_path
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
        for source, target in demand_dict:
            P = self.tunnel_dict[(source, target)]

            # m.addConstr(demand_dict[(source, target)] <= quicksum(
            #     flow_p[p] for p in P), "flow_{}_{}".format(source, target))
            m.addConstr(
                1 == quicksum(flow_p[p] for p in P),
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
        for link_i, (link_source, link_target) in tqdm(
            enumerate(list(super_graph.edges())),
            desc="Initializing link utilization constraints.",
            total=len(super_graph.edges()),
        ):
            link_tunnels = []
            for tunnel_i, tunnel in enumerate(super_paths_list):
                if link_on_path(tunnel, [link_source, link_target]):
                    link_tunnels.append(tunnel_i)

            m.addConstr(
                link_util[link_i] == quicksum(flow_p[i] for i in link_tunnels),
                "link_demand_{}".format(link_i),
            )

            # m.addConstr(self.LINK_CAPACITY >= quicksum(
            #     flow_p[i] for i in link_tunnels), "link_utilization_{}".format(link_i))
            m.addConstr(
                1 >= quicksum(flow_p[i] for i in link_tunnels),
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

    def run_model_minimize_cost(self):
        self.BUDGET = 1
        while self.BUDGET <= len(self.candidate_links):
            print(
                "Attempting to solve optimization with Budget: {}".format(
                    self.BUDGET
                )
            )
            self.run_model()
            if self.model.status == GRB.Status.OPTIMAL:
                return
            self.BUDGET += 1
        print("Model was infeasible.")
        return

    def run_model_minimize_cost_v1(self, LOAD_FACTOR=1.0):
        """Runs Multi-objective solver to minimize number of new links added while satisfying
           minimum max link utilization

        Args:
            LOAD_FACTOR (float, optional): Artificial constraint on link utilization. 0.5 ensures any solution
            will not load any link with more than 50% of it's potential capacity. Defaults to 0.1.
        """
        super_graph = self.super_graph
        candidate_links = self.candidate_links
        super_paths_list = self.tunnel_list
        LINK_CAPACITY = self.LINK_CAPACITY
        demand_dict = self.demand_dict
        super_paths_list = self.tunnel_list
        self.model = m = Model("ONSET")
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

        candid_link_vars = m.addVars(
            range(len(candidate_links)), vtype=GRB.BINARY, name="b_link"
        )

        path_vars = m.addVars(
            range(len(super_paths_list)), vtype=GRB.BINARY, name="b_path"
        )

        link_util = m.addVars(
            range(len(super_graph.edges())),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=LOAD_FACTOR * LINK_CAPACITY,
            name="link_util",
        )

        flow_p = m.addVars(
            range(len(super_paths_list)),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="flow_p",
        )

        flow_p_variance = m.addVars(
            range(len(super_paths_list)),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="flow_p_variance",
        )

        M = m.addVar(vtype=GRB.CONTINUOUS, name="M")

        # ### Objective
        # Minimize $M$

        # m.addConstr(M == max_(link_util), name="max_link_util")
        m.setObjective(M, sense=GRB.MINIMIZE)

        # m.setObjectiveN(M, 0, 0) # index 0, priority low (0)
        # m.setObjectiveN(quicksum(candid_link_vars), 1, 1) # index 1, priority high (1)

        # ### Budget Constraint
        #
        # $\sum_{\hat{e} \in E} b_\hat{e} \leq B$
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
        for p_i in range(len(path_vars)):
            m.addConstr(
                flow_p[p_i] <= path_vars[p_i] * LINK_CAPACITY,
                "flow_binary_{}".format(p_i),
            )

            path_i_links = list(
                zip(super_paths_list[p_i], super_paths_list[p_i][1:])
            )
            candidate_links_on_path = []
            for link_l in path_i_links:
                l1 = list(link_l)
                l2 = [l1[1], l1[0]]
                if l1 in candidate_links or l2 in candidate_links:
                    candidate_index = (
                        candidate_links.index(l1)
                        if l1 in candidate_links
                        else candidate_links.index(l2)
                    )
                    candidate_links_on_path.append(candidate_index)

            if candidate_links_on_path:
                path_candidate_link_vars = [
                    candid_link_vars[var_i]
                    for var_i in candidate_links_on_path
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
        for source, target in demand_dict:
            P = self.tunnel_dict[(source, target)]

            m.addConstr(
                demand_dict[(source, target)]
                == quicksum(flow_p[p] for p in P),
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
        for link_i, (link_source, link_target) in tqdm(
            enumerate(list(super_graph.edges())),
            desc="Initializing link utilization constraints.",
            total=len(super_graph.edges()),
        ):
            link_tunnels = []
            for tunnel_i, tunnel in enumerate(super_paths_list):
                if link_on_path(tunnel, [link_source, link_target]):
                    link_tunnels.append(tunnel_i)

            m.addConstr(
                link_util[link_i] == quicksum(flow_p[i] for i in link_tunnels),
                "link_demand_{}".format(link_i),
            )

            m.addConstr(
                self.LINK_CAPACITY
                >= quicksum(flow_p[i] for i in link_tunnels),
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

    def get_links_to_add(self):
        if len(self.links_to_add) > 0:
            return self.links_to_add
        else:
            return []


def main():
    if __name__ == "__main__":
        # G = nx.read_gml("./data/graphs/gml/sprint.gml")
        # G = nx.read_gml("./data/graphs/gml/linear_3.gml")
        GML_File = (
            "/home/mhall/network_stability_sim/data/graphs/gml/sprint_test.gml"
        )
        network = "sprint"
        demand_matrix = "/home/mhall/network_stability_sim/data/traffic/sprint_benign_0Gbps_targets_5_iteration_2_strength_100_mix"

        G = nx.read_gml(GML_File)

        # demand_matrix = "./data/traffic-2022-01-29/sprint_240Gbps.txt"
        # demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_50Gbps_targets_5_iteration_2_strength_200"
        # demand_matrix = "/home/matt/network_stability_sim/temp/sprint_benign_50Gbps_5x200Gbps_3_oneShot.txt"
        # demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_0Gbps_targets_3_iteration_1_strength_100_atk"
        # demand_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_benign_0Gbps_targets_3_iteration_1_strength_150_atk"

        # demand_matrix = "/home/mhall/network_stability_sim/data/traffic/linear_3_benign_0Gbps_targets_1_iteration_1_strength_150_atk"

        # demand_matrix = "/home/matt/network_stability_sim/26495ffb2405ed09de0cf24bc1d54c5d0eb56579"
        # network = "linear_3"
        optimizer = Link_optimization(G, 3, demand_matrix, network)
        # optimizer.run_model()

        # optimizer.run_model_minimize_cost()
        optimizer.run_model_mixed_objective()

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
