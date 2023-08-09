from gurobipy import Model, GRB, quicksum, tupledict, min_
from collections import defaultdict
from itertools import permutations, product
from onset.utilities.plot_reconfig_time import calc_haversine
import numpy as np
import networkx as nx


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
    ):
        self.G = G
        if isinstance(demand_matrix_file, str):
            self.demand_matrix_file = demand_matrix_file
            self.demand_matrix = np.loadtxt(demand_matrix_file)
            self.demand_dict = {}
        elif isinstance(demand_matrix_file, dict):
            self.demand_matrix_file = None
            self.demand_matrix = None
            self.demand_dict = demand_matrix_file
        else:
            print("unknow demand type")
        self.network = network
        if isinstance(core_G, nx.Graph):
            self.core_G = core_G
            self.txp_count = [
                len(core_G.nodes[x]["transponder"]) for x in core_G.nodes
            ]
        else:
            self.core_G = self.G.copy(as_view=True)
        if txp_count is None:
            self.txp_count = [
                (len(self.core_G[node]) + 1) for node in self.core_G.nodes
            ]
        else:
            self.txp_count = txp_count
        self.nodes = self.core_G.nodes
        self.use_cache = use_cache
        self.PARALLEL = parallel_execution
        self.k = 1
        self.MAX_DISTANCE = 5000  # km
        self.MAX_DISTANCE = float("inf")  # km
        self.LINK_CAPACITY = 100  # bps
        # self.LINK_CAPACITY = 100 * 10**9  # bps
        # self.BUDGET = BUDGET
        # self.G = nx.read_gml("./data/graphs/gml/sprint.gml")
        self.super_graph = nx.Graph()

        self.all_node_pairs = list(permutations(self.G.nodes, 2))
        # self.demand_matrix = np.loadtxt("./data/traffic/sprint_240Gbps.txt")

        self.model = None
        self.candidate_links = []
        self.tunnel_list = []
        self.links_to_add = []
        self.tunnel_dict = defaultdict(list)
        self.original_tunnel_list = []
        self.core_shortest_path_len = defaultdict(lambda: np.inf)
        if self.demand_dict is None:
            self.initialize_demand()
        self.initialize_candidate_links()
        self.add_candidate_links_to_super_graph()
        # self.get_shortest_paths()

    def add_candidate_links_to_super_graph(self):
        self.super_graph.add_edges_from(self.candidate_links)

    def update_shortest_path_len(self, this_path_len, source, target):
        prev_path_len = self.core_shortest_path_len[(source, target)]
        self.core_shortest_path_len[
            (source, target)
        ] = self.core_shortest_path_len[(target, source)] = min(
            prev_path_len, this_path_len
        )

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

    def optimize(self):
        # Create a new model
        m = Model("MulticommodityFlow")
        super_graph = self.super_graph.to_directed()
        initial_graph = self.G.to_directed()
        demand_dict = self.demand_dict
        nodes = self.nodes # list of nodes
        txp_count = self.txp_count # list of max-degree for each node.
        # Define the set of edges in the network
        super_graph_edges = list(super_graph.edges)

        # Define the set of commodities
        commodities = [(source, target) for source, target in demand_dict]

        # Add binary variables for initial links
        initial_links = [
            1 if e in initial_graph.edges else 0 for e in super_graph.edges
        ]

        # Add binary variables for candidate links
        candid_link_vars = m.addVars(
            len(super_graph_edges), vtype=GRB.BINARY, name="candidate_link"
        )

        # Add binary variables for link intersection with initial links
        link_intersection = m.addVars(
            len(super_graph_edges), vtype=GRB.BINARY, name="link_intersection"
        )
        for i in range(len(super_graph_edges)):
            m.addConstr(link_intersection[i] <= candid_link_vars[i])
            m.addConstr(link_intersection[i] <= initial_links[i])

        # Add integer variables for node degree
        node_degree_vars = m.addVars(
            len(nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
        )
        for v_idx, vertex in enumerate(nodes):
            m.addConstr(
                node_degree_vars[v_idx]
                == sum(
                    candid_link_vars[eid]
                    for eid, (v1, v2) in enumerate(super_graph_edges)
                    if v1 == vertex or v2 == vertex
                )
            )
            m.addConstr(node_degree_vars[v_idx] <= txp_count[v_idx])

        # Define dictionary for commodity flow variables
        flow_vars = tupledict()
        edge_capacity_vars = tupledict()
        edge_utilization_vars = tupledict()
        edge_flow_utilization_vars = {}

        # Add edge capcity variables and constraints
        for e_idx, (u, v) in enumerate(super_graph_edges):
            edge_capacity_vars[u, v] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=self.LINK_CAPACITY,
                name=f"capacity_{u}_{v}",
            )
            # edge capacity is 0 if edge is not present
            m.addConstr(
                edge_capacity_vars[u, v]
                == candid_link_vars[e_idx] * self.LINK_CAPACITY
            )

        # Make capacity bi-directional
        for e_idx, (u, v) in enumerate(super_graph_edges):
            m.addConstr(edge_capacity_vars[u, v] == edge_capacity_vars[v, u])
            if e_idx == len(super_graph_edges) / 2:
                break

        # Add utilization capcity variables and constraints
        for e_idx, (u, v) in enumerate(super_graph_edges):
            edge_utilization_vars[u, v] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=self.LINK_CAPACITY,
                name=f"utilization_{u}_{v}",
            )
            m.addConstr(
                edge_utilization_vars[u, v] <= edge_capacity_vars[u, v]
            )

        # # Add continuous variables for flow
        # for source, target in commodities:
        #     flow_vars[source, target] = m.addVars(
        #         len(super_graph_edges),
        #         vtype=GRB.CONTINUOUS,
        #         lb=0,
        #         name=f"flow_{source}_{target}",
        #     )
        #     m.addConstr(
        #         demand_dict[source, target]
        #         <= quicksum(flow_vars[source, target])
        #     )

        # Add continuous variables for flow
        for (source, target), (u, v) in product(commodities, super_graph_edges):
            flow_vars[source, target, u, v] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                name=f"flow_{source}_{target}_{u}_{v}",
            )

        for source, target in commodities:
            m.addConstr(
                demand_dict[source, target]
                <= flow_vars.sum(source, target, '*', '*')
            )

        for u, v in super_graph_edges:
            m.addConstr(
                flow_vars.sum('*', '*', u, v) <= edge_capacity_vars.sum(u, v)
            )

        # # Add continuous variables for edge utiliztion from a particular flow
        # for (u, v), (source, target) in product(edge_capacity_vars, flow_vars):
        #     edge_flow_utilization_vars[u, v, source, target] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.LINK_CAPACITY, name=f"utilization_{u}_{v}_from_demand_{source}_{target}")
        #     m.addConstr(edge_flow_utilization_vars[u, v, source, target] == quicksum())

        # Set objective to minimize the sum of link intersections
        M = m.addVar(vtype=GRB.INTEGER, name="M")
        m.addConstr(M == link_intersection.sum())
        m.setObjective(M, sense=GRB.MINIMIZE)

        # # Add constraints for commodity flow conservation
        # for source, target in commodities:
        #     for e_idx, (u, v) in enumerate(super_graph_edges):
        #         inflow = quicksum(
        #             flow_vars[source, target][i] for i in range(e_idx)
        #         )
        #         outflow = quicksum(
        #             flow_vars[source, target][i] for i in range(e_idx + 1)
        #         )
        #         if u == source:
        #             m.addConstr(inflow == 0)
        #             m.addConstr(outflow <= demand_dict[source, target])
        #         elif v == target:
        #             m.addConstr(inflow <= demand_dict[source, target])
        #             m.addConstr(outflow == 0)
        #         else:
        #             m.addConstr(inflow == outflow)


        # m.addConstrs(
        #     (flow_vars.sum(source, dest, '*', j) + flow_vars.sum(source, dest, j, '*') == flow_vars.sum(source, dest, j, '*')
        #     for (source, dest) in commodities for j in nodes), "node")


        # Add constraints for commodity flow conservation
        for source, target in commodities:
            for node in nodes:
                inflow = flow_vars.sum(source, target, '*', node)
                outflow = flow_vars.sum(source, target, node, '*')
                if node == source:
                    m.addConstr(inflow == 0)
                    m.addConstr(outflow <= demand_dict[source, target])
                elif node == target:
                    m.addConstr(outflow == 0)
                    m.addConstr(inflow <= demand_dict[source, target])                    
                else:
                    m.addConstr(inflow == outflow)

        # Optimize the model
        m.optimize()
        # print(f"demand_dict[('4', '9')]: {demand_dict[('4', '9')]}")
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
            add_edges = [e for e in resultant_edges if e not in self.G.edges]
            drop_edges = [e for e in self.G.edges if e not in resultant_edges]
            return (resultant_edges, add_edges, drop_edges)
        else:
            m.computeIIS()
            # Print out the IIS constraints and variables
            print("\nThe following constraints and variables are in the IIS:")
            for c in m.getConstrs():
                if c.IISConstr:
                    print(f"\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}")

            for v in m.getVars():
                if v.IISLB:
                    print(f"\t{v.varname} <= {v.LB}")
                if v.IISUB:
                    print(f"\t{v.varname} <= {v.UB}")

            m.write("iismodel-mcf.ilp")


def mcf(G_0:nx.Graph, G_prime:nx.Graph, demand:tupledict):
    LINK_CAPACITY = 20

    m = Model("MulticommodityFlow")
    
    
    # Pull out directionless edges for a constraint later
    directionless_edges = G_prime.edges()

    # Convert the graph to a directed graph
    G_0 = G_0.to_directed()
    G_prime = G_prime.to_directed()
    
    # Graphs should only differ in edges
    assert set(G_0.nodes) == set(G_prime.nodes)

    # Get the list of nodes and edges
    nodes = list(G_0.nodes)
    txp_count = [2 * (len(G_0[n]) + 1) for n in nodes] 
    
    # Add integer variables for node degree
    node_degree_vars = m.addVars(
        len(nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
    )

    initial_edges = list(G_0.edges)
    prime_edges = list(G_prime.edges)
    
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
        edge_capacity[u, v] == edge_capacity[v, u]
    
    # Add binary constraint on edges
    for (u, v) in prime_edges:        
        m.addConstr(edge_capacity[u,v] == edge_vars[(u,v)] * LINK_CAPACITY)

    # Set the objective to minimize the total flow
    # m.setObjective(quicksum(flow_vars[source, target, u, v] for (source, target) in demand.keys() for (u, v) in edges), sense=GRB.MINIMIZE)
    # m.setObjective(quicksum(flow_vars[source, target, u, v] for (source, target) in demand.keys() for (u, v) in prime_edges), sense=GRB.MINIMIZE)
    m.setObjective(quicksum(link_intersection[u,v] for (u,v) in prime_edges), sense=GRB.MINIMIZE)
    
    # Optimize the model
    m.optimize()
    flow_paths = tupledict()
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
    else:
        print("No optimal solution found.")
    
    return flow_paths


def optimize(initial_graph, super_graph, demand_dict):
    # Create a new model
    m = Model("MulticommodityFlow")
    super_graph = super_graph.to_directed()
    initial_graph = initial_graph.to_directed()
    demand_dict = demand_dict
    nodes = list(initial_graph.nodes) # list of nodes
    txp_count = [len(initial_graph[n]) + 1 for n in nodes] # list of max-degree for each node.
    LINK_CAPACITY = 100

    # Define the set of edges in the network
    candidate_edges = list(super_graph.edges)

    # Define the set of commodities
    commodities = [(source, target) for source, target in demand_dict]

    # Add binary variables for initial links
    initial_links = [
        1 if e in initial_graph.edges 
        else 0 
        for e in candidate_edges
    ]

    # Add binary variables for candidate links
    candid_link_vars = m.addVars(
        len(candidate_edges), vtype=GRB.BINARY, name="candidate_link"
    )

    # Add binary variables for link intersection with initial links
    link_intersection = m.addVars(
        len(candidate_edges), vtype=GRB.BINARY, name="link_intersection"
    )

    for i in range(len(candidate_edges)):
        m.addConstr(link_intersection[i] <= candid_link_vars[i])
        m.addConstr(link_intersection[i] <= initial_links[i])

    # Add integer variables for node degree
    node_degree_vars = m.addVars(
        len(nodes), vtype=GRB.INTEGER, name="node_degree", lb=0
    )

    for v_idx, vertex in enumerate(nodes):
        m.addConstr(
            node_degree_vars[v_idx]
            == sum(
                candid_link_vars[eid]
                for eid, (v1, v2) in enumerate(candidate_edges)
                if v1 == vertex or v2 == vertex
            )
        )
        m.addConstr(node_degree_vars[v_idx] <= txp_count[v_idx])

    # Define dictionary for commodity flow variables
    flow_vars = tupledict()
    edge_capacity_vars = tupledict()
    edge_utilization_vars = tupledict()

    # Add edge capcity variables and constraints
    for e_idx, (u, v) in enumerate(candidate_edges):
        edge_capacity_vars[u, v] = m.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=LINK_CAPACITY,
            name=f"capacity_{u}_{v}",
        )
        # edge capacity is 0 if edge is not present
        m.addConstr(
            edge_capacity_vars[u, v]
            == candid_link_vars[e_idx] * LINK_CAPACITY
        )

    # Make capacity bi-directional
    for e_idx, (u, v) in enumerate(candidate_edges):
        m.addConstr(edge_capacity_vars[u, v] == edge_capacity_vars[v, u])
        if e_idx == len(candidate_edges) / 2:
            break

    # Add utilization capcity variables and constraints
    for e_idx, (u, v) in enumerate(candidate_edges):
        edge_utilization_vars[u, v] = m.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=LINK_CAPACITY,
            name=f"utilization_{u}_{v}",
        )
        m.addConstr(
            edge_utilization_vars[u, v] <= edge_capacity_vars[u, v]
        )

    # Add continuous variables for flow
    for (source, target), (u, v) in product(commodities, candidate_edges):
        flow_vars[source, target, u, v] = m.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            name=f"flow_{source}_{target}_{u}_{v}",
        )
    
    # Relate flow to demand
    for source, target in commodities:
        m.addConstr(
            demand_dict[source, target]
            <= flow_vars.sum(source, target, '*', '*')
        )
    
    #  Relate flow to capacity
    for u, v in candidate_edges:
        m.addConstr(
            flow_vars.sum('*', '*', u, v) <= edge_capacity_vars.sum(u, v)
        )

    # Add constraints for commodity flow conservation
    for source, target in commodities:
        for node in nodes:
            inflow = flow_vars.sum(source, target, '*', node)
            outflow = flow_vars.sum(source, target, node, '*')
            if node == source:
                m.addConstr(inflow == 0)
                # m.addConstr(outflow <= demand_dict[source, target])
            elif node == target:
                m.addConstr(outflow == 0)
                # m.addConstr(inflow <= demand_dict[source, target])                    
            else:
                m.addConstr(inflow == outflow)

    # Set objective to minimize the sum of link intersections
    # m.setObjective(link_intersection.sum(), sense=GRB.MINIMIZE)
    m.setObjective(1, sense=GRB.MINIMIZE)

    # Optimize the model
    m.optimize()
    # print(f"demand_dict[('4', '9')]: {demand_dict[('4', '9')]}")
    if m.status == GRB.Status.OPTIMAL:
        # links_to_add = []
        # for clv in candid_link_vars:
        #     if candid_link_vars[clv].x == 1:
        #         links_to_add.append(candidate_links[clv])
        # self.links_to_add = links_to_add
        # resultant_edges = [
        #     list(super_graph.edges)[i]
        #     for i in range(len(candid_link_vars))
        #     if candid_link_vars[i].x == 1
        # ]
        # add_edges = [e for e in resultant_edges if e not in self.G.edges]
        # drop_edges = [e for e in self.G.edges if e not in resultant_edges]
        # return (resultant_edges, add_edges, drop_edges)
        print("objective solved.")
    else:
        m.computeIIS()
        # Print out the IIS constraints and variables
        print("\nThe following constraints and variables are in the IIS:")
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}")

        for v in m.getVars():
            if v.IISLB:
                print(f"\t{v.varname} <= {v.LB}")
            if v.IISUB:
                print(f"\t{v.varname} <= {v.UB}")

        m.write("iismodel-mcf.ilp")



def main():
    G_0 = nx.Graph()
    G_0.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    G_prime = nx.Graph()
    G_prime.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("A", "C"), ("B", "D")])
    demand_dict = {}
    for u in G_0.nodes:
        for v in G_0.nodes:
            if u == v: 
                pass
            else:   
               demand_dict[u, v] = 5
    
    # optimize(G_0, G_prime, demand_dict)
    mcf(G_0, G_prime, demand_dict)


if __name__ == "__main__":
    main()
