from onset.utilities.graph_utils import read_json_graph, write_json_graph
from onset.utilities.sysUtils import postfix_str
from onset.utilities import logger

from onset.constants import SEED
from onset.constants import IPV4

import matplotlib.pyplot as plt
import networkx as nx

import ipaddress
import random
from collections import defaultdict


class Network:
    """
    Network
    """

    def __init__(self, graph_file: str, output_file="", prefix=24):
        """
        graph_file (str): network topology file to build this network.
        output(str, optional): name of the graph file when the network topology is exported. Default is graph_file+"_netout".
        prefix(int, optional): Address space prefix of the network.
        """

        self.prefix = prefix
        self.ip_address_space = self.random_ipv4_address_space()
        self.available_ip_interfaces = self.interfaces_list()
        self.used_ip_interfaces = []
        self.node_types = [
            "Amplifier",
            "ROADM",
            "Transponder",
            "Router",
            "Virtual",
            "Client",
        ]
        self.link_types = [
            "Span",
            "Fiber",
            "Optical",
            "Layer3",
            "Virtual",
            "Client",
        ]
        self.graph_file = graph_file
        self.router_count = 0
        self.transponder_count = 0
        self.layer3_link_count = 0
        self.optical_link_count = 0
        self.virtual_node_count = 0
        self.virtual_link_count = 0

        # self.fdL = {} # Links flow density.
        # self.fdN = {} # Nodes flow density. Sum of flow densities for all incoming links to the node.

        # if not output_file:
        #   output_file = self.graph_file + '_netout' + '.gml'
        if not output_file:
            output_file = postfix_str(self.graph_file, "_modeled")

        self.output_file = output_file
        self.graph = self.import_graph(self.graph_file)

        self.init_client_links()
        
    
    def init_client_links(self):
        G = self.graph
        for node in self.get_router_nodes():
            client_id = f"client_{node}"

            client_node_attr = {
                "node_type": self.node_types[5],
                "router_id": f"router_{node}",
                "virtual_router_list": [],
            }
            G.add_node(client_id, **client_node_attr)

            client_link_attr = {
                "link_type": self.link_types[5],
                "capacity": 1000,  # Gb/s, Sum of optical. Current value needs to changed.
                "carrier": [],  # list of "Optical" link objects which carries the L3 link.
                "virtual_layer3_list": [],
                # "interface_map": {node, G.nodes[node]["client_interface"]},
                "fdL": defaultdict(
                    float
                ),  # Flow density of link {(src, dest): fd}. Set when calculate_fdL is called.
            }

            G.add_edge(client_id, node, **client_link_attr)

            G.nodes[node]["client_id"] = client_id

    def random_ipv4_address_space(self):
        """
        Generates random IPV4 address space.
        """
        random.seed(SEED)
        ip_addr = ipaddress.IPv4Address(random.randint(0, IPV4))
        address_space = ipaddress.IPv4Network(
            f"{ip_addr}/{self.prefix}", strict=False
        )  # setting strict=False will mask with the given prefix.
        return address_space

    def interfaces_list(self):
        """
        Output: A list containing all host interfaces for the current address space.
        """
        return list(self.ip_address_space.hosts())

    def get_available_interfaces(self):
        """
        Output: List of available ip interfaces.
        """
        return self.available_ip_interfaces

    def remove_available_interface(self, interface_addr):
        """
        interface_addr: ip address of interface to remove from available interfaces.
                        Typically used after getting available interface and assigning it.
        """
        try:
            self.available_ip_interfaces.remove(interface_addr)
            self.used_ip_interfaces.append(interface_addr)
        except Exception as e:
            # sometimes it reaches here although the interface is in the available interfaces
            logger.error(
                f"Unable to remove interface {interface_addr}. \
            Available interfaces: {self.available_ip_interfaces}, \
            Full interfaces {self.interfaces_list()}]\
            type(interface_addr): {type(interface_addr)}",
                exc_info=1,
                stack_info=1,
            )

    def get_available_interface(self):
        """
        Output: Random available address that is in the same address space of the network.
                If there aren't any interfaces available None is returned.
        """
        try:
            interface = random.choice(self.get_available_interfaces())        
            return interface
        except:
            print("No interface(s) available.")
            return

    def get_graph(self):
        """
        Returns graph of the network.
        """
        return self.graph

    def import_graph(self, file_path):
        """
        file_path: graph to import from.
        Reads input graph and add appropriate attributes to it to describe a
            physical and logic network.
        """
        if file_path.endswith(".gml"):
            G = nx.read_gml(file_path, label="id")
        elif file_path.endswith(".json"):
            G = read_json_graph(file_path)
        else:
            raise BaseException(
                f"Expected file of type .gml or .json. Got: {file_path}"
            )

        # nx.relabel_nodes(
        #     G,
        #     {n: str(n) for n, _ in zip(G.nodes(), range(len(G.nodes())))},
        #     copy=False,
        # )
        nx.relabel_nodes(G, lambda x: str(x), copy=False)
        nodes_attr = (
            {}
        )  # attributes that will be added for each node. {node : {attribute : value, ...} }
        nodes = G.nodes
        for node in nodes:
            # Assign ip address for this interface and
            # remove it from the available ip addresses in this ip address space.
            # total_transponders = G.degree[node] # default, degree of node read from file.
            client_interface = self.get_available_interface()
            self.remove_available_interface(client_interface)
            node_attr = {
                "node_type": self.node_types[3],
                "router_id": f"router_{str(node)}",
                "interface_map": {},
                "virtual_router_list": [],
                # "total_transponders": total_transponders,
                # "transponder_list": [],
                "client_interface": str(client_interface),
                "fdN": 0,  # flow density for the node. Set when calculate_fdN is called.
                "flows": set() 
            }
            nodes_attr[node] = node_attr
            self.router_count += 1
        nx.set_node_attributes(G, nodes_attr)

        links = G.edges
        links_attr = {}

        transponder_nodes = []
        optical_links = []

        for u, v in links:
            assert u != v, "BAD GRAPH!!"
            link_attr = {
                "link_type": self.link_types[3],
                "layer3_id": self.layer3_link_count,
                "capacity": 100,  # Gb/s, Sum of optical. Current value needs to changed.
                "carrier": [],  # list of "Optical" link objects which carries the L3 link.
                "virtual_layer3_list": [],
                "fdL": defaultdict(
                    float
                ),  # Flow density of link {(src, dest): fd}. Set when calculate_fdL is called.
                "flows": set()
            }
            links_attr[(u, v)] = link_attr
            self.layer3_link_count += 1
            # if G.is_directed() and (G.has_edge(u,v) or G.has_edge(v,u)):
            #   continue
            interface_u = self.get_available_interface()
            interface_v = self.get_available_interface()
            self.remove_available_interface(interface_u)
            self.remove_available_interface(interface_v)
            # setup interface map for both nodes.
            G.nodes[u]["interface_map"][v] = str(interface_v)
            G.nodes[v]["interface_map"][u] = str(interface_u)

            router_u_id = G.nodes[u]["router_id"]
            router_v_id = G.nodes[v]["router_id"]

            # for the sake of testing EqualNet transponders and optical links are commented out
            if 0:
                transponder_u_id = self.transponder_id
                transponder_v_id = self.transponder_id + 1

                transponder_u_attr = {
                    "node_type": self.node_types[2],
                    "transponder_id": transponder_u_id,
                    "capacity": 100,  # default 100 Gb/s
                    "status": "active",
                    "router_interface": router_u_id,
                    "partner": transponder_v_id,
                }
                transponder_u = f"transponder_{transponder_u_id}"
                transponder_nodes.append((transponder_u, transponder_u_attr))
                G.nodes[u]["transponder_list"].append(transponder_u_id)

                transponder_v_attr = {
                    "node_type": self.node_types[2],
                    "transponder_id": transponder_v_id,
                    "capacity": 100,  # default 100 Gb/s
                    "status": "active",
                    "router_interface": router_v_id,
                    "partner": transponder_u_id,
                }
                transponder_v = f"transponder_{transponder_v_id}"
                transponder_nodes.append((transponder_v, transponder_v_attr))
                G.nodes[v]["transponder_list"].append(transponder_v_id)

                # for the two transponders that just got created and linked with each other.
                optical_link_attr = {
                    "link_type": self.link_types[2],
                    "optical_id": self.optical_link_id,
                    "wavelength": 0,  # default 0 nm
                    "frequency": 0,  # default 0 Thz
                    "width": 1,  # default 1, number of channels the wave occupies on a fiber
                    "data_rate": 100,  # default 100 Gb/s
                }

                optical_links.append(
                    (transponder_u, transponder_v, optical_link_attr)
                )
                self.optical_link_id += 1
                self.transponder_id += 2
        nx.set_edge_attributes(G, links_attr)
        # G.add_edges_from(optical_links)
        # print(transponder_nodes)
        # print(G.nodes['transponder_0'])
        # G.add_nodes_from(transponder_nodes)
        return G

    def _calculate_fd(self, flows):
        """
        Given flows calculates flow density for links and nodes.
        """
        fdL = self.calculate_fdL(flows)
        fdN = self.calculate_fdN()
        return (fdL, fdN)

    def calculate_fd(self, flows):
        G = self.graph
        nx.set_node_attributes(G, 0, "fdN")
        for src, dest, tracing_flows in flows:
            shortest_paths = list(nx.all_shortest_paths(G, src, dest))
            num_shortest_paths = len(shortest_paths)
            print(f"{num_shortest_paths} paths between {src} and {dest}")
            for path in shortest_paths:
                print(path)
                path_len = len(path)
                # if there multiple shortest paths.
                path_tf = tracing_flows // num_shortest_paths
                print(f"path flow: {path_tf}")
                for i in range(1, path_len):
                    prev_node = path[i - 1]
                    current_node = n_p_i = path[i]
                    link = (prev_node, current_node)
                    # if "client" in link[0] or "client" in link[1]: continue
                    G.nodes[current_node]["fdN"] += path_tf
                    G[prev_node][current_node]["fdL"][str(link)] += path_tf
        return self.get_fdN()

    # def calculate_fdL(self, flows):
    #     """
    #     flows (List[tuple]): [(src, dest, tracing_flows), ...]
    #     Output: Links flow density.
    #     """
    #     G = self.graph
    #     links = G.edges
    #     # fdL = {}
    #     for src, dest, tracing_flows in flows:
    #         shortest_paths = list(nx.all_shortest_paths(G, src, dest))
    #         num_shortest_paths = len(shortest_paths)
    #         path_tf = tracing_flows / num_shortest_paths
    #         for path in shortest_paths:
    #             path_len = len(path)
    #             for i in range(path_len-1): # what if it self loops?
    #                 link = (path[i], path[i+1])
    #                 try:
    #                     # Maybe a check needed to see if L3/virtual link or not.
    #                     links[link]['fdL'][str(link)] += path_tf # networkx doesn't like tuples as keys when exporting so turn it into string.
    #                     # self.fdL[link] += path_tf
    #                 except KeyError as e: # set tracing flow for link when there isn't one.
    #                     links[link]['fdL'][str(link)] = path_tf # networkx doesn't like tuples as keys when exporting so turn it into string.
    #                     # self.fdL[link] = path_tf
    #                     # print(f"KeyError: {e}. Flow has been assigned not added to.")
    #     return self.get_fdL()

    # def calculate_fdN(self):
    #     """
    #     Assigns each node a flow density by summing the flow densities for all incoming links to the node.
    #     Note calculate_fdL must be called before this function.
    #     Output: Flow densities for the nodes.
    #     """
    #     # fdN = {}
    #     G = self.graph
    #     nodes = G.nodes
    #     for node in nodes:
    #         if nodes[node]['node_type'] in ["Router", 'Virtual']: # could add virtual nodes here as well.
    #             current_fdN = 0
    #             node_links = G.edges(node)
    #             for u, v in node_links:
    #                 # print(G[u][v]['fdL'])
    #                 in_link = (v, u)
    #                 # if not G.is_directed():
    #                 #   in_link = (v, u)
    #                 # else:
    #                 #   in_link = (u, v)
    #                 try:
    #                     current_fdN += G[u][v]['fdL'][str(in_link)]
    #                 except KeyError as e:
    #                     G[u][v]['fdL'][str(in_link)] = 0 # should be fine to do this.
    #                     # self.fdL[in_link] = 0
    #                     # current_fdN += G[u][v]['fdL'][str(in_link)]
    #                     # print(f'KeyError: {e} for incoming link {in_link}, this link has no tracing flows. Setting link tracing flow to 0.')
    #             # print(f'Node: {node} fdN: {current_fdN}')
    #             G.nodes[node]['fdN'] = current_fdN
    #             # self.fdN[node] = current_fdN
    #     return self.get_fdN()

    # def create_transponders(self, router_id, total_transponders):
    #   """
    #   Create transponders for a certain router.
    #   """
    #   pass

    def create_virtual_nodes(self, fixed):
        """
        Creates fixed amount of virtual nodes for each router on the graph.
        Splits the tracing flow among the node created.
        """
        virtual_nodes = []
        G = self.graph
        nodes = G.nodes
        current_fdN = self.get_fdN()
        for i in range(fixed):
            for node in nodes:
                if nodes[node]["node_type"] == "Router":
                    node_tf = current_fdN[node] // (
                        fixed + 1
                    )  # plus 1 because we are counting real node as well.
                    # print(node, node_tf)
                    physical_id = nodes[node]["router_id"]
                    virtual_client_interface = self.get_available_interface()
                    self.remove_available_interface(virtual_client_interface)
                    virtual_node_id = f"vn_{self.virtual_node_count}"
                    virtual_node_attr = {
                        "node_type": self.node_types[4],
                        "virtual_id": virtual_node_id,
                        "interface_map": {},
                        "physical_id": physical_id,
                        "client_interface": str(virtual_client_interface),
                        "fdN": node_tf,
                    }
                    nodes[node]["virtual_router_list"].append(virtual_node_id)
                    virtual_nodes.append((virtual_node_id, virtual_node_attr))
                    self.virtual_node_count += 1
                    G.nodes[node]["fdN"] -= node_tf
                    # update actual node fdN
        # pp.pprint(virtual_nodes)
        G.add_nodes_from(virtual_nodes)
        # self.fdN = self.get_fdN() # update fdN dict.
        # pp.pprint(self.fdN)

        # add virtual links to clients
        for vn, attrs in virtual_nodes:
            p_node = G.nodes[vn]["physical_id"]

            client_node = G.nodes[p_node]["client_id"]
            client_link_attr = {
                "link_type": self.link_types[5],
                "capacity": 1000,  # Gb/s, Sum of optical. Current value needs to changed.
                "carrier": [],  # list of "Optical" link objects which carries the L3 link.
                "virtual_layer3_list": [],
                "fdL": defaultdict(
                    float
                ),  # Flow density of link {(src, dest): fd}. Set when calculate_fdL is called.
            }
            G.add_edge(client_node, vn, **client_link_attr)

        return None

    def create_virtual_node(self, node, vn_interface):
        """
        Creates virtual node for the given node.
        node: node to create the virtual node on.
        vn_interface (IPv4Address): client interface IP address that the virtual node will be assigned to.
        Output virtual node label (this way you can index the nodes dict using it).
        Note: this does not equalize the flow when the virtual node is created.
        equalize_flows_for_prev_nodes & equalize_flows_for_next_nodes in mimic_EqualNet.py
        create links and nodes for the neighbors and equalizes flows.
        """
        G = self.graph
        nodes = G.nodes
        self.remove_available_interface(vn_interface)
        physical_id = nodes[node]["router_id"]
        vn_id = f"vn_{self.virtual_node_count}"
        virtual_node_attr = {
            "node_type": self.node_types[4],
            "virtual_id": vn_id,
            "interface_map": {},
            "physical_id": physical_id,
            "client_interface": str(vn_interface),  #
            "fdN": 0,
        }
        nodes[node]["virtual_router_list"].append(vn_id)
        G.add_node(vn_id, **virtual_node_attr)
        self.virtual_node_count += 1

        # add virtual link to client
        client_node = G.nodes[node]["client_id"]
        client_link_attr = {
            "link_type": self.link_types[5],
            "capacity": 1000,  # Gb/s, Sum of optical. Current value needs to changed.
            "carrier": [],  # list of "Optical" link objects which carries the L3 link.
            "virtual_layer3_list": [],
            "fdL": defaultdict(
                float
            ),  # Flow density of link {(src, dest): fd}. Set when calculate_fdL is called.
        }
        G.add_edge(client_node, vn_id, **client_link_attr)

        return vn_id

    def create_virtual_links(self, fixed):
        """
        Creates fixed amount of virtual links
        for the fixed amount of virtual nodes created.
        Connects virtual nodes with each other similar to how
        physical layer3 links are connected to router nodes,
        splits the tracing flow among the links created.
        p.s I think there is a better way to do this,
        by getting router nodes and getting the virtual_router_list and going
        """
        virtual_links = []
        G = self.graph
        nodes = G.nodes
        links = G.edges
        current_fdL = self.get_fdL()
        # print(current_fdL)
        for i in range(fixed):
            for u, v in links:
                link = (u, v)
                if links[(u, v)]["link_type"] == "Layer3":
                    u_v = str(link)
                    v_u = str((v, u))
                    link_u_v_tf = current_fdL[link][u_v] // (
                        fixed + 1
                    )  # plus 1 because we are counting real link as well.
                    link_v_u_tf = current_fdL[link][v_u] // (
                        fixed + 1
                    )  # plus 1 because we are counting real link as well.
                    # links[(u, v)]['fdL']
                    vn_u_id = nodes[u]["virtual_router_list"][i]
                    virtual_node_u = nodes[vn_u_id]
                    vn_u_interface = self.get_available_interface()
                    self.remove_available_interface(vn_u_interface)

                    vn_v_id = nodes[v]["virtual_router_list"][i]
                    virtual_node_v = nodes[vn_v_id]
                    vn_v_interface = self.get_available_interface()
                    self.remove_available_interface(vn_v_interface)

                    # setup virtual node interface based on the nodes connected.
                    virtual_node_u["interface_map"][vn_v_id] = str(
                        vn_v_interface
                    )
                    virtual_node_v["interface_map"][vn_u_id] = str(
                        vn_u_interface
                    )

                    virtual_link_fdL = defaultdict(float)
                    u_v_fdL = str((vn_u_id, vn_v_id))
                    v_u_fdL = str((vn_v_id, vn_u_id))

                    virtual_link_fdL[u_v_fdL] = link_u_v_tf
                    virtual_link_fdL[v_u_fdL] = link_v_u_tf

                    physical_link_id = links[link]["layer3_id"]
                    virtual_link_attr = {
                        "link_type": self.link_types[4],
                        "virtual_id": self.virtual_link_count,
                        "carrier": [],  # list of "Optical" link objects which carries the L3 link.
                        "physcial_id": physical_link_id,
                        "fdL": virtual_link_fdL,  # Flow density of link {(src, dest): fd}
                    }
                    G[u][v]["virtual_layer3_list"].append(
                        self.virtual_link_count
                    )
                    self.virtual_link_count += 1
                    virtual_links.append((vn_u_id, vn_v_id, virtual_link_attr))
                    G[u][v]["fdL"][u_v] -= link_u_v_tf
                    G[u][v]["fdL"][v_u] -= link_v_u_tf
            G.add_edges_from(virtual_links)

    def create_virtual_link(self, real_link, current_vn, neighbor_vn):
        """
        Creates virtual link between two virtual nodes, current_vn and neighbor_vn
        Also sets up interface between the two virtual nodes.
        The adjustment of flow density for link and node occurs after this function is called.
        current_vn: virtual node
        neighbor_vn: The real node of this virtual node is a neighbor or the real node of current_vn.
        """
        G = self.graph
        links = G.edges
        nodes = G.nodes

        # check here might be needed so that link can't
        # be created if no interface can be assigend.
        current_vn_interface = self.get_available_interface()
        self.remove_available_interface(current_vn_interface)
        neighbor_vn_interface = self.get_available_interface()
        self.remove_available_interface(neighbor_vn_interface)

        nodes[current_vn]["interface_map"][neighbor_vn] = str(
            neighbor_vn_interface
        )
        nodes[neighbor_vn]["interface_map"][current_vn] = str(
            current_vn_interface
        )

        physical_link_id = links[real_link]["layer3_id"]
        u, v = real_link
        virtual_link_attr = {
            "link_type": self.link_types[4],
            "virtual_id": self.virtual_link_count,
            "carrier": [],  # list of "Optical" link objects which carries the L3 link.
            "physcial_id": physical_link_id,
            "fdL": defaultdict(
                float
            ),  # Flow density of link {(src, dest): fd}
        }
        G[u][v]["virtual_layer3_list"].append(self.virtual_link_count)
        G.add_edge(current_vn, neighbor_vn, **virtual_link_attr)
        self.virtual_link_count += 1

    def get_virtual_nodes(self, physical_id):
        """
        Gets virtual node ids from a certain router given its id.
        """
        G = self.graph
        # nodes = G.nodes
        vn_ids = nx.get_node_attributes(G, "virtual_router_list")[physical_id]
        virtual_nodes = [vn_id for vn_id in vn_ids]
        return virtual_nodes

    def get_all_virtual_nodes(self):
        """
        Gets all virtual nodes in the graph.
        Returns list of list of vn_{virtual node id}.
        """
        G = self.graph
        # nodes = G.nodes
        nodes_vn_ids = nx.get_node_attributes(
            G, "virtual_router_list"
        ).values()
        virtual_nodes = [
            vn_id for node_vn_ids in nodes_vn_ids for vn_id in node_vn_ids
        ]
        return virtual_nodes

    def get_connected_virtual_node(self, real_node, virtual_node):
        """
        real_node: Node to check if it has a virtual node connected to virtual_node.
        virtual_node: node that will be checked to see if a virtual node from real_node is connected to it.
        Output: Returns the virtual node from real_node this is connected to virtual_node.
        """
        G = self.graph
        physical_id = G.nodes[real_node]["router_id"]
        prev_virtual_nodes = self.get_virtual_nodes(physical_id)
        connected_vn = next(
            (vn for vn in prev_virtual_nodes if G.has_edge(vn, virtual_node)),
            None,
        )
        return connected_vn

    def get_router_nodes(self):
        G = self.graph
        return [
            n
            for n in G.nodes()
            if G.nodes[n]["node_type"] == self.node_types[3]
        ]

    def get_client_nodes(self):
        G = self.graph
        return [
            n
            for n in G.nodes()
            if G.nodes[n]["node_type"] == self.node_types[5]
        ]

    def get_fdN(self):
        """
        Get fdN for the graph.
        """
        try:
            return nx.get_node_attributes(self.graph, "fdN")
        except Exception as e:
            print(e)
            print("No fdN found. Empty dict returned.")
            return defaultdict(float)

    def get_router_fdN(self):
        """
        Get fdN for router nodes.
        """
        G = self.graph
        nodes = G.nodes
        router_fdN = defaultdict(float)
        try:
            router_fdN.update(
                {
                    node: nodes[node]["fdN"]
                    for node in nodes
                    if nodes[node]["node_type"] == "Router"
                }
            )
            return router_fdN
        except:
            return router_fdN

    def get_virtual_fdN(self):
        """
        Get fdN for virtual nodes.
        """
        G = self.graph
        nodes = G.nodes
        virtual_fdN = defaultdict(float)
        try:
            virtual_fdN.update(
                {
                    node: G.nodes[node]["fdN"]
                    for node in nodes
                    if G.nodes[node]["node_type"] == "Virtual"
                }
            )
            return virtual_fdN
        except:
            return virtual_fdN

    def get_fdL(self):
        """
        Get fdL for the graph.
        """
        try:
            return nx.get_edge_attributes(self.graph, "fdL")
        except Exception as e:
            print(e)
            print("No fdL found. Empty dict returned.")
            return defaultdict(float)

    def get_L3_fdL(self):
        """
        Get fdL for L3 links.
        """
        G = self.graph
        links = G.edges
        l3_fdL = defaultdict(float)
        try:
            l3_fdL.update(
                {
                    (u, v): G[u][v]["fdL"]
                    for u, v in links
                    if G[u][v]["link_type"] == "Layer3"
                }
            )
            return l3_fdL
        except:
            return l3_fdL

    def get_virtual_fdL(self):
        """
        Get fdL for virtual links.
        """
        G = self.graph
        links = G.edges
        virtual_fdL = defaultdict(float)
        try:
            virtual_fdL.update(
                {
                    (u, v): G[u][v]["fdL"]
                    for u, v in links
                    if G[u][v]["link_type"] == "Virtual"
                }
            )
            return virtual_fdL
        except Exception as e:
            print(e)
            return virtual_fdL

    def export_network_gml(self, postfix=""):
        out_file = postfix_str(self.output_file, postfix)
        nx.write_gml(self.graph, out_file)

    def export_network_json(self, postfix=""):
        out_file = postfix_str(self.output_file, postfix)
        write_json_graph(self.graph, out_file)

    def export_network_plot(self, postfix=""):
        # nx.draw(self.graph, with_labels=True, font_weight='bold')
        nx.draw_spectral(self.graph, with_labels=False, font_weight="bold")
        out_file = postfix_str(self.output_file, postfix)
        fig_name = f"{out_file}_plot.jpg"
        plt.savefig(fig_name)
        plt.clf()


def main():    
    topology_file = "data/graphs/json/campus/campus_ground_truth.json"
    Network(topology_file)
    
if __name__ == "__main__":
    main()
