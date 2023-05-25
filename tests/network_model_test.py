import unittest
from onset.network_model import Network


class network_model_test(unittest.TestCase):
    def test(self):
        topology_file = "data/graphs/json/campus/campus_ground_truth.json"

        net = Network(topology_file)

        G = net.graph

        for node in G.nodes():                        
            if G.nodes[node]["node_type"] == "Router":
                self.assertEqual(G.nodes[node]["router_id"], node)
                self.assertEqual(G.nodes[node]["client_id"], f"client_{node}")

            if G.nodes[node]["node_type"] == "Client":
                self.assertEqual(
                    G.nodes[node]["router_id"], node.replace("client", "router")
                )
