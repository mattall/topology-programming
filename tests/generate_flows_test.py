import unittest
from onset.utilities.flows import generate_flows

class generate_flows_test(unittest.TestCase):
    def test(self):
        topology_file = "data/graphs/json/campus/campus_ground_truth.json"
        
        G, flows = generate_flows(topology_file, 1000, 1000)        
        str_nodes = [str(n) for n in G.nodes()]

        for x, y, z in flows:
            x = x.split('_')[1]
            y = y.split('_')[1]
            self.assertIn(x, str_nodes)
            self.assertIn(y, str_nodes)
            
        
        