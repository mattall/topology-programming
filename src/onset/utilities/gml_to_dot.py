from ipaddress import ip_address
from sys import argv, exit
from os import path
from networkx import read_gml
from networkx.classes.function import get_edge_attributes
from networkx.classes.graph import Graph
from .logger import logger


class Gml_to_dot():
    def __init__(self, gml, outFile):
        # Get providers
        print("[Gml_to_dot] inFile: {}, outFile: {}".format(gml, outFile))
        self.write_gml_to_dot(gml, outFile)

    # def __call__(self, inFile, outfile):

    def reduce_range(self, nodes, links, link_capacity):
        return nodes, links, link_capacity
        
        # links and nodes are sets.
        
        nodes = sorted(nodes)
        links = sorted(links)
        
        node_map = {} # map input node number to a 1 to len(nodes) range
        for n in range(len(nodes)):
            node_map[nodes[n]] = n+1
        
        g_nodes = [node_map[n] for n in nodes]
        g_links = [(node_map[a], node_map[b]) for a,b in links]

        g_link_capacity = { (node_map[a], node_map[b]) : link_capacity[(a,b)] for a, b in links }
        
        return g_nodes, g_links, g_link_capacity

    def mac_range(self, max):
        # https://stackoverflow.com/questions/8484877/mac-address-generator-in-python
        mac = '00:00:00:'
        mac_list = []
        for number in range(max):
            hex_num = hex(number)[2:].zfill(6)
            mac_list.append("{}{}{}:{}{}:{}{}".format(mac,*hex_num))
        return mac_list

    def write_dot_graph(self, nodes, links, link_capacity,  name):
        nodes = list(nodes)
        links = list(links)
        switch_ips = [str(ip_address(a)) for a in range(len(nodes))]
        host_ips = [str(ip_address(a)) for a in range(2**16+len(nodes))]
        mac_addrs = self.mac_range(len(nodes)*2)
        #with open("./" + name + '.dot', 'w') as fob:
        with open(name, 'w') as fob:
            fob.write("digraph topology {\n\n")
            for x in sorted(nodes):
                mac = mac_addrs.pop()
                ip = switch_ips.pop()
                fob.write('s{0}\t[type=switch,id={0},mac="{1}",ip="{2}"];\n'\
                            .format(x, mac, ip))
            
            fob.write("\n")
            for x in sorted(nodes):
                mac = mac_addrs.pop()
                ip = host_ips.pop()
                fob.write('h{0}\t[type=host,mac="{1}",ip="{2}"];\n'\
                            .format(x, mac, ip))
            
            fob.write("\n")
            for x in sorted(nodes):
                capacity = max(link_capacity.values()) # ensure congestion happens "in network," not at hosts. 
                fob.write('h{0} -> s{0}\t[src_port=0, dst_port=0, cost=1, capacity="{1}Gbps"];\n'.format(x, capacity))
                fob.write('s{0} -> h{0}\t[src_port=0, dst_port=0, cost=1, capacity="{1}Gbps"];\n'.format(x, capacity))

            fob.write("\n")
            for (a,b) in sorted(links):
                try:
                    capacity = link_capacity[(a, b)]
                except:
                    for l in link_capacity:
                        print(l, link_capacity[l])
                    exit()

                fob.write('s{0} -> s{1}\t[src_port={1}, dst_port={0}, cost=1, capacity="{2}Gbps"];\n'\
                        .format(a, b, capacity))
                fob.write('s{0} -> s{1}\t[src_port={1}, dst_port={0}, cost=1, capacity="{2}Gbps"];\n'\
                        .format(b, a, capacity))

            fob.write("}")

    def write_gml_to_dot(self, gml, out_file):
        if type(gml) is str and path.isfile(gml):
            G = read_gml(gml)
        else:
            G = gml

        links = set(G.edges)
        nodes = set(G.nodes)

        try:
            link_capacity = get_edge_attributes(G, 'capacity')
            assert link_capacity
        except AssertionError:
            link_capacity = {}
            for link in links:
                link_capacity[link]=10

        vertices, edges, edge_capacity = self.reduce_range(nodes, links, link_capacity)

        self.write_dot_graph(vertices, edges, edge_capacity, out_file)


def main(argv):
    try:
        gml_file = argv[1]
    except:
        print("usage: python gml_to_dot.py gml_file [output_file]")
        exit()
        
    try:
        outfile = argv[2]
    except:
        outfile = gml_file.split('.')[0]

    Gml_to_dot(gml_file, outfile)

if __name__ == "__main__":
    main(argv)
