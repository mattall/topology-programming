# Created by MNH
# Create random graph. A node has some chance p dt of failing 
# at some time t. 
# 
# A node's chance of failing increases as more nodes around it 
# fail. 
# 
# p dt ~ h(n) where h(n) is the health of the neighbors around it. 

from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.bipartite.basic import color
import numpy as np
from networkx import Graph, draw
from networkx import read_gml
from networkx.drawing.nx_pydot import read_dot
from numpy import exp, random
from os import path, makedirs, listdir, rmdir, remove, system, chdir
from pprint import pprint
from onset.utilities.links_as_nodes import verbose_tranform_link_to_node
from onset.utilities.logger import logger
from onset.utilities.sysUtils import delete_dir, export_vid, test_ssh
from onset.utilities import SCRIPT_HOME


def _sanitize(u):
    if type(u) == str:
        u = u.replace(" ", "")
        u = u.replace(",", "_")
    return u

def sanitize(u, v):
    u = _sanitize(u)
    v = _sanitize(v)
    return(u, v)    

def pack_link_node(u,v):
    return '(s{},s{})'.format(u,v)

def unpack_link_node(e):
    u, v = e.strip( '()' ).split( ',' )
    u = u.strip('s')
    v = v.strip('s')
    return u, v

def swap_order_link_node(e):
    u, v = unpack_link_node(e)
    return pack_link_node(v, u)

def edge_downshift(e):
    u, v = unpack_link_node(e)
    u = str(int(u) - 1)
    v = str(int(v) - 1)
    return pack_link_node(u,v)

class FiberGraph(object):
    def __init__(self, name='unnamed_fiber_graph', G=None):
        '''Generates a Fiber Graph object.
            Can be initialized empty. Some methods have strick dependency on others being called first. 
            Create G with FiberGraph.random_graph, FiberGraph.complete_graph, or load via FiberGraph.import_* methods.
            G must be loaded before calling draw_* or _init_link_graph

        Args:
            name (str, optional): name used to reference the FiberGraph in files created by methods of the class. Defaults to 'unnamed_fiber_graph'.
            G (nx.Graph, optional): networkx object that contains the node, links, and their attributes. Defaults to None.
        '''        

        self.name = name
        self.G = G
        self.link_graph = None

    def draw_graph(self, G, node_color='blue', with_labels=False, edge_color='black'):
        '''Draws graph object, placing nodes according to Longitude and Latitude attributes.

        Args:
            G (Graph): [description]
            node_color (str, optional): [description]. Defaults to 'blue'.
        '''        
        pos = {}
        for node in G.nodes():
            pos[node] = (G.nodes()[node]['Longitude'], G.nodes()[node]['Latitude'])

        # pprint(G.nodes())
        # pos = spring_layout(G)
        # pprint(pos)
        if node_color == 'white':
            draw(G, pos, alpha=0)
        else:
            draw(G, pos, node_color=node_color, with_labels=with_labels, edge_color=edge_color)

    def set_weights(self, weight_file):
        '''Reads weights from file and assigns colors to nodes in the link graph
        based on the weights read. 
            0.0 <= weight < 0.2 : color Blue
            0.2 <= weight < 0.4 : color Green
            0.4 <= weight < 0.6 : color yellow
            0.6 <= weight < 0.8 : color Orange
            0.8 <= weight < 1.0 : color Red

        Args:
            weight_file (str) : Expects file formatted with lines 
                following this example...
                    (node,node) : weight
                    (node,node) : weight
                    ...
        '''        

        link_graph = self.link_graph
        color_map = {}
        edge_weight = defaultdict(lambda : 0)
        with open(weight_file, 'r') as fob: 
            for line in fob: 
                line=line.strip()
                if line.startswith( '(' ) and 'h' not in line: # silently ignores everything that's NOT presumably a core edge.. 
                    edge, weight = line.split( ':' )
                    # Yates indexes from 1. GML files index from 0.
                    # Hack to shift edges from file. 
                    edge = edge.strip()
                    # edge = edge_downshift(edge)
                    weight = float(weight.strip())
                    if edge not in link_graph.nodes():
                        edge = swap_order_link_node(edge)
                    if edge not in link_graph.nodes():
                        logger.error('edge {} not in graph {}.'.format(edge, link_graph.nodes()))
                        raise KeyError
                    if weight > edge_weight[edge]:
                        edge_weight[edge] = weight
                   
      
        # assign weight from edge_weights
        for n in link_graph.nodes:
            weight = edge_weight[n]
            if weight < 0.2:
                color_map[n] = 'blue'
            elif weight < 0.4:
                color_map[n] = 'green'
            elif weight < 0.6: 
                color_map[n] = 'yellow'
            elif weight < 0.8:
                color_map[n] = 'orange'
            elif weight < 1:
                color_map[n] = 'red'
            else:
                color_map[n] = 'black'

        nx.set_node_attributes(link_graph, color_map, 'color')

    def _init_link_graph(self):
        '''Creates a link graph for the object's graph. 
            Every node in the link graph corresponds to 
            a link in the original graph. Edges in the link
            graph connect edges that are adjacent in the 
            original graph. 

            Example: 
                A <--> B --> C.
                Becomes
                (B->C) <- (A->B) <- (B->A) <-
                            |               |
                            ----------------                 
        '''        
        G = self.G
        self.link_graph = Graph()
        H = self.link_graph
        for edge in G.edges():
            # u, v = edge[0], edge[1]
            u, v = sanitize(edge[0], edge[1])
            # s = u.replace(' ', '_')
            # t = v.replace(' ', '_')
            new_node = pack_link_node(u,v)
            latitude = (G.nodes()[u]['Latitude'] + G.nodes[v]['Latitude']) / 2.0
            longitude = (G.nodes()[u]['Longitude'] + G.nodes[v]['Longitude']) / 2.0
            H.add_node(new_node, Longitude=longitude, Latitude=latitude)
            # for adjacent_node in G.adj[s]:

        # pprint(H.nodes())
        for m in H.nodes():
            m_first, m_last = unpack_link_node(m)
            for n in H.nodes():
                if n == m: continue
                n_first, n_last = unpack_link_node(n)
                # logger.debug('{} {}'.format(m, n))
                if m_first in n or m_last in n: # must be able to walk from M to N
                    H.add_edge(m, n)
                # if n_first in m or n_last == m: # must be able to walk from M to N
                #     H.add_edge(m, n)

    def draw_graphs(self, FIGURES_DIR=None,link_graph=False):
        ''' Draws three graphs, the base graph, the link graph, and a combo of the two.
            Saves images to ./figures/
        '''        
        PROJECT_HOME=path.realpath(path.curdir)
        name = self.name
        base_graph = self.G

        self.draw_graph(base_graph, node_color='black') 
        plt.savefig(path.join(FIGURES_DIR))
        
        plt.clf()

        if FIGURES_DIR is None:
            FIGURES_DIR = path.join(PROJECT_HOME, 'figures', 'graphs')
        
        if link_graph:
            if self.link_graph is None: raise('Link graph undefined. First call FiberGraph._init_link_graph')
            link_graph = self.link_graph
            link_graph_colors = list(nx.get_node_attributes(link_graph, "color").values())
            self.draw_graph(link_graph, node_color=link_graph_colors, with_labels=False, edge_color='white')
            plt.savefig(path.join(FIGURES_DIR, name+'_base_and_link'))

            self.draw_graph(base_graph, node_color='white') 
            self.draw_graph(link_graph, node_color=link_graph_colors, edge_color='white')
            plt.savefig(path.join(FIGURES_DIR, name+'_link'))
                
        # base_graph_colors = nx.get_node_attributes(base_graph, "color")
        plt.clf()

    def random_graph(self, n):
        # n: Number of nodes in the graph
        # Generates a random graph on n nodes. 
        # Gives nodes a health attribute, initially set to up. 
        # returns graph G, node_color_map, and position for nodes. 
        self.G = nx.Graph()
        for i in range(n):
            self.G.add_node(i, health='up')

        for node_s in self.G:
            for node_t in self.G:
                # generate edges by flipping a coin. 
                if random.random() < 0.2:
                    if node_s is not node_t:
                        self.G.add_edge(node_s, node_t, color='black')

        self.color = ['green'] * n
        initial_pos = {} 
        for i in range(n):
            initial_pos[i] = (1-(i/n), i/n)

        position = nx.spring_layout(self.G, pos = initial_pos)                        
        nx.set_node_attributes(self.G, position, 'position')

    def import_gml_graph(self, path):#, label=None, destringizer=None):
        # MOVING THIS TO UTILITIES
        # self.G = read_gml(path, label, destringizer)
        self.G = read_gml(path, label="id")
        # Note: Because of something weird in read_gml, must remake node IDs into strings manually.
        if min(self.G.nodes)==0:
            node_to_str_map = {node: str(node+1) for (node) in self.G.nodes}
            # node_to_str_map = {node: ("sw" + str(node)) for (node) in self.G.nodes}
            nx.relabel_nodes(self.G, node_to_str_map, copy=False)

        # nx.relabel_nodes(self.G, )
        # nx.relabel_nodes(self.G, lambda x: _sanitize(x))
        position = {}
        for node in self.G.nodes():
            position[node] = (self.G.nodes()[node]['Longitude'], self.G.nodes()[node]['Latitude'])
        
        nx.set_node_attributes(self.G, position, 'position')

        color = {}
        for node in self.G.nodes():
            color[node] = 'blue'
        nx.set_node_attributes(self.G, color, 'color')
        return self.G


    def import_dot_graph(self, path):
        self.G = read_dot(path)
        
        position = {}
        for node in self.G.nodes():
            position[node] = (self.G.nodes()[node]['Longitude'], self.G.nodes()[node]['Latitude'])
        nx.set_node_attributes(self.G, position, 'position')
        
        color = {}
        for node in self.G.nodes():
            color[node] = 'blue'         
        nx.set_node_attributes(self.G, color, 'color')
              

    def complete_graph(self, n):
        # n: Number of nodes in the graph
        # Generates a complete graph on n nodes. 
        # Gives nodes a health attribute, initially set to up. 
        # returns graph G, node_color_map, and position for nodes. 
        position = {}
        self.G = nx.Graph()
        for i in range(n):
            self.G.add_node(i, health='up')
        
        for node_s in self.G:
            for node_t in self.G:
                if node_s is not node_t:
                    self.G.add_edge(node_s, node_t, color='black')
        
        color = {}
        for node in self.G.nodes():
            color[node] = 'blue'         
        nx.set_node_attributes(self.G, color, 'color')

        initial_pos = {} 
        for i in range(n):
            initial_pos[i] = (1-(i/n), i/n)

        position = nx.spring_layout(self.G, pos = initial_pos)   
        nx.set_node_attributes(self.G, position, 'position')


    def update_health(self, t, folder, alpha=0.8, beta=0.1, start_clean=False):
        # health is gauged in two phases. 
        # First, everynode has alpha probability of being
        # down. Then, every node has alpha * N change of being down,
        # Where N is the number of down neighbors.

        # set all edges to black
        for u,v in self.G.edges():
            self.G[u][v]['color'] = 'black'

        # Find contageons 
        for node in self.G:
            if random.random() < alpha:
                self.G.nodes()[node]['health'] = 'down'
                self.G[node]['color'] = 'red'

        self.print_g(folder, start_clean=start_clean, \
            title='Contagion\nt = {}    alpha = {}    beta = {}'\
                .format(t, alpha, beta))   

        # Spread contageon
        for node in self.G:
            # count the down neighbors of node
            down_neighbors = []
            for neighbor in self.G[node]:
                if self.G.nodes()[neighbor]['health'] == 'down':
                    down_neighbors.append(neighbor)
            
            # update node's health according to the health of the neighbors
            if random.random() < alpha * len(down_neighbors):
                self.G.nodes()[node]['health'] = 'down'
                self.G[node]['color'] = 'red'
                for dn in down_neighbors:
                    self.G[node][dn]['color'] = 'red'
            
        self.print_g(folder, title='Spread\nt = {}    alpha = {}    beta = {}'.format(t, alpha, beta))   

        # set all edges to black
        for u,v in self.G.edges():
            self.G[u][v]['color'] = 'black'

        # Spontaneous recovery
        for node in self.G:
            if self.G.nodes()[node]['health'] == 'down':
                if random.random() < beta:
                    self.G.nodes()[node]['health'] = 'up'
                    self.G[node]['color'] = 'green'

        self.print_g(folder, title='Recovery\nt = {}    alpha = {}    beta = {}'.format(t, alpha, beta))   

    def print_g(self, folder, start_clean=False, title=None):
        # Params:
        # G: graph
        # pos: position of nodes in graph (dictionary: node_index -> (x,y))
        # node_color_map: list of colors of nodes
        # start_clean: True values deletes all previous files and folders from pwd/figures/
        # title: str to annotate graph.
        if title is not None:
            plt.title(title)

        # edge_color = [self.G[u][v]['color'] for u,v in self.G.edges()]
        edge_color = nx.get_edge_attributes(self.G, 'color')
        node_color = nx.get_node_attributes(self.G, 'color')
        position = nx.get_node_attributes(self.G, 'position')
        nx.draw(self.G, position, node_color=node_color, edge_color=edge_color)

        if start_clean:
            delete_dir(path.join(SCRIPT_HOME, folder))

        makedirs(path.join(SCRIPT_HOME, folder), exist_ok=True)
        fig_num = len(listdir(path.join(SCRIPT_HOME, folder)))
        # print('title: \t', title)
        # print('start clean: ', start_clean)
        # print('fig num: ', fig_num)

        plt.savefig( path.join(SCRIPT_HOME, folder, str(fig_num)))
        plt.clf()            

    def simulate_random_starts_and_failures(self, steps=10, kind='random', alpha=0.1, beta=0.2):
        # steps = 100
        # kind = 'random'
        # alpha = 0.1
        # beta = 0.1
        out_folder = path.join(\
                'figures', 'node_{}_steps_{}_class_{}_alpha_{}_beta_{}'\
                    .format(len(self.G.nodes()), steps, kind, alpha, beta)\
                        )

        for t in range(steps):
            if t == 0:
                self.update_health(t, out_folder, alpha=alpha, beta=beta, start_clean=True)
            else:
                self.update_health(t, out_folder, alpha=alpha, beta=beta, start_clean=False)

        export_vid(out_folder)

def main():
    if 0: 
        myG = FiberGraph()    
        myG.random_graph(20)
        myG.simulate_random_starts_and_failures(steps=10, kind='random', alpha=0.1, beta=0.2)

    if 0: 
        myG = FiberGraph()
        myG.complete_graph(10)
        myG.simulate_random_starts_and_failures(steps=10, kind='complete', alpha=0.1, beta=0.2)

    if 0: 
        test_ssh()

    if 1: 
        myG = FiberGraph()
        gml_file = path.join(SCRIPT_HOME, 'graphs', 'gml', 'Abilene.gml')
        myG.import_gml_graph(gml_file)
        

    # Todo:
    # 1. read GML into FiberGraph. DONE
    # 2. run yates on file and map congestion to the FiberGraph
    



if __name__ == '__main__':
    main()