from networkx import Graph, read_gml, draw, spring_layout
from pprint import pprint
import matplotlib.pyplot as plt
from os import path
from sys import argv
from networkx.algorithms.bipartite import projection
from networkx.generators.degree_seq import expected_degree_graph

def gml_reader(path, label="label", destringizer=None):
    G = read_gml(path, label, destringizer)
    return G

def my_draw(G:Graph, node_color='blue'):
    pos = {}
    for node in G.nodes():
        pos[node] = (G.nodes()[node]['Longitude'], G.nodes()[node]['Latitude'])
    # pprint(G.nodes())
    # pos = spring_layout(G)
    # pprint(pos)
    if node_color == 'white':
        draw(G, pos, alpha=0)
    else:
        draw(G, pos, node_color=node_color)

def edge_to_node(G:Graph):
    H = Graph()
    for edge in G.edges():
        u, v = edge[0], edge[1]
        # s = u.replace(" ", "_")
        # t = v.replace(" ", "_")
        new_node = u + "<->" + v
        latitude = (G.nodes()[u]["Latitude"] + G.nodes[v]["Latitude"]) / 2.0
        longitude = (G.nodes()[u]["Longitude"] + G.nodes[v]["Longitude"]) / 2.0
        H.add_node(new_node, Longitude=longitude, Latitude=latitude)
        # for adjacent_node in G.adj[s]:

    # pprint(H.nodes())
    for m in H.nodes():
        m_first, m_last = m.split('<->')
        for n in H.nodes():
            if n == m: continue
            if m_first in n or m_last in n:
                H.add_edge(m, n)
    return H

def simple_tranform_link_to_node(name, input_type):
    PROJECT_HOME=path.realpath(path.curdir)
    GRAPH_DIR=path.join(PROJECT_HOME, 'graphs', input_type)
    G = gml_reader(path.join(GRAPH_DIR, name+'.'+input_type))
    H = edge_to_node(G)
    return H

def verbose_tranform_link_to_node(name, input_type):
    PROJECT_HOME=path.realpath(path.curdir)
    GRAPH_DIR=path.join(PROJECT_HOME, 'graphs', input_type)
    G = gml_reader(path.join(GRAPH_DIR, name+'.'+input_type))
    H = edge_to_node(G)
    
    pos = {}
    for node in H.nodes():
        pos[node] = (H.nodes()[node]['Longitude'], H.nodes()[node]['Latitude'])

    node_color_map = {} 
    for node in H.nodes():
        node_color_map[node] = 'green'

    return (H, pos, node_color_map)

def links_as_nodes_viewer(name, input_type):
    # name: Name of graph - e.g., Sprint. Case sensitve. Must match a file name.
    # input_type: file tpye, e.g., dot, gml. 
    # Folder for the given type must exist under PROJECT_HOME/graphs/<input_type>
    #
    PROJECT_HOME=path.realpath(path.curdir)
    GRAPH_DIR=path.join(PROJECT_HOME, 'graphs', input_type)
    FIGURES_DIR=path.join(PROJECT_HOME, 'figures', 'graphs')
    
    G = gml_reader(path.join(GRAPH_DIR, name+'.'+input_type))
    my_draw(G, node_color='orange') 
    plt.savefig(path.join(FIGURES_DIR, name+"_0"))
    # ax = plt.gca()
    # print(ax)
    H = edge_to_node(G)
    my_draw(H, node_color='green')
    plt.savefig(path.join(FIGURES_DIR, name+"_1"))
    plt.clf()
    my_draw(G, node_color='white') 
    # print(ax)
    my_draw(H, node_color='green')
    plt.savefig(path.join(FIGURES_DIR, name+"_2"))
    
    return H

if __name__ == "__main__":
    try:
        name = argv[1]
        type = argv[2]
    except:
        raise("Usage: {} name type".format(argv[0]))
    links_as_nodes_viewer(name, type)
