"""
def print_g(G, pos, node_color_map, folder, start_clean=False, title=None):
    #
    # DEPRICATED --- Moved into class FiberGraph as method
    #
    # Params:
    # G: graph
    # pos: position of nodes in graph (dictionary: node_index -> (x,y))
    # node_color_map: list of colors of nodes
    # start_clean: True values deletes all previous files and folders from pwd/figures/
    # title: str to annotate graph.
    if title is not None:
        plt.title(title)

    edge_color = [G[u][v]['color'] for u,v in G.edges()]
    nx.draw(G, pos, node_color=node_color_map, edge_color=edge_color)

    if start_clean:
        delete_dir(path.join(SCRIPT_HOME, folder))

    makedirs(path.join(SCRIPT_HOME, folder), exist_ok=True)
    fig_num = len(listdir(path.join(SCRIPT_HOME, folder)))
    # print("title: \t", title)
    # print("start clean: ", start_clean)
    # print("fig num: ", fig_num)

    plt.savefig( path.join(SCRIPT_HOME, folder, str(fig_num)))
    plt.clf()

def random_graph(n):
    #
    # DEPRICATED --- Moved into class FiberGraph as method
    #
    # n: Number of nodes in the graph
    # Generates a random graph on n nodes. 
    # Gives nodes a health attribute, initially set to up. 
    # returns graph G, node_color_map, and position for nodes. 
    pos = []
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, health='up')
    
    for node_s in G:
        for node_t in G:
            # generate edges by flipping a coin. 
            if random.random() < 0.2:
                if node_s is not node_t:
                    G.add_edge(node_s, node_t, color='black')
    
    node_color_map = ["green"] * n
    initial_pos = {} 
    for i in range(n):
        initial_pos[i] = (1-(i/n), i/n)

    pos = nx.spring_layout(G, pos = initial_pos)        
            
    return G, pos, node_color_map    

def update_health(t, G, pos, node_color_map, folder, alpha=0.8, beta=0.1, start_clean=False):    
    #
    # DEPRICATED --- Moved into class FiberGraph as method
    #
    # health is gauged in two phases. 
    # First, everynode has aplha probability of being
    # down. Then, every node has apha * N change of being down,
    # Where N is the number of down neighbors.

    # set all edges to black
    for u,v in G.edges():
        G[u][v]['color'] = 'black'

    # Find contageons 
    for node in G:
        if random.random() < alpha:
            G.nodes()[node]['health'] = 'down'
            node_color_map[node] = 'red'

    print_g(G, pos, node_color_map, folder, start_clean=start_clean, title="Contagion\nt = {}    alpha = {}    beta = {}".format(t, alpha, beta))   

    # Spread contageon
    for node in G:
        # count the down neighbors of node
        down_neighbors = []
        for neighbor in G[node]:
            if G.nodes()[neighbor]['health'] == 'down':
                down_neighbors.append(neighbor)
        
        # update node's health according to the health of the neighbors
        if random.random() < alpha * len(down_neighbors):
            G.nodes()[node]['health'] = 'down'
            node_color_map[node] = 'red'
            for dn in down_neighbors:
                G[node][dn]['color'] = 'red'
         
    print_g(G, pos, node_color_map, folder, title="Spread\nt = {}    alpha = {}    beta = {}".format(t, alpha, beta))   

    # set all edges to black
    for u,v in G.edges():
        G[u][v]['color'] = 'black'

    # Spontanious recovery
    for node in G:
        if G.nodes()[node]['health'] == 'down':
            if random.random() < beta:
                G.nodes()[node]['health'] = 'up'
                node_color_map[node] = 'green'
    print_g(G, pos, node_color_map, folder, title="Recovery\nt = {}    alpha = {}    beta = {}".format(t, alpha, beta))   

def delete_dir(d):
    # Delete all files of directory d, and d itself.
    # If the directory doesn't exist nothing happens. 

    try: 
        contents = listdir(d)
        for c in contents:
            remove(path.join(d, c))
        rmdir(d)

    except FileNotFoundError:
        pass
      
def export_vid(folder): 
    chdir(folder)
    movie_name = path.basename(folder)
    system("ffmpeg -r 1 -i %00d.png -vcodec mpeg4 -y {}.mp4".format(movie_name))
    chdir(SCRIPT_HOME)

"""
