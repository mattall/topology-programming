import networkx as nx
import matplotlib.pyplot as plt
''' 
G1

0 --- 3 ---\         /-- 8  --- 11 
|  X  |     \       /    |   X  |
1 --- 4 --- 6 ---- 7 --- 9  --- 12
|  X  |     /       \    |   X  |
2 --- 5 ---/         \-- 10 --- 13
'''
G1 = nx.Graph()
nodes = range(14)
G1.add_nodes_from(nodes)


edges1 = [(0, 3), (0, 4), (0,1), (1, 3), (1, 4), (2, 5), (2, 4), (2,1), (1,5), (3, 4), (3, 6), (4, 5),
         (4, 6), (5,6), (6, 7),  (7, 8), (7, 9), (7, 10), (8, 9), (9, 10),
         (8, 11), (9, 12), (10, 13), (11, 12), (12, 13), (8, 12), (9, 11), (9, 13), (10, 12)]
G1.add_edges_from(edges1)

# nx.draw(G1, with_labels=True)
# plt.show()

'''
G2

0 --- 3 ---\         /-- 8  --- 11 
            \       /    
1 --- 4 --- 6 ---- 7 --- 9  --- 12
            /       \     
2 --- 5 ---/         \-- 10 --- 13
'''


G2 = nx.Graph()
nodes = range(14)
G2.add_nodes_from(nodes)
edges2 = [(0, 3), (1, 4), (2, 5), (3, 6),
         (4, 6), (5, 6), (6, 7),  (7, 8), (7, 9), (7, 10),
         (8, 11), (9, 12), (10, 13)]
G2.add_edges_from(edges2)
plt.clf()
# Draw the graph
# nx.draw(G2, with_labels=True)
# plt.show()

S1 = set(edges1)
S2 = set(edges2)

overlap_sim = len(S1.intersection(S2)) / len(S1.union(S2))

print("overlap similarity: ", overlap_sim )

TED_0 = 16

TED_1 = len(G1.edges) + len(G1.nodes)

TED_2 = len(G2.edges) + len(G2.nodes)

ProTo_sim = 1 - (TED_1 / (TED_1 + TED_2))

print("ProTo/EqualNet Similarity: ", ProTo_sim)

