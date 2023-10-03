import networkx as nx
import json
import numpy as np
from sklearn.cluster import AffinityPropagation
import distance

def get_clusters(strings:list):
    strings = np.asarray(strings) #So that indexing with a list will work
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in strings] for w2 in strings])
    lev_similarity = np.array(lev_similarity, dtype=float)
    affprop = AffinityPropagation(affinity="precomputed", damping=.75)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = strings[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(strings[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = "\n\t ".join(cluster)
        print(" - *%s:*\n\t %s\n" % (exemplar, cluster_str))
    
    return cluster

json_files = [
    "/home/m/src/topology-programming/data/NOC/sox.edu.iu.grnoc.routerproxy/SIDCO_Nashville-TN/showint-update.json",
    "/home/m/src/topology-programming/data/NOC/sox.edu.iu.grnoc.routerproxy/CODAC_Atlanta_GA/showint-update.json",
    "/home/m/src/topology-programming/data/NOC/sox.edu.iu.grnoc.routerproxy/365DC_Nashville-TN/showint-update.json",
    "/home/m/src/topology-programming/data/NOC/sox.edu.iu.grnoc.routerproxy/56MAR_Atlanta_GA/showint-ae.json"
]
json_data = {}
for j in json_files:
    with open(j, 'r') as fob:        
        this_data = json.load(fob)
        name = this_data["name"]
        json_data[name] = this_data


G = nx.MultiGraph()
G.add_nodes_from(json_data.keys())

for rtr_key in G.nodes():
    print(rtr_key)   
    descriptions = [json_data[rtr_key]["physical_interfaces"][interface]["description"].strip() for interface in json_data[rtr_key]["physical_interfaces"]]
    # descriptions = [json_data[rtr_key]["description"] for rtr_key in json_data.keys()]
    clusters = get_clusters(descriptions)

