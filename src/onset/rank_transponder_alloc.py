# read file formatted like below to create transponder 
# allocation files for each network at nodes where rank
# is greater than 'n' for all 'n' up to the make rank'd link
# of that netwrok (skip files where no link has a given rank)
# 
# Network,Attacks,Link,Rank,Percentile
# sprint,10,(s1,s3),1,10.0
# sprint,10,(s1,s8),1,10.0
# 
# Alpwolf uses 0-indexed nodes with no 's', so first convert
# them to the right format (strip the s and reduce the 
# numeric potion by 1)
# 
# Result files go to .data/txp/[Network]_[n].txt 
# 
# Result is formatted with lines, [node_id], [x] 
# where [x] is the number of fallow transpnders at the node
# 
# Default [x] is 2.

import networkx as nx
from collections import defaultdict

def clean(node:str) -> str:
    # expects string like '(s45' or 's45)'
    # returns '44'
    return str(int(node.replace(")","").replace("(","").replace("s","")) - 1)

fallow_txp = 2
link_rank_file = "/home/mhall/network_stability_sim/plot/cdf_link_rank_Crossfire.csv"
network = ''
net_node_ranks = defaultdict(lambda: defaultdict(int))
with open(link_rank_file, 'r') as fob:
    fob.readline() # skip first line
    for line in fob:                
        net, _, messy_source, messy_target, rank, _ = line.strip().split(',')            
        rank = int(rank)
        source = clean(messy_source)
        target = clean(messy_target)
        net_node_ranks[net][source] = max(rank, net_node_ranks[net][source])
        net_node_ranks[net][target] = max(rank, net_node_ranks[net][target])

net_n_cost = defaultdict(lambda: defaultdict(int))

for net, node_rank in net_node_ranks.items():
    max_rank_node = max(node_rank, key=node_rank.get)
    max_rank = node_rank[max_rank_node]
    for n in range(max_rank + 1):
        if n not in node_rank.values(): continue            
        with open(f"data/txp/{net}_{n}.txt", 'w') as fob:             
            for node, rank in node_rank.items():
                if rank >= n:
                    net_n_cost[net][n] += fallow_txp
                    fob.write(f"\t{node},{fallow_txp}\n")
                else:
                    fob.write(f"\t{node},0\n")
                    pass
            
with open("cost_by_rank.csv", 'w') as fob:
    for net, n_dict, in net_n_cost.items():
        for n, cost in n_dict.items():
            fob.write(f"{net},{n},{cost}\n")

    