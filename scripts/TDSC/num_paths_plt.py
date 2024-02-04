import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from os import stat
from onset.utilities import plotters
data = {}

if 0:
    with open("data/paths/optimization/pre-astar/bellCanada_original_tunnel_dict.json", 'r') as fob: 
        data['Original Paths'] = json.load(fob)

    with open("data/paths/optimization/pre-astar/bellCanada_conservative.json", 'r') as fob: 
        data['K-Shortest Paths'] = json.load(fob)

    with open("data/paths/optimization/bellCanada_max.json", 'r') as fob: 
        data['A* Paths'] = json.load(fob)

if 0:
    with open("data/paths/optimization/pre-astar/CRL_original_tunnel_dict.json", 'r') as fob: 
        data['Original Paths'] = json.load(fob)
        data['Original Paths']["size"]= stat("data/paths/optimization/pre-astar/CRL_original_tunnel_dict.json").st_size

    with open("data/paths/optimization/pre-astar/CRL_max-djikstra.json", 'r') as fob: 
        data['K-Shortest Paths'] = json.load(fob)
        data['K-Shortest Paths']['size'] = stat("data/paths/optimization/pre-astar/CRL_max-djikstra.json").st_size

    with open("data/paths/optimization/CRL_max.json", 'r') as fob: 
        data['A* Paths'] = json.load(fob)
        data['A* Paths']['size'] = stat("data/paths/optimization/CRL_max.json").st_size

    for d in data: 
        # print()
        print(d)
        # print(type(data[d]))
        # print(data[d].keys())
        # print(type(data[d]["tunnels"]))
        # print(data[d]["tunnels"].keys())
        data[d]['total'] = 0
        for s in data[d]["tunnels"]: 
            for t in data[d]["tunnels"][s]:
                for p in data[d]["tunnels"][s][t]:
                    data[d]['total'] += 1


fig, ax = plt.subplots(figsize=(8,6))

xs = ["Original", "Original", "K-Shortest", "K-Shortest", "A*", "A*"]
hue = ["Total Paths", "Size (B)"] * 3
# ys = [data['Original Paths']['total'], data['Original Paths']["size"], 
#       data['K-Shortest Paths']['total'], data['K-Shortest Paths']['size'], 
#       data['A* Paths']['total'], data['A* Paths']['size'] ]
ys = [231,7825,2779000,325585282,12616,1362418,]


g = sns.barplot(x=xs, y=ys, hue=hue, edgecolor=".2")
g.set_yscale("log")


hatches = ["X"] * 3 + ["o"] * 3
for hatch, patch in zip(hatches, g.patches):
    patch.set_hatch(hatch)

ax.legend(loc='best')

# for i, bar in enumerate(g.patches):
#     # print(thisbar)
#     # Set a different hatch for each bar
#     # first set
#     print(i, bar)
#     if i < len(xs):
#         bar.set_hatch("X")
#     else:
#         bar.set_hatch("/")    

# the non-logarithmic labels you want
# ticks = [1, 10, 100, 1000]
# g.set_yticks(ticks)
# g.set_yticklabels(ticks)

_ = g.set(xlabel="", ylabel="")

plt.savefig("paths.pdf")
