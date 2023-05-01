import json
from os import system

with open("notes_v2.json", 'r') as fob:
    experiment = json.load(fob)

def copy(source, target):
    print("cp {} {}".format(source, target))
    system("cp {} {}".format(source, target))

for network in experiment:
    for item in experiment[network]:
        if "GML" in item:
            filename = experiment[network][item].split('/')[-1]
            copy(experiment[network][item], "/home/matt/ripple/simulator/topologies/{}/{}".format(network, filename))
