from os import makedirs, popen
from itertools import product


NETWORKS = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
VOLUME = ["100E9", "200E9"]    
TE_METHODS = ["-ecmp", "-mcf"]    
arg_dir = "scripts/TDSC/args/crossfire" 
arg_file = f"{arg_dir}/run_all.txt"
makedirs(arg_dir, exist_ok=True)
with open(arg_file, 'w') as fob:
    for network, attack_string, te_method in product(NETWORKS, VOLUME, TE_METHODS):     
        traffic_file    = f"data/archive/traffic/traffic-05-16-2022/{network}_crossfire_every_node_{attack_string}.txt"
        iterations      = int(popen(f'wc -l < {traffic_file}').read())
        for i in range(iterations):         
            fob.write(f"scripts/TDSC/crossfire_consumer.py " + " ".join(str(arg) for arg in (network, te_method, attack_string, i, iterations)) + "\n")

print(f"wrote: {arg_file}")
