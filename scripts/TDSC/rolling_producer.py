from os import makedirs, popen
from itertools import product


NETWORKS = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
TE_METHODS = ["-ecmp", "-mcf"]
TP_METHODS = ["onset_v3", "baseline"]

experiment = "rolling"
arg_dir = "scripts/TDSC/args/rolling" 
arg_file = f"{arg_dir}/run_all.txt"

makedirs(arg_dir, exist_ok=True)
with open(arg_file, 'w') as fob:
    for network, te_method, tp_method in product(NETWORKS, TE_METHODS, TP_METHODS):        
        fob.write(f"scripts/TDSC/{experiment}_consumer.py " + " ".join(str(arg) for arg in (network, te_method, tp_method)) + "\n")

print(f"wrote: {arg_file}")
