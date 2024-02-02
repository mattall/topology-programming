from os import makedirs
from itertools import product

def main():
    NETWORKS = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
    VOLUME = ["100", "150", "200"]
    N_TARGETS = [1, 2, 3, 4, 5]
    TE_METHODS = ["-ecmp", "-mcf"]
    CANDIDATE_SET = ["max"]
            
    makedirs("scripts/TDSC/args/multi-attack", exist_ok=True)
    with open("scripts/TDSC/args/multi-attack/run_all.txt", 'w') as fob:
        for network, n_targets, te_method, volume_per_target, candidate_set in product(NETWORKS, N_TARGETS, TE_METHODS, VOLUME, CANDIDATE_SET):        
            fob.write(f"scripts/TDSC/link_x_cap_worker.py " + " ".join(str(arg) for arg in (network, n_targets, te_method, volume_per_target, candidate_set)) + "\n")
    print("wrote: ", "scripts/TDSC/args/multi-attack/run_all.txt")
    with open("scripts/TDSC/args/multi-attack/prep_json_paths.txt", 'w') as fob:
        for network in NETWORKS:        
            n_targets, te_method, volume_per_target, candidate_set = max(N_TARGETS), "-ecmp", max(VOLUME), "max"
            fob.write(f"scripts/TDSC/link_x_cap_worker.py " + " ".join(str(arg) for arg in (network, n_targets, te_method, volume_per_target, candidate_set)) + "\n")
    print("wrote: ", "scripts/TDSC/args/multi-attack/prep_json_paths.txt")

if __name__ == "__main__":
    main()