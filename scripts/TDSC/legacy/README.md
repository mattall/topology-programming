# ShakeRoute_eval.py

This script used the `net_sim.py` from the parent directory. 


The intended flow of the program is as follows. 

### 1. Give a network name (assumes this network is in `./data/graphs/gml/`).

- The program loads a network topology from a `.gml` file and induces every possible fiber cut with at least `k` links. 

### 2. Load the graph and call `enumerate_cut_scenarios(G:nx.Graph, k:int, out_dir:str, net_name:str)`

- The results are stored in `../data/graphs/shakeroute/{TOPOLOGY}/` where `{TOPOLOGY}` is the name of the network. 

- Each fiber cut is saved as its own `.gml` file where the name is coded as `{TOPOLOGY}_{INDEX}_{TOTAL_EDGES_CUT}_edges.gml`.

### 3. Call `get_performance_after_cuts(network:str, cut_dir:str, result_file:str)`

- The program calls instances a simulation from `net_sim.py` and evaluates the performance under each of the cut scenarios.

- These results are saved to `./data/results/{TOPOLOGY}_shakeroute.csv`

4. Call `enumerate_alt_paths_from_scenarios(G:nx.Graph, net_name:str)`.

- For each fiber cut scenario, the program finds an alternate path in the physical network topology if one exists. 

- The the length of these paths, their number, and the number of cuts for which no alternate paths exists are pickled in a dictionary. 