# convert paths to JSON file.

"""
{
    "paths":{
        "path{}".foramt(id): {
            "dst": node_id,
            "src": node_id,
            "nhop": int, 
            "hops": [
                node_id,
                ...
            ]
            }
        },
        ...
}
"""
import json
from os import makedirs, path

from onset.utilities.graph_utils import convert_paths_onset_to_json

if __name__ == "__main__":
    if 1:  # Reroute Paths
        convert_paths_onset_to_json(
            "/home/matt/network_stability_sim/data/results.old/surfNet_add_circuit_heuristic_10/9_20/paths/ecmp_0",
            "/home/matt/ripple/simulator/topologies/surfNet/9_20_paths.json",
        )


