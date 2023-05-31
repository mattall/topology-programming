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
    if 0:  # Base Paths
        convert_paths_onset_to_json(
            "/home/matt/network_stability_sim/data/results/sprint_add_circuit_heuristic_10/__0/paths/ecmp_0",
            "/home/matt/ripple/simulator/topologies/sprint/path.json",
        )

    if 1:  # Reroute Paths
        convert_paths_onset_to_json(
            "/home/matt/network_stability_sim/data/results.old/surfNet_add_circuit_heuristic_10/9_20/paths/ecmp_0",
            "/home/matt/ripple/simulator/topologies/surfNet/9_20_paths.json",
        )

    if 0:
        with open("/home/matt/network_stability_sim/notes.json", "r") as fob:
            config = json.load(fob)

        for network in config:
            makedirs(
                path.join("/home/matt/ripple/simulator/topologies", network),
                exist_ok=True,
            )
            # txt_path = config[network]["BasePathsONSET"]
            # json_path = config[network]["BasePathsJSON"]
            # convert_paths_onset_to_json(txt_path, json_path)
            txt_path = config[network]["AdaptedPathsONSET"]
            json_path = config[network]["AdaptedPathsJSON"]
            print(json_path)
            convert_paths_onset_to_json(txt_path, json_path)

